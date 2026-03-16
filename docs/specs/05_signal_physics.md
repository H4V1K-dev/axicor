# 05. Физика Сигнала и I/O (Signal Physics)

> Часть архитектуры [Genesis](../../README.md). Полный путь сигнала: вход → распространение → выход.

---

## 1. Физика Сигнала (Signal Physics)

Движение сигнала и реакция синапсов реализуются через целочисленную арифметику индексов. Логика оптимизирована для минимизации чтений глобальной памяти (Global Memory Reads).

### 1.0. Пайплайн дневного тика (Day Phase Pipeline)

Каждый такт выполняется строго последовательная цепочка CUDA kernels. Порядок критичен: каждый kernel зависит от результатов предыдущего.

```mermaid
sequenceDiagram
    participant Host
    participant GPU as GPU / CUDA Stream

    Host->>GPU: 1. InjectInputs<br/>(управление виртуальными аксонами)
    GPU->>GPU: Читает input_bitmask<br/>Пишет axon_heads[virtual_axons] = 0
    GPU-->>GPU: Signal birth: virtual spikes

    GPU->>GPU: 2. ApplySpikeBatch<br/>(сетевые спайки)
    GPU->>GPU: Читает ghost_indices из соседних шардов<br/>Пишет axon_heads[ghost_axons] = 0
    GPU-->>GPU: Signal birth: network spikes

    GPU->>GPU: 3. PropagateAxons<br/>(распространение всех сигналов)
    GPU->>GPU: Читает axon_heads (все)<br/>Вычисляет head += v_seg<br/>Пишет обновлённые axon_heads
    GPU-->>GPU: Сигналы движутся по аксонам

    GPU->>GPU: 4. UpdateNeurons (GLIF kernel)<br/>(интеграция и пороговая логика)
    GPU->>GPU: Читает voltage, flags, axon_heads, dendrite struct.<br/>Выполняет: гомеостаз → рефрактерность → GLIF leak<br/>→ columnar dendrite loop → threshold check → fire
    GPU->>GPU: Записывает: voltage, flags (новые спайки!)<br/>Запускает собственные аксоны
    GPU-->>GPU: Soma fires or stays silent

    GPU->>GPU: 5. ApplyGSOP (пластичность)<br/>(STDP на основе спайков)
    GPU->>GPU: Читает flags (спайки) + dendrite_timers (контакты)<br/>Вычисляет: potentiation/depression по причинности<br/>Пишет обновлённые dendrite_weights
    GPU-->>GPU: Синапсы усилены или ослаблены

    GPU->>GPU: 6. RecordReadout (запись выходов)<br/>(сбор моторных команд)
    GPU->>GPU: Читает flags (soma spikes)<br/>Пишет output_history[current_tick] = spikes
    GPU-->>GPU: Выходы аккумулированы в батче

    Host<<-GPU: Конец тика (stream synchronization point)
```

**Порядок выполнения (Dependency Chain):**

1. **InjectInputs** → виртуальные аксоны
2. **ApplySpikeBatch** → сетевые аксоны
3. **PropagateAxons** → все аксоны движутся
4. **UpdateNeurons** → дендриты собирают → soma fires → собственные аксоны рождаются
5. **ApplyGSOP** → веса обновляются (основано на флаге спайка из UpdateNeurons)
6. **RecordReadout** → результаты записываются в буфер вывода

**Критические зависимости:**

- Шаг 4 (UpdateNeurons) **ДОЛЖЕН** запуститься перед шагом 5 (ApplyGSOP), т.к. GSOP читает информацию о спайках и контактах (dendrite timers).
- Шаг 3 (PropagateAxons) **ДОЛЖЕН** запуститься перед шагом 4 (UpdateNeurons), т.к. UpdateNeurons читает обновлённые головы аксонов.

**Синхронизация:**

- Весь день запускается как один CUDA stream (или несколько синхронизированных потоков).
- После каждого kernel'а есть soft sync (при необходимости данных из предыдущего).
- После RecordReadout - барьер: Host ждёт завершения дня перед началом ночи (если есть).

---

### 1.1. Движение: «Пулеметная очередь» (Burst Train Model)

Сигнал - не точка, а пачка импульсов (Burst), скользящая по сегментам аксона.

- **Состояние аксона:** Хранится массив из 8 голов `BurstHeads8` (32 байта, 1 кэш-линия).
- **Константа:** `v_seg` (скорость в сегментах/тик).
- **Update Loop:** Каждый тик для каждой головы $h_i$: если $h_i \neq AXON\_SENTINEL$, то $h_i \mathrel{+}= v\_seg$.
- **Active Tail:** Сегмент $i$ считается активным, если он попадает в хвост хотя бы одной из 8 голов:
  $(h_k - Tail\_Length) \le i \le h_k$

При новом спайке сомы, регистр голов сдвигается: старейшая голова `h7` стирается, на место `h0` записывается `0`. Это позволяет нейрону стрелять очередями без затирания предыдущих сигналов.

### 1.2. Инференс: Фильтрация (Inference Pipeline)

Выполняется каждый тик для каждого дендрита. Стратегия **Early Exit** для разгрузки шины памяти.

**Шаг 1. Refractory Gate (Отсечение на первом чтении):**

```cuda
// Refractory timer - 1 байт на дендрит. 32 потока × 1 байт = 32 байта (1 сектор L1)
u8 timer = refractory_timer[slot * N_padded + tid];
if (timer > 0) {
    refractory_timer[slot * N_padded + tid] = timer - 1;
    return; // Early Exit: ~90% тиков дендрит «спит» → НЕ читаем Global Memory аксона
}
```

**Шаг 2. Overlap Check (Проверка перекрытия):**

- Читаем `Axon_Head_Index` из глобального массива аксонов.
- Проверяем: попадает ли подключённый сегмент в интервал Active Tail.
- Если **да:**
  - `Soma_Voltage += Synapse_Weight` (атомарное сложение или агрегация в warp).
  - `dendrite.timer = const_mem.variants[variant_id].synapse_refractory` (значение из Constant Memory, 1 такт).

* **Защита от наложения (Burst Gating):** Если хотя бы одна из 8 голов аксона касается дендрита (Branchless OR), дендрит вбрасывает свой вес в сому и **мгновенно уходит в синаптическую рефрактерность** (`synapse_refractory_period`). Если следом летят вторая и третья головы из той же очереди, они попадут на закрытый шлюз (Шаг 1) и будут проигнорированы. Это аппаратная защита от эпилепсии при плотных очередях.

### 1.2.1. Инъекция Сетевых Спайков: ApplySpikeBatch

Ядро #2 Day Phase. Принимает массив `schedule_indices[]` из `SpikeBatch` (см. [06_distributed.md §2.5](./06_distributed.md)). Sender-Side Mapping: индексы уже готовые `u32`, никакого перевода ID.

> **Early Exit:** Если `num_spikes == 0` (нет спайков от соседей в этом тике), ядро мгновенно завершается - ноль математики, ноль транзакций памяти.

```cuda
__global__ void apply_spike_batch_kernel(u32 num_spikes,
                                         const u32* schedule_indices,
                                         u32* axon_heads,
                                         u32 total_axons) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_spikes) return;

    // O(1) routing. schedule_indices[tid] = абсолютный индекс в axon_heads[].
    // Bounds checking (index < total_axons) гарантированно выполнен на CPU
    // во время Map Phase (06_distributed.md §2.8).
    // Сброс → 0 = рождение сигнала.
    u32 ghost_id = schedule_indices[tid];
    if (ghost_id < total_axons) {
        axon_heads[ghost_id] = 0;
    }
}
```

Обучение запускается **только** когда сома генерирует спайк (`Is_Spiking == true`). Вместо анализа истории («кто стрелял 5 мс назад?»), анализируем **текущее пространственное состояние** Active Tail.

- **Принцип:** Сома стреляет → хвост аксона всё ещё касается дендрита → значит, этот аксон участвовал в возбуждении. Причинно-следственная связь через перекрытие, не через временные метки.

**Constant Memory:** `GenesisConstantMemory` (см. [07_gpu_runtime.md §1.5](./07_gpu_runtime.md)). Содержит array из 16 `VariantParameters` structs (по одному на каждый тип нейрона из blueprints). Milestone 2: при `adaptive_leak_mode != 0` используется dopamine-driven effective leak. Variant ID распаковывается из флагов как `(flags >> 4) & 0xF` (биты 4-7 = 16 типов).

### 1.3. Инференс: Пространственный GSOP и Нейромодуляция

Ядро пластичности лишено временных меток. Время вычисляется через пространственную дистанцию `dist = min(h_k - seg_idx)`.

Влияние пулеметной очереди (Burst) из 8 голов **строго не суммируется**, чтобы избежать экспоненциального взрыва весов. Применяется паттерн **Winner-Takes-All**:

**Механика Нелинейного R-STDP (Zero-cost Local Trace):**

1. **Сома генерирует спайк.** Перебираем 128 дендритов.

2. **Branchless Unroll (Поиск свежей головы):** Читаем 32 байта `BurstHeads8`. Поиск `min_dist` выполняется строго без ветвлений (`#pragma unroll`), чтобы избежать Warp Divergence. Захватывается только **самая свежая (ближняя)** голова, остальные игнорируются.

   ```cuda
   min_dist = min(min_dist, (d < len) ? d : 0xFFFFFFFF);
   ```

3. **Остывание (Cooling Shift):** Чем дальше ушла выбранная голова, тем слабее связь запоминает совпадение. Вычисляем сдвиг: `cooling_shift = dist >> 4` (каждые 16 тиков сила падает в 2 раза).

4. **Asymmetric Dopamine Modulation (D1/D2 Receptors):** В процесс интегрирован глобальный сигнал Дофамина (Dopamine), хранящийся в `__constant__` памяти. Базовая пластичность модулируется нелинейно на основе аффинности рецепторов конкретного типа нейрона:

   ```cpp
   int32_t pot_mod = (current_dopamine * p.d1_affinity) >> 7;
   int32_t dep_mod = (current_dopamine * p.d2_affinity) >> 7;

   // D1 увеличивает потенциацию при награде, D2 подавляет депрессию при награде (сохраняя связи)
   int32_t final_pot = max(0, p.gsop_potentiation + pot_mod);
   int32_t final_dep = max(0, p.gsop_depression - dep_mod);
   ```

4.5. **Burst-Dependent Plasticity (BDP):** В процесс обучения интегрирован аппаратный счетчик бурстов `burst_count` (биты `[3:1]` массива `flags`).

- **Фильтрация шума:** Если нейрон спайкует случайно (`burst_count = 1`), он получает базовую пластичность.
- **Квадратичная кристаллизация:** Если нейрон истерит по делу (`burst_count > 1`), базовая дельта умножается на количество спайков в серии (до x7).
- **Branchless Math:**

  ```cpp
  // Инкремент в UpdateNeurons (без warp divergence)
  burst_count += final_spike * (burst_count < 7);

  // Умножение в ApplyGSOP
  int32_t burst_mult = (burst_count > 0) ? burst_count : 1;
  int32_t delta_pot = (raw_pot * inertia * burst_mult) >> 7;
  ```

  Это позволяет работать в режиме "Near-Zero Economy", где фоновый дофамин отрицателен (выжигает случайные связи), а BDP-множитель пробивает эту эрозию только для плотных, причинно-следственных паттернов моторики.

5. **Двухфакторный STDP:**
   - Вычисляем ранг инерции синапса: `rank = abs(weight) >> 11`.
   - Достаём инерцию: `inertia = p.inertia_curve[rank]`.
   - Базовая дельта (Branchless причинность): `delta = is_active ? ((final_pot * inertia) >> 7) : -((final_dep * inertia) >> 7);`
   - Экспоненциальное затухание (пространственное): `delta >>= cooling_shift`.

6. **Slot Decay и Применение:**
   - Учитываем деградацию слота: `delta = (delta * slot_decay) >> 7;`
   - Безопасное применение: `weight = sign(weight) * clamp(abs(weight) + delta, 0, 32767)`.

**Инвариант Winner-Takes-All:** Только одна (ближняя) голова влияет на GSOP за тик. Остальные 7 голов в `BurstHeads8` полностью игнорируются. Это предотвращает суммирование потенциаций и гарантирует стабильность пластичности при высокочастотных всплесках.

**Инвариант Branchless:** Ноль FPU-операций, ноль условных ветвлений. Экспоненциальная кривая биологического STDP аппроксимируется через битовые сдвиги (`>>`). Компилятор становит `#pragma unroll(8)` над циклом поиска минимума для полной развёртки на железе.

### 1.4. Zero-Cost Оптимизации

| Паттерн             | Реализация                                                                                                                       | Эффект                                                                                                      |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Early Exit**      | Если сома не спайкует (`is_spiking == 0`), поток варпа мгновенно завершает работу (`return`).                                    | В ~99% тиков GSOP не выполняет ни одной математической операции и не трогает шину памяти.                   |
| **Branchless Math** | Вычисление `is_active` через целочисленную дистанцию `head - seg_idx < propagation_length`. Нет условных переходов - только ALU. | Нет дивергенции варпов (Warp Divergence). Все 32 потока в варпе выполняют один путь.                        |
| **Integer Physics** | Веса хранятся как `i16`, изменения применяются через целочисленное сложение/вычитание с saturate clamp `±32767`.                 | 0% Float операций. Максимальная плотность целочисленных ALU. Нет FPU конвейеров - ALU выполняют всю работу. |

**Результат:** Kernel ApplyGSOP при спайке - это одна загрузка флага, одна проверка бита, затем цикл по 128 слотам с целочисленными чтениями и сложениями. Никакие деревья, никакие приоритетные очереди, никакие кольцевые буферы. Чистая Data-Oriented Design.

### 1.5. Главный Тик: UpdateNeurons (GLIF Kernel)

Ядро, которое собирает всю физику в один проход: GLIF leak, гомеостаз, Early Exit, суммация дендритов, threshold check, fire/reset. Параметры читаются из `GenesisConstantMemory` (см. [07_gpu_runtime.md §1.5](./07_gpu_runtime.md)).

**Milestone 2 — Dopamine-Driven Adaptive Leak:** при `adaptive_leak_mode != 0` и `leak_min < leak_max`:

```text
leak_mod = (dopamine * dopamine_leak_gain) >> 7
effective_leak = clamp(base_leak_rate + leak_mod, leak_min, leak_max)
leak_used = max(effective_leak, 1)
```

Эффект виден в телеметрии через `ready_dopamine` и изменение spike rate при варьировании dopamine.

```cuda
__constant__ GenesisConstantMemory const_mem;

__global__ void update_neurons_kernel(
    u32 padded_n,           // Padded neuron count
    i32* voltage,           // Membrane voltage
    i32* threshold_offset,  // Homeostasis offset
    u8* refractory_timer,   // Soma refractory countdown
    u8* flags,              // [31..8] reserved | [7..4] variant | [0] is_spiking
    const u32* soma_to_axon,// Soma → Axon mapping
    const u32* dendrite_targets,  // Packed: [31..10] Axon_ID | [9..0] Segment
    i16* dendrite_weights,
    u8* dendrite_timers,
    u32* axon_heads
) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= padded_n) return;

    // 1. Распаковка типа + загрузка параметров (1 такт L1)
    u8 f = flags[tid];
    u8 type_mask = f >> 4;
    u8 variant = type_mask & 0xF;   // Биты 4-7 = Variant (16 типов)
    VariantParameters p = const_mem.variants[variant];

    // 2. Гомеостаз (Soft Limit) - выполняется ВСЕГДА, даже когда сома спит
    i32 t_off = threshold_offset[tid];
    i32 decayed = t_off - p.homeostasis_decay;
    t_off = decayed & ~(decayed >> 31);       // Branchless max(0, val)

    // 3. Рефрактерность сомы - Early Exit (~90% потоков)
    u8 ref_timer = refractory_timer[tid];
    if (ref_timer > 0) {
        refractory_timer[tid] = ref_timer - 1;
        threshold_offset[tid] = t_off;
        flags[tid] = f & ~0x1;
        return;
    }

    // 4. GLIF: Утечка заряда (Branchless clamp ≥ rest)
    i32 v = voltage[tid];
    i32 leaked = v - p.leak;
    i32 diff = leaked - p.rest_potential;
    v = p.rest_potential + (diff & ~(diff >> 31));  // max(rest, leaked)

    // 5. Columnar Loop: 128 дендритных слотов (Coalesced Access)
    for (int slot = 0; slot < 128; ++slot) {
        u32 col_idx = slot * padded_n + tid;

        // 5a. Refractory Gate (Early Exit - экономим 3 global reads)
        u8 d_timer = dendrite_timers[col_idx];
        if (d_timer > 0) {
            dendrite_timers[col_idx] = d_timer - 1;
            continue;  // ~90% тиков: дендрит спит → skip target/head/weight
        }

        // 5b. Пустой слот - BREAK, не continue!
        // Инвариант: после Night Phase (Baking §4.2 Columnar Defrag) все пустые
        // слоты (target=0) гарантированно в хвосте массива.
        u32 target_packed = dendrite_targets[col_idx];
        if (target_packed == 0) break;

        // 5c. Branchless Active Tail Check (8 Heads)
        u32 axon_id = target_packed >> 10;
        u32 seg_idx = target_packed & 0x3FF;
        BurstHeads8 h = axon_heads[axon_id]; // 32-byte coalesced read

        u32 prop = p.signal_propagation_length;
        bool hit = ((h.h0 - seg_idx) < prop) |
                   ((h.h1 - seg_idx) < prop) |
                   ((h.h2 - seg_idx) < prop) |
                   ((h.h3 - seg_idx) < prop) |
                   ((h.h4 - seg_idx) < prop) |
                   ((h.h5 - seg_idx) < prop) |
                   ((h.h6 - seg_idx) < prop) |
                   ((h.h7 - seg_idx) < prop);

        if (hit) {
            // 5d. Voltage accumulation
            i16 w = dendrite_weights[col_idx];
            v += (i32)w;
            dendrite_timers[col_idx] = p.synapse_refractory_period;
        }
    }

    // 6. Threshold Check, DDS Heartbeat & Fire (Branchless)
    i32 eff_threshold = p.threshold + t_off;
    i32 is_glif_spiking = (v >= eff_threshold) ? 1 : 0;

    // DDS Phase Accumulator (§4 neuron_model.md)
    // Пространственное рассеивание фазы: tid * 104729 (простое число)
    u32 phase = (current_tick * p.heartbeat_m + tid * 104729) & 0xFFFF;
    i32 is_heartbeat = (phase < p.heartbeat_m) ? 1 : 0;

    // Итоговый спайк (ГЛИФ ИЛИ Heartbeat)
    i32 final_spike = is_glif_spiking | is_heartbeat;

    // Мембрана и гомеостаз сбрасываются ТОЛЬКО от GLIF-спайка (Heartbeat их не трогает)
    v     = is_glif_spiking * p.rest_potential + (1 - is_glif_spiking) * v;
    ref_timer = is_glif_spiking * p.refractory_period;
    t_off += is_glif_spiking * p.homeostasis_penalty;

    // Флаг активности устанавливается от ЛЮБОГО спайка (нужно для GSOP)
    f     = (f & 0xFE) | (u8)final_spike;

    // 7. Сдвиг голов аксона при спайке (Burst Shift)
    if (final_spike) {
        u32 my_axon = soma_to_axon[tid];
        if (my_axon != 0xFFFFFFFF) {
            BurstHeads8 h = axon_heads[my_axon];
            h.h7 = h.h6; h.h6 = h.h5; h.h5 = h.h4; h.h4 = h.h3;
            h.h3 = h.h2; h.h2 = h.h1; h.h1 = h.h0; h.h0 = 0;
            axon_heads[my_axon] = h; // 32-byte coalesced write
        }
    }

    // 8. Запись в VRAM
    voltage[tid] = v;
    threshold_offset[tid] = t_off;
    refractory_timer[tid] = ref_timer;
    flags[tid] = f;
}
```

### 1.6. Пропагация Сигнала: PropagateAxons

Безусловный сдвиг **всех** аксонов (Local + Ghost + Virtual). Количество аксонов `A ≠ N` (сом). Запускается каждый тик **до** `UpdateNeurons`.

**Sentinel:** `AXON_SENTINEL = 0x80000000`. При инициализации и после Night Phase все неактивные аксоны инициализируются этим значением. `dist = 0x80000000 - seg_idx` = огромное число → `is_active = false`. Overflow до 0 произойдёт через ~59.6 часов.

> **⚠️ Sentinel Refresh:** Для зон с `night_interval_ticks = 0` (или редким сном) host каждые ~50 часов сбрасывает переполненные `axon_heads` обратно в `AXON_SENTINEL`. Подробности: [07_gpu_runtime.md §2.2](./07_gpu_runtime.md).

```cuda
#define AXON_SENTINEL 0x80000000u

__global__ void propagate_axons_kernel(u32 total_axons, u32* axon_heads, u32 v_seg) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_axons) return;

    // 100% Coalesced Access. 1 такт IADD. 0 ветвлений.
    u32 head = axon_heads[idx];
    if (head != AXON_SENTINEL) {
        axon_heads[idx] = head + v_seg;
    }
}
```

- **Рождение сигнала = сброс `axon_heads[id] = 0`:**
  - Локальный спайк: `UpdateNeurons` (шаг 7, predicated store)
  - Сетевой спайк: `ApplySpikeBatch` (Ghost аксоны)
  - Внешний стимул: `InjectInputs` (виртуальные аксоны)
- **1 спайк в полёте:** `signal_propagation_length < soma_refractory_period` → первый поезд успевает доехать до конца до рождения второго.

---

### 1.6. Безопасность Сигналов при Обслуживании (Sentinel Refresh Safety)

Фаза «Ночь» и регламентные очистки **никогда не прерывают активные сигналы**, циркулирующие в сети.

#### 1.6.1. Механика защиты SENTINEL_DANGER_THRESHOLD

При обслуживании массива `axon_heads` применяется фильтр, предотвращающий стирание активных сигналов:

- **Фильтр Сохранения:** Если значение `axon_heads[id]` меньше порога `SENTINEL_DANGER_THRESHOLD` (например, `0x70000000`), это означает, что спайк произошёл недавно и сигнал физически находится внутри проводящей части аксона (Active Tail). Такое значение **не изменяется**.

- **Инвариант Причинности:** Только "давно остывшие" аксоны, чьи головы превысили порог, могут быть безопасно возвращены в `AXON_SENTINEL` для предотвращения целочисленного переполнения `u32` [1, 4].

- **Результат:** Краткосрочная память (Active Tails) и текущий контекст распространения сигналов бесшумно переживают фазу Maintenance, не требуя синхронизации с GPU-циклом.

**Пример:**

```cpp
#define SENTINEL_DANGER_THRESHOLD 0x70000000u

void refresh_axon_sentinels(u32* axon_heads, u32 count) {
    for (u32 i = 0; i < count; ++i) {
        u32 head = axon_heads[i];
        // Только холодные аксоны могут быть обнулены
        if (head > SENTINEL_DANGER_THRESHOLD) {
            axon_heads[i] = AXON_SENTINEL;
        }
        // Активные сигналы (head <= threshold) остаются нетронутыми
    }
}
```

### 1.7. Warp-Aggregated Telemetry (Zero Atomics)

Извлечение IDs сработавших нейронов (для IDE и метрик) традиционно требует использования глобальных атомарных операций (`atomicAdd`). Если 100,000 нейронов стреляют одновременно, 100,000 потоков блокируют шину L2 кэша, пытаясь инкрементировать один счётчик `out_count`.

В Genesis применяется паттерн **Warp-Aggregated Atomics**, переносящий синхронизацию на аппаратный уровень варпа:

1. **Ballot Sync:** Каждый поток варпа выставляет свой бит спайка (`is_spiking`) через аппаратный регистр `__ballot_sync(0xFFFFFFFF, is_spiking)`.

2. **Population Count:** Лидер варпа (lane 0) считает общее количество спайков в варпе с помощью инструкции `__popc(active_mask)`.

3. **Single Atomic:** Только лидер варпа делает единственный `atomicAdd` в глобальную память, получая базовый `offset` для всего варпа.

4. **Shuffle Sync:** Лидер раздаёт `offset` остальным потокам через `__shfl_sync`.

5. **Write:** Потоки параллельно записывают свои ID по вычисленным смещениям.

```cuda
// genesis-compute/src/cuda/telemetry.cu

__global__ void record_spikes_aggregated(
    const u8* soma_flags,           // [padded_n]
    u32* out_spike_ids,             // (ring buffer)
    u32* out_count,                 // Shared atomic counter
    u32 padded_n
) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 lane = threadIdx.x % 32;

    // Step 1: Ballot - каждый поток выставляет бит если спайкнул
    bool is_spiking = (tid < padded_n) && (soma_flags[tid] & 0x80);
    u32 active_mask = __ballot_sync(0xFFFFFFFF, is_spiking);

    // Step 2: Лидер варпа (lane 0) считает количество спайков в варпе
    u32 warp_pop = (lane == 0) ? __popc(active_mask) : 0;

    // Step 3: Лидер делает ОДИН atomicAdd за весь варп
    u32 warp_offset = 0;
    if (lane == 0) {
        warp_offset = atomicAdd(out_count, warp_pop);
    }

    // Step 4: Раздать offset всем потокам варпа
    warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);

    // Step 5: Каждый поток параллельно записывает свой ID (если спайкнул)
    if (is_spiking) {
        u32 local_rank = __popc(active_mask & ((1u << lane) - 1));  // Count popcount before me
        out_spike_ids[warp_offset + local_rank] = tid;
    }
}
```

**Результат:** Количество атомарных транзакций в глобальную память сокращается в **32 раза**. Если 1024 потока (32 варпа) 99% из них спайкуют, вместо 1000+ atomicAdd операций выполняется всего 32 операции (по одной на варп). Шина памяти не блокируется, обеспечивая HFT-производительность.

---

## 2. Входные Интерфейсы (Input Interfaces)

Входы реализуются как **Виртуальные Аксоны** (Virtual Axons). Конфигурация задаётся на уровне ноды (`io.toml`), т.к. зоны могут быть распределены по разным машинам.

### 2.1. Топология: Spatial Input Maps

Входной сигнал - не сырая картинка, а результат препроцессинга (Feature Extraction). Каждый вход описывается как **Input Map** - 2D-матрица пикселей, каждый из которых представлен виртуальным аксоном.

- **Множественные маппинги:** На одну зону может приходить N маппингов разных размеров (64×64, 32×32, 128×16). Каждый - отдельный сенсорный канал.
- **target_type:** Каждый маппинг привязан к типу нейрона из `blueprints.toml` (или `"ALL"` без ограничений). Baker ищет сомы этого типа для размещения.

#### Baking: Размещение через Seeded Hash

Для каждого пикселя каждого маппинга:

1. Вычислить пространственный регион: `region_xy = pixel_xy * (zone_size / map_size)`. Все Z внутри региона.
2. Собрать кандидатов (сомы `target_type` в регионе).
3. Выбрать сому: `seed = master_seed ^ fnv1a(input_name) ^ pixel_idx`, `chosen = candidates[hash(seed) % len]`.
4. Если нужен retry (вторичная коллизия): `seed' = ... ^ (pixel_idx + attempt)`.
5. **Вырастить** виртуальный аксон с параметрами `target_type` (velocity, propagation_length, steering). Аксон прорастает от вокселя сомы, создавая пятно влияния - не точечный 1-сегментный стаб.
6. Проверки коллизий вокселей **не требуется**: хеш разводит пиксели одной матрицы в разные регионы, а разные матрицы дают разный seed (из-за имени).

#### Multi-Shard

- Каждый шард получает свой файл маппинга `{shard_name}.gxi`.
- Baker определяет шард по координатам региона.
- GhostPacket при прорастании через границу шарда (штатный механизм).
- Host при подаче батча режет матрицы и адресует каждому шарду свой кусок.

### 2.2. Протокол Входа: Single-Tick Pulse

Виртуальные аксоны управляются драйвером через bitmask. Протокол:

- **Пульс:** Драйвер устанавливает бит = 1 на **ровно 1 тик**. `InjectInputs` сбрасывает `axon_heads[id] = 0` (рождение сигнала).
- **Virtual Refractory:** Минимум `signal_propagation_length` тиков между пульсами на один канал. **Ответственность хоста** - runtime не проверяет.
- **Регуляция:** Защита от перевозбуждения лежит на принимающей стороне (дендритная рефрактерность, §1.2).

> **⚠️ Нельзя удерживать бит = 1 на несколько тиков!** InjectInputs каждый тик сбрасывает head = 0, не давая сигналу пропагироваться (голова застревает на сегменте 0-1).

### 2.3. Пирог Признаков (Feature Pyramid Batching)

Видеокадр (например, 30 fps) раскладывается препроцессором на **100+ матриц признаков** (edges, color, motion, corners, ...). Каждая матрица - отдельный слой «пирога». Батч подаётся покадрово: **1 слой = 1 тик**.

```
Кадр 30fps → препроцессор → 100+ масок
  batch[0]  = edges_mask         (тик 0)
  batch[1]  = color_mask         (тик 1)
  batch[2]  = motion_left_mask   (тик 2)
  ...
  batch[99] = corners_mask       (тик 99)
```

Преимущество: мозг получает информацию **постепенно**, а не одним взрывом. Между батчами - тишина (или следующий кадр). Размер батча и периодичность - на усмотрение внешнего интерфейса.

Host загружает весь батч одним DMA. Runtime каждый тик берёт `batch[tick_in_batch]`.

### 2.4. Драйвер Ввода: DMA Bitmask Injection + InjectInputs Kernel

**Host (CPU):** Формирует плотную битовую маску `Input_Bitmask` (1 бит = 1 виртуальный аксон). Все маппинги конкатенированы в один flat массив. Каждый тик в батче может содержать свою маску.

**Transfer:** `cudaMemcpyAsync(Input_Bitmask_GPU, Input_Bitmask_Host, size, stream)` каждый батч - микросекунды.

**Kernel InjectInputs (Ядро #1 Day Phase):**

```cuda
__global__ void inject_inputs_kernel(
    u32 *axon_heads,                    // Soma axons to inject into
    const u32 *input_bitmask,           // Dense bitmask from host
    const u32 *map_pixel_to_axon,       // Pixel → Axon ID mapping
    u32 num_pixels,                     // Pixels in this shard
    u32 pixel_offset,                   // Base offset in global pixel counting
    u32 total_num_pixels,               // Total pixels across all shards
    u32 tick_in_batch,                  // Current position in batch
    u32 input_stride,                   // Stride: only run every N ticks
    u32 max_ticks_in_batch              // Batch size
) {
    // Early exit if this tick is not an input tick (input_stride can throttle)
    if (input_stride == 0) return;
    if (tick_in_batch % input_stride != 0) return;

    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pixels) return;

    // Calculate effective tick index within bitmask data
    u32 effective_tick = tick_in_batch / input_stride;
    if (effective_tick >= max_ticks_in_batch) return;

    // Dense packing: words_per_tick = ceil(total_num_pixels / 32)
    // All shards share the same total but have offset within global pixel numbering
    u32 words_per_tick_total = (total_num_pixels + 31) / 32;
    u32 tick_offset = effective_tick * words_per_tick_total;

    // Absolute bit position in global numbering + local offset
    u32 abs_bit = pixel_offset + tid;
    u32 word_idx = abs_bit / 32;
    u32 bit_in_word = abs_bit % 32;

    // Broadcast-read: 32 threads in a warp read same u32 from mask
    u32 mask = input_bitmask[tick_offset + word_idx];
    u32 is_active = (mask >> bit_in_word) & 1;

    // Conditional write: Signal birth (axon_heads[id] = 0)
    if (is_active) {
        axon_heads[map_pixel_to_axon[pixel_offset + tid]] = 0;
    }
}
```

**Параметры:**

- `input_stride`: Если `= 2`, kernel запускается через тик (скорость ввода вдвое меньше). Позволяет контролировать частоту входных импульсов.
- `words_per_tick_total`: Плотная упаковка - каждый бит отвечает за один пиксель. На `N=64K` пикселей требуется `2K` слов u32 per tick.
- **Multi-Shard:** Каждый шард знает свой `pixel_offset` (глобальная нумерация). Маска общая для всех шардов, но каждый берёт свой кусок.

**Почему это эффективно:**

- 1 broadcast-read = 32 потока читают 1 слово (ширина пропускной способности L1).
- Запись только если `is_active = 1` (~5-10% тиков на канал).
- Нет необходимости в atomics.

---

## 3. Система Вывода (Readout Interface)

Вывод реализуется как **Direct Memory Access** к состоянию конкретных сом. Формируем временнýю матрицу активности.

### 3.1. Топология: Проекция «Победитель получает всё»

В конфиге `io.toml` определяется зона среза (Volume Slice). Строгое требование уникальности.

**Baking Phase (Compiler Tool):**

1. Зона разбивается на 2D-сетку (каналы вывода).
2. В каждой ячейке сетки ищутся все попадающие в неё сомы (вертикальный столб).
3. **Разрешение коллизий (Z-Sort):**
   - Сортируем кандидатов по высоте Z.
   - Выбираем одного с **наименьшим Z** (ближайшего к дну слоя / Z0).
   - Остальные игнорируются (они работают как часть сети, но к выходу не подключены).

**Результат:** Статический массив `Map_Channel_to_SomaID[]`. Размер = число пикселей выходной матрицы. Маппинг 1-к-1, без конфликтов.

### 3.2. Рантайм: Накопление (Batch Accumulation)

Мы не останавливаем симуляцию каждый тик ради вывода. Данные накапливаются в VRAM.

- **Буфер:** `Output_History[Batch_Size × Num_Channels]` (`u8`) - выделяется при старте.
- **Kernel RecordReadout (каждый тик, после ApplyGSOP - итоговый kernel дня):**

```cuda
__global__ void record_readout_kernel(
    const u8* flags,                    // Soma spike flags
    const u32* mapped_soma_ids,         // [channel] → soma_id mapping
    u8* output_history,                 // Output buffer
    u32 total_mapped_somas,             // Number of output channels
    u32 current_tick_in_batch,          // Current batch position
    u32 padded_n                        // VRAM padding (for bounds checking)
) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_mapped_somas) return;

    // O(1) channel → soma mapping (baked during initialization)
    u32 soma_id = mapped_soma_ids[tid];
    u8 is_spiking = 0;

    // Safety bounds check
    if (soma_id < padded_n) {
        is_spiking = flags[soma_id] & 0x01;
    }

    // Coalesced Write: threads 0..31 write consecutive u8 values
    // Stride = 32-thread groups per warp for maximum bandwidth
    output_history[current_tick_in_batch * total_mapped_somas + tid] = is_spiking;
}
```

**Почему u8, а не биты?** Bit-packing потребовал бы `atomicOr` при записи - медленно и создаёт race conditions. `u8` на канал = 32 потока варпа пишут 32 непрерывных байта в один транзакцию (максимальная пропускная способность).

**Маппинг** строго 1-к-1 (Z-Sort из §3.1). Нет конфликтов записи, нет need для atomics.

**Бакинг:** Массив `mapped_soma_ids[]` создаётся во время Baking phase (см. [09_baking_pipeline.md](./09_baking_pipeline.md)). Содержит soma IDs отсортированные по output channel. Фиксирован на весь день.

### 3.3. Синхронизация: Выгрузка (Flush)

В конце цикла `sync_batch` (вместе с обменом граничными аксонами):

1. **Transfer:** Единый `cudaMemcpy` копирует весь `Output_History` в RAM хоста.
2. **Latency:** Внешний мир получает данные с задержкой в размер батча (например, 10 мс).
   - **Биологическая валидность:** Сигнал от моторной коры до мышцы идёт 10–20 мс. Эта задержка естественна.

### 3.4. Интерпретатор: Population Coding (External Hub)

На выходе Hub получает матрицу `output_history[Time × Channels]` (`u8`: 0 или 1).

**Принцип: Population Coding (не Rate Coding / ШИМ).**

Для каждого действия выделяется **популяция** выходных нейронов. Сила действия пропорциональна количеству одновременно активных нейронов в популяции. Мозг сам учится через GSOP рекрутировать нужное количество.

| Пример          | Популяция                 | Сила              |
| --------------- | ------------------------- | ----------------- |
| Сгибание руки   | 100 нейронов → 23 активны | `23 / 100 = 0.23` |
| Разгибание руки | 100 нейронов → 87 активны | `87 / 100 = 0.87` |

**Интерпретация на Hub:**

```
// На микроконтроллере (ESP32 / Teensy / STM32):
let active = popcount(population_mask);   // 1 инструкция
let strength = active as f32 / population_size as f32;  // 0.0 .. 1.0
set_motor_output(strength);               // Прямой аналоговый выход
```

- **Мгновенный отклик:** Не нужно ждать N тиков для подсчёта частоты - `popcount` в текущем кадре даёт силу.
- **Биологичность:** В природе сила мышцы = количество рекрутированных моторных единиц.
- **Замкнутая петля:** Моторный выход → робот двинул рукой → датчики изменились → Input обновился → аксоны в L4 выстрелили.
- **Универсальность:** Аудио, моторика, любой выход - одинаковый формат (битовая маска популяции).

### 3.5. Moving Window: Управляемая Дискретность (Control FPS)

`output_history` - двумерная матрица `[sync_batch_ticks × num_channels]`. Hub **не обязан** читать весь батч одним куском.

**Протокол скользящего окна:**

```
window_size = tick_duration_us × sync_batch_ticks / (1_000_000 / target_fps)

// Пример: tick=100мкс, batch=100тиков → батч = 10 мс
// target_fps = 60 → window = 10.000мкс / 16.667мкс ≈ 6 тиков / кадр
// target_fps = 120 → window = 10.000мкс / 8.333мкс ≈ 1-2 тика / кадр
```

Hub читает `output_history` строка за строкой с шагом `window_size` и выдаёт команду приводу с частотой `target_fps`. Настраивается в `io.toml`.

> **Антагонисты (Biceps / Triceps и т.п.):** Разрешение конфликтующих сигналов - ответственность Hub-интерпретатора, не Genesis-движка. Пример: `signal = Strength(A) - Strength(B)` → ненулевой дифференциал = направление и сила. Мозг обучится через GSOP формировать корректные корреляции самостоятельно.

---

## 4. Night Phase: Обслуживание (Maintenance & Synaptic Pruning)

Периодически (раз в несколько дней или недель симуляции) запускается **Night Phase** - фаза обслуживания. Основные задачи:

1. **Synaptic Pruning:** Удаление слабых синапсов (контактов с `|weight| < threshold`).
2. **Columnar Defrag:** Переупорядочивание слотов, чтобы пустые (удалённые) синапсы скопились в хвосте, а живые - в начале.
3. **Sentinel Refresh:** Сброс переполненных `axon_heads` обратно в `AXON_SENTINEL` (если зона работает ~60+ часов без перезапуска).

### 4.1. Sort & Prune Kernel

Запускается один раз за Night: сортирует и подрезает все синапсы для всех нейронов.

```cuda
#define MAX_DENDRITE_SLOTS 128

struct alignas(8) DendriteSlot {
    u32 target_packed;   // [31..10] Axon_ID | [9..0] Segment
    i16 weight;
    u8 timer;
    u8 _pad;
};

__global__ void sort_and_prune_kernel(
    u32 padded_n,                      // Padded neuron count
    u32 *dendrite_targets,             // Columnar: [slot][neuron]
    i16 *dendrite_weights,             // Columnar: [slot][neuron]
    u8 *dendrite_timers,               // Columnar: [slot][neuron]
    i16 prune_threshold                // Minimum |weight| to keep
) {
    // 32 нейрона per block (1 warp). Shared memory = 32 × 128 × 8 = 32KB.
    __shared__ DendriteSlot smem[32][MAX_DENDRITE_SLOTS];

    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;
    bool active = (tid < padded_n);

    // 1. Coalesced Read от Global → Shared Memory
    for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
        if (active) {
            u32 idx = slot * padded_n + tid;
            smem[lane][slot].target_packed = dendrite_targets[idx];
            smem[lane][slot].weight = dendrite_weights[idx];
            smem[lane][slot].timer = dendrite_timers[idx];
        }
    }
    __syncthreads();

    if (active) {
        // 2. Sequential Insertion Sort по |weight| в убывающем порядке
        // Empty slots (target_packed == 0) собираются в конце
        for (int i = 1; i < MAX_DENDRITE_SLOTS; ++i) {
            DendriteSlot key = smem[lane][i];
            i16 key_abs = (key.weight >= 0) ? key.weight : -key.weight;
            int j = i - 1;

            while (j >= 0) {
                i16 w_j = smem[lane][j].weight;
                i16 abs_j = (w_j >= 0) ? w_j : -w_j;

                bool key_is_empty = (key.target_packed == 0);
                bool j_is_empty = (smem[lane][j].target_packed == 0);

                bool should_swap = false;
                if (j_is_empty && !key_is_empty) {
                    should_swap = true;  // Пустой slot сдвигается вниз
                } else if (!j_is_empty && !key_is_empty) {
                    if (key_abs > abs_j) {
                        should_swap = true;  // Более сильный вес → вверх
                    }
                }

                if (should_swap) {
                    smem[lane][j + 1] = smem[lane][j];
                    j = j - 1;
                } else {
                    break;
                }
            }
            smem[lane][j + 1] = key;
        }

        // 3. Pruning: уничтожить синапсы с |weight| < threshold
        for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
            i16 w = smem[lane][slot].weight;
            i16 abs_w = (w >= 0) ? w : -w;
            if (abs_w < prune_threshold) {
                smem[lane][slot].target_packed = 0;  // Пометить как удалённый
            }
        }
    }
    __syncthreads();

    // 4. Coalesced Write от Shared → Global Memory
    for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
        if (active) {
            u32 idx = slot * padded_n + tid;
            dendrite_targets[idx] = smem[lane][slot].target_packed;
            dendrite_weights[idx] = smem[lane][slot].weight;
            dendrite_timers[idx] = smem[lane][slot].timer;
        }
    }
}
```

**Инвариант после Sort & Prune:**

- Слоты 0..K-1 содержат живые синапсы, отсортированные по `|weight|` в убывающем порядке.
- Слоты K..127 содержат пустые записи (`target_packed = 0`).
- Day Phase использует это: `if (target == 0) break` → O(K) вместо O(128).

**Сложность:**

- Per-thread Insertion Sort: $O(K^2)$ в худшем случае, $O(K)$ в среднем (в начале слотов много пустых).
- Per-neuron: ~10мкс на 128 слотов (CPU Shared Memory, локальный).
- Всего блюдо за ночь: мила секунда на ~100K нейронов.

---

## Связанные документы

| Документ                                         | Что связывается                                                           |
| ------------------------------------------------ | ------------------------------------------------------------------------- |
| [01_foundations.md](./01_foundations.md)         | §1: Grundlagen нейрона (GLIF модель), Spike definition                    |
| [03_neuron_model.md](./03_neuron_model.md)       | §2: VariantParameters, threshold, refractory_period, GSOP parameters      |
| [04_connectivity.md](./04_connectivity.md)       | §1.2: Dendrite topology, synapse mapping, columnar layout                 |
| [06_distributed.md](./06_distributed.md)         | §2.5, §2.8: SpikeBatch protocol, Ghost sync, Sender-side mapping          |
| [07_gpu_runtime.md](./07_gpu_runtime.md)         | §1.5: Constant Memory structure, CUDA stream orchestration                |
| [08_ide.md](./08_ide.md)                         | Visualization: Real-time monitoring of signal propagation, spike heatmaps |
| [09_baking_pipeline.md](./09_baking_pipeline.md) | §4.2: Columnar defrag, Night phase integration, Pruning thresholds        |
| [project_structure.md](../project_structure.md)  | Overview: Role of Signal Physics in Genesis architecture                  |

---

## Changelog

| Дата       | Версия | Описание изменений                                                                                                                                                                                                                                                                                                                           |
| ---------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-02-28 | 2.1    | Синхронизация с реальным кодом (physics.cu, readout.cu, inject_inputs.cu, sort_and_prune.cu): исправлены variant bits (6-7 → 4-7), обновлены сигнатуры ApplySpikeBatch, UpdateNeurons, PropagateAxons. Добавлены полные CUDA kernel examples для RecordReadout, InjectInputs, SortAndPrune. Добавлена Mermaid диаграмма Day Pipeline (§1.0). |
| TBD        | 2.0    | Первая версия спеки (30 мес назад)                                                                                                                                                                                                                                                                                                           |
