# 06. Распределённая Архитектура (Distributed)

> Часть архитектуры [Genesis](../../README.md). Шардинг, BSP, границы, атлас-маршрутизация.

**Статус:** [MVP] = реализовано в V1, [PLANNED] = дорабатывается после V1.0 release.

---

## 1. Планарный Шардинг (Tile-Based Sharding)

### 1.1. Обоснование [MVP]

1. **Топология коры:** Кора — это «простыня» (Sheet). 2–4 мм толщины (Z) против метров ширины (X/Y). Пилить 4 мм на разные сервера — безумие. Пилить метры ширины на тайлы — необходимость.
2. **Колоночная архитектура:** Вся оптимизация памяти (Columnar Layout, см. [02_configuration.md §3](./02_configuration.md)) заточена на то, что вертикальная колонка (L1–L6) лежит в памяти рядом. Внутри колонки самый горячий трафик (L4 → L2/3 → L5 → L6). Разрезать Z = превратить локальные связи в сетевые пакеты.
3. **Атлас вместо стэка:** Для соединения V1 (затылок) с PFC (лоб) используем маршрутизацию через Атлас (§3), а не физическое вертикальное наложение.

### 1.2. Топология Соседей [MVP]

> [!NOTE]
> **[Planned: Macro-3D]** Расширить топологию соседей до 3D (6 граней).
// Сейчас шард имеет 4 соседа (X+, X-, Y+, Y-). Для изотропного макро-куба необходимо добавить `Z+ (Roof)` и `Z- (Floor)`.
// 
// Механика Z-Handovers:
// Если аксон при росте пробивает `Z_max` (потолок шарда), он не должен останавливаться или разворачиваться. 
// Он должен формировать `GhostPacket` и улетать в шард-сосед сверху (`Z+`).

Каждый шард — плоская плитка (Tile). Максимум **4 соседа:**

```
        [North (Y+)]
            │
[West (X-)]─┼─[East (X+)]
            │
        [South (Y-)]
```

Вертикальных соседей нет. Все связи по Z (L2 → L5, L4 → L6) — **всегда локальные** (Zero-Copy).

### 1.3. Ghost Connections (Ghost Axon Metadata) [MVP]

При прорастании аксона через боковую границу шарда формируется контакт (Connection). В текущей MVP реализации (V1) сохраняется только метаинформация о связи:

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GhostConnection {
    pub local_axon_id: u32,    // Индекс аксона на целевом шарде (Ghost)
    pub paired_src_soma: u32,  // Soma ID на источнике (для трассировки)
}
```

**Жизненный цикл:**
- **Baking Phase:** Baker создаёт файл `.ghosts` с массивом Connections, отсортированными по целевому каналу.
- **Runtime:** Каждая Connection описывает Ghost Axon, который уже **физически растёт** в целевом шарде (как обычный локальный аксон).
- **Sender-Side Mapping:** При спайке soma на источнике, спайк кодируется как предвычисленный `ghost_id` (из таблицы маппинга), передаётся по сети, и на целевом шарде используется прямым индексом в массив `axon_heads`.

**[MVP] Полная AxonHandover (V2):**
При включении полнофункционального роста аксонов через границы, будет расширено до:

```rust
#[repr(C, packed)]
pub struct AxonHandoverEvent {
    pub local_axon_id: u32,    // ID локального аксона-исходника
    pub entry_x: u16,          // 3D-координаты входа в новый шард
    pub entry_y: u16,
    pub vector_x: i8,          // Вектор роста (направление + скорость)
    pub vector_y: i8,
    pub vector_z: i8,
    pub type_mask: u8,         // Geo(1) | Sign(1) | Var(2)
    pub remaining_length: u16, // TTL роста (оставшиеся сегменты)
    pub entry_z: u8,           // Z-координата входа
    pub _padding: u8,
} // 16 bytes
```

**Отличие:** In V1, аксон уже вырос и зафиксирован. In V2, аксон продолжал бы расти через границы динамически.

### 1.4. Ghost Axons (Планарная Версия) [PLANNED]

> [!NOTE]
> **[Planned: Macro-3D]** Аппаратная маршрутизация спайков.
// При переходе на оптику / RDMA, `BspBarrier` должен поддерживать многоуровневый роутинг: 
// Шард -> Локальная Нода -> Top-of-Rack Switch -> Целевая Нода -> Целевой Шард.

Механика из [04_connectivity.md §1.7](./04_connectivity.md) с поправкой на 2D-границы:

- **Рождение:** При получении `AxonHandover` шард создаёт Ghost Axon от стены внутрь.
- **Цепочка:** Если Ghost дорастает до **противоположной** стены (X- → X+), экспортируется дальше.
- **Вертикальная свобода:** Ghost Axon может расти вверх/вниз по Z свободно — это локальная память, без сетевого трафика.

### 1.5. Тороидальная Топология (Periodic Boundaries) [PLANNED]

Для ядерных неслоистых структур (таламус, базальные ганглии) применяются **Периодические Граничные Условия**. Физика шара через математику плоскости.

**Принцип:** GPU по-прежнему считает плоскую сетку. Замыкание решается **конфигурацией соседей:**

- **1 шард (весь объём на одной GPU):** `Neighbor_X+ = Self`, `Neighbor_X- = Self`, `Neighbor_Y+ = Self`, `Neighbor_Y- = Self`. Аксон, вышедший за правый край, появляется на левом как Ghost Axon **того же шарда**.
- **Сетка шардов (например, 2×2):** Правый крайний шард → `Neighbor_X+ = левый крайний шард` (кольцевое замыкание).

**Ось Z:** Для слоистой коры Z жёстко ограничен (L1 сверху, L6 снизу) — замыкание **запрещено**. Для ядерных зон (где `height_pct = 1.0` и нет слоёв) — Z замыкается аналогично X/Y.

**Защита от Ouroboros (бесконечных циклов):** Поле `remaining_segments` в `AxonHandover` (§1.3) продолжает уменьшаться при каждом пересечении границы. Когда `remaining_segments == 0` — рост останавливается. Аксон физически не может замкнуться на себя, если его длина меньше периметра зоны.

**Итог:** Нет «мёртвых зон» на краях. Изотропия сохранена. Ноль нового кода в Hot Loop.

---

## 2. Протокол Синхронизации (Time Protocol) [MVP]

Строгий BSP (Bulk Synchronous Parallel) с Zero-Copy передачей сырой памяти и маппингом ID на стороне отправителя.

### 2.1. Модель: Strict BSP [MVP]

- Шард вычисляет `sync_batch_ticks` (например, 100 тиков = 10 мс) **абсолютно автономно**.
- В конце батча — жёсткий сетевой **барьер**:
  1. Flush исходящих буферов соседям.
  2. Ждём пакеты от всех `neighbors`.
  3. Нельзя начать батч N+1, пока не закрыт батч N.

### 2.2. Latency Hiding (Компенсация Задержек) [MVP]

Сетевой лаг **легализуется** как физическая задержка сигнала (аксона).

- Полученные спайки из батча N не применяются мгновенно.
- Они укладываются в **кольцевой буфер** (`Schedule`) получателя и срабатывают в батче N+1 на тике, равном их `tick_offset`.
- **Результат:** Задержка в 1 батч (10 мс) биологически естественна — сигнал от одной зоны коры до другой идёт 5–20 мс.

#### 2.2.1. HFT Log Throttling & BSP Timings

**Проблема:** Day Phase на GPU крутится 100+ раз в секунду (sync_batch_ticks = 100, 10 мс батч). Без троттлирования каждый спайк, каждый Egress-пакет, каждый Self-Heal event создаёт систему вывод в stdout → блокирует реактор.

**Решение: Atomic Throttling (Lock-Free)**

Счётчики сообщений хранятся в `AtomicUsize` (thread-safe без Mutex) и выводятся **один раз на 100 сообщений**:

```rust
pub struct HftLogger {
    dopamine_count: AtomicUsize,
    egress_packet_count: AtomicUsize,
    self_heal_event_count: AtomicUsize,
    throttle_rate: usize,  // 100
}

impl HftLogger {
    pub fn log_dopamine(&self) {
        let count = self.dopamine_count.fetch_add(1, Ordering::Relaxed);
        if count % self.throttle_rate == 0 {
            eprintln!("[Dopamine] Batch #{}, cumulative: {}", batch_id, count);
        }
    }
    
    pub fn log_egress(&self, packet_size: u32) {
        let count = self.egress_packet_count.fetch_add(1, Ordering::Relaxed);
        if count % self.throttle_rate == 0 {
            eprintln!("[Egress] {} packets sent, last size: {} bytes", count, packet_size);
        }
    }
    
    pub fn log_self_heal(&self, node_id: u32) {
        let count = self.self_heal_event_count.fetch_add(1, Ordering::Relaxed);
        if count % self.throttle_rate == 0 {
            eprintln!("[SelfHeal] Node {} recovered, cumulative: {} events", node_id, count);
        }
    }
}
```

**Ключевые инварианты:**

1. **Нулевой блокинг:** `fetch_add(Ordering::Relaxed)` — чистая CAS-операция на ALU, ~2 цикла. Без шпинделя, без семафора.
2. **Выборочный вывод:** Только каждое 100-е сообщение идёт в stderr. За 100 тиков (1 батч батча) система хранит в памяти только счётчик, затем однажды sпечатает.
3. **Истина в logs:** Настоящие счётчики растут монотонно. Пользователь видит правду («cumulative: 1247»), не лепространные фрагменты.

#### 2.2.2. BSP Synchronization Timeout

**Инвариант:** **BSP_SYNC_TIMEOUT_MS = 500 ms** (вместо 50 ms).

**Почему увеличиваем:**

- Медленная нода в кластере может лагать на 10–20 мс дополнительно (контекст, I/O, CPU scheduler jitter).
- При `sync_batch_ticks = 100` (10 мс батч) это = 1–2 батча запаса.
- Таймаут в 50 мс = 5 батчей, но первый лаг ноды → timeout → false panic.
- **500 мс = ~50 батчей запаса** → защита даже от медленных дисков и CPU thrashing.

**Фиксация в константе:**

```rust
const BSP_SYNC_TIMEOUT_MS: u64 = 500;
const MAX_BATCHES_LATENCY: u64 = BSP_SYNC_TIMEOUT_MS / BATCH_DURATION_MS;  // ~50 батчей

pub fn wait_for_neighbors(
    wait_strategy: WaitStrategy,
    start_time: Instant,
) -> Result<Vec<SpikeBatch>, BspTimeoutError> {
    loop {
        if let Some(batch) = try_recv_all_neighbors() {
            return Ok(batch);
        }
        
        if start_time.elapsed().as_millis() > BSP_SYNC_TIMEOUT_MS {
            return Err(BspTimeoutError {
                timeout_ms: BSP_SYNC_TIMEOUT_MS,
                batch_count_allowed: MAX_BATCHES_LATENCY,
            });
        }
        
        match wait_strategy {
            WaitStrategy::Aggressive => std::hint::spin_loop(),
            WaitStrategy::Balanced => std::thread::yield_now(),
            WaitStrategy::Eco => std::thread::sleep(Duration::from_millis(1)),
        }
    }
}
```

**Эффект:**

| Параметр | 50 мс | 500 мс |
|---|---|---|
| Батчей запаса | 5 | 50 |
| Защита от контекст-свичей | Низкая (одного спайка достаточно) | Высокая (система может глючить 0.5 сек) |
| Performance Impact | Ноль | Ноль (timeout — редкое событие) |

**Как чтение?** Если нода зависнет на диске > 500 мс → система обнаружит deadlock и спаникует с логом. Это лучше, чем молчаливая потеря данных.

### 2.3. Fast Path & L7 Fragmentation (V2)

UDP имеет жёсткий лимит MTU (максимальный размер полезной нагрузки `MAX_UDP_PAYLOAD = 65507`). Батчи спайков могут превышать этот лимит. Мы фрагментируем их на уровне приложения (L7) перед отправкой.

#### 2.3.1. Параметры L7 Фрагментации

- **Максимум спайков в UDP пакете:** `MAX_EVENTS_PER_PACKET = 8186`
- **Размер пакета:** 16 байт (заголовок V2) + 8186 × 8 байт (события) = 65504 байта (идеально влезает в MTU 65535).

#### 2.3.2. Бинарный Контракт V2: SpikeBatchHeaderV2 & SpikeEventV2

```rust
#[repr(C, align(8))]
pub struct SpikeBatchHeaderV2 {
    pub src_zone_hash: u32,    // Hash отправляющей зоны
    pub dst_zone_hash: u32,    // Hash принимающей зоны
    pub epoch: u32,            // Глобальный счётчик батчей (Epoch)
    pub is_last: u32,          // 0 = Чанк, 1 = Последний чанк (Heartbeat), 2 = ACK-ответ
}  // = 16 bytes

#[repr(C)]
pub struct SpikeEventV2 {
    pub ghost_id: u32,    // ПРЯМОЙ индекс в памяти принимающего шарда
    pub tick_offset: u32, // Смещение внутри батча (0..sync_batch_ticks)
}  // = 8 bytes
```

**Инвариант Heartbeat:** Даже если за батч не было ни одного спайка, маршрутизатор обязан отправить пустой пакет с `is_last = 1`. Это пробивает BSP-барьер на стороне получателя и гарантирует лассинхронизацию за счёт `epoch`.

#### 2.3.3. Zero-Copy Pipeline (Legacy MVP, работает как раньше)

**Пайплайн:** `cudaMemcpy (VRAM → RAM)` → UDP сокет (с фрагментацией) → RAM соседа → аппаратный каст сырых байтов в `&[SpikeEventV2]`.

Отправитель формирует пакеты с `SpikeBatchHeaderV2` + `SpikeEventV2[]` (до 8186 событий). Если коулических спайков > `MAX_EVENTS_PER_PACKET`, формируется несколько пакетов, все кроме последнего с `is_last = 0`, последний — `is_last = 1`.

### 2.4. Sender-Side Mapping [MVP]

Принимающий шард **не занимается маршрутизацией** внутри горячего цикла (Hot Loop).

- **Отправитель** при генерации спайка сам переводит свой локальный ID в `receiver_ghost_id` (ID Ghost Axon на целевом сервере). Таблица маппинга формируется при Handshake (создание Ghost Axon).
- **Получатель** берёт `receiver_ghost_id` из пакета и использует как **прямой `u32` индекс** для инъекции в массив. Ноль ветвлений, O(1).

### 2.5. Жизненный Цикл Ghost Axon [MVP: Рождение + Активность, PLANNED: Смерть]

4 фазы: Рождение → Активность → Пропагация → Смерть.

**Шаг 1: Рождение (Slow Path / Night Phase)**

**Шаг 1: Рождение и Абсорбция (Slow Path / Night Phase)**

1. Аксон шарда A достигает 3D-границы (X, Y или Z). Ограничения шарда пробиваются, и в очередь SHM формируется `AxonHandoverEvent`.
2. В фазу «Ночь» (внутри `run_sprouting_pass`) шард B читает входящую очередь `handovers_count`.
3. CPU шарда B выполняет **Абсорбцию Ghost Axons**: происходит O(N) сканирование диапазона `padded_n .. padded_n + total_ghosts` (с идеальной локальностью L1 кэша) для поиска свободных слотов (`tip == 0`, что означает мёртвого призрака).
4. Найдя свободный слот, шард распаковывает `entry_x/y/z` и `vector_x/y/z`, записывает их в `axon_tips_uvw` и `axon_dirs_xyz` и сбрасывает длину в 0. Новый Ghost Axon физически готов к росту и синаптогенезу.

**Шаг 2: Активность (Fast Path / Day Phase)**

1. Аксон шарда A стреляет → CPU формирует `SpikeBatch` с готовым `ghost_id` (Sender-Side Mapping).
2. Шард B получает массив `ghost_indices[]` → ядро `ApplySpikeBatch`: `axon_heads[ghost_id] = 0` за O(1).

**Шаг 3: Пропагация**

`PropagateAxons` безусловно сдвигает Ghost аксоны (`+= v_seg`). `UpdateNeurons` в шарде B проверяет Active Tail через `dist = head - seg_idx` — математика идентична локальным аксонам.

**Шаг 4: Смерть (Night Phase)**

Шард A решает убить аксон (Pruning) → отправляет `PruneAxon(ghost_id)` в шард B. В следующую Ночь шард B освобождает слот и восстанавливает `axon_heads[ghost_id] = AXON_SENTINEL` (0x80000000).

### 2.6. Slow Path: Геометрия [PLANNED]

Передача `AxonHandover` (§1.3) происходит **раз в K батчей** (например, раз в 10–50 мс). Рост аксонов — процесс медленный. Обновлять геометрию каждые 100 мкс не нужно. Это существенно разгружает канал.

### 2.7. Main Loop [MVP: Compute + Network, PLANNED: Geometry]

```python
while simulation_running:
    # 1. Compute Phase (GPU)
    for t in range(sync_batch_ticks):
        current_tick += 1
        apply_remote_spikes(current_tick)   # Из Schedule (Ring Buffer)
        physics_step()
        collect_outgoing_spikes()

    # 2. Network Phase (CPU/IO) — Барьер
    send_buffers_to_neighbors()             # Zero-Copy Flush
    incoming_data = wait_for_neighbors()    # Блокировка

    # 3. Map Phase
    schedule_spikes(incoming_data)           # → Ring Buffer
```

### 2.8. Ring Buffer Schedule [MVP]

Плоский 2D массив — никаких Priority Queue, деревьев или сортировки:

```rust
// Выделяется при старте (на CPU, затем → VRAM)
struct SpikeSchedule {
    // schedule[tick_offset][slot] = ghost_id
    buffer: Vec<Vec<u32>>,      // [sync_batch_ticks][MAX_SPIKES_PER_TICK]
    counts: Vec<u32>,           // Количество спайков в каждом тике
}
```

**Запись (Map Phase, CPU):** Спайки из `SpikeBatch` раскладываются по `tick_offset` из `SpikeEvent`:

```rust
for event in incoming_batch {
    let ghost_id = event.receiver_ghost_id as usize;
    let tick = event.tick_offset as usize;
    
    // Защита от мусорных ID (Инъекция/Баги сети/Десинхронизация)
    // Валидация происходит 1 раз на CPU, разгружая Hot Loop на GPU
    if ghost_id >= local_axons && ghost_id < (local_axons + ghost_axons) && tick < sync_batch_ticks {
        schedule.buffer[tick][schedule.counts[tick]] = ghost_id as u32;
        schedule.counts[tick] += 1;
    } else {
        // Log telemetry: dropped invalid spike (security/corruption warning)
    }
}
```

**Чтение (Day Phase, GPU):** На каждом тике `ApplySpikeBatch` читает **один слот** `schedule[current_tick_in_batch]`, запускается только для `counts[tick]` спайков.

### 2.8.1. Epoch Synchronization & Biological Amnesia

Вместо слепого пинг-понга, BSP-барьер опирается на строгий счётчик `epoch` (номер батча). Поскольку UDP не гарантирует порядок доставки, рантайм реализует два механизма выживания:

#### Biological Amnesia (Drop)

Если приходит пакет с `header.epoch < current_epoch`, он **мгновенно отбрасывается**.

```rust
fn process_spike_batch(
    packet: &SpikeBatchHeaderV2,
    current_epoch: Atomic<u32>,
) {
    let curr = current_epoch.load(Ordering::Acquire);
    
    if packet.epoch < curr {
        // Пакет "из прошлого" — биологически бессмыслен
        // Отброс: нет логирования, нет паники, просто DROP
        return;
    }
    
    // Обработать как обычно
    apply_spike_events_to_schedule(packet);
}
```

**Почему это работает:** Сигнал "из прошлого" физиологически не имеет смысла и только разрушит причинно-следственные связи GSOP (долгосрочная пластичность зависит от точного временного порядка). Отброс — это **легальное биологическое поведение** (аналог забывчивости нейрона при задержниках).

#### Self-Healing (Fast-Forward)

Если приходит пакет с `header.epoch > current_epoch`, это означает, что **наш шард завис или пропустил Heartbeat**. Шард принудительно сбрасывает свой барьер, делает прыжок во времени и синхронизируется с сетью, жертвуя потерянными локальными тиками:

```rust
fn self_heal_from_network(
    packet: &SpikeBatchHeaderV2,
    current_epoch: &AtomicU32,
) {
    let curr = current_epoch.load(Ordering::Acquire);
    
    if packet.epoch > curr {
        // Нас "отставили" в сети. Fast-forward:
        eprintln!("[SelfHeal] Epoch jumped from {} to {}", curr, packet.epoch);
        
        // Сбросить Ring Buffer (потеря локальных спайков за пропущенные батчи)
        flush_spike_schedule();
        
        // Atomically обновить epoch
        current_epoch.store(packet.epoch, Ordering::Release);
    }
}
```

**Эффект:** Если кластер лагает, одна быстрая нода может "разбудить" отставшие, заставив их пересинхронизироваться. Потеря данных (Biological Amnesia) — это цена синхронизма, и она **приемлема для нейросети** (спайки теряются на биологических синапсах при глубоком сне).

**Инвариант:** `epoch` никогда не движется назад (монотонно возрастает). Это гарантирует, что система движется вперёд во времени, даже если теряет данные.

### 2.9. Bulk DMA & Autonomous Batch Execution [MVP]

Сетевой реактор Tokio и GPU полностью изолированы через Ping-Pong буферы (`BspBarrier`). Шина PCIe не дергается каждый тик.

**Стратегия: Минимум пересечений Host-Device**

1. **Bulk H2D (Host-to-Device):**
   Перед началом батча `Input_Bitmask` и `SpikeSchedule` заливаются в VRAM за **одну** асинхронную транзакцию `cudaMemcpyAsync`. 
   - Размер: `(input_bitmask_size + schedule_size)` байт
   - Ожидание: <1 мс на PCIe 4.0 x16 при ≈100 МБ батча
   - Запуск: `cudaMemcpyAsync(d_input, h_input, size, 0, stream)` с non-blocking флагом

2. **Autonomous GPU Loop:**
   GPU крутит **6-ядерный цикл** (`sync_batch_ticks` шагов) полностью независимо от хоста:
   ```
   for tick_in_batch in 0..sync_batch_ticks:
       InjectInputs        (Virtual Axon pulse)
       ApplySpikeBatch     (Ghost Axon activation)
       PropagateAxons      (Безусловный IADD на все аксоны)
       UpdateNeurons       (GLIF + spike check)
       ApplyGSOP           (Plasticity)
       RecordReadout       (Output snapshot)
   ```
   - **Ни одного вызова `cudaDeviceSynchronize()`** внутри цикла
   - Ни одного обращения к хосту
   - **Результат:** Полная утилизация GPU SM, без простоев

3. **Pointer Offsetting (O(1)):**
   Внутри горячего цикла CUDA ядра получают смещенные указатели на данные текущего тика:
   ```cuda
   __global__ void PropagateAxons(...) {
       int tid = blockIdx.x * blockDim.x + threadIdx.x;
       if (tid >= total_axons) return;
       
       // tick_input_ptr = base_ptr + (tick_in_batch * stride)
       // O(1) арифметика, никакого поиска
       uint32_t *tick_schedule = schedule_base + (tick_in_batch * max_spikes_per_tick);
       int head = axon_heads[tid];
       axon_heads[tid] = head + v_seg[tid];  // Propagate
   }
   ```

4. **Bulk D2H (Device-to-Host):**
   По окончании батча — одна асинхронная транзакция:
   ```cuda
   cudaMemcpyAsync(h_output, d_output_history, output_size, cudaMemcpyDeviceToHost, stream);
   ```
   - Перекрытие: пока host обрабатывает батч N, GPU готовит батч N+1
   - Network Phase парирует это естественно через BSP barrier

**Инвариант:** Нет микротранзакций. Нет пинг-понга каждый тик. Только 4 макро-операции за весь батч (100 ms → 4 DMA).

### 2.10. WaitStrategy: CPU Profiles [MVP]

**Задача:** Управление планировщиком ОС в горячих циклах (BSP Barrier, сетевой ввод). Выбор между спин-локом, yield и дремой при ожидании данных от соседей.

**3 профиля (флаг `--cpu-profile`):**

| Профиль | Стратегия | Латентность | Использование CPU | Сценарий |
|---|---|---|---|---|
| **Aggressive** | `spin_loop()` | ~1 нс | 100% ядро | Production / HFT |
| **Balanced** | `yield_now()` | ~1–15 мс | Разд. с OS | Дебаг, локальная сеть |
| **Eco** | `sleep(1ms)` | ~1–5 мс | ~0% детально | Ноутбуки, батарея |

**Реализация в Network Phase:**

```rust
pub enum WaitStrategy {
    Aggressive,  // spin_loop()
    Balanced,    // yield_now()
    Eco,         // sleep(1ms)
}

pub fn wait_for_neighbors(wait_strategy: WaitStrategy) -> Vec<SpikeBatch> {
    loop {
        // Попытка неблокирующего чтения из буфера сокетов
        if let Some(batch) = try_recv_all_neighbors() {
            return batch;
        }
        
        match wait_strategy {
            WaitStrategy::Aggressive => std::hint::spin_loop(),
            WaitStrategy::Balanced => std::thread::yield_now(),
            WaitStrategy::Eco => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}
```

**Ключевые Инварианты:**

1. **Spin-loop безопасен:** BSP барьер — единственное место, где хост ждёт физиологического события (приход сетки). Нет Mutex, нет atomics-loop.
2. **Не цепляется за GPU:** Network Phase — чисто CPU, GPU работает независимо (Autonomous Loop, §2.9).
3. **Портативность:** Выбор профиля — runtime, переносится через config или CLI. Ядро физики одинаково во всех режимах.

### 2.11. Почему Это Сработает [MVP]

| Принцип | Эффект |
|---|---|
| **Масштабируемость** | Платим за латентность сети один раз за 100 тиков, не 100 раз |
| **Детерминизм** | При тех же сидах и `sync_batch_ticks` — бит-в-бит одинаковый результат, независимо от скорости сети |
| **Безопасность** | Если сеть лагает — система притормаживает (Wall Clock Speed падает), но физика (GSOP, спайки) остаётся математически точной |
| **Zero-Copy** | Hot Loop получателя: O(1) вставка по готовому индексу, без десериализации |

---

## 3. Интеграция Атласа (White Matter Routing) [PLANNED]

Атлас (например, на базе FLNe-матриц мармозетки) — **статичная таблица маршрутизации**. Используется только на CPU во время фазы Baking. В горячем цикле (Hot Loop) на GPU Атласа не существует. Пространство между зонами **не симулируется**.

### 3.1. Квотирование (Hard Quotas)

Отказываемся от генерации связей «по вероятности». Используем **жёсткие квоты**.

- Если атлас говорит, что доля V1 во входящих связях V2 составляет 17%, и слою нужно 100 000 входов — Compiler Tool обязан выбрать **ровно 17 000** сом-отправителей из V1.
- Выбор отправителей производится **детерминированным шаффлом** на базе `master_seed` (см. [02_configuration.md §5.3](./02_configuration.md)).

### 3.2. Топографический Маппинг (UV-Projection)

Зоны имеют разные физические размеры. Прямой перенос координат невозможен.

- **UV-нормализация:** Координаты сомы-отправителя переводятся в диапазон `0.0..1.0` (`U`, `V`).
- **Проекция:** Целевая координата = `U × Target_Width`, `V × Target_Height`.
- **Результат:** Пространственная топология проекции сохраняется — то, что рядом в V1, рядом и в V2.

### 3.3. Детерминированное Рассеивание (Jitter)

Никаких проверок коллизий для Ghost Axons на целевой стороне.

- Чтобы аксоны не попадали в один математический пиксель идеальной сетки, к целевой координате добавляется **детерминированный шум:**

```
Target_X += Hash(master_seed + soma_id) % jitter_radius
Target_Y += Hash(master_seed + soma_id + 1) % jitter_radius
```

- Шум воспроизводим при том же `master_seed` — детерминизм сохранён.

### 3.4. Результат: Zero-Cost Routing

1. **Передающий шард (V1):** Создаётся выходной порт (`Port Out`).
2. **Принимающий шард (V2):** Создаётся массив Ghost Axons в рассчитанных координатах (`Target_X`, `Target_Y`). Они начинают **локально прорастать** вглубь целевого слоя, подчиняясь обычной физике Cone Tracing (§4 из [04_connectivity.md](./04_connectivity.md)).
3. **Задержка:** `Delay_Ticks` рассчитывается жёстко по физическому расстоянию между зонами в Атласе. Реальный сетевой лаг (пинг) **прячется** внутри этой математической задержки — сеть перестаёт быть проблемой.

### 3.5. Dynamic Routing (Read-Copy-Update)

Статическая маршрутизация не подходит для отказоустойчивых кластеров. Адреса шардов могут меняться при их воскрешении (Resurrection) на новых нодах.

**Таблица маршрутизации** (`RoutingTable`) работает по принципу **RCU (Read-Copy-Update)**, гарантируя **0 блокировок в горячем цикле:**

- **Read:** Egress-потоки читают адреса через `AtomicPtr` (O(1), без блокировок).
- **Copy-Update:** При получении пакета `RouteUpdate` (Magic: `0x54554F52` / "ROUT"), оркестратор клонирует хэш-таблицу, обновляет IP:Port, и делает атомарный `swap` указателя.
- **Deferred Cleanup:** Старая таблица удаляется через `tokio::spawn` с задержкой 100 мс, гарантируя, что все отставшие Egress-потоки успели завершить чтение.

#### Формат пакета RouteUpdate

```rust
#[repr(C)]
pub struct RouteUpdate {
    pub magic: u32,      // 0x54554F52 ("ROUT")
    pub zone_hash: u32,  // ID перемещенной зоны
    pub new_ipv4: u32,   // Новый IP в u32 (network byte order)
    pub new_port: u16,   // Новый порт
    pub _padding: u16,   // Выравнивание; ignore
}  // = 16 bytes
```

#### RCU Implementation

```rust
pub struct RoutingTable {
    // Указатель на текущую таблицу маршрутизации
    // Читается без блокировок через load(Ordering::Acquire)
    ptr: AtomicPtr<HashMap<u32, SocketAddr>>,
}

impl RoutingTable {
    pub fn lookup(&self, zone_hash: u32) -> Option<SocketAddr> {
        // O(1) чтение без мьютекса
        let table = unsafe { &*self.ptr.load(Ordering::Acquire) };
        table.get(&zone_hash).copied()
    }
    
    pub fn update(&self, zone_hash: u32, new_addr: SocketAddr) {
        // Copy: клонировать текущую таблицу
        let old_ptr = self.ptr.load(Ordering::Acquire);
        let mut new_table = unsafe { (*old_ptr).clone() };
        
        // Update: вставить новый адрес
        new_table.insert(zone_hash, new_addr);
        let new_ptr = Box::into_raw(Box::new(new_table));
        
        // Swap: атомарно поменять указатель
        let old_ptr = self.ptr.swap(new_ptr, Ordering::Release);
        
        // Cleanup (deferred): удалить старую таблицу через 100 мс
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            drop(unsafe { Box::from_raw(old_ptr) });
        });
    }
}
```

**Инварианты:**

1. **Горячий цикл (Egress):** `lookup()` — чистое чтение без атомиков, без блокировок. CAS + Acquire ordering = ~3 цикла на x86.
2. **Холодный цикл (Control Plane):** `update()` работает редко (при воскрешении шарда); может заполнять CPU на copy+swap, но не влияет на рендеринг спайков.
3. **Безопасность:** Deferred cleanup гарантирует, что нет Use-After-Free: старая таблица удаляется только когда все читатели заведомо завершили свои операции (100 мс >> max jitter).

---

## 4. Текущая Реализация в V1 (MVP)

### 4.1. Что Работает Сейчас [MVP]

**Single-Node (Одна GPU):**
- Columnar Memory Layout сохраняется.
- CUDA kernels вычисляют весь день без синхронизации с сетью.
- Шардирование по X/Y отключено — весь мозг на одной GPU VRAM.

**BSP Barrier (Mock):**
- `BspBarrier::new()` создаёт ping-pong буферы (`schedule_a`, `schedule_b`).
- `sync_and_swap()` выполняет жёсткую синхронизацию (swap буферов).
- Сокет **отключен** (`socket: None`) — для offline тестов.

**SpikeSchedule:**
```rust
pub struct SpikeSchedule {
    pub sync_batch_ticks: usize,
    pub buffer: Vec<u32>,      // [sync_batch_ticks × MAX_SPIKES_PER_TICK]
    pub counts: Vec<u32>,      // [sync_batch_ticks]
}
```
- Flat 1D array для Single DMA to VRAM.
- Каждый тик: GPU берёт `buffer[tick * MAX_SPIKES_PER_TICK .. +counts[tick]]`.
- Map Phase (CPU) раскладывает спайки по `tick_offset` из `SpikeEvent`.

**GhostConnection (Metadata):**
```rust
pub struct GhostConnection {
    pub local_axon_id: u32,     // Индекс в axon_heads[] на целевом шарде
    pub paired_src_soma: u32,   // Soma ID на источнике (для трассировки)
}
```
- Загружается из файла `.ghosts` (Baker output).
- Используется только для трансляции soma_id → ghost_id при спайке.

### 4.2. Что НЕ Работает Пока [PLANNED]

| Функция | Статус | Причина |
|---|---|---|
| **Multi-Node (Network)** | PLANNED | Сокет не подключен, одна GPU |
| **Dynamic AxonHandover** | PLANNED | Аксоны не растут через границы; они бакутся заранее |
| **Periodic Boundaries (Toroidal)** | PLANNED | Замыкание Z и кольцо X/Y требуют изменения конфига |
| **Slow Path Geometry Sync** | PLANNED | Геометрия вычисляется только раз при Baking |
| **Pruning & Night Phase Networking** | PLANNED | Удаление Ghost Axons на целевом шарде не реализовано |
| **Latency Hiding Variance** | PLANNED | Фиксированная задержка 1 батч; адаптивная задержка не реализована |
| **Atlas-Based Routing** | PLANNED | Маршрутизация между зонами фиксирована в `.ghosts` файлах |

---

## 5. Структуры Данных (Реальные Сигнатуры)

### 5.1. SpikeBatchHeader & SpikeEvent

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeBatchHeader {
    pub batch_id: u32,      // Номер батча (защита от рассинхрона)
    pub spikes_count: u32,  // Количество событий в батче
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeEvent {
    pub receiver_ghost_id: u32,  // Ghost Axon ID на целевом шарде (прямой индекс)
    pub tick_offset: u8,         // Тик внутри батча (0..sync_batch_ticks)
    pub _pad: [u8; 3],           // Выравнивание до 8 байт
}
```

**Размер:** 8 байт на спайк (Coalesced Access на GPU).

**Передача:** Заголовок (8 байт) + массив SpikeEvent (8 байт × count).

### 5.2. BspBarrier Internals

```rust
pub struct BspBarrier {
    pub schedule_a: SpikeSchedule,              // Буфер 1
    pub schedule_b: SpikeSchedule,              // Буфер 2
    pub writing_to_b: bool,                     // Куда пишет сеть
    pub outgoing_batches: HashMap<u32, Vec<SpikeEvent>>,  // Исходящие спайки по шардам
    pub socket: Option<NodeSocket>,             // Сокет (None в MVP)
    pub peer_addresses: HashMap<u32, SocketAddr>,  // Маршрут: shard_id → IP:port
}
```

**Жизненный цикл sync_and_swap():**
1. Flip `writing_to_b` (GPU читает из одного, сеть пишет в другой).
2. Если `socket.is_some()`: отправить `outgoing_batches` всем соседям (async).
3. Ждём прихода батчей от соседей (await).
4. `ingest_spike_batch()` раскладывает входящие спайки в текущий Schedule.
5. Сброс Schedule на следующий батч.

### 5.3. GhostConnection File Format

Файл `{shard_name}.ghosts` (бинарный, Little-Endian):

```
[0..3]   magic: u32 = 0x47485354 ("GHST")
[4]      version: u8 = 1
[5]      padding: u8
[6..7]   width: u16 (измерение X/Y стыка)
[8..9]   height: u16 (измерение Z или второе строящееся направление)
[10..11] padding: u16
[12..15] src_zone_hash: u32 (идентификатор Zona-источника)
[16..19] dst_zone_hash: u32 (идентификатор Zona-приёма)
[20..]   connections: [GhostConnection; width × height]
         Каждый: local_axon_id (u32) + paired_src_soma (u32) = 8 байт
```

**Примечание:** Массив Connections отсортирован по (Y, Z) если X-граница или (X, Z) если Y-граница. Это позволяет GPU применять `ApplySpikeBatch` с хорошей локальностью доступа.

---

## Связанные документы

| Документ | Что связывается |
|---|---|
| [04_connectivity.md](./04_connectivity.md) | §1.7: Ghost Axons, Cone Tracing, Axon Growth |
| [05_signal_physics.md](./05_signal_physics.md) | §1.2.1: ApplySpikeBatch kernel, Ghost indices |
| [07_gpu_runtime.md](./07_gpu_runtime.md) | §2: Batch Orchestration, sync_and_swap timing, Stream management |
| [09_baking_pipeline.md](./09_baking_pipeline.md) | §3: Ghost Connection generation, Z-sort for boundaries |
| [02_configuration.md](./02_configuration.md) | §5.3: master_seed, deterministic topology |
| [project_structure.md](../project_structure.md) | Distributed arch role in overall Genesis design |

## 6. Autonomous Node Recovery (The Great Resurrection)

Кластер Genesis проектируется под постоянные аппаратные отказы. Вместо остановки симуляции применяется механизм **Zero-Downtime Shard Recovery**, гарантирующий восстановление вычислений за миллисекунды.

### 6.1. Shadow Buffers (Теневые Реплики)

Для избежания потери синаптических весов при смерти ноды применяется асинхронная репликация в POSIX Shared Memory.

1. **Периодичность:** Каждые 500 батчей оркестратор вызывает `replicate_shards`.

2. **Zero-Copy Transfer:** Массивы `dendrite_weights` и `dendrite_targets` напрямую копируются из `/dev/shm/genesis_shard_{zone_hash}` в `/dev/shm/{zone_hash}.shadow` на соседних резервных нодах.

3. **OS-Level Transport:** Используется `tokio::io::copy`, который на Linux транслируется в системный вызов `sendfile` (или `splice`), обеспечивая копирование из сокета в файловый дескриптор в пространстве ядра, минуя user-space аллокации.

```rust
// genesis-runtime/src/recovery.rs

pub async fn replicate_shards(
    shards: &[ShardState],
    backup_nodes: &[BackupNode],
) -> Result<Vec<ReplicaStatus>> {
    for shard in shards {
        let shm_path = format!("/dev/shm/genesis_shard_{:x}", shard.zone_hash);
        let shadow_path = format!("/dev/shm/{:x}.shadow", shard.zone_hash);
        
        // Open source (shared memory)
        let mut src = tokio::fs::File::open(&shm_path).await?;
        
        // Send to backup nodes asynchronously
        for backup in backup_nodes {
            let mut dst = TcpStream::connect(backup.addr).await?;
            tokio::io::copy(&mut src, &mut dst).await?;  // kernel-level sendfile
        }
    }
    Ok(replicas)
}
```

### 6.2. 500ms Isolation Detection

`BspBarrier` отслеживает время ожидания от каждого соседа.

- Если барьер заблокирован более чем на `BSP_SYNC_TIMEOUT_MS` (500 мс), срабатывает детектор изоляции.
- Оркестратор генерирует ошибку `BspError::NodeIsolated(dead_zone_hash)`.
- Узел, хранящий теневую реплику (`.shadow`) упавшей зоны, назначается координатором воскрешения.

```rust
// genesis-runtime/src/orchestrator.rs

pub fn detect_isolation(
    current_epoch: u32,
    barrier_start_epoch: u32,
    timeout_ms: u64,
) -> Option<IsolatedZone> {
    let elapsed = (current_epoch - barrier_start_epoch) * TICK_DURATION_US as u32 / 1000;
    
    if elapsed > timeout_ms as u32 {
        return Some(IsolatedZone {
            zone_hash: last_missing_peer,
            detection_epoch: current_epoch,
        });
    }
    None
}
```

### 6.3. The Great Resurrection (Протокол Воскрешения)

При активации координатор выполняет следующий конвейер:

1. **VRAM Re-allocation:** Выделяется новая память на GPU под `padded_n` и `total_axons`.

2. **Shadow Restore:** Данные из `/dev/shm/*.shadow` заливаются в VRAM через `cudaMemcpyAsync`.

3. **RCU Route Patching:** Координатор рассылает всем пирам широковещательный пакет `RouteUpdate` (см. [06_distributed.md §3.5](./06_distributed.md#35-dynamic-routing-read-copy-update), `ROUT_MAGIC = 0x54554F52`). Пиры атомарно обновляют свои Egress-таблицы (RCU Swap), перенаправляя трафик на новый IP/Port. **Ноль мьютексов в горячем цикле**.

```rust
// genesis-runtime/src/recovery.rs

pub struct ResurrectionCoordinator {
    shadow_path: String,
    zone_hash: u32,
}

impl ResurrectionCoordinator {
    pub async fn execute(&self) -> Result<()> {
        // Step 1: VRAM allocation
        let new_vram = allocate_vram(padded_n, total_axons)?;
        
        // Step 2: Shadow restore
        let shadow_data = read_shadow_buffer(&self.shadow_path)?;
        cuda_memcpy_async(new_vram, shadow_data)?;
        cuda_synchronize()?;
        
        // Step 3: RCU route patching
        self.broadcast_route_update(new_addr).await?;
        
        Ok(())
    }
}
```

### 6.4. Stabilization (Warmup Loop)

Сохранённый `.shadow` дамп содержит веса дендритов и топологию аксонов, но **не содержит мембранные потенциалы** (вольтаж сомы) — они слишком волатильны и их сброс на диск убьёт PCIe шину.

При воскрешении все нейроны имеют `voltage = 0`. Если сразу пустить шард в сеть, произойдёт эпилептический шторм из-за потери гомеостатических порогов или, наоборот, полное молчание.

**Механика:**

Команда `ComputeCommand::Resurrect(zone_hash)` переводит шард в режим **Warmup** длительностью 100 тиков (Day Phase):

1. **Входящие сетевые спайки** обрабатываются нормально (`ApplySpikeBatch`).
2. **Собственные исходящие спайки** (External Outputs через Virtual Axons) **гасятся** (не отправляются в сеть).
3. **Мембранные потенциалы** (`voltage` и `threshold_offset`) медленно напитываются токами и стабилизируются до биологической нормы.
4. **По окончании 100 тиков** шард переходит в режим Normal и начинает полноценно взаимодействовать с кластером.

```rust
// genesis-runtime/src/compute.rs

pub struct ShardMode {
    pub variant: ShardVariant,
    pub warmup_ticks_remaining: u32,
}

pub enum ShardVariant {
    Normal,
    Warmup,  // Входящие спайки ОК, исходящие гасятся
}

pub fn record_readout_warmup(
    out_spike_ids: &mut Vec<u32>,
    out_count: &mut u32,
    shard_mode: &ShardMode,
) {
    if shard_mode.variant == ShardVariant::Warmup {
        // Отбросить все исходящие спайки
        out_spike_ids.clear();
        *out_count = 0;
        return;
    }
    // Normal mode: запись как обычно
    // ... записать spike_ids в out_spike_ids
}
```

**Результат:** После 100 тиков (1–10 мс реального времени) напряжение стабилизируется, пороги учитывают локальный контекст активности, и шард может безопасно взаимодействовать с остальным кластером без риска паралича или неконтролируемых всплесков.

---

## Changelog

| Дата | Версия | Описание изменений |
|---|---|---|
| 2026-03-02 | 1.2 | Добавлено требование по Autonomous Node Recovery (Fault Tolerance) |
| 2026-02-28 | 1.1 | Разделение на [MVP] vs [PLANNED] маркеры. Уточнение GhostConnection (V1 реальность). Обновлены SpikeEvent, BspBarrier, GhostConnection структуры согласно реальному коду. Добавлен раздел "Текущая реализация в V1" и таблица функций. |
| TBD | 1.0 | Первая версия спеки |

