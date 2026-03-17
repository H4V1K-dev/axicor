# 07. GPU Runtime (GPU & Storage)

> Часть архитектуры [Genesis](../../README.md). Как данные лежат в VRAM, как грузятся, как переключаются фазы работы.

---

## 1. Архитектура Памяти и Данных (GPU & Storage)

**Инвариант:** Полный отказ от объектов (AoS) в горячей памяти. Данные лежат плоскими векторами (SoA) для обеспечения 100% Coalesced Memory Access на GPU и векторизации на CPU/MCU. 

Движок поддерживает **Dual-Backend** архитектуру: нативный CUDA и нативный HIP. Все вычисления — исключительно в целых числах (Integer Physics).

### 1.1. Строгий FFI-Контракт VRAM (Headerless SoA)

Движок не парсит данные при старте. Бинарные файлы `.state` и `.axons` — это чистые дампы памяти (Headerless), готовые к `cudaMemcpyAsync` или `spi_flash_mmap` (ESP32).

**Раскладка ShardVramPtrs (строгий побайтовый порядок в .state блобе):**

Размеры массивов зависят от `N` (`padded_n`, выровненного по 32).
*   `soma_voltage`       [N] × `i32`   (4N bytes)
*   `soma_flags`         [N] × `u8`    (1N bytes)
*   `threshold_offset`   [N] × `i32`   (4N bytes)
*   `timers`             [N] × `u8`    (1N bytes)
*   `soma_to_axon`       [N] × `u32`   (4N bytes)
*   `dendrite_targets`   [128 × N] × `u32` (512N bytes)
*   `dendrite_weights`   [128 × N] × `i16` (256N bytes)
*   `dendrite_timers`    [128 × N] × `u8`  (128N bytes)

*Примечание для Edge (ESP32): При дистилляции 128 слотов урезаются до 32 (WTA Distillation).*

**Аксоны (.axons блоб):**
Вынесены в отдельный файл, так как количество аксонов `A` (`total_axons` = Local + Ghost + Virtual) не равно `N`.
*   `axon_heads`         [A] × `BurstHeads8` (32A bytes)

**Инвариант 32-байтного Выравнивания Аксонов:** 
Структура `BurstHeads8` обязана быть выровнена по 32 байтам. На Xtensa LX7 (ESP32) чтение невыровненных 32-байтных блоков через векторные инструкции вызовет аппаратное исключение.

```cpp
// 32-byte alignment гарантирует загрузку 8 голов за 1 транзакцию L1 кэша.
struct alignas(32) BurstHeads8 {
    uint32_t h0; uint32_t h1; uint32_t h2; uint32_t h3;
    uint32_t h4; uint32_t h5; uint32_t h6; uint32_t h7;
};
```

Битовая семантика `soma_flags` (1 байт): Запрещено перезаписывать байт целиком. Чтение и запись строго через битовые маски.

*   `[7:4]` type_mask (u4): Индекс типа нейрона (0..15). Прямой индекс в VARIANT_LUT.
*   `[3:1]` burst_count (u3): Счетчик серийных спайков для Burst-Dependent Plasticity (BDP).
*   `[0:0]` is_spiking (u1): Флаг спайка в текущем тике (1 = fired).

Очистка флага спайка без уничтожения BDP-аккумулятора: `flags[tid] &= ~0x01;`.

### 1.2. Константная Память (VariantParameters)
Параметры мембраны и пластичности загружаются в constant память GPU (или лежат во Flash-памяти MCU) один раз при старте.
Размер структуры строго 64 байта для идеального попадания в L1 Cache. Любое смещение убьет Coalesced Access.

```cpp
// Строго 64 байта (1 кэш-линия L1)
struct alignas(64) VariantParameters {
    // === Блок 1: 32-bit (Смещения 0..20) ===
    int32_t threshold;                        // 0..4
    int32_t rest_potential;                   // 4..8
    int32_t leak_rate;                        // 8..12
    int32_t homeostasis_penalty;              // 12..16
    uint32_t spontaneous_firing_period_ticks; // 16..20

    // === Блок 2: 16-bit (Смещения 20..28) ===
    uint16_t initial_synapse_weight;          // 20..22
    uint16_t gsop_potentiation;               // 22..24
    uint16_t gsop_depression;                 // 24..26
    uint16_t homeostasis_decay;               // 26..28

    // === Блок 3: 8-bit (Смещения 28..32) ===
    uint8_t refractory_period;                // 28..29
    uint8_t synapse_refractory_period;        // 29..30
    uint8_t signal_propagation_length;        // 30..31
    uint8_t is_inhibitory;                    // 31..32

    // === Блок 4: Массивы (Смещения 32..48) ===
    uint8_t inertia_curve[16];                // 32..48 (ранги: abs(w) >> 11)

    // === Блок 5: Adaptive Leak Hardware (Смещения 48..58) ===
    int32_t adaptive_leak_max;                // 48..52
    uint16_t adaptive_leak_gain;              // 52..54
    uint8_t adaptive_mode;                    // 54..55
    uint8_t _leak_pad[3];                     // 55..58

    // === Блок 6: Pad (Смещения 58..64) ===
    uint8_t _pad[6];                          // 58..64 (Выравнивание)
};

// Контейнер для 16 вариантов (Ровно 1024 байта)
struct GenesisConstantMemory {
    VariantParameters variants[16];
};
```

Доступ из горячего цикла: Variant ID извлекается из `soma_flags` за 1 такт ALU: `u8 var_id = (flags[tid] >> 4) & 0xF;` Далее — прямое чтение параметров из памяти без ветвлений `const_mem.variants[var_id].threshold;`.

### 1.6. Cross-Platform IPC & Zero-Copy Mmap

**Инвариант:** Идеал Zero-Copy Загрузки (§1.4) реализуется одинаково эффективно на всех платформах. Отказ от Linux-exclusive конструкций (`/dev/shm`, Unix Sockets).

**Инвариант:** Идеал Zero-Copy Загрузки (§1.4) реализуется через Dual-Backend компиляцию: `nvcc` для NVIDIA и `hipcc` для AMD. Система автоматически выбирает реализацию на этапе сборки через feature-флаг `amd`.

#### 1.6.1. Архитектура: Платформа-Специфичные Фолбэки

| Платформа | Память (Night Phase) | Синхронизация (Network) |
|---|---|---|
| **Linux** | POSIX `shm_open()` → `/dev/shm/*.state.shm` | Unix Domain Sockets (**UDS**); Fast-Path UDP для Data Plane |
| **Windows** | File-backed mmap в `%TEMP%` → файлы `*.state.bin.mmap` | TCP/IP на портах `19000 + (hash % 1000)` для Control & Data Plane |
| **Darwin (macOS)** | POSIX `shm_open()` (аналог Linux) | UDS + TCP fallback для Legacy Systems |

**Выбор платформы:** Compile-time через `cfg!(target_os)`. Runtime auto-detection переносится на stage bootstrap (инициализация Node в Distributed Cluster).

#### 1.6.2. Page-Aligned Memory Guarantee (4096 bytes)

**Жёсткий C-ABI контракт:** Всегда `mmap` выравнивается по границе **4096 байт** (минимальный page size современных ОС).

```rust
// generic_ipc.rs
pub fn allocate_shared_memory(size: usize) -> Result<SharedMemoryRegion, CAbiBoundaryError> {
    let aligned_size = (size + 4095) / 4096 * 4096; // Align UP to 4096
    
    #[cfg(target_os = "linux")]
    {
        let shm = unsafe { libc::shm_open(name, libc::O_CREAT | libc::O_RDWR, 0o644) }?;
        let addr = unsafe { libc::mmap(
            std::ptr::null_mut(),
            aligned_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            shm,
            0,
        )};
        // addr guaranteed to be % 4096 == 0
        
        assert_eq!(addr as usize % 4096, 0, "FATAL C-ABI BOUNDARY: mmap address not page-aligned!");
    }
    
    #[cfg(target_os = "windows")]
    {
        let file = File::create(temp_path)?;
        file.set_len(aligned_size as u64)?;
        let handle = unsafe { CreateFileMappingA(file, aligned_size)? };
        let addr = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, aligned_size) };
        
        assert_eq!(addr as usize % 4096, 0, "FATAL C-ABI BOUNDARY: MapViewOfFile address not page-aligned!");
    }
    
    Ok(SharedMemoryRegion { addr, size: aligned_size })
}
```

**Гарантия:** Все `.state` и `.axons` блобы лежат **целиком в выровненной по 4096 памяти**.

```
Mmap:   [0x10000000 aligned to 4096]
         ┌─────────────────────────────────┐
         │ SoA Payload (782B × N)          │
         │ + Burst Architecture (32B × A)  │ ← All @ offset % 4096 == 0
         └─────────────────────────────────┘ ← Also @ offset % 4096 == 0
         [страница 4096 байт, no gaps]
```

#### 1.6.3. Legalized bytemuck::cast_slice (Zero-Copy)

**Почему это работает:**

1. **Mmap гарантирует выравнивание:** Любой указатель внутри mmap-региона = `base_addr + offset`. Если `base_addr % 4096 == 0` и `offset % align_of::<T>() == 0`, то полученный указатель гарантированно выровнен.

2. **Baking Tool Determinism (Compile-Time):** Baker гарантирует, что SoA-массивы начинаются ровно на границе требуемого выравнивания (`align_of::<T>()`). Для типов как `VariantParameters` (align 64) это означает: `offset % 64 == 0`.

3. **Host-Side Zero-Copy:**

```rust
// memory.rs (Night Phase CPU-side)
let shared_region = allocate_shared_memory(total_state_bytes)?;

// ZERO allocations, ZERO copies
let soma_voltage: &[i32] = unsafe {
    bytemuck::cast_slice(std::slice::from_raw_parts(
        (shared_region.addr + offset_soma_voltage) as *const i32,
        neuron_count,
    ))
};

// This is SAFE because:
// 1. (shared_region.addr + offset_soma_voltage) % align_of::<i32>() == 0 (baker-enforced)
// 2. shared_region.addr % 4096 == 0 (mmap-guaranteed)
// 3. bytemuck::Pod trait ensures no padding, no Option, no refcells
```

**CUDA-Side (Device Pointers):**

```cuda
// kernel.cu
__global__ void example_kernel(CudaShardVramPtrs ptrs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ZERO device-side alignment overhead
    int32_t v = ptrs.soma_voltage[tid];  // Guaranteed coalesced access: 32 threads read 128 bytes @ 32-byte boundary
}
```

#### 1.6.4. Паника FATAL C-ABI BOUNDARY (Нарушение Контракта)

При любом нарушении выравнивания **во время загрузки** мотор обязан паниковать с сообщением `FATAL C-ABI BOUNDARY`:

```rust
// validation.rs (baking_tool output check)
pub fn validate_shard_memory_contract(header: &ShardStateHeader) -> Result<()> {
    let vram_base = header.vram_base_ptr as usize;
    
    if vram_base % 4096 != 0 {
        panic!(
            "FATAL C-ABI BOUNDARY: VRAM base address 0x{:x} is not 4096-byte page-aligned. \
            This violates the Zero-Copy Mmap contract and will cause uncoalesced access & cache thrashing.",
            vram_base
        );
    }
    
    for soa_field in &header.soa_fields {
        let offset = soa_field.offset as usize;
        if offset % soa_field.required_align != 0 {
            panic!(
                "FATAL C-ABI BOUNDARY: SoA field '{}' @ offset 0x{:x} breaks {} alignment. \
                Baker must enforce columnar layout alignment from compile-time.",
                soa_field.name, offset, soa_field.required_align
            );
        }
    }
    
    Ok(())
}
```

**Взаимодействие с Node Runtime:**

- При `load_shard(shard_id)` хост проверяет контракт перед `cudaMemcpy`.
- Если контракт нарушен → **сразу паника**, ноль попыток работать с невыровненными данными.
- Это **legalization** логики аппаратуры в коде: нет хаков, нет `__pack__`, нет скрытых аллокаций.

---

### 1.7. SHM Binary Contract (Night Phase IPC v4)

Связь между `genesis-runtime` и `genesis-baker-daemon` происходит через Shared Memory. 
Нулевой оффсет файла всегда содержит 64-байтный `ShmHeader` (Little-Endian, C-ABI). 
Каждый массив данных начинается строго по границе **64 байт**.

| Смещение | Поле | Тип | Описание |
| :--- | :--- | :--- | :--- |
| `0x00` | `magic` | `u32` | 0x47454E53 ("GENS") |
| `0x04` | `version` | `u8` | Текущая версия = **2** |
| `0x05` | `state` | `u8` | State Machine (0=Idle, 1=NightStart, 2=Sprouting, 3=NightDone, 4=Error) |
| `0x06` | `_pad` | `u16` | Выравнивание |
| `0x08` | `padded_n` | `u32` | Количество нейронов (кратно 32) |
| `0x0C` | `dendrite_slots` | `u32` | Всегда 128 |
| `0x10` | `weights_offset` | `u32` | Смещение до i16 массива весов (кратно 64) |
| `0x14` | `targets_offset` | `u32` | Смещение до u32 массива целей (кратно 64) |
| `0x18` | `epoch` | `u64` | Глобальный счетчик батчей (BSP Epoch) |
| `0x20` | `total_axons` | `u32` | Local + Ghost + Virtual аксоны |
| `0x24` | `handovers_offset` | `u32` | Смещение до очереди `AxonHandoverEvent` (кратно 64) |
| `0x28` | `handovers_count` | `u32` | Количество событий в очереди |
| `0x2C` | `zone_hash` | `u32` | FNV-1a хэш имени зоны |
| `0x30` | `prunes_offset` | `u32` | Смещение до очереди `AxonHandoverPrune` (кратно 64) |
| `0x34` | `prunes_count` | `u32` | Количество исходящих прунов |
| `0x38` | `incoming_prunes_count`| `u32` | Количество входящих прунов от соседей |
| `0x3C` | `flags_offset` | `u32` | Смещение до `soma_flags` (кратно 64) |

**Общий размер: ровно 64 байта. Ни одного свободного байта больше нет. Любое расширение потребует перехода на 128-байтный заголовок.**

#### Семантика Полей v2

- **`version = 2`:** Поддежка `prunes_offset`, `incoming_prunes_count` и `flags_offset` для механики Living Axons.
- **`prunes_offset`:** Очередь `AxonHandoverPrune` - события удаления аксонов при Pruning (ночная фаза). Рантайм читает и отправляет соседям, которые удаляют Ghost Axons.
- **`incoming_prunes_count`:** Счётчик входящих прун-событий от соседних зон. Рантайм обновляет этот счётчик для синхронизации с Baker.
- **`flags_offset`:** Прямое смещение до массива `soma_flags[padded_n]` для быстрого доступа вне стандартной раскладки SoA.

**Инвариант:** Все смещения (offset) должны быть кратны 64 байтам. Baker гарантирует это выравнивание при формировании блобов.

## 2. Архитектура Цикла: День и Ночь (Day/Night Cycle)

Фундаментальное решение: разделение вычислений во времени. Разрешает конфликт между жёсткой статической памятью (Coalesced Access на GPU) и структурной пластичностью графа (динамические аллокации).

**Инвариант:** Night Phase - **локальная операция на уровне зоны**. Замораживается только конкретная зона - остальные продолжают работать. Глобального останова нет.

### 2.1. Фаза «День» (Online / Hot Loop)

Выполняется **исключительно на GPU**. Максимальная пропускная способность, полностью лишена структурной логики.

- **Read-Only Топология:** Геометрия аксонов и массив подписок дендритов заморожены. Никаких `malloc`/`free` внутри ядра.
- **Изменяемое Состояние:** Веса синапсов (GSOP), `axon_heads[]`, таймеры, вольтаж, флаги.

**Порядок запуска ядер (каждый тик):**

| # | Kernel | Описание | Оисточник |
|---|---|----|----|
| 1 | `InjectInputs` | Bitmask Injection для виртуальных аксонов (single-tick pulse) | [05_signal_physics.md §2.4](./05_signal_physics.md) |
| 2 | `ApplySpikeBatch` | Чтение Ghost indices из Schedule, сброс `axon_heads[ghost_id] = 0` | [05_signal_physics.md §1.2.1](./05_signal_physics.md) |
| 3 | `PropagateAxons` | Безусловный `axon_heads[tid] += v_seg` для **всех** аксонов (Local + Ghost + Virtual) | [05_signal_physics.md §1.6](./05_signal_physics.md) |
| 4 | `UpdateNeurons` | GLIF + дендритный цикл + проверка порога + срыв спайка | [05_signal_physics.md §1.5](./05_signal_physics.md) |
| 5 | `ApplyGSOP` | Пластичность: Timer-as-Contact-Flag режим STDP | [05_signal_physics.md §1.3](./05_signal_physics.md) |
| 6 | `RecordReadout` | Чтение spike flags из mapped_soma_ids, запись в output_history | [05_signal_physics.md §3.2](./05_signal_physics.md) |


### 2.2. Фаза «Ночь» (Per-Zone Offline Maintenance)

Выполняется на **CPU**. Каждая зона имеет **свой цикл сна** - независимый от остальных.

**Триггеры засыпания:**
Проверяются оркестратором на CPU **только в конце каждого батча** `sync_batch_ticks` (во время сетевого барьера). Внутри рантайма GPU проверок сна нет - такты не тратятся.

| Триггер | Источник | Пример |
|---|---|---|
| **Таймер** | `night_interval_ticks` в конфиге зоны | V1: каждые 5 мин, гиппокамп: каждые 2 мин |
| **Внешний сигнал** | `sleep_zone(zone_id)` через API оркестратора | Массовый сон существа (моторика → сенсоры → ассоциация) |
| **Никогда** | `night_interval_ticks = 0` | Статические зоны (таламус, ствол - Variant = Fixed/Relay, GSOP заморожен) |

> **⚠️ Sentinel Refresh (зоны с `night_interval_ticks = 0`):**
> `AXON_SENTINEL = 0x80000000` ≈ 59.6 часов при v_seg=1. Без Night Phase неактивные аксоны переполнятся → фантомные спайки. **Решение:** Каждые ~50 часов (`SENTINEL_REFRESH_TICKS = 1_800_000_000`) host запускает лёгкий проход: все `axon_heads[id]` со значением `> SENTINEL_DANGER_THRESHOLD` принудительно сбрасываются в `AXON_SENTINEL`. Активные сигналы (head < propagation_length × 10) не затрагиваются.

**Конвейер Maintenance (5 шагов):**

| Шаг | Где | Название | Описание |
|---|---|---|---|
| **1** | **GPU** | **Sort & Prune** | Segmented Radix Sort: 128 слотов по `abs(weight)` (descending). Слоты с `abs(w) < threshold` обнуляются. Шина PCIe не забивается мусором. |
| **2** | **PCIe** | **Download** (VRAM → RAM) | `cudaMemcpyAsync` только изменённых массивов (веса + targets). Статическая геометрия уже известна хосту. |
| **3** | **CPU** | **Sprouting & Nudging** | Тяжёлая фаза. Cone Tracing для пустых слотов (Spatial Hash), рост отростков, создание Ghost Axons для межшардовых путей. Длительность зависит от железа и turnover rate. |
| **4** | **CPU** | **Baking** | Дефрагментация топологии → новый `.axons`. Подготовка SoA-массивов с выравниванием по 32 (Warp Alignment). |
| **5** | **PCIe** | **Upload** (RAM → VRAM) | `cudaMemcpyAsync` свежих данных. Шард мгновенно встраивается в барьер Strict BSP. |

**Длительность фазы Maintenance - плавающая.** Зависит от количества нейронов, turnover rate и мощности CPU. Быстрый CPU = быстрый метаболизм, короткий сон. Это легализовано через [Structural Determinism](./01_foundations.md) (§2.3).

**Возвращение по готовности:** Как только `cudaMemcpy` завершается, шард мгновенно встраивается в текущий барьер Strict BSP и продолжает работу.

#### 2.2.1. Step 1: GPU Sort & Prune (Детали)

**Проблема:** 128 дендритов лежат поколонно (Columnar Layout, stride = N). Глобальный Radix Sort с таким stride убьёт кэш.

**Решение: Shared Memory Staging**

1. Ядро загружает 128 слотов для 32 нейронов варпа в Shared Memory (AoS [Neuron][Slot])
   - Per slot: `weight` (i16, 2B) + `target` (u32, 4B) + `timer` (u8, 1B) = **7 bytes**
   - Per neuron: 128 × 7 = **896 bytes**
   - Per warp (32 neurons): **~28 KB** → идеально в Shared Memory (48-96 KB/SM)

2. **Bitonic Sort** (лучше Radix для N=128 на GPU) по `abs(weight)` descending - целочисленный, без float

3. **Auto LTM/WM Promotion:** Сортировка автоматически ставит сильнейшие связи в слоты 0-79 (LTM, low decay), слабые - в 80-127 (WM, high decay). Никакой ручной логики перемещения.

4. **Pruning:** Слоты в хвосте с `abs(weight) < prune_threshold` → `target_packed = 0` (Sentinel пустого слота)

5. Запись обратно в глобальную память в Columnar Layout

```
Shared Memory (AoS, per warp):
┌─────────────────────────────────────────────────┐
│ Neuron 0: [slot0_w, slot0_t, slot0_tmr] ... ×128│
│ Neuron 1: [slot0_w, slot0_t, slot0_tmr] ... ×128│
│ ...                                      ×32    │
└───────── Sort per neuron, write back ───────────┘
```

> **`weight = 0` ≠ `target = 0`:** Днём вес может упасть до 0 через GSOP depression - связь электрически молчит, но структурно жива (target ≠ 0). GSOP может поднять её обратно. Физическое удаление (`target = 0`) - только здесь, при Pruning.

#### 2.2.3. Step 3: Sprouting & Nudging (CPU, f32 легален)

Порядок строго последовательный: сначала растим кабели, потом ищем розетки.

**a) Nudging (Growth Step):**
- Аксоны с `remaining_length > 0` делают шаг через `step_and_pack()` (см. [04_connectivity.md §4.3](./04_connectivity.md)).
- Математика: `V_global + V_attract + V_noise` → `normalize` → `quantize` → `PackedPosition`.

**b) Boundary Check → NewAxon Handover:**
- Если координата вылетает за габариты шарда → аксон обрезается, формируется `NewAxon { entry_point, vector, type_mask }` в Slow Path очередь соседу (см. [06_distributed.md §2.5](./06_distributed.md)).

**c) Spatial Grid Rebuild:**
- Новые сегменты прописываются в 3D хэш-сетку (ключи из PackedPosition X|Y|Z). Обязателен до Sprouting - иначе `get_in_radius()` не увидит свежие аксоны.

**d) Sprouting (Slot Filling):**
- CPU сканирует массив `targets[]`. Если `target_packed == 0` - слот пуст.
- Cone Query: `calculate_v_attract()` в Spatial Grid (FOV + Lookahead).
- Фильтрация: тип владельца аксона = `seg_val >> 28` (4 бита из PackedPosition). Без обращения к соме.
- **Выбор кандидата** - тройной скоринг `sprouting_score()` по дистанции, `soma_power_index` и exploratory-шуму (см. [04_connectivity.md §1.6.1](./04_connectivity.md)). Веса конфигурируются по типу нейрона.
- Новый `target_packed` записывается, вес = базовый (74), слот попадает в WM (индексы 80-127).

#### 2.2.4. Step 4: Baking & Defragmentation (CPU)

**a) f32 → u32 Quantization:**
- Float-координаты квантуются через `step_and_pack()` → `PackedPosition` (4 bytes/segment).

#### 2.2.4. Step 4: Baking & Defragmentation (CPU)

**a) f32 → u32 Quantization:**
- Float-координаты квантуются через `step_and_pack()` → `PackedPosition` (4 bytes/segment).

**b) DenseIndex Generation:**
- GPU работает с dense indices (0..N-1), не с PackedPosition.
- CPU строит маппинг: `PackedPosition → dense_id` для всех `target_packed` в массиве дендритов.
- В массив `targets[]` вписываются DenseIndex + segment offset.

**c) Columnar Layout Defrag:**
- Новые связи вписываются в транспонированную матрицу `Column[Slot_K]`, не в конец массива.

**d) Warp Alignment:**
- `padded_n = align_to_warp(neuron_count)` → padding до кратного 32.
- Итоговые `.state` и `.axons` блобы байт-в-байт совпадают с VRAM layout → Step 5: `cudaMemcpyAsync`.

### 2.3. External I/O Server (UDP для входов/выходов)

Отдельный Tokio-сервер (на третьем ядре) для взаимодействия с External Hub. Обрабатывает I/O неблокирующе.

```rust
pub struct ExternalIoServer {
    sock_in: Arc<UdpSocket>,        // Port N: ресивер Input Bitmasks
    sock_out: Arc<UdpSocket>,       // Port N+1: сендер Output History
    last_client_addr: Option<SocketAddr>, // Память о клиенте
}

// Протокол пакета
#[repr(C)]
pub struct ExternalIoHeader {
    pub zone_hash: u32,     // идентификатор Zone
    pub matrix_hash: u32,   // идентификатор Input/Output матрицы
    pub payload_size: u32,  // размер пайлоада
}
```

**Дисфрагментация:** UDP пакеты больше 65KB автоматически дропятся (отсутствует EMSGSIZE отравления сокета). Полные передачи когда батч готов.

**Плагин**:
- На каждом батче (когда `current_tick_in_batch == 0`) сервер выслыла UDP датаграмму с `output_history` предыдущего батча клиенту (робоцика, визуализация).
- Одновременно вычитывает входящие `Input Bitmask` из датаграмм, сканирует через `try_recv_input()` в неблокирующем положении и ассоциирует пиксели с Virtual Axons (`InjectInputs`).

### 2.3.1. WaitStrategy: Управление CPU в Горячих Циклах

**Контекст:** День-фаза (GPU) автономна, но ночная фаза (Night Phase) и сетевой ввод (Network Phase, см. [06_distributed.md §2.10](./06_distributed.md)) требуют синхронизации с OS scheduler.

**Сценарий:** CPU ждёт данных от соседних шардов в BSP-барьере или дождается завершения I/O асинхронного. Без явного управления spin/yield ядро растрачивает кванты впустую.

**3 профиля (флаг `--cpu-profile`):**

| Профиль | Стратегия | Эффект | Сценарий |
|---|---|---|---|
| **Aggressive** | `std::hint::spin_loop()` | ~1 нс латентность, 100% CPU | Production, HFT, локальный кластер |
| **Balanced** | `std::thread::yield_now()` | OS берёт квант, процесс в очереди (~1–15 мс) | Дебаг, SSH-сессии, многопроцессный хост |
| **Eco** | `std::thread::sleep(1ms)` | ~0% CPU в холостую, батарея сохранена | Ноутбуки, мобильные, фоновые процессы |

```rust
pub enum WaitStrategy {
    Aggressive,
    Balanced,
    Eco,
}

impl WaitStrategy {
    pub fn poll_neighbors_until_ready(&self) -> Vec<SpikeBatch> {
        loop {
            if let Some(batch) = try_recv_all_neighbors() {
                return batch;
            }
            match self {
                Self::Aggressive => std::hint::spin_loop(),
                Self::Balanced => std::thread::yield_now(),
                Self::Eco => std::thread::sleep(Duration::from_millis(1)),
            }
        }
    }
}
```

**Инварианты:**

1. **Выбор на стартапе:** WaitStrategy фиксируется при инициализации runtime в `OnceLock<WaitStrategy>`. Нулевой cost для горячего цикла.
2. **Безопасность:** BSP-барьер - единственное место, где CPU физически ждёт события. Нет Mutex, нет CAS-loop.
3. **Портативность:** Ядро физики (GSOP, спайки, диффузия) идентично во всех профилях. Меняется только OS-level scheduling поведение.


### 2.4. Легализованная Амнезия (Spike Drop)

Пока зона спит, остальные зоны продолжают работать и слать спайки (Fast Path).

- Хост спящей зоны принимает TCP/UDP пакет, видит статус `SLEEP` → **мгновенный Drop**.
- Ноль копирований в VRAM. Ноль ветвлений. Информация теряется **физиологически достоверно**.
- **Биологический аналог:** Человек во сне не обрабатывает зрительный вход. Это нормальное поведение живой системы, не ошибка инференс-сервера.


## Connected Documents

| Document | Connection |
|---|---|
| [05_signal_physics.md](./05_signal_physics.md) | Day Pipeline kernels (§1.0), Constant Memory variant parameters |
| [06_distributed.md](./06_distributed.md) | Ring Buffer, Ghost Axons, BSP sync, network I/O |
| [02_configuration.md](./02_configuration.md) | Variant definitions, blueprints, parameter validation |
| [09_baking_pipeline.md](./09_baking_pipeline.md) | .state/.axons file format, Sort&Prune during Night |
| [project_structure.md](../project_structure.md) | Architecture overview |

---

## Changelog

| Дата | Версия | Описание |
|---|---|---|
| 2026-03-17 | 2.2 | Полная переработка раскладки VRAM и VariantParameters. Введен BurstHeads8 (32-byte alignment). Уточнена битовая семантика soma_flags. Исправлен inertia_curve (u8[16]). |
| 2026-02-28 | 2.1 | Синхронизирована VramState с реальным memory.rs (добавлены I/O Matrix поля, readout буферы). Обновлена таблица Day Phase с 6 kernels и ссылками на источники. Добавлен раздел External I/O Server для UDP мультиплексирования. |
| TBD | 2.0 | Первая версия |

---

## 3. Инварианты Жизненного Цикла (Lifecycle Invariants)

### 3.1. Cold Start: Sentinel Assert

> **⚠️ Baking Tool Assert:** Перед записью `.state` блоба Baking Tool обязан убедиться что весь массив `axon_heads` заполнен `AXON_SENTINEL` (`0x80000000`), а не нулями (`calloc`-default). Нули при старте вызовут эпилептический разряд всей коры в Тик 1 - гомеостатические пороги задерутся и система умрёт на старте.

```rust
// baking_compiler/src/validate.rs
assert!(
    axon_heads.iter().all(|&h| h == AXON_SENTINEL),
    "CRITICAL: axon_heads must be initialized to AXON_SENTINEL, not zero!"
);
```

### 3.2. Reset: O(1) Сброс и Блокирующая Ночь

При команде `reset_zone(zone_id)`:

1. **Если зона спит (Night Phase):** Сброс **блокирующий** - CPU дожидается завершения Maintenance pipeline (до Step 5 Upload включительно). Прерывание в середине оставит VRAM с дырявыми матрицами дендритов.
2. **Ring Buffer инвалидация (O(1)):** Обнуляются только `counts` обоих Ping-Pong буферов. Сами `ghost_id` не важны - GPU читает ровно `counts[tick]` записей. Предотвращает фантомные сигналы из прошлой жизни.

```rust
// O(1) - достаточно обнулить счётчики, не весь буфер
memset(schedule_a.counts, 0, batch_size * size_of::<u32>());
memset(schedule_b.counts, 0, batch_size * size_of::<u32>());
```

> **Phantom Signals & Input Bleed:** Фантомные сигналы из Ring Buffer при перезапуске - **легализованное биологическое поведение** (аналог дежавю при пробуждении). Input Bleed от асинхронного сенсора - аналогично. Не дефекты архитектуры.

### 3.3. Hot Checkpoint (Периодический Дамп на Диск)

Помимо дампа геометрии после каждой Night Phase, оркестратор делает **периодический снапшот** (`dendrite_weights` + `dendrite_targets`) в холодное хранилище:

```rust
const CHECKPOINT_INTERVAL_BATCHES: u32 =
    300_000_000 / TICK_DURATION_US / SYNC_BATCH_TICKS; // ≈ 5 минут

if batch_counter % CHECKPOINT_INTERVAL_BATCHES == 0 {
    cudaMemcpyAsync(host_buf, vram_weights, ..., DeviceToHost);
    // Атомарная запись: сначала .tmp, потом rename() - защита от краша
    write_to_disk("checkpoint_weights.bin.tmp");
    rename("checkpoint_weights.bin.tmp", "checkpoint_weights.bin");
}
```

| Тип дампа | Триггер | Файл |
|---|---|---|
| **Геометрия** (`axons`) | После каждой Night Phase | `.axons` |
| **Состояние** (`weights` + `targets`) | Каждые ~5 минут | `checkpoint_weights.bin` |

### 3.4. Crash Tolerance (Mmap Page Cache Flush)

При работе с Zero-Copy Mmap (`.geom`, `.paths`) ОС использует Page Cache. В случае неожиданного отключения питания или Kernel Panic, "грязные" (dirty) страницы могут не успеть сброситься на энергонезависимый носитель (NVMe SSD), что приведёт к рассинхронизации топологии.

**Асинхронный сброс (Flush Async):**
Сразу после успешного структурного обновления в Night Phase (вызов `run_sprouting_pass`), Baker Daemon инициирует `flush_async()` для `mmap_geom` и `mmap_paths`. Это не блокирует дальнейший ход симуляции, но заставляет ОС приоритетно отправить грязные страницы на диск, минимизируя окно уязвимости до миллисекунд.
