# 02. Конфигурация (Configuration)

> Часть архитектуры [Genesis](../../README.md). Файловая структура, пайплайн данных и TOML-спецификации.

---

## 1. Иерархия Конфигурации (Human Readable)

Конфиги делятся на три уровня ответственности. Разделяем, чтобы можно было менять «железо» нейрона, не ломая структуру слоёв.

**Законы Вселенной → Спецификация Зоны → Параметры Инстанса**

### 1.1. Глобальная Физика (`simulation.toml`)

Лежит в корне. Это константы, которые **не могут отличаться** между зонами, иначе сломается синхронизация времени и пространства.

- `tick_duration_us` - Шаг времени (в микросекундах).
- `voxel_size_um` - Размер вокселя (квант пространства).
- `signal_speed_um_tick` - Скорость распространения сигнала.
- `sync_batch_ticks` - Количество тиков автономного расчета между синхронизациями шардов.
- `master_seed` - Глобальный сид (`String`, хэшируется в `u64` при старте) для детерминированной генерации.
- `max_dendrites` - Хард-лимит (128).

**Влияние на память:** Это глобальные `const` переменные (Uniforms в CUDA). Грузятся один раз, всегда в кеше L1.

### 1.2. Конфиг Зоны (Папка `Genesis_Models/{project}/zones/V1/`)

Все файлы конкретной зоны лежат в ее директории. Это позволяет загружать в Runtime несколько зон параллельно из разных папок.
Определяет, **что** мы строим.

1. **`anatomy.toml`** - Слои (L1–L6) в процентах высоты. Позволяет менять масштаб (`world.height`), не переписывая конфиг.
   - Для V1: список слоёв, их высота в %, плотность нейронов.
   - Для «неслоистых» зон: один слой `"Nuclear"` высотой 100%.
   - Распределение типов (4-битная маска: Geo × Sign × Variant) в процентах для каждого слоя.
2. **`blueprints.toml`** - Типы нейронов: параметры мембраны (`threshold`, `leak`) и правила связности (матрица).
   - Параметры GSOP (`74`, `-2`, `64000`).
   - Специфические параметры мембран для каждого из 4-х типов.
3. **`io.toml`** - Карта входов/выходов:
   - Маппинг входных каналов (External Axons → слой L4).
   - Маппинг выходных каналов (аксоны из L5/L6 → Output Ports).
   - Разрешение, стратегия сбора и модальность для каждого канала.

### 1.3. Конфиг Инстанса (`runtime/shard_04.toml` или CLI аргументы)

> [!NOTE]
> **[Planned: Macro-3D]** Обновить DTO конфигурации шардов для 3D кластеров.
> // В `InstanceConfig` (`genesis-core/src/config/instance.rs`):
> // 1. Поля `world_offset` (x, y, z) должны стать `u64`.
> // 2. В `Neighbors` добавить `z_plus: Option<String>` и `z_minus: Option<String>`.

Определяет, **где** и **какой кусок** считает этот процесс.

- `zone_id` - `"V1"` (ссылка на папку зоны).
- `world_offset` - `{ x: 1000, y: 0, z: 0 }` (смещение этого шарда в глобальном мозге).
- `dimensions` - `{ w: 500, h: 2000 }` (размеры этого куска).
- `neighbors` - Список адресов (`IP:Port` или Shared Memory ID) для соседей (`X+`, `X-`, `Y+`, `Y-`). Поддерживает **Self-loop** (`= Self`) для реализации тороидальной топологии ядерных зон (см. [06_distributed.md §1.5](./06_distributed.md)).

#### 1.3.1. Секция `[settings]` (Runtime Optimization)

Параметры, влияющие на поведение конкретного инстанса, но не на биологическую логику.

- `save_checkpoints_interval_ticks` - Интервал (в тиках) между сохранениями горячего состояния (`checkpoint.state` / `.axons`).
  - **По умолчанию:** `100 000` (для высоконагруженных систем 2000+ TPS).
  - **Зачем:** Балансировка между частотой сохранения и нагрузкой на SSD (I/O).
  - **Расчет:** Рантайм вычисляет количество батчей как `interval_ticks / batch_size`.

---

## 2. Пайплайн: Запекание (Baking)

**Закон: Движок (Runtime) никогда не читает TOML файлы напрямую.**

Парсинг текстовых конфигов на GPU - смерть производительности. Вводится этап «Запекания» (Baking), который разделяет «Человеческие конфиги» (TOML) и «Машинные данные» (Binary Blobs). Все результаты сохраняются в `Genesis_Models/`.

### 2.1. Конвейер

```
TOML конфиги ──→ [Compiler Tool (CPU)] ──→ Бинарные файлы ──→ [Runtime (GPU)]
                  │                          │
                  ├─ Парсит TOML             ├─ .axons (геометрия)
                  ├─ Генерирует нейроны      ├─ .state (начальное состояние)
                  ├─ Растит аксоны           └─ .dendrites (матрица связей)
                  └─ Раскладывает в SoA
```

- **Compiler Tool:** Утилита на CPU - читает TOML, генерирует нейроны, растит аксоны (Cone Tracing), раскладывает данные в плоские массивы.
- **Zero-Copy Loading:** Движок делает `mmap` (отображение файла в память) или прямой `cudaMemcpy` из бинарника в VRAM. Никакого парсинга при старте.

### 2.2. Выравнивание (Alignment)

- Данные на диске лежат **байт-в-байт** так, как они нужны видеокарте.
- Если структура данных занимает 28 байт - добиваем паддингом до **32 байт** для Coalesced Access.
- Время загрузки миллиарда нейронов = скорость чтения SSD.

### 2.3. Валидация Конфигурации (Baking Tool Asserts)

При парсинге `simulation.toml` Baking Tool вычисляет производные величины и проверяет инварианты:

```rust
// baking_compiler/src/physics_constants.rs

let segment_length_um = config.voxel_size_um * config.segment_length_voxels;
let v_seg = config.signal_speed_um_tick / segment_length_um;

// Runtime assert: v_seg обязан быть целым (01_foundations.md §1.6)
assert!(
    config.signal_speed_um_tick % segment_length_um == 0,
    "CRITICAL: signal_speed_um_tick must be a multiple of segment_length_um!"
);
```

### 2.4. Формат `.state`: ShardStateSoA

Бинарный файл `.state` - плоская склейка массивов. Структура **байт-в-байт** совпадает с раскладкой VRAM.

```rust
// baking_compiler/src/layout.rs

/// Warp Alignment: размер кратен 32
pub const fn align_to_warp(count: usize) -> usize {
    (count + 31) & !31
}

pub struct ShardStateSoA {
    pub padded_n: usize,       // Нейроны (aligned to 32)
    pub total_axons: usize,    // Local + Ghost + Virtual (aligned to 32)

    // Soma arrays (size = padded_n)
    pub voltage: Vec<i32>,           // 4 bytes × N
    pub flags: Vec<u8>,              // 1 byte × N (upper nibble: Type, bit 0: Is_Spiking)
    pub threshold_offset: Vec<i32>,  // 4 bytes × N
    pub refractory_counter: Vec<u8>, // 1 byte × N
    pub soma_to_axon: Vec<u32>,      // 4 bytes × N (маппинг Soma → Local Axon ID)

    // Dendrite arrays - Columnar Layout (size = 128 × padded_n)
    // Обращение в CUDA: data[slot * padded_n + tid]
    pub dendrite_targets: Vec<u32>,  // 4 bytes × 128 × N (Packed: 22b Axon_ID | 10b Segment_Index)
    pub dendrite_weights: Vec<i16>,  // 2 bytes × 128 × N (Signed: знак = E/I)
    pub dendrite_timers: Vec<u8>,    // 1 byte × 128 × N

    // Axon arrays (size = total_axons, NOT padded_n!)
    pub axon_heads: Vec<u32>,        // 4 bytes × A (PropagateAxons: += v_seg)
}
```

**Seed-детерминированная генерация:** PackedPosition (`u32`) используется как `entity_id` для хэширования:

```rust
pub fn entity_seed(master_seed: u64, packed_pos: u32) -> u64 {
    wyhash::wyhash(&packed_pos.to_le_bytes(), master_seed)
}
```

### 2.5. Ночной Цикл (Night Baking)

Baking - не одноразовая операция. Каждую «Ночь» (см. [07_gpu_runtime.md](./07_gpu_runtime.md)) Compiler Tool перегенерирует бинарные файлы с учётом изменений (sprouting, nudging, дефрагментация), и движок загружает свежий `.axons` как чистый лист. Все бинарные артефакты обновляются в `Genesis_Models/{project}/baked/{zone}/`.

---

## 3. Закон Хранения Данных: SoA (Structure of Arrays)

**Полный отказ от объектов в памяти GPU.** Это не рекомендация, это закон архитектуры.

### 3.1. Проблема: AoS (Array of Structures)

```
// Плохо (Cache Miss)
struct Neuron { pos: Vec3, voltage: f32, type_id: u8 }
neurons: [Neuron; N]
```

Чтобы прочитать `voltage` всех нейронов, GPU загружает в кеш `pos` и `type_id` - **мусор**, который не нужен. Полезная нагрузка кеш-линии ≈ 15%.

### 3.2. Решение: SoA

```
// Хорошо (Cache Hit, Coalesced Access)
all_voltages:  [f32; N]    // Непрерывный массив
all_types:     [u8; N]     // Непрерывный массив
all_positions: [Vec3; N]   // Непрерывный массив
```

Варп GPU (32 потока) читает 32 подряд идущих `f32` за **одну транзакцию памяти**. Полезная нагрузка кеш-линии = 100%.

### 3.3. Транспонированные Дендриты (Columnar Layout)

Самое узкое место по памяти: 128 дендритов на нейрон.

- **Не** `Neuron.Dendrites[0..127]` (построчно).
- **А** `Column[Slot_K]` - K-й дендрит для **всех** нейронов подряд (поколонно).

```
Slot_0:   [Нейрон_0.Д0, Нейрон_1.Д0, Нейрон_2.Д0, ...]   // Массив размером N
Slot_1:   [Нейрон_0.Д1, Нейрон_1.Д1, Нейрон_2.Д1, ...]   // Массив размером N
...
Slot_127: [Нейрон_0.Д127, Нейрон_1.Д127, ...]              // Массив размером N
```

В цикле `for slot in 0..128` все потоки GPU обращаются к `Slot_K` - идеально линейное чтение. Bandwidth используется на 100%.

---

## 4. Спецификация: `simulation.toml`

Определяет «песочницу», в которой живет система.

```toml
[world]
# Физические размеры пространства (в микрометрах)
# Миниколонка во-первых V1: ~1mm × 1mm × 2.5mm (коры)
width_um = 1000    # = 40 voxels × 25 um
depth_um = 1000    # = 40 voxels × 25 um
height_um = 2500   # = 100 voxels × 25 um (толщина коры)

[simulation]
# --- Глобальные параметры (единые для всех зон) ---

# Временное разрешение
tick_duration_us = 100  # 1 тик = 100 мкс (0.1 мс). Необходимое разрешение для GSOP (01_foundations.md §1.4).

# Пространственная дискретизация
voxel_size_um = 25       # Единица квантирования пространства (01_foundations.md §1.1).
                         # 25 μm - компромисс между разрешением и памятью для кортикальных нейронов.

# Параметры аксонов
segment_length_voxels = 2   # 1 сегмент = 2 вокселя = 50 мкм. Максимум: 8 вокселей (01_foundations.md §1.2).
signal_speed_um_tick = 50   # Скорость сигнала в микрометрах за тик (01_foundations.md §1.5).
                            # Вычисляется как: v_seg = signal_speed_um_tick ÷ (voxel_size_um × segment_length_voxels) = 50 ÷ 50 = 1 ✓

# Синхронизация
sync_batch_ticks = 100      # Количество тиков между синхронизациями шардов (100 тиков × 100 мкс = 10 мс).

# Контроль и инициализация
total_ticks = 0             # 0 = бесконечная симуляция (до завершения программы)
master_seed = "GENESIS"     # String → хэшируется в u64 при старте (01_foundations.md §2.1).
                            # Единственная точка входа энтропии. Используйте читаемые значения: "GENESIS", "DEBUG_RUN_42", т.д.

# [DOD FIX] global_density УДАЛЕН.
# Плотность теперь вычисляется строго per-layer в anatomy.toml (bottom-up аллокация).

# Рост аксонов (Baking/Sprouting)
axon_growth_max_steps = 500     # Максимум итераций Cone Tracing при росте (500 шагов × 50 мкм = 25 мм)
```

### 4.1. Оптимизация Типов Данных (Memory Optimization)

Понижаем битность счётчиков, опираясь на архитектуру День/Ночь ([07_gpu_runtime.md](./07_gpu_runtime.md)):

| Счётчик                                      | Было  | Стало | Обоснование                                                                         |
| -------------------------------------------- | ----- | ----- | ----------------------------------------------------------------------------------- |
| **Global Clock**                             | `u64` | `u32` | Хватает на ~5 дней непрерывной работы (при тике 100 мкс). Сбрасывается в фазу Ночи. |
| **Local Timers: Refractory** (soma, synapse) | `u64` | `u8`  | Max 255 тиков = 25 мс. Физиологически достаточно для рефрактерности.                |
| **Local Timers: Delays** (Delay_Counter)     | `u64` | `u16` | 65 535 тиков = 6.5 сек. Межзональные задержки (Projectors).                         |
| **Batch Size**                               | `u64` | `u32` | Пакет синхронизации шарда не может быть больше 4 млрд тиков.                        |

### 4.2. Архитектурные Константы (Immutable)

Параметры, которые «запекаются» при старте и не меняются в рантайме:

| Параметр               | Тип      | Значение    | Примечание                                                                        |
| ---------------------- | -------- | ----------- | --------------------------------------------------------------------------------- |
| `tick_duration_us`     | `u32`    | `100`       | 0.1 мс                                                                            |
| `voxel_size_um`        | `f32`    | `25.0`      | Квант пространства                                                                |
| `signal_speed_um_tick` | `u16`    | `50`        | Рассчитывается при старте. `int` для быстрого сложения.                           |
| `max_dendrites`        | `const`  | `128`       | Hard Constraint. Гарантирует выравнивание памяти по 128 байт (кеш-линии + варпы). |
| `master_seed`          | `String` | `"GENESIS"` | Хэшируется в `u64` при старте (см. [01_foundations.md §2](./01_foundations.md)).  |

### 4.3. Правило Выравнивания (Warp Alignment)

Требование к Compiler/Baking Tool (§2):

- **Padding Rule:** Количество активных нейронов в бинарном файле шарда (`.state`) всегда добивается пустышками до числа, **кратного 32**.
- **Зачем:** При запуске ядра `UpdateNeurons<<<N/32, 32>>>` все потоки в варпах заняты делом, доступ к памяти строго Coalesced. Без паддинга последний варп может содержать «мусорные» потоки, которые читают за пределы массива.

---

## 5. Спецификация: `anatomy.toml`

Определяет анатомию конкретной зоны: слои, плотность, состав.

### 5.1. Вертикальная Метрика (Relative Height)

- Слои задаются строго в процентах (`height_pct`) от `0.0` до `1.0`.
- **Инвариант:** `L1 + L2 + ... + L6 = 1.0`.
- **Зачем:** Можно менять физическую высоту зоны (растянуть кору с 2 мм до 3 мм) через `world.height`, не переписывая файл анатомии.

### 5.2. Управление Плотностью (Absolute Layer Density)

Плотность задаётся строго для каждого слоя:

1. **`density`** (в `anatomy.toml`): Процент заполненности вокселей именно в этом физическом объеме. Бюджет нейронов больше не является глобальным.
   - Пример: L4 имеет `density = 0.08` (8% плотность), L5 имеет `density = 0.02` (2% плотность).

### 5.3. Композиция Типов (Hard Quotas)

> **[!IMPORTANT]**
> Отказ от вероятностной генерации (RNG) в пользу **жёсткого квотирования**.

- **Было (черновик):** «Вероятность спавна типа А = 80%» → при малом N результат плавает.
- **Стало:** «Бюджет типа А = ровно 80% от населения слоя».
  - Если в слое 1000 мест → система **обязана** создать ровно 800 нейронов типа А.
- **Размещение:** Типы перемешиваются в пространстве **детерминированным** алгоритмом (`Shuffle` на базе `master_seed`), чтобы не было кучности. Количество математически точно совпадает с конфигом.

### 5.4. Универсальность Структуры

Схема применима не только к коре:

| Зона                           | Слоёв | Подход                                                          |
| ------------------------------ | ----- | --------------------------------------------------------------- |
| **Кора** (V1, V2, Motor)       | 6     | `L1..L6` с уникальными `height_pct` и `composition`             |
| **Таламус, Базальные ганглии** | 1     | Один слой `"Nuclear"` с `height_pct = 1.0` и своим миксом типов |

Каждая зона (`V1`, `V2`, `Motor`, `Thalamus`) имеет свою отдельную папку и свой `anatomy.toml`.

### 5.5. Пример и Логика Расчёта (Математика Baker'а)

```toml
# config/zones/CortexPie/anatomy.toml

[[layer]]
name = "L4_Sensory"
height_pct = 0.2
# [DOD FIX] Абсолютная плотность вокселей в этом слое (8% вокселей содержат сому)
density = 0.08
composition = { "L4_spiny_MTG_1" = 0.8, "L4_aspiny_MTG_11" = 0.2 }

[[layer]]
name = "L23_Hidden"
height_pct = 0.5
density = 0.04  # Средняя плотность (4%)
composition = { "L23_spiny_MTG_1" = 0.8, "L23_aspiny_MTG_1" = 0.2 }

[[layer]]
name = "L5_Motor"
height_pct = 0.3
density = 0.02  # Гигантские пирамиды требуют много места, плотность сом низкая (2%)
composition = { "L5_spiny_MTG_1" = 0.8, "L5_aspiny_MTG_11" = 0.2 }
```

**Логика расчёта при старте (Baking):**

- Определение общих габаритов в вокселях: `Max_X = Width_um / voxel_size`, `Max_Y = Depth_um / voxel_size`, `Max_Z = Height_um / voxel_size`
- Обнуление глобального счетчика `Total_Capacity = 0`.
- Для каждого слоя `[[layer]]`:
  - `Z_Start = Σ(предыдущих height_pct) × Max_Z`
  - `Z_End = Z_Start + height_pct × Max_Z`
  - `Layer_Volume_Voxels = Max_X × Max_Y × (Z_End - Z_Start)`
  - `Layer_Budget = floor(Layer_Volume_Voxels × layer.density)`
  - `Total_Capacity += Layer_Budget`
  - Для каждого типа в `composition`:
    - `Type_Count = floor(quota × Layer_Budget)`
- Warp Alignment: `Padded_N = align_to_warp(Total_Capacity)` (добиваем пустышками до кратного 32).

---

## 6. Спецификация: `blueprints.toml`

Определяет физические «ТТХ» для каждого из 4-х типов нейронов. Используется **Целочисленная Физика (Integer Physics)** для гарантии детерминизма и снижения нагрузки на FPU. Свойства мембраны не зависят от размера коры - абсолютные величины.

### 6.1. Структура Конфигурации

```toml
[[neuron_type]]
# ID 0 (Биты: 00) - Основной возбуждающий
name = "Vertical_Excitatory"

# --- Параметры Мембраны (Units: microVolts / absolute integers) ---
threshold = 42000               # Порог срабатывания
rest_potential = 10000           # Потенциал покоя
leak_rate = 1200                 # Скорость утечки (вычитание каждый тик)

# --- Adaptive Leak (LTC Milestone 3, optional, off by default) ---
# adaptive_leak_mode = 0          # 0 = disabled, 1 = dopamine+burst modulation
# dopamine_leak_gain = 0          # (dopamine * gain) >> 7 added to leak
# burst_leak_gain = 0             # burst_count * gain added to leak
# leak_min = 0                   # Clamp lower bound (must be < leak_max to enable)
# leak_max = 0                   # Clamp upper bound

# --- Тайминги (Units: Ticks) ---
refractory_period = 15           # u8. Абсолютная рефрактерность сомы.
synapse_refractory_period = 15   # u8. Рефрактерность входного порта (дендрита).

# --- Физика Сигнала (Units: Integer geometry) ---
conduction_velocity = 200        # [Архитектурно под вопросом] Скорость (дискретное смещение за тик)
signal_propagation_length = 10   # Длина "хвоста" сигнала в сегментах (Active Tail, per-variant)

# --- Гомеостаз (Adaptive Threshold) ---
homeostasis_penalty = 5000       # +5000 к порогу после спайка
homeostasis_decay = 10           # Вычитается из порога каждый тик, пока threshold не вернётся к базовому

# --- Slot Decay (GSOP множители, Fixed-point: 128 = 1.0×) ---
slot_decay_ltm = 160             # Слоты 0-79: 160/128 = 1.25× (усиленное удержание)
slot_decay_wm = 96               # Слоты 80-127: 96/128 = 0.75× (ускоренный распад)

# --- Рост аксона (Steering & Arborization) ---
steering_fov_deg = 60.0              # Угол обзора конуса (Cone Tracing)

# Параметры терминального ветвления (Terminal Arborization, см. 04_connectivity.md §1.4):
arborization_target_layer = "+1"     # Целевой слой ("+1" = следующий вверх по Z, или имя слоя, например "L4")
arborization_radius_um = 150.0       # Радиус разрастания "кроны" по достижении цели
arborization_density = 0.8           # Коэффициент петляния (0.0 = нет арборизации, 1.0 = максимальный хаос V_noise)

sprouting_weight_distance = 0.5  # f32. Ближний = лучше
sprouting_weight_power   = 0.4  # f32. soma_power_index (См. 04_connectivity.md §1.6.1)
sprouting_weight_explore = 0.1  # f32. Шум по эпохе (защита от повторных выборов)

# --- Спонтанная активность (Фоновый шум) ---
spontaneous_firing_period_ticks = 500  # Спайк в среднем раз в 500 тиков (0 = отключено)

[[neuron_type]]
# ID 3 (Биты: 11) - Тормозной интернейрон
name = "Horizontal_Inhibitory"

# --- Мембрана ---
threshold = 40000                # Чуть ниже порог (легче возбуждается)
rest_potential = 10000
leak_rate = 1500                 # Быстрее «остывает» (Leak выше)

# --- Тайминги ---
refractory_period = 10           # Быстрее восстанавливается
synapse_refractory_period = 5

# --- Физика Сигнала ---
conduction_velocity = 100        # [Архитектурно под вопросом] Медленный сигнал

# --- Гомеостаз ---
homeostasis_penalty = 3000
homeostasis_decay = 15

# --- Slot Decay ---
slot_decay_ltm = 140             # 140/128 = 1.09× (чуть слабее удержание)
slot_decay_wm = 80               # 80/128 = 0.625× (агрессивный распад)

# --- Sprouting Score (интернейроны меньше ценят мощные хабы) ---
sprouting_weight_distance = 0.6  # Больше ценят локальные связи
sprouting_weight_power   = 0.3  # Меньше зависимость от хабов
sprouting_weight_explore = 0.1

# --- Спонтанная активность (Фоновый шум) ---
spontaneous_firing_period_ticks = 200  # Тормозные - чаще срабатывают (раз в 200 тиков)
```

### 6.3. Компиляция DDS Heartbeat

Baker преобразует человеко-читаемый `spontaneous_firing_period_ticks` во фракционный множитель `heartbeat_m` для GPU (см. [03_neuron_model.md §4.1](./03_neuron_model.md#41-математика-zero-cost-branchless)):

- Если `period == 0`: `heartbeat_m = 0` (спонтанная активность отключена).
- Иначе: `heartbeat_m = clamp(65536 / period, 1, 65535)` (распределение фазы DDS).

**Примеры расчёта:**

- `period = 500` → `heartbeat_m = 65536 / 500 ≈ 131` → В среднем 1 спайк на 500 тиков
- `period = 200` → `heartbeat_m = 65536 / 200 ≈ 328` → В среднем 1 спайк на 200 тиков
- `period = 1` → `heartbeat_m = 65535` → Каждый тик срабатывает (spike generator, не рекомендуется)

**Реализация (Baking Tool):**

```rust
// genesis-baker/src/compile_heartbeat.rs

pub fn compile_dds_heartbeat(period_ticks: u32) -> u16 {
    if period_ticks == 0 {
        return 0;  // Отключено
    }

    // DDS фаза: 0..65535 (16 битов)
    let heartbeat_m = (65536 / period_ticks.max(1)).min(65535) as u16;

    // Проверка пределов
    assert!(heartbeat_m > 0, "period_ticks > 65536 is invalid (heartbeat too rare)");
    assert!(heartbeat_m <= 65535, "Critical: heartbeat_m overflow");

    heartbeat_m
}
```

### 6.4. Алгоритмическая деривация D1/D2 рецепторов

В `blueprints.toml` для каждого типа нейрона (Variant) вычисляются параметры аффинности к дофамину.

1. **d1_affinity (LTP-like):**
   - Поток: `Is_Excitatory` → `High` (1.5x), `Is_Inhibitory` → `Low` (0.5x).
   - Расчет: `d1_affinity = variant.is_excitatory ? 192 : 64`.
2. **d2_affinity (LTD-like):**
   - Поток: `Is_Excitatory` → `Medium` (1.0x), `Is_Inhibitory` → `High` (2.0x).
   - Расчет: `d2_affinity = variant.is_excitatory ? 128 : 256`.

**Инвариант:** Эти значения запекаются в `VariantParameters` (1 байт каждое) и используются ядром `ApplyGSOP` для нелинейной модуляции весов в зависимости от глобального уровня дофамина.

---

## 7. Спецификация: `brain.toml` (Multi-Zone Architecture)

> [!NOTE]
> **[Planned: Macro-3D]** В `brain.toml`:
> // Добавить сущность `[cluster]` для описания макро-геометрии (размеры изотропного куба, лимиты нод), которая будет выше уровня `[simulation]`.

Определяет топологию мультизонального мозга: какие зоны присутствуют, где вычисляются данные и как они синхронизируются. Это **корневой конфиг** для всей распределённой системы (см. [06_distributed.md](./06_distributed.md)).

### 7.1. Структура и Иерархия

```
brain.toml (корень)
├─ [simulation] → simulation.toml (глобальная физика)
├─ [[zone]] × N (определяет зоны)
│  ├─ name: "SensoryCortex"
│  ├─ blueprints: "config/zones/.../blueprints.toml"
│  └─ baked_dir: "baked/SensoryCortex/"
└─ [[connection]] × M (межзональные связи через Ghost Axons)
   ├─ from: "SensoryCortex"
   ├─ to: "HiddenCortex"
   └─ output_matrix: "sensory_out"
```

**Инвариант:** Каждая зона ссылается на папку `baked_dir` внутри `Genesis_Models`. Отсутствие файла в `baked_dir` → **критическая ошибка при инициализации** (Must have: `.state`, `.axons`, `.gxo`).

**Smart Path Resolution:** Рантайм поддерживает умный поиск. Если указан `--brain mouse_agent`, нода попробует найти `Genesis_Models/mouse_agent/brain.toml`. Путь также может быть задан через переменную окружения `GENESIS_MODELS_PATH`.

### 7.2. Секция [simulation]

Ссылка на единственный `simulation.toml` - все зоны синхронизированы одинаковым `tick_duration_us` и пространственной метрикой.

```toml
[simulation]
config = "config/simulation.toml"  # Абсолютный или относительный путь
```

**Обоснование:** Нельзя, чтобы SensoryCortex считал тики быстрее, чем HiddenCortex, иначе входная синхронизация рассинхронизируется (см. [06_distributed.md §2.4](./06_distributed.md#24-batch-synchronization)).

### 7.3. Секция [[zone]]: Определение Зон

```toml
[[zone]]
name = "SensoryCortex"                                 # Имя зоны (уникально)
blueprints = "Genesis_Models/mouse_agent/zones/SensoryCortex/blueprints.toml"  # Путь к типам
baked_dir = "Genesis_Models/mouse_agent/baked/SensoryCortex/"                 # Путь к бинарникам

[[zone]]
name = "HiddenCortex"
blueprints = "config/zones/HiddenCortex/blueprints.toml"
baked_dir = "baked/HiddenCortex/"

[[zone]]
name = "MotorCortex"
blueprints = "config/zones/MotorCortex/blueprints.toml"
baked_dir = "baked/MotorCortex/"
```

#### Поля

| Поле         | Тип      | Описание                                                                                                                                                                                                                   |
| ------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`       | `String` | Уникальный идентификатор зоны. Используется в `[[connection]]` и шардовых конфигах. Примеры: `"V1"`, `"V2"`, `"Motor"`, `"Thalamus"`.                                                                                      |
| `blueprints` | `String` | Путь к `blueprints.toml` этой зоны. Содержит определения 4-х типов нейронов (§6). Может быть абсолютным или относительным к позиции `brain.toml`.                                                                          |
| `baked_dir`  | `String` | Папка, содержащая бинарные файлы: `.state` (начальное состояние сомы), `.axons` (геометрия), `.gxo` (выходная матрица, опционально), `.gxi` (входная матрица, опционально). Создаётся инструментом Baking (genesis-baker). |

### 7.4. Секция [[connection]]: Межзональные Связи

Определяет Ghost Axon Projections - как выходные аксоны одной зоны подключаются в качестве входных к соседней (см. [06_distributed.md §2.5](./06_distributed.md)).

```toml
[[connection]]
from = "SensoryCortex"                  # Исходящая зона
to = "HiddenCortex"                     # Целевая зона
output_matrix = "sensory_out"           # Имя выходного матрикса в исходящей зоне
                                        # (debe existir в SensoryCortex/blueprints.toml)
width = 64                              # Ширина матрицы в пикселях
height = 64                             # Высота матрицы в пикселях
entry_z = "top"                         # Глубина спавна: "top" | "mid" | "bottom" | число в мкм
target_type = "All"                     # Целевые типы нейронов: "All" | конкретный тип
growth_steps = 1000                     # Максимум итераций Cone Tracing для Ghost Axon Growth

# Опциональные поля (Planned, не реализовано в текущей версии)
# synapse_weight = 5000                 # Начальный вес синапса Ghost Axon
# latency_ticks = 2                     # Задержка распространения (межзональная)
```

#### Интерпретация

1. **from/to:** Зона с `from` должна иметь `output_matrix` с именем `sensory_out` в её `blueprints.toml`.
2. **width/height:** Разрешение проекции. Вычисляется матричный масштаб: `spatial_scale_x = zone_width / width` (см. [08_io_matrix.md](./08_io_matrix.md)).
3. **entry_z:** На какой высоте (слой) спавнятся входные аксоны в целевую зону. `"top"= слой L1, "bottom" = слой L5/6`.
4. **target_type:** К каким нейронам прикрепляются дендриты Ghost Axon. `"All"` = ко всем типам.
5. **growth_steps:** Параметр алгоритма синтеза связей. При значении 0 - связи **заранее известны** (читаются из бинарника). При значении > 0 - **динамический рост** (Cube Tracing в целевой зоне).

### 7.5. Пример Полной Конфигурации

```toml
[simulation]
config = "config/simulation.toml"

[[zone]]
name = "SensoryCortex"
blueprints = "config/zones/SensoryCortex/blueprints.toml"
baked_dir = "baked/SensoryCortex/"

[[zone]]
name = "HiddenCortex"
blueprints = "config/zones/HiddenCortex/blueprints.toml"
baked_dir = "baked/HiddenCortex/"

[[zone]]
name = "MotorCortex"
blueprints = "config/zones/MotorCortex/blueprints.toml"
baked_dir = "baked/MotorCortex/"

# --- Межзональная контрастивная экономика ---

[[connection]]
# V1 → Hidden: сенсорные входы
from = "SensoryCortex"
to = "HiddenCortex"
output_matrix = "sensory_out"
width = 64
height = 64
entry_z = "top"
target_type = "All"
growth_steps = 1000

[[connection]]
# Hidden → Motor: моторные ответы
from = "HiddenCortex"
to = "MotorCortex"
output_matrix = "hidden_out"
width = 32
height = 32
entry_z = "mid"
target_type = "All"
growth_steps = 800
```

### 7.6. Runtime Инициализация (Startup Sequence)

Движок при инициализации:

1. **Парсит `brain.toml`.**
2. **Для каждой `[[zone]]`:**
   - Загружает `blueprints.toml` → структура нейротипов
   - Загружает бинарные файлы из `baked_dir/` в VRAM (`.state`, `.axons`)
   - Опционально загружает `.gxo` (выходные матрицы) и инициирует UDP output servers
   - Создаёт `ZoneRuntime` с этими данными
3. **Для каждой `[[connection]]`:**
   - Проверяет существование `from` и `to` зон в `zones`
   - Создаёт `IntraGpuChannel` (voir [06_distributed.md §2.6](./06_distributed.md#26-intra-gpu-channel-ghost-axon-routing)) для маршрутизации Ghost Axons
   - Синтезирует связи через Cone Tracing (если `growth_steps > 0`) или загружает из бинарника
4. **Инициирует BSP Barrier** между зонами (§1.2.1, [06_distributed.md §1.3](./06_distributed.md#13-bsp-barrier))

**Fail-Fast Policy:** Если какой-то путь недоступен или файл повреждён → immediate panic с диагностикой. Сломанный `baked_dir/` → **not bootable**.

---

## Сводка Иерархии

```
simulation.toml (Laws of Physics)
    ↓
anatomy.toml (Layer heights, Density per layer)
    ↓
blueprints.toml (Neuron types, Synapse rules)
    ↓
brain.toml (Multi-zone topology, Inter-zone connections)
    ├→ baked/ (Compiled, immutable binary snapshots)
    └→ [Shard configs] (Instance-specific: offset, dimensions)
```

### 6.2. Типы Данных в Runtime

Пояснения по типам, которые используются в GPU-массивах после Baking (§2):

| Группа          | Параметры                                        | Тип Runtime   | Обоснование                                                                 |
| --------------- | ------------------------------------------------ | ------------- | --------------------------------------------------------------------------- |
| **Potentials**  | `threshold`, `rest_potential`, `voltage`         | `i32`         | Запас от переполнения при суммации входов                                   |
| **Timers**      | `refractory_period`, `synapse_refractory_period` | `u8`          | Значения > 255 тиков (25 мс) физиологически бессмысленны для рефрактерности |
| **Geometry**    | `conduction_velocity`                            | `u16`         | [Архитектурно под вопросом] Дискретные единицы, диапазон до 65535           |
| **Homeostasis** | `homeostasis_penalty`, `homeostasis_decay`       | `i32` / `u16` | Penalty суммируется с threshold (i32), decay - малое значение               |

---

## 8. Roadmap: Расширенный Hot-Reload (Manifest Extensions)

> [!NOTE]
> **[Planned]** Реализовать расширение `manifest.toml` для управления параметрами в горячем цикле.

### 8.1. Динамические интервалы (Night Phase)

- **Цель**: Позволить менять `night_interval_ticks` индивидуально для каждого шарда без перезапуска.
- **Обоснование**: Частая «Ничья Фаза» (консолидация, прунинг, спраутинг) на ранних этапах обучения может сбивать динамику формирования функциональных цепочек. Возможность отодвинуть «ночь» в горячем режиме позволит сети дольше находиться в фазе эксплорации.

### 8.2. Параметры Пластичности

- Перенос `gsop_potentiation`, `gsop_depression` и `prune_threshold` из константной памяти в текстурные буферы или динамические переменные, обновляемые через манифест.
