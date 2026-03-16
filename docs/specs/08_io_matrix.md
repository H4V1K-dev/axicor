# 08. Матричный I/O Интерфейс

> Часть архитектуры [Genesis](../../README.md). Единая абстракция ввода/вывода и межзональных связей.

---

## 1. Концепция

Внешний мир и соседние зоны взаимодействуют с зоной мозга через **матрицы** - 2D-сетки фиксированного размера W×H.

- **Входная матрица** - каждый пиксель = виртуальный аксон внутри зоны
- **Выходная матрица** - каждый пиксель = привязанная сома внутри зоны
- **Межзональная связь** - выходная матрица зоны A → входная матрица (ghost axons) зоны B

Снаружи все интерфейсы выглядят одинаково: плоский массив ID, адресуемый как `matrix[y * W + x]`.

> **Инвариант: Baking Freeze.** Топология I/O (какие матрицы, какие аксоны, какие сомы) **статична после Baking**. Нельзя добавить, убрать или изменить матрицу в runtime. Перекомпиляция = повторный Baking.

---

## 2. Входная Матрица (Virtual Axons)

### 2.1. Размещение

Матрица W×H **растягивается** на X-Y плоскость зоны. Каждый пиксель отображается в пространственный регион зоны:

```
region_x = pixel_x * (zone_width  / W)
region_y = pixel_y * (zone_depth  / H)
```

Один пиксель = один кластер нейронов. Сколько сом попадает в один пиксель - вопрос планирования архитектуры зоны (размер зоны, плотность нейронов, резолюция матрицы).

### 2.2. Spawning по Z

Виртуальный аксон **спавнится** в вычисленной (x, y) позиции на конфигурируемой высоте Z:

```toml
# io.toml
[[input]]                 # входная матрица
name = "retina"           # имя входа
zone = "SensoryCortex"    # зона, в которую спавнится вход
width = 64                # ширина матрицы в пикселях
height = 64               # высота матрицы в пикселях
entry_z = "top"           # "top" | "mid" | "bottom" | конкретное значение в um
target_type = "All"       # "All" | конкретный тип нейрона
growth_steps = 1500       # максимальная длина роста аксона (шаги Cone Tracing)
empty_pixel = "skip"      # "skip" | "nearest" - что делать если в регионе пикселя нет нейронов target_type
stride = 1                # каждый N-й тик: 0 = снимок (тик 0 только), 1 = каждый тик, 2 = каждый 2-й, ...
```

Варианты `entry_z`:
- `"top"` - Z_max (вход сверху, как thalamo-cortical)
- `"mid"` - Z_mid (вход в среднюю часть столба)
- `"bottom"` - Z_min
- Число - конкретная координата в микрометрах (um)

Варианты `empty_pixel`:
- `"skip"` - пиксель без кандидатов не создаёт аксон (потеря ID в матрице)
- `"nearest"` - назначается ближайшая сома target_type из соседнего региона

### 2.3. Рост

От точки спавна виртуальный аксон **растёт на общих основаниях** - используя тот же механизм Cone Tracing что и локальные аксоны:

- Длина роста задаётся параметром `growth_steps` в блоке `[[input]]` (io.toml)
- Cone Tracing придаёт органичность - аксон не идёт строго прямо, а «бродит». Параметры брожения (cone angle, steering weight, и т.д.) берутся из `blueprints.toml`, блок `[virtual_axon_growth]`. Этот блок переопределяет глобальные настройки `[cone_tracing]`, позволяя задавать отдельное поведение для виртуальных аксонов
- Аффинитет к `target_type` работает штатно (steering к указанному типу нейронов)
- По пути создаются дендритные контакты - стандартная механика

> **Стоимость:** Виртуальный аксон = одна запись в `axon_heads[]` (4 байта). `PropagateAxons` = один `IADD` на аксон - микросекунды даже для миллионов. Количество дендритных контактов ограничено числом `сом × 128`, а не числом аксонов. Лишние аксоны-«изгои» не стоят ничего кроме 4 байт на голову.

> **Night Phase:** Виртуальные аксоны **не подлежат pruning**. Они вечная инфраструктура. Дендриты, подписанные на виртуальный аксон, могут отмереть штатно (pruning по весу), но сам аксон никогда не удаляется и не перерастает. Baker помечает их специальным флагом (`is_virtual = true`), Night Phase их пропускает.

Результат: внутри зоны пиксель = аксон, который **физически проросший** и имеет реальные синаптические контакты с окружающими нейронами.

> // TODO: [V2 Architecture] Расширить систему маппинга матриц в Baker.
> Внедрить нормализованную UV-проекцию (uv_rect) для поддержки режимов: Canvas, Chunked, Pie.

### 2.4. Пространственный Маппинг (uv_rect) и Мульти-Матрицы

На одну зону может быть подключено **N входных/выходных матриц** произвольных размеров. Их пространственная проекция на физическую сетку зоны задается параметром `uv_rect = [u, v, width, height]`, где значения - это доли от 0.0 до 1.0.

Эта абстракция покрывает три фундаментальных сценария маппинга без оверхеда в рантайме:

**1. Pie (Пирог / Feature Overlap)**
Наслоение нескольких матриц признаков (RGB, Edge, Motion) на одни и те же нейроны. Матрицы растягиваются на всю зону независимо друг от друга.
`uv_rect = [0.0, 0.0, 1.0, 1.0]` (По умолчанию)

**2. Canvas (Холст)**
Размещение независимых сенсоров (например, левый и правый глаз) рядом в пространстве зоны без пересечений.
Левый глаз: `uv_rect = [0.0, 0.0, 0.5, 1.0]`
Правый глаз: `uv_rect = [0.5, 0.0, 0.5, 1.0]`

**3. Chunked (L7-Фрагментация)**
Разрезание огромной матрицы (например, 4K видео) на чанки для обхода лимита UDP MTU (65507 байт). Клиент шлет 4 независимых UDP пакета, а Baker склеивает их в пространстве.
Чанк 1 (левый верхний): `uv_rect = [0.0, 0.0, 0.5, 0.5]`
Чанк 2 (правый верхний): `uv_rect = [0.5, 0.0, 0.5, 0.5]`

Baker конкатенирует все виртуальные аксоны от всех матриц в один плоский массив. Для ноды не существует понятия "матрица" - она работает с единым потоком байт.

### 2.5. Внешний интерфейс

Снаружи входные матрицы - один плоский массив бит:

```
Input_Bitmask[tick][pixel_id / 32] - u32 слова
pixel_id = matrix_offset + y * W + x
```

Установка бита = сброс `axon_heads[virtual_offset + pixel_id] = 0` (рождение сигнала). Далее всё по штатной физике - сигнал распространяется, дендриты реагируют, GSOP обучает.

### 2.6. Bulk DMA & Stride (Autonomous Batch Execution)

**Bulk DMA Архитектура (см. [06_distributed.md §2.9](./06_distributed.md)):**

Входная маска **не потокируется** каждый тик. **Весь батч** загружается в VRAM **одной асинхронной операцией** `cudaMemcpyAsync` перед началом вычислений (<1 мс на PCIe 4.0 x16). GPU крутит **6-ядерный Autonomous Loop** полностью независимо от хоста, используя смещённые указатели (`tick_input_ptr`) внутри батча за **O(1)** без обращения к хосту или выхода из цикла.

**Stride Parameter (Intra-Batch Frequency):**

Параметр `stride` определяет частоту обновления входа в батче:

```
stride = 0   → тик 0 только (Снимок, Single-Tick Pulse)
stride = 1   → каждый тик (Поток)
stride = 2   → каждый 2-й тик
stride = N   → каждый N-й тик
```

При `stride = S`, ядро `InjectInputs` получает смещённый указатель на маску каждый S-й тик. Эффективное количество инъекций в батче:
$$\text{effective\_ticks} = \lceil \text{sync\_batch\_ticks} / S \rceil$$

Размер `input_bitmask_buffer`:
$$\text{size} = \lceil \text{total\_virtual\_axons} / 32 \rceil \times 4 \times \text{effective\_ticks}$$

**Примеры:**
- `stride = 0, sync_batch_ticks = 100`: 1 снимок (t=0), buffer = `N/32 × 4 × 1` байт - вход **замораживается** на весь батч
- `stride = 1, sync_batch_ticks = 100`: 100 инъекций (поток), buffer = `N/32 × 4 × 100` байт - sensor data каждый тик
- `stride = 2, sync_batch_ticks = 100`: 50 инъекций, buffer = `N/32 × 4 × 50` байт - downsampling каждый 2-й тик

**Early Exit:** Нулевая маска на тике → `InjectInputs` пропускает все 128 потоков варпа. Ноль FLOPS. Ноль работы.

**Инвариант Bulk:** Четыре DMA за батч (100 ms): H2D input, H2D schedule, D2H output, D2H activity. Ноль микротранзакций в горячем цикле. Хост полностью деспотичен.

### 2.7. UDP Протокол (External I/O Communication)

Хост формирует массивный батч данных, превышающий UDP MTU, и режет его на фрагменты. Нода потребляет их как непрерывный **Data-Stream**.

**Структура каждого UDP чанка:**

```rust
// Header (20 байт, <IIIIhH)
#[repr(C)]
pub struct ExternalIoHeader {
    pub magic: u32,          // 0x4F495347 (GSIO) или 0x4F4F5347 (GSOO)
    pub zone_hash: u32,      // FNV-1a hash имени zone
    pub matrix_hash: u32,    // FNV-1a hash matrix name
    pub payload_size: u32,   // Размер payload в ЭТОМ чанке (байт)
    pub global_reward: i16,  // Сигнал подкрепления (Dopamine)
    pub _padding: u16,       // Добивка
}

// Payload: Кусок данных батча
```

> [!IMPORTANT]
> Заголовок прикрепляется к **каждому** UDP-пакету. Поле `payload_size` указывает размер конкретного куска. Это позволяет Ноде корректно собирать мега-батчи из разрозненных фрагментов.

**Примеры хеширования (FNV-1a):**
```python
import struct

def fnv1a_32(data: bytes) -> int:
    hash_val = 0x811c9dc5
    for b in data:
        hash_val ^= b
        hash_val = (hash_val * 0x01000193) & 0xFFFFFFFF
    return hash_val

zone_hash = fnv1a_32(b"SensoryCortex")        # zone в io.toml
matrix_hash = fnv1a_32(b"retina")             # name матрицы в io.toml
```

**Портовые конвенции:**

| Направление | Порт (пример) | Источник | Приёмник | Протокол |
|---|---|---|---|---|
| **Входы** (Input) | 8081 | Хост (сенсор) | Genesis Runtime | UDP |
| **Выходы** (Output) | 8082 | Genesis Runtime | Хост (мотор) | UDP |

Runtime слушает на входящем (`sock_in`) и отправляет на исходящем (`sock_out`). Клиент запоминает адрес первого входящего пакета и отправляет выходы по этому адресу.

**Синхронизация:**

- **Батч получен** → `cudaMemcpyAsync` в VRAM, старт Day Phase
- **Батча нет** → Day Phase запускается с предыдущей маской (или нулевой - конфигурируемо)
- Runtime проверяет наличие нового батча **в конце каждого текущего батча** (синхр. точка BSP Barrier)

**Фрагментация:** UDP лимит = 65507 байт. Пакеты больше 65KB автоматически дропятся (защита от EMSGSIZE отравления сокета). Для очень больших батчей требуется либо фрагментация на хосте, либо переход на TCP/shared memory.

### 2.8. Асимметрия и Feature Pyramid Batching (Абстракция)

Клиент формирует входы асимметрично относительно частоты среды. Рекомендуемый паттерн - **Feature Pyramid**, где многослойная матрица признаков (токены, цвета, грани) разворачивается по оси времени (тикам).

**Концепция:**
- **Тик 0:** Активация матрицы фильтров "краев" (Edges).
- **Тик 1:** Активация цветовых признаков (Color).
- **Тик 2:** Семантические токены.
- **Тики 3..N:** Пауза (Propagation Tail).

Ноде всё равно на семантическое разделение - она видит только битовый поток виртуальных аксонов. GSOP (пластичность) самостоятельно обучит дендриты различать слои на основе задержек прилета сигнала.

### 2.9. Реальный пример: CartPole

Стандартная тестовая среда `CartPole-v0` (OpenAI Gym): балансировка палки на тележке. 4 входных переменных, 2 дискретных действия.

**Входная сторона (SensoryCortex):**

4 физических переменных → **населённое кодирование** (Gaussian population code):
- `cart_position` ∈ [-2.4, 2.4]
- `cart_velocity` ∈ [-3.0, 3.0]
- `pole_angle` ∈ [-0.41, 0.41] (≈ ±12°)
- `pole_angular_velocity` ∈ [-2.0, 2.0]

Каждая переменная кодируется **16 нейронами** (tuning width σ = 0.15):

```python
# Encoding (хост)
def encode_variable(val: float, bounds: list, num_neurons: int) -> list:
    """Gaussian receptive fields, центры в равномерной сетке."""
    val_norm = (val - bounds[0]) / (bounds[1] - bounds[0])  # нормализация в [0, 1]
    centers = [i / (num_neurons - 1) for i in range(num_neurons)]
    sigma = 0.15
    spikes = []
    for center in centers:
        distance_sq = (val_norm - center) ** 2
        prob = math.exp(-distance_sq / (2 * sigma ** 2))
        spike = 1 if prob > 0.5 else 0  # пороговый
        spikes.append(spike)
    return spikes

# Итого: 4 переменных × 16 нейронов = 64 входных аксона
```

`io.toml`:
```toml
[[input]]
name = "cartpole_sensors"
zone = "SensoryCortex"
width = 4
height = 4
# Интерпретация: матрица 4×4 = 16 пикселов,
# каждый пиксель связан с 4 нейронами (все переменные, 1 центр)
# Или: линейно - пиксели 0-15 от уменьшенной топологии
virtual_axon_count = 64
```

**Выходная сторона (MotorCortex):**

2 действия (левый/правый толчок) → **population decoding**:

```python
# Decoding (хост)
def decode_action(output_history: bytes, batch_ticks: int) -> int:
    """Winner-takes-all из последних спайков."""
    # output_history - concat выходов, размер = sum(W×H) / 8 × batch_ticks
    # для 2 действий 8×8 = 64 нейрона per action
    
    total_spikes = sum(output_history)
    left_spikes = sum(output_history[:len(output_history)//2])
    right_spikes = total_spikes - left_spikes
    
    return 0 if left_spikes > right_spikes else 1  # action: left=0, right=1
```

`io.toml`:
```toml
[[output]]
name = "motor_left"
zone = "MotorCortex"
width = 8
height = 8
target_type = "Excitatory"     # only excitatory neurons

[[output]]
name = "motor_right"
zone = "MotorCortex"
width = 8
height = 8
target_type = "Excitatory"
```

**Синхронизация (хост):**

```python
INPUT_PORT = 8081
OUTPUT_PORT = 8082
SYNC_BATCH_TICKS = 100

while True:
    # 1. Получить сенсоры из CartPole
    cart_x, cart_v, pole_a, pole_av = env.step(action)
    
    # 2. Кодировать в спайки
    spikes = (
        encode_variable(cart_x, [-2.4, 2.4], 16) +
        encode_variable(cart_v, [-3.0, 3.0], 16) +
        encode_variable(pole_a, [-0.41, 0.41], 16) +
        encode_variable(pole_av, [-2.0, 2.0], 16)
    )
    
    # 3. Отправить батч (100 тиков)
    bitmask = pack_ticks(spikes, sync_batch_ticks=100)
    send_to_runtime(INPUT_PORT, bitmask)
    
    # 4. Получить выходы
    output_history = receive_from_runtime(OUTPUT_PORT)
    
    # 5. Декодировать действие
    action = decode_action(output_history, batch_ticks=100)
```

Цизнь менструального цикла у подопытного мозга: **100 ms вычисления ≈ 10 логических шагов**. Реальная скорость зависит от CUDA-конфигурации и задержки сети.

---

## 3. Выходная Матрица (Soma Readout)

### 3.1. Размещение

Аналогично входу: матрица W×H растягивается на X-Y плоскость зоны. Каждый пиксель покрывает пространственный регион.

### 3.2. Множественные выходные матрицы

На одну зону может быть подключено **M выходных матриц** разных размеров:

```toml
[[output]]
name = "motor_left"
zone = "MotorCortex"
width = 8
height = 8
target_type = "Excitatory"

[[output]]
name = "motor_right"
zone = "MotorCortex"
width = 8
height = 8
target_type = "Excitatory"

[[output]]
name = "attention_map"
zone = "MotorCortex"
width = 32
height = 32
target_type = "All"
```

> **Инвариант: Порядок конкатенации = порядок объявления** блоков `[[output]]` в `io.toml`. Baker записывает offset'ы в `.gxo` файл.

### 3.3. Захват сомы

В каждом пикселе (регионе) обычно находится **больше одной сомы**. Выбор - **детерминированно-случайный** (master_seed + output_name + pixel_index):

```
candidates = сомы target_type в регионе пикселя
seed = master_seed ^ fnv1a(output_name) ^ pixel_index
chosen_soma = candidates[hash(seed) % len(candidates)]
```

Маппинг 1 пиксель → 1 сома. Статический, вычисляется при Baking.

### 3.4. Выходные данные

Выход = **глубокий батч спайков** выбранных сом:

```
Output_History[tick][pixel_id] - u8 (0 или 1)
```

Размер: `sync_batch_ticks × total_output_pixels` байт.

Каждый тик ядро `RecordOutputs` проверяет `flags[soma_id] & 0x01` для каждого привязанного пикселя и пишет результат.

### 3.5. Внешняя интерпретация

Снаружи `Output_History` - это просто 2D-матрица активности с временной глубиной. Как её интерпретировать - **ответственность внешнего Hub'а**:

- **Population Coding**: `popcount(pixels_group)` → сила действия
- **Rate Coding**: среднее количество спайков на пиксель за N тиков
- **Spatial Pattern**: паттерн активности как карта значимости

Genesis не навязывает интерпретацию. Он отдаёт сырую матрицу спайков.

---

## 4. Межзональные Связи (Ghost Axon Matrix)

### 4.1. Унификация

Межзональная связь - это **тот же матричный интерфейс**: выходная матрица зоны-источника проецируется как входная матрица (ghost axons) в зону-приёмник.

```toml
# brain.toml
[[connection]]
from = "SensoryCortex"
to = "HiddenCortex"
output_matrix = "sensory_out"    # имя выходной матрицы зоны-источника
width = 16                       # резолюция проекции в зоне-приёмнике
height = 16
entry_z = "top"
target_type = "All"
growth_steps = 750
```

Это означает:
1. Зона `SensoryCortex` имеет `[[output]]` с `name = "sensory_out"` (например 32×32)
2. Connection **проецирует** эту выходную матрицу на зону `HiddenCortex` как ghost-матрицу 16×16
3. Baker создаёт 256 ghost axons в `HiddenCortex`, каждый проросший Cone Tracing
4. Runtime синхронизирует: `output_pixel[i]` из SensoryCortex → `ghost_axon_head[i]` в HiddenCortex

> **Масштабирование:** Если выходная матрица (32×32) больше проекции (16×16), каждый ghost-пиксель захватывает **группу** выходных пикселей (pooling). Если меньше - upsampling. Маппинг = детерминированный.

### 4.2. Чем ghost axons отличаются от virtual axons

| Свойство | Virtual Axon (вход) | Ghost Axon (межзональный) |
|----------|---------------------|---------------------------|
| Управление | Хост (Input_Bitmask) | Runtime (sync axon_heads) |
| Источник сигнала | Внешний мир (UDP) | Выходная матрица другой зоны |
| Синхронизация | `cudaMemcpyAsync` маски | D2H + H2D (4 байта × N) |
| Night Phase | Не pruning | Не pruning |
| `soma_idx` | `usize::MAX` (нет сомы) | `usize::MAX` (нет сомы) |

Концептуально идентичны. Различие только в **источнике управления головой аксона**.

---

## 5. Коллизии

Один и тот же регион зоны может быть одновременно входом и выходом (как сенсомоторная кора в биологии). Виртуальные аксоны и привязанные сомы сосуществуют штатно - конфликтов архитектуры нет.

---

## 6. Симметрия

| Свойство | Вход (GXI) | Выход (GXO) | Ghost |
|----------|------------|-------------|-------|
| Абстракция | 2D матрица W×H | 2D матрица W×H | 2D матрица W×H |
| Мульти-резолюция | Да | Да | Да (проекция) |
| Внутри зоны | Виртуальный аксон | Привязанная сома | Ghost аксон |
| Размещение | Cone Tracing | Seeded хеш | Cone Tracing |
| Стоимость | 4 байта/голову | 0 (сома есть) | 4 байта/голову |
| Night Phase | Не pruning | N/A | Не pruning |
| Данные | Бит/пиксель (in) | u8/пиксель (out) | Sync axon_head |
| Формат файла | `.gxi` | `.gxo` | `.ghosts` |

Единая модель. Один механизм Baking. Симметричный протокол.

---

## 7. Ограничения (UDP && VRAM)

Размеры батчей и матриц ограничены физикой сети и памяти. Квоты мягкие (warning) и жёсткие (drop).

| Параметр | Формула | Номинал | Природа | Решение |
|----------|---------|---------|---------|---------|
| **Max UDP Input** | `⌈total_virtual_axons / 32⌉ × 4 × effective_ticks` | < 65507 | IP/UDP MTU | Фрагментация на хосте или TCP/SHM |
| **Max UDP Output** | `∑(W×H для всех [[output]]) × sync_batch_ticks` | < 65507 | IP/UDP MTU | Разбить выходы на разные зоны |
| **Input Bitmask Buffer** | `virtual_axon_count / 32 × 4` (per tick) | 256 КБ типично | VRAM (input_bitmask) | Снизить total_virtual_axons |
| **Output History Buffer** | `∑ mapped_soma_ids × sync_batch_ticks` | 256 КБ типично | VRAM (output_history) | Снизить цель пикселей или batch_ticks |
| **Ghost Schedule Buffers** | 2 × (max spike_events за 1 батч) | ~4 МБ | Ring Buffer pinned | Снизить количество зон или ghost_density |

**Примеры:**

- **CartPole (4 inputs 64 neurons)**: Input size = 8 bytes/tick × 100 ticks = 800 B ✅ OK
- **Retina (1920×1080)**: Input size = 64 КБ/tick (если все 8M пикселей) ✅ OK (64 КБ < 65.5 КБ)
- **4 zones × 64М ghost spikes/batch**: Schedule buffer ~256 МБ в ring ⚠️ Требует дополнительного конфигурирования
- **Output: 8 matrice × 256×256**: Output size = 128 КБ/batch ❌ Overflow - разбить или снизить resolution

Все лимиты **конфигурируемы** в `simulation.toml` (VRAM params) и `brain.toml` (virtual_axon_count, ghost_density).

---

## Connected Documents

| Document | Purpose | Status |
|----------|---------|--------|
| [05_signal_physics.md](05_signal_physics.md) | RecordReadout kernel, Input spike injection, Output readout kernels | ✅ MVP |
| [06_distributed.md](06_distributed.md) | BSP sync, output_history aggregation across shards | ⏳ MVP |
| [07_gpu_runtime.md](07_gpu_runtime.md) | VramState I/O fields, ExternalIoServer UDP | ✅ MVP |
| [02_configuration.md](02_configuration.md) | io.toml [[input]] / [[output]] schema | ⏳ TODO |
| [04_connectivity.md](04_connectivity.md) | Ghost axon placement via Cone Tracing | ⏳ TODO |
| [09_baking_pipeline.md](09_baking_pipeline.md) | .gxi / .gxo / .ghosts file format | ⏳ TODO |

---

## Changelog

**v1.0 (2026-02-28)**

- Added UDP protocol details (**§2.7**): ExternalIoHeader struct, zone_hash/matrix_hash computation (FNV-1a), port conventions, fragmentation behavior
- Added CartPole real example (**§2.9**): 4-input Gaussian population coding (16 neurons per variable), 2-output population decoding, TOML config, host synchronization pattern
- Added Constraints table (**§7**): UDP payload limits (65507 B), VRAM buffer examples (CartPole, Retina, 8-output), configuration paths
- Clarified stride formula impact on bitmask size (**previous §2.6**): effective_ticks = ⌈sync_batch_ticks / stride⌉
- Verified all sections against 07_gpu_runtime.md ExternalIoServer, CartPole client reference

**Known Issues**

- Ghost axon fragmentation (> 65K schedule events) requires explicit mem allocation; not covered in MVP
- TCP/Shared Memory I/O for bandwidth > 1 Gbps; UDP-only in V1
