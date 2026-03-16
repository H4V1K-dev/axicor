# 🐍 Axicor Client SDK: HFT Python Guide

Добро пожаловать в Client SDK. Забудьте про статические графы вычислений, backpropagation и `loss.backward()`. Вы подключаетесь к живому биологическому реактору, который работает в реальном времени.

Ваша задача как инженера - написать HFT-шлюз (Agent), который будет переводить данные среды в спайки и успевать делать это за доли миллисекунды, чтобы не сорвать **Lockstep-барьер** кластера.

---

## 1. Sensory-Motor Loop (HFT-Пайплайн)

Классический цикл обучения заменяется на непрерывный обмен сырыми байтами через UDP Fast-Path. Мы не передаем JSON или списки. Мы шлём битовые маски.

> [!IMPORTANT]
> **ГЛАВНЫЙ ЗАКОН (Zero-Garbage)**
> Внутри горячего цикла `while True:` **запрещено создавать объекты**. Никаких списков `[]`, никаких кастов `float()`, никаких циклов `for` по сенсорам. Любая аллокация разбудит сборщик мусора (GC) Питона, и вы убьете жесткий бюджет в 10 мс.

### Эталонный цикл (CartPole Agent)

```python
import time
import numpy as np
from genesis.client import GenesisMultiClient
from genesis.encoders import PopulationEncoder
from genesis.decoders import PwmDecoder
from genesis.brain import fnv1a_32

# Константы R-STDP (Time-Scaled)
DOPAMINE_PULSE = -25         # Фоновая эрозия 
DOPAMINE_REWARD = 30         # Пульс за правильные действия
DOPAMINE_PUNISHMENT = -255   # Death Signal

BATCH_SIZE = 10 # Strict BSP Sync: 1 пакет = 10 тиков (1 мс)

# 1. Преаллокация и инициализация (ВНЕ горячего цикла)
zone_hash = fnv1a_32(b"SensoryCortex")
matrix_hash = fnv1a_32(b"cartpole_sensors")
input_payload_size = (64 * BATCH_SIZE) // 8

client = GenesisMultiClient(
    addr=("127.0.0.1", 8081),
    matrices=[{'zone_hash': zone_hash, 'matrix_hash': matrix_hash, 'payload_size': input_payload_size}]
)

# Энкодер: 4 переменных -> 64 рецептора. Декодер: 128 моторных выходов.
encoder = PopulationEncoder(variables_count=4, neurons_per_var=16, batch_size=BATCH_SIZE)
decoder = PwmDecoder(num_outputs=128, batch_size=BATCH_SIZE)

# Векторизованная нормализация (Матрица диапазонов среды)
bounds = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.41, 0.41], [-2.0, 2.0]], dtype=np.float16)
range_diff = bounds[:, 1] - bounds[:, 0]

while True:
    # --- ZERO-GARBAGE HOT LOOP ---

    # 1. Zero-Cost Normalization [0.0, 1.0]
    norm_state = np.clip((state - bounds[:, 0]) / range_diff, 0.0, 1.0).astype(np.float16)

    # 2. Определение дофаминового сигнала
    if terminated or truncated:
        # THE DEATH SIGNAL: Пролонгированный шок для выжигания связей
        for _ in range(15):
            client.step(DOPAMINE_PUNISHMENT)
        env.reset()
        continue
    elif score > 0 and score % 10 == 0:
        dopamine_signal = DOPAMINE_REWARD
    else:
        dopamine_signal = DOPAMINE_PULSE

    # 3. Zero-Copy Encoding (Прямо в сетевой буфер)
    encoder.encode_into(norm_state, client.payload_views, 0)

    # 4. Lockstep Barrier (Блокирующий пинг-понг с GPU)
    rx = client.step(dopamine_signal)

    # 5. Zero-Copy Decoding (Смещение 0, т.к. step() уже отрезал C-ABI заголовок)
    total_motor = decoder.decode_from(rx, 0)
    
    # Winner-Takes-All
    action = 0 if np.sum(total_motor[:64]) > np.sum(total_motor[64:]) else 1
    state, reward, terminated, truncated, _ = env.step(action)
```

### ⚙️ Как это работает под капотом (DOD)

*   `PopulationEncoder` и `PwmDecoder` используют векторизованные операции `np.packbits` и `np.frombuffer`. Они пишут биты напрямую в `memoryview` сырого байтового буфера.
*   Метод `client.step()` пакует дофамин в 20-байтовый заголовок `ExternalIoHeader`, блокируется на `recvfrom`, отрезает заголовок ответа и возвращает чистый `memoryview` выходов.
*   Время выполнения Python-обвязки - менее 0.1 мс. Оставшееся время тратится исключительно на физику среды (Gymnasium).

---

## 2. The Neurosurgeon (Нейрохирургия коннектома)

Если вам нужно грубо вмешаться в память сети (или извлечь аналитику), SDK предоставляет модуль `GenesisMemory` и `GenesisSurgeon`. Они общаются с памятью строго через Zero-Copy mmap файлов ОС (`/dev/shm`), минуя оркестратор на Rust.

> [!WARNING]
> ПРЕДОХРАНИТЕЛЬ: Запрещено вызывать методы memory внутри горячего цикла каждый шаг. Обработка матриц в сотни мегабайт нарушит бюджет HFT. Вызывайте их только в моменты падения агента или при холодном старте.

### Извлечение и Трансплантация Навыков (Grafting)
Благодаря Структурному Детерминизму (если зоны скомпилированы с одинаковым `master_seed`), нейроны лежат по одним и тем же адресам. Вы можете дистиллировать обученный навык из одного агента в другого:

```python
from genesis.memory import GenesisMemory
from genesis.surgeon import GenesisSurgeon

# 1. Подключаемся к VRAM обученного агента (read_only=False)
mem_donor = GenesisMemory(zone_hash=0xDEADBEEF, read_only=False)
surgeon = GenesisSurgeon(mem_donor)

# Экстракция: извлекаем плоские 1D-оффсеты самых сильных синапсов
offsets, signs = surgeon.extract_graft(threshold=25000)

# 2. Подключаемся к "чистому" агенту
mem_recipient = GenesisMemory(zone_hash=0xBEEFDEAD, read_only=False)
surgeon_rec = GenesisSurgeon(mem_recipient)

# Мгновенная пересадка навыка (Монументализация)
surgeon_rec.inject_graft(offsets, signs)
```

**Под капотом:** Мы не обходим графы и ООП-узлы. `extract_graft` использует маску `np.where()` над mmap массивом в полгигабайта, извлекая сырые индексы памяти. `inject_graft` записывает максимальную силу веса (32767) по этим оффсетам. Скорость выполнения ограничена только шиной RAM.
