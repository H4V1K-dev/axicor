# High-Performance Encoders & Decoders (Python SDK)

В Axicor мы работаем с тысячами сенсоров и моторных выходов на частотах 100+ Гц. Классический подход с циклами `for` и созданием ООП-объектов `Spike` - это смертный приговор для производительности (GIL и сборщик мусора убьют бюджет в 10 мс).

Секрет HFT-интеграции - **Data-Oriented Design (DOD)**. Мы конвертируем float-состояния среды в сырые битовые маски через плоские операции NumPy и пересылаем их через `memoryview` без единой аллокации в Hot Loop.

SDK предоставляет готовые Zero-Copy кодировщики.

## 1. Входящие сигналы (Encoders)

Сенсорные данные (`float16`) преобразуются в спайки на стороне Python-клиента и отправляются на порт ноды (`8081`).

### 1.1. PopulationEncoder (Пространственное кодирование)
Разворачивает одну `float` переменную в популяцию из `M` рецепторов (Gaussian Receptive Fields). Идеально для координат, углов и скоростей (например, в CartPole).

```python
from genesis.encoders import PopulationEncoder

# 4 переменные среды по 16 нейронов на каждую, батч = 10 тиков
encoder = PopulationEncoder(variables_count=4, neurons_per_var=16, batch_size=10)

# В Hot Loop:
# 1. Векторизованная нормализация в [0.0, 1.0] (In-place)
norm_state = np.clip(state / max_bounds, 0.0, 1.0)

# 2. Кодирование прямо в сетевой буфер сокета!
# offset=20 оставляет место под C-ABI заголовок ExternalIoHeader
encoder.encode_into(norm_state, client.payload_views, offset=20)
```

**DOD-магия:** Метод `encode_into` использует `broadcast_to` и векторизованный расчет дистанций прямо в преаллоцированный массив `self._expanded_buffer`. Ни одного объекта в куче не создается.

### 1.2. PwmEncoder (Частотное кодирование / Rate Coding)
Для плотных аналоговых потоков (RGB-камеры, аудио). Значение f16 (0.0 - 1.0) кодируется в частоту спайков одного виртуального аксона.

> [!IMPORTANT]
> **Защита от Burst Gating:** `PwmEncoder` аппаратно смещает фазу (Golden Ratio Dither), чтобы сенсоры не выстрелили одновременно. Это предотвращает блокировку дендритов через `synapse_refractory_period`.

```python
from genesis.encoders import PwmEncoder

# 1024 пикселя, глубина батча = 100 тиков
pwm = PwmEncoder(num_sensors=1024, batch_size=100)

# В Hot Loop:
pwm.encode_into(camera_frame_f16, client.payload_views, offset=20)
```

---

## 2. Исходящие команды (Decoders)
Когда нода завершает расчет батча, она возвращает `Output_History` (Матрица Тики × Моторы). Ядро `RecordReadout` пишет 1 байт (u8) на каждый спайк.

### PwmDecoder (Rate Decoding)
Сжимает временную развертку батча обратно в плотные float-значения усилий (Duty Cycle) для моторов среды.

```python
from genesis.decoders import PwmDecoder

# 128 моторных каналов, батч = 10 тиков
decoder = PwmDecoder(num_outputs=128, batch_size=10)

# В Hot Loop (после sock.recvfrom_into в преаллоцированный буфер):
# Декодер читает сырые байты, делает виртуальный reshape(10, 128) 
# и суммирует спайки по оси времени (axis=0) без копирования!
motor_forces = decoder.decode_from(rx_view, offset=20)

# Использование (Winner-Takes-All)
left_force = np.sum(motor_forces[:64])
right_force = np.sum(motor_forces[64:])
action = 0 if left_force > right_force else 1
```

### Золотые правила HFT-Скриптинга (Pro Tips)

1. Zero-Copy Sockets: Никогда не используйте sock.recv(65535). Это выделяет новый bytes объект ОС. Используйте sock.recvfrom_into(rx_buf) с предварительно выделенным bytearray и работайте через memoryview. (Класс GenesisMultiClient делает это под капотом).
2. C-ABI Offset: Всегда помните, что первый 20 байт любого пакета Data Plane - это ExternalIoHeader (<IIIIhH). Данные масок и выходов начинаются строго со смещения 20.
3. In-Place Math: Если нужно нормализовать массив, используйте аргумент out= в функциях NumPy: np.multiply(raw, 0.5, out=preallocated_array).