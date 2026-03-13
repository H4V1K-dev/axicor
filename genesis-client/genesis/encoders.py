import math
import numpy as np

class PwmEncoder:
    """
    Temporal PWM Encoding (Rate Coding) для непрерывных аналоговых сигналов.
    Размазывает спайки по батчу через фазовый сдвиг, предотвращая Burst Gating.
    """
    def __init__(self, num_sensors: int, batch_size: int):
        self.N = num_sensors
        self.B = batch_size
        
        # GPU ожидает массив u32 (по 32 виртуальных аксона в слове).
        # Строка каждого тика обязана быть кратна 4 байтам (32 битам).
        self.padded_N = math.ceil(self.N / 32) * 32
        self.bytes_per_tick = self.padded_N // 8
        self.total_bytes = self.bytes_per_tick * self.B
        
        # Временная ось и фазовый сдвиг (Golden Ratio Dither)
        t = np.linspace(0, 1, self.B, endpoint=False, dtype=np.float16)[:, None]
        phase = (np.arange(self.N, dtype=np.float16) * 0.618033) % 1.0
        self.pwm_wave = (t + phase) % 1.0
        
        # Преаллоцированный буфер для избежания аллокаций в Hot Loop
        self._bool_buffer = np.zeros((self.B, self.padded_N), dtype=np.bool_)

    def encode_into(self, sensors_f16: np.ndarray, tx_view: memoryview, offset: int) -> int:
        """
        Broadcasting сравнение и Zero-copy запись в сетевой буфер.
        Возвращает количество записанных байт.
        """
        # Сравнение только для реальных сенсоров (остальные биты останутся 0)
        self._bool_buffer[:, :self.N] = self.pwm_wave < sensors_f16[None, :]
        
        # bitorder='little' критичен! CUDA делает (word >> bit_idx) & 1
        packed = np.packbits(self._bool_buffer, bitorder='little', axis=1)
        
        # Копирование плоского массива байт прямо в UDP сокет-буфер
        tx_view[offset : offset + self.total_bytes] = packed.ravel()
        return self.total_bytes


class PopulationEncoder:
    """
    Пространственное кодирование (Gaussian Receptive Fields).
    Разворачивает 1 float переменную в популяцию из M нейронов.
    """
    def __init__(self, variables_count: int, neurons_per_var: int, batch_size: int):
        self.V = variables_count
        self.M = neurons_per_var
        self.N = self.V * self.M
        self.B = batch_size
        
        self.padded_N = math.ceil(self.N / 32) * 32
        self.bytes_per_tick = self.padded_N // 8
        self.total_bytes = self.bytes_per_tick * self.B
        
        # Предрасчет радиуса активации для Гауссианы. prob > 0.5 эквивалентно abs(dist) < R
        # Это избавляет нас от вызова медленного np.exp в горячем цикле.
        sigma = 0.15
        self.radius = sigma * math.sqrt(-2.0 * math.log(0.5))
        
        # Центры рецептивных полей (M центров для каждой из V переменных)
        centers_1d = np.linspace(0.0, 1.0, self.M, dtype=np.float16)
        self.centers = np.tile(centers_1d, self.V)
        
        self._bool_buffer = np.zeros(self.padded_N, dtype=np.bool_)
        self._batch_bool_buffer = np.zeros((self.B, self.padded_N), dtype=np.bool_)

    def encode_into(self, states_f16: np.ndarray, tx_view: memoryview, offset: int) -> int:
        """
        states_f16: массив нормализованных [0..1] значений (размер V)
        В Population кодировании маска одинакова для всего батча (или обновляется раз в кадр)
        Мы посылаем импульс (1) только в первый тик батча (Соблюдение закона Single-Tick Pulse), 
        остальные тики заполняем нулями, чтобы хвост сигнала прошел по графу.
        """
        # Растягиваем переменные: [v1, v2] -> [v1, v1.. M раз, v2, v2.. M раз]
        expanded_states = np.repeat(states_f16, self.M)
        
        # Векторизованный поиск попаданий в рецептивные поля
        np.abs(expanded_states - self.centers, out=expanded_states) # In-place
        self._bool_buffer[:self.N] = expanded_states < self.radius
        
        # Обнуляем весь батч, затем пишем только в нулевой тик (Single-Tick Pulse)
        self._batch_bool_buffer.fill(False)
        self._batch_bool_buffer[0, :] = self._bool_buffer
        
        packed = np.packbits(self._batch_bool_buffer, bitorder='little', axis=1)
        tx_view[offset : offset + self.total_bytes] = packed.ravel()
        return self.total_bytes
