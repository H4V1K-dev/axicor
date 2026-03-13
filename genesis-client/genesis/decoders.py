import numpy as np

class PwmDecoder:
    """
    Temporal PWM Decoding (Rate Coding) для моторного кортекса.
    Конвертирует бинарную историю спайков (Output_History) за батч
    в плотный f16 массив аналоговых усилий (Duty Cycle / 0.0 - 1.0).
    """
    def __init__(self, num_outputs: int, batch_size: int):
        self.N = num_outputs
        self.B = batch_size
        
        # Размер полезной нагрузки: B тиков * N моторов (1 байт = 1 флаг спайка)
        self.payload_size = self.N * self.B
        self._inv_b = np.float16(1.0 / self.B)
        
        # Преаллокация для HFT-цикла (Zero-Garbage)
        self._sum_buffer = np.zeros(self.N, dtype=np.float16)
        self._out_buffer = np.zeros(self.N, dtype=np.float16)

    def decode_from(self, rx_view: memoryview, offset: int) -> np.ndarray:
        """
        Извлекает данные из сырого UDP буфера без копирования памяти.
        rx_view: memoryview сокета
        offset: размер заголовка C-ABI (обычно 20 байт)
        """
        # 1. Zero-copy каст байтов. Под капотом создается только view на память ОС.
        raw_bytes = np.frombuffer(rx_view, dtype=np.uint8, count=self.payload_size, offset=offset)
        
        # 2. Виртуальный reshape (Тики, Моторы). Меняет только strides, без копирования.
        spikes_2d = raw_bytes.reshape((self.B, self.N))
        
        # 3. Векторизованная сумма по оси тиков (по времени). Запись прямо в преаллоцированный буфер!
        np.sum(spikes_2d, axis=0, dtype=np.float16, out=self._sum_buffer)
        
        # 4. Нормализация к диапазону [0.0, 1.0] (In-place)
        np.multiply(self._sum_buffer, self._inv_b, out=self._out_buffer)
        
        # Возвращаем ссылку на внутренний буфер. Данные валидны до следующего вызова decode_from.
        return self._out_buffer
