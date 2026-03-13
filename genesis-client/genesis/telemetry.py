import struct
import threading
import asyncio
import numpy as np
from typing import Optional

try:
    import websockets
except ImportError:
    raise ImportError("Для телеметрии требуется пакет websockets (pip install websockets)")

# Контракт из genesis-core/src/ipc.rs и genesis-ide/src/telemetry.rs
# 0..4: Magic "SPIK" (0x4B495053)
# 4..12: Tick (u64)
# 12..16: Spikes Count (u32)
HEADER_FORMAT = "<IQI"
HEADER_SIZE = 16
TELE_MAGIC = int.from_bytes(b"SPIK", "little")

class TelemetryListener:
    """
    Lock-Free фоновый слушатель спайков.
    Изолирует медленный I/O WebSocket от HFT-цикла среды.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 9003, max_neurons: int = 1_000_000):
        self.ws_url = f"ws://{host}:{port}/ws"
        self.max_neurons = max_neurons
        
        # Разделяемое состояние. 
        # Фоновый поток пишет единицы, главный применяет decay и читает.
        self._heatmap = np.zeros(self.max_neurons, dtype=np.float32)
        self._latest_tick = 0
        
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="Genesis-Telemetry-Rx")
        self._thread.start()

    def _run_loop(self):
        """Точка входа для изолированного OS-потока."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ws_listener())

    async def _ws_listener(self):
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.ws_url) as ws:
                    print(f"🔌 [Telemetry] Connected to {self.ws_url}")
                    async for message in ws:
                        if self._stop_event.is_set():
                            break
                        if isinstance(message, bytes):
                            self._process_frame(message)
            except Exception:
                # Тихий реконнект, чтобы не спамить в консоль
                await asyncio.sleep(1.0)

    def _process_frame(self, frame: bytes):
        if len(frame) < HEADER_SIZE:
            return

        magic, tick, count = struct.unpack_from(HEADER_FORMAT, frame, 0)
        
        if magic != TELE_MAGIC:
            return

        expected_size = HEADER_SIZE + count * 4
        if len(frame) < expected_size:
            return

        # Zero-Copy каст хвоста пакета в u32
        spikes = np.frombuffer(frame, dtype=np.uint32, count=count, offset=HEADER_SIZE)
        
        # Обновляем состояние
        self._latest_tick = tick
        if count > 0:
            # Массовая (векторизованная) запись без блокировок
            # Фильтруем мусор из сети, чтобы не словить SegFault
            valid_spikes = spikes[spikes < self.max_neurons]
            self._heatmap[valid_spikes] = 1.0

    def get_snapshot(self, decay: float = 0.8) -> tuple[int, np.ndarray]:
        """
        Вызывается в главном потоке (RL Agent / Dashboard).
        Применяет затухание (fade-out) к тепловой карте и возвращает её.
        decay: множитель сохранения активности (0.0 = только спайки этого тика, 1.0 = бесконечное свечение)
        """
        # Умножаем in-place для избежания аллокаций
        self._heatmap *= decay
        return self._latest_tick, self._heatmap

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
