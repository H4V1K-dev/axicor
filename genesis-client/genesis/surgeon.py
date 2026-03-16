import numpy as np
from .memory import GenesisMemory

class GenesisSurgeon:
    """
    Data-Oriented скальпель для прямого вмешательства в VRAM.
    Работает строго через Zero-Copy mmap, без вызовов Rust-оркестратора.
    """
    def __init__(self, memory: GenesisMemory):
        self.mem = memory

    def incubate_gaba(self, baseline_weight: int = -30000) -> int:
        """
        Защита от эпилептических штормов при Tabula Rasa.
        Благодаря Закону Дейла (знак веса == тип нейрона-источника), 
        мы просто находим все отрицательные веса и максимизируем их.
        """
        # (target != 0) гарантирует, что мы не трогаем пустые слоты
        mask = (self.mem.targets != 0) & (self.mem.weights < 0)
        
        self.mem.weights[mask] = baseline_weight
        return int(np.sum(mask))

    def extract_graft(self, threshold: int = 25000) -> tuple[np.ndarray, np.ndarray]:
        """
        Экстракция подграфа (навыка).
        Возвращает плоские оффсеты (1D индексы) и знаки весов для самых сильных связей.
        """
        mask = (self.mem.targets != 0) & (np.abs(self.mem.weights) > threshold)
        
        # Получаем плоские смещения в памяти
        # Используем ravel() для получения 1D представления
        weights_flat = self.mem.weights.ravel()
        targets_flat = self.mem.targets.ravel()
        
        flat_mask = (targets_flat != 0) & (np.abs(weights_flat) > threshold)
        offsets = np.where(flat_mask)[0].astype(np.uint32)
        signs = np.sign(weights_flat[offsets]).astype(np.int8)
        
        return offsets, signs

    def inject_graft(self, offsets: np.ndarray, signs: np.ndarray):
        """
        Мгновенная трансплантация (включение) навыка в чистой зоне.
        Используется эффект монументализации (ранг инерции 15 -> вес 32767).
        """
        # Прямая запись поверх VRAM дампа
        self.mem.weights.ravel()[offsets] = signs.astype(np.int16) * 32767
