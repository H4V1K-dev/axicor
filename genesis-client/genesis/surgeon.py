import numpy as np
import struct
from .memory import GenesisMemory

class GenesisSurgeon:
    """
    Data-Oriented скальпель для прямого вмешательства в VRAM.
    Работает строго через Zero-Copy mmap, без вызовов Rust-оркестратора.
    """
    def __init__(self, memory: GenesisMemory):
        self.mem = memory

    def incubate_gaba(self, baseline_weight: int = -30000) -> int:
        mask = (self.mem.targets != 0) & (self.mem.weights < 0)
        self.mem.weights[mask] = baseline_weight
        return int(np.sum(mask))

    def extract_reflex_path(self, root_soma_ids: np.ndarray, prune_threshold: int = 15000) -> dict:
        """
        [DOD] Vectorized Back-Tracing для изоляции конкретного навыка.
        Ищет все сильные связи (возбуждающие и тормозные), которые питают целевые нейроны.
        """
        padded_n = self.mem.padded_n
        
        # 1. O(1) Inverse Mapping: Axon_ID -> Soma_ID
        # Извлекаем total_axons (смещение 0x20 в 64-байтном ShmHeader v2)
        total_axons = struct.unpack_from("<I", self.mem._mm, 0x20)[0]
        axon_to_soma = np.full(total_axons, -1, dtype=np.int32)
        
        # [DOD FIX] Находим живые аксоны и мапим их обратно на сомы
        valid_somas = np.where(self.mem.soma_to_axon != 0xFFFFFFFF)[0]
        valid_axons = self.mem.soma_to_axon[valid_somas]
        
        # Защита от выхода за пределы массива при поврежденном дампе
        valid_mask = valid_axons < total_axons
        axon_to_soma[valid_axons[valid_mask]] = valid_somas[valid_mask]

        # 2. Глобальные маски выживания (Zero-Garbage BFS)
        surviving_somas = np.zeros(padded_n, dtype=np.bool_)
        surviving_somas[root_soma_ids] = True
        
        surviving_synapses = np.zeros_like(self.mem.targets, dtype=np.bool_)
        frontier_somas = root_soma_ids

        # 3. Hot Loop обратного обхода по графу (strictly vectorized)
        while len(frontier_somas) > 0:
            # Вырезаем колонки для текущих нейронов фронтира
            frontier_targets = self.mem.targets[:, frontier_somas]
            frontier_weights = self.mem.weights[:, frontier_somas]

            # Выживают только сильные связи (Закон Дейла: abs() захватывает тормозные)
            # Признак жизни: target != 0 и вес выше порога
            valid_mask = (frontier_targets != 0) & (np.abs(frontier_weights) > prune_threshold)

            # Находим координаты выживших синапсов в локальном окне фронтира
            row_idx, col_idx = np.where(valid_mask)
            
            # Конвертируем локальные ID фронтира в глобальные Soma ID
            actual_somas = frontier_somas[col_idx]
            surviving_synapses[row_idx, actual_somas] = True

            # Извлекаем Axon IDs через Zero-Index Trap
            # axon_id = (target & 0x00FFFFFF) - 1
            packed_targets = frontier_targets[valid_mask]
            source_axon_ids = (packed_targets & 0x00FFFFFF).astype(np.int32) - 1

            # Мапим аксоны обратно на сомы-источники
            source_somas = axon_to_soma[source_axon_ids]
            
            # Отсекаем "виртуальные" аксоны или внешние входы (у них нет сомы в этом шарде)
            source_somas = source_somas[source_somas != -1]

            # Оставляем только сомы, которые еще не посещали (защита от циклов)
            new_somas = source_somas[~surviving_somas[source_somas]]
            
            if len(new_somas) == 0:
                break
                
            surviving_somas[new_somas] = True
            frontier_somas = np.unique(new_somas)

        # 4. Формируем "Картридж Навыка" (Payload)
        return {
            "soma_mask": surviving_somas,
            "synapse_mask": surviving_synapses,
            "flags": self.mem.flags[surviving_somas].copy(),
            "threshold_offset": self.mem.threshold_offset[surviving_somas].copy(),
            "targets": self.mem.targets[surviving_synapses].copy(),
            "weights": self.mem.weights[surviving_synapses].copy()
        }

    def inject_subgraph(self, payload: dict):
        """
        [DOD] Surgical Grafting с Монументализацией. 
        Физически стирает целевые нейроны и вживляет имплант с весами в Rank 15.
        """
        soma_mask = payload["soma_mask"]
        syn_mask = payload["synapse_mask"]

        # 1. Стираем текущее состояние в местах инъекции (убиваем конфликты)
        # Битовые маски (np.bool_) позволяют делать это мгновенно через NumPy
        self.mem.flags[soma_mask] &= 0xF0 # Оставляем типы (биты 4-7), сбрасываем спайки/BDP
        self.mem.threshold_offset[soma_mask] = 0
        self.mem.targets[syn_mask] = 0
        self.mem.weights[syn_mask] = 0

        # 2. Инъекция состояния сом
        self.mem.flags[soma_mask] = payload["flags"]
        self.mem.threshold_offset[soma_mask] = payload["threshold_offset"]

        # 3. Восстановление топологии
        self.mem.targets[syn_mask] = payload["targets"]
        
        # 4. Монументализация (Ранг 15 -> 32767)
        # Сохраняем знак источника (Dale's Law), но выкручиваем силу на максимум
        implant_weights = payload["weights"]
        signs = np.sign(implant_weights).astype(np.int16)
        self.mem.weights[syn_mask] = signs * 32767
