import os
import mmap
import struct
import numpy as np

class GenesisMemory:
    # Строгий C-ABI v2 (64 bytes) -> genesis-core/src/ipc.rs
    # magic(I), version(B), state(B), pad(H)
    # padded_n(I), dendrite_slots(I), weights_off(I), targets_off(I)
    # epoch(Q)
    # total_axons(I), handovers_off(I), handovers_count(I), zone_hash(I)
    # prunes_off(I), prunes_count(I), incoming_prunes(I), flags_off(I)
    SHM_HEADER_FMT = "<IBBHIIIIQIIIIIIII"
    SHM_HEADER_SIZE = 64
    MAGIC = 0x47454E53 # "GENS"

    def __init__(self, zone_hash: int):
        self.zone_hash = zone_hash
        path = f"/dev/shm/genesis_shard_{zone_hash:08X}"
        
        # Открываем строго на чтение (Zero-Copy Introspection)
        # Если открыть на запись, numpy мутации могут крашнуть GPU-рантайм
        fd = os.open(path, os.O_RDONLY)
        self._mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        os.close(fd)
        
        # 1. Читаем заголовок
        header = struct.unpack_from(self.SHM_HEADER_FMT, self._mm, 0)
        assert header[0] == self.MAGIC, f"Invalid SHM Magic in {path}"
        
        self.padded_n = header[4]
        self.dendrite_slots = header[5]
        self.weights_offset = header[6]
        self.targets_offset = header[7]
        self.flags_offset = header[16]
        
        assert self.dendrite_slots == 128, "C-ABI violation: dendrite_slots != 128"
        
        # 2. Натягиваем матрицы прямо на оперативную память ОС
        # [128, padded_n] - Columnar Layout (Coalesced GPU access)
        self.weights = np.ndarray(
            (self.dendrite_slots, self.padded_n), 
            dtype=np.int16, 
            buffer=self._mm, 
            offset=self.weights_offset
        )
        
        self.targets = np.ndarray(
            (self.dendrite_slots, self.padded_n), 
            dtype=np.uint32, 
            buffer=self._mm, 
            offset=self.targets_offset
        )
        
        self.soma_flags = np.ndarray(
            (self.padded_n,),
            dtype=np.uint8,
            buffer=self._mm,
            offset=self.flags_offset
        )

    def extract_topology(self, soma_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Векторизованное извлечение геометрии для конкретного нейрона.
        Решает проблему Zero-Index Trap без циклов Python.
        Возвращает: (axon_ids, segment_offsets, weights)
        """
        # Срез колонки дендритов для одной сомы (O(1) pointer view)
        raw_targets = self.targets[:, soma_idx]
        raw_weights = self.weights[:, soma_idx]
        
        # Фильтруем пустые слоты
        valid_mask = raw_targets != 0
        valid_targets = raw_targets[valid_mask]
        active_weights = raw_weights[valid_mask]
        
        # Распаковываем u32
        # [31..24] seg_offset | [23..0] axon_id + 1
        axon_ids = (valid_targets & 0x00FFFFFF).astype(np.int32) - 1
        seg_offsets = valid_targets >> 24
        
        return axon_ids, seg_offsets, active_weights

    def get_network_stats(self) -> dict:
        """Сканирует VRAM (Zero-Copy) и возвращает статистику графа."""
        # Убираем нулевые цели (пустые слоты)
        valid_mask = self.targets != 0
        active_weights = self.weights[valid_mask]
        
        if len(active_weights) == 0:
            return {"active_synapses": 0, "avg_weight": 0.0, "max_weight": 0}
            
        return {
            "active_synapses": int(np.sum(valid_mask)),
            "avg_weight": float(np.mean(np.abs(active_weights))),
            "max_weight": int(np.max(np.abs(active_weights)))
        }

    def close(self):
        self._mm.close()
