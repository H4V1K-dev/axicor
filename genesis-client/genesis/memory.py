import mmap
import os
import struct
import tempfile

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

    def __init__(self, zone_hash: int, read_only: bool = False):
        self.zone_hash = zone_hash
        path = self._resolve_path(zone_hash)
        
        # Если read_only=True, открываем строго на чтение (Zero-Copy Introspection).
        # Если открыть на запись, numpy мутации могут крашнуть GPU-рантайм, 
        # используйте осторожно для дистилляции (Pruning).
        mode = os.O_RDONLY if read_only else os.O_RDWR
        
        fd = os.open(path, mode)
        if os.name == "nt":
            access = mmap.ACCESS_READ if read_only else mmap.ACCESS_WRITE
            self._mm = mmap.mmap(fd, 0, access=access)
        else:
            prot = mmap.PROT_READ if read_only else mmap.PROT_READ | mmap.PROT_WRITE
            self._mm = mmap.mmap(fd, 0, prot=prot)
        os.close(fd)
        
        # 1. Читаем заголовок
        header = struct.unpack_from(self.SHM_HEADER_FMT, self._mm, 0)
        assert header[0] == self.MAGIC, f"Invalid SHM Magic in {path}"
        
        self.padded_n = header[4]
        self.dendrite_slots = header[5]
        self.weights_offset = header[6]
        self.targets_offset = header[7]
        self.flags_offset = header[16]
        self.voltage_offset = self.SHM_HEADER_SIZE
        self.threshold_offset_offset = self.voltage_offset + self.padded_n * 4 + self.padded_n
        self.timers_offset = self.threshold_offset_offset + self.padded_n * 4
        
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

        self.soma_voltage = np.ndarray(
            (self.padded_n,),
            dtype=np.int32,
            buffer=self._mm,
            offset=self.voltage_offset
        )

        self.threshold_offset = np.ndarray(
            (self.padded_n,),
            dtype=np.int32,
            buffer=self._mm,
            offset=self.threshold_offset_offset
        )

        self.timers = np.ndarray(
            (self.padded_n,),
            dtype=np.uint8,
            buffer=self._mm,
            offset=self.timers_offset
        )

    @staticmethod
    def _resolve_path(zone_hash: int) -> str:
        name = f"genesis_shard_{zone_hash:08X}"
        if os.name == "nt":
            return os.path.join(tempfile.gettempdir(), name)
        return os.path.join("/dev/shm", name)

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

    def get_network_stats(self, saturation_threshold: int = 31129) -> dict:
        """Сканирует VRAM (Zero-Copy) и возвращает статистику графа."""
        # Убираем нулевые цели (пустые слоты)
        valid_mask = self.targets != 0
        active_weights = self.weights[valid_mask]
        abs_weights = np.abs(active_weights)
        burst_count = (self.soma_flags >> 1) & 0x07
        spiking_mask = (self.soma_flags & 0x01) != 0
        
        if len(active_weights) == 0:
            return {
                "active_synapses": 0,
                "avg_weight": 0.0,
                "max_weight": 0,
                "saturated_weight_share": 0.0,
                "spike_rate": float(np.mean(spiking_mask)),
                "mean_burst_count": float(np.mean(burst_count)),
            }
            
        return {
            "active_synapses": int(np.sum(valid_mask)),
            "avg_weight": float(np.mean(abs_weights)),
            "max_weight": int(np.max(abs_weights)),
            "saturated_weight_share": float(np.mean(abs_weights >= saturation_threshold)),
            "spike_rate": float(np.mean(spiking_mask)),
            "mean_burst_count": float(np.mean(burst_count)),
        }

    def distill_graph(self, prune_threshold: int) -> int:
        """
        Zero-Copy дистилляция графа (In-place Pruning).
        Выжигает исследовательский шум (слабые связи) напрямую в памяти ОС (VRAM-дамп).
        
        :param prune_threshold: Порог отсечения. Связи с abs(weight) < prune_threshold удаляются.
        :return: Количество выжженных (удаленных) связей.
        """
        # Векторизованный поиск (SIMD-accelerated).
        # self.targets != 0 гарантирует, что мы не трогаем уже пустые слоты (Zero-Index Trap)
        weak_mask = (self.targets != 0) & (np.abs(self.weights) < prune_threshold)
        
        pruned_count = int(np.sum(weak_mask))
        
        # Физическое уничтожение связей в Shared Memory (Zero-Copy).
        # Для CUDA-ядра target == 0 означает аппаратный Early Exit.
        self.targets[weak_mask] = 0
        self.weights[weak_mask] = 0
        
        return pruned_count

    def clear_weights(self):
        """
        Tabula Rasa: Полное обнуление всех весов в шарде.
        Оставляет топологию (связи), но делает их "невесомыми".
        """
        self.weights.fill(0)

    def close(self):
        self._mm.close()
