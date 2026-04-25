import numpy as np
import struct
from .memory import AxicorMemory

class AxicorSurgeon:
    """
    Data-Oriented scalpel for direct VRAM intervention.
    Operates strictly via Zero-Copy mmap, without calling the Rust orchestrator.
    """
    def __init__(self, memory: AxicorMemory):
        self.mem = memory

    def incubate_gaba(self, baseline_weight: int = -2000000000) -> int:
        mask = (self.mem.targets != 0) & (self.mem.weights < 0)
        self.mem.weights[mask] = baseline_weight
        return int(np.sum(mask))

    def extract_reflex_path(self, root_soma_ids: np.ndarray, prune_threshold: int = 15000) -> dict:
        """
        [DOD] Vectorized Back-Tracing for isolating specific skills.
        Finds all strong connections (excitatory and inhibitory) that feed the target neurons.
        """
        padded_n = self.mem.padded_n
        
        # 1. O(1) Inverse Mapping: Axon_ID -> Soma_ID
        # Extract total_axons (offset 0x20 in 64-byte ShmHeader v2)
        total_axons = struct.unpack_from("<I", self.mem._mm, 0x20)[0]
        axon_to_soma = np.full(total_axons, -1, dtype=np.int32)
        
        # [DOD FIX] Invariant 14: Local axons have identity mapping soma_id == axon_id
        # The first padded_n axons in VRAM are local axons for the neurons.
        axon_to_soma[:padded_n] = np.arange(padded_n, dtype=np.int32)

        # 2. Global survival masks (Zero-Garbage BFS)
        surviving_somas = np.zeros(padded_n, dtype=np.bool_)
        surviving_somas[root_soma_ids] = True
        
        surviving_synapses = np.zeros_like(self.mem.targets, dtype=np.bool_)
        frontier_somas = root_soma_ids

        # 3. Graph traversal Hot Loop (strictly vectorized)
        while len(frontier_somas) > 0:
            # Slice columns for current frontier neurons
            frontier_targets = self.mem.targets[:, frontier_somas]
            frontier_weights = self.mem.weights[:, frontier_somas]

            # Only strong connections survive (Dale's Law: abs() captures inhibitory)
            # Life sign: target != 0 and weight above threshold
            # [DOD FIX] Compare Mass Domain weights against shifted threshold
            valid_mask = (frontier_targets != 0) & (np.abs(frontier_weights) > (prune_threshold << 16))

            # Find coordinates of surviving synapses in the local frontier window
            row_idx, col_idx = np.where(valid_mask)
            
            # Convert local frontier IDs to global Soma IDs
            actual_somas = frontier_somas[col_idx]
            surviving_synapses[row_idx, actual_somas] = True

            # Extract Axon IDs via Zero-Index Trap
            # axon_id = (target & 0x00FFFFFF) - 1
            packed_targets = frontier_targets[valid_mask]
            source_axon_ids = (packed_targets & 0x00FFFFFF).astype(np.int32) - 1

            # Map axons back to source somas
            source_somas = axon_to_soma[source_axon_ids]
            
            # Exclude "virtual" axons or external inputs (no local soma in this shard)
            source_somas = source_somas[source_somas != -1]

            # Retain only somas not yet visited (loop protection)
            new_somas = source_somas[~surviving_somas[source_somas]]
            
            if len(new_somas) == 0:
                break
                
            surviving_somas[new_somas] = True
            frontier_somas = np.unique(new_somas)

        # 4. Construct "Skill Cartridge" (Payload)
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
        [DOD] Surgical Grafting with Monumentalization. 
        Physically erases target neurons and implants the subgraph with Rank 15 weights.
        """
        soma_mask = payload["soma_mask"]
        syn_mask = payload["synapse_mask"]

        # 1. Erase current state at injection sites (kill conflicts)
        # Bitmasks (np.bool_) enable instantaneous NumPy operations
        self.mem.flags[soma_mask] &= 0xF0 # Retain types (bits 4-7), reset spikes/BDP
        self.mem.threshold_offset[soma_mask] = 0
        self.mem.targets[syn_mask] = 0
        self.mem.weights[syn_mask] = 0

        # 2. Inject soma state
        self.mem.flags[soma_mask] = payload["flags"]
        self.mem.threshold_offset[soma_mask] = payload["threshold_offset"]

        # 3. Restore topology
        self.mem.targets[syn_mask] = payload["targets"]
        
        # 4. Monumentalization (Rank 15 -> 2.14B)
        # Preserve source sign (Dale's Law), maximize strength
        implant_weights = payload["weights"]
        signs = np.sign(implant_weights).astype(np.int32)
        self.mem.weights[syn_mask] = signs * 2140000000
