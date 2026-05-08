import os
import toml
from typing import Callable, Any

from .axic import AxicReader
from .platform import get_manifest_path
from .utils import fnv1a_32 # DOD FIX: Fixed Circular Import

class AxicorControl:
    """
    Control Plane SDK. 
    Manages runtime parameters on the fly via atomic rewriting of manifest.toml.
    """
    def __init__(self, axic_path: str, zone_name: str):
        self.zone_hash = fnv1a_32(zone_name.encode('utf-8'))
        # [DOD FIX] Node exports the manifest into SHM. SDK modifies it there!
        self.manifest_path = get_manifest_path(self.zone_hash)
        
        if not os.path.exists(self.manifest_path):
            reader = AxicReader(axic_path)
            data = reader.read_file(f"baked/{zone_name}/manifest.toml")
            if data:
                with open(self.manifest_path, "wb") as f:
                    f.write(data)
            else:
                raise FileNotFoundError(f"manifest.toml not found in {axic_path} for zone {zone_name}")
                    
        with open(self.manifest_path, "r") as f:
            self.manifest = toml.load(f)

    def _update_manifest(self, mutator_func: Callable[[dict[str, Any]], None]):
        # 1. Read current state
        with open(self.manifest_path, "r") as f:
            data = toml.load(f)

        # 2. Apply mutation
        mutator_func(data)
        self.manifest = data # Update cache

        # 3. Atomic write (Zero-downtime for the Rust orchestrator)
        tmp_path = self.manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            toml.dump(data, f)
        
        # os.replace guarantees atomic inode replacement at the filesystem level
        os.replace(tmp_path, self.manifest_path)

    def set_night_interval(self, ticks: int):
        """Changes the frequency of the sleep phase (Consolidation/Pruning). 0 = disable sleep."""
        def mutate(d):
            if "settings" not in d:
                d["settings"] = {}
            d["settings"]["night_interval_ticks"] = ticks
        self._update_manifest(mutate)

    def set_prune_threshold(self, threshold: int):
        """Changes the pruning aggressiveness threshold (removal of weak connections)."""
        def mutate(d):
            if "settings" not in d:
                d["settings"] = {}
            if "plasticity" not in d["settings"]:
                d["settings"]["plasticity"] = {}
            d["settings"]["plasticity"]["prune_threshold"] = threshold
        self._update_manifest(mutate)

    def set_checkpoints_interval(self, ticks: int):
        """Zero-Downtime trigger: Forces a VRAM dump flush to disk (D2H DMA)."""
        def mutate(d):
            if "settings" not in d: d["settings"] = {}
            d["settings"]["save_checkpoints_interval_ticks"] = ticks
        self._update_manifest(mutate)

    def set_dopamine_receptors(self, variant_id: int, d1_affinity: int = None, d2_affinity: int = None):
        """Configures the sensitivity of a specific neuron type to rewards (R-STDP)."""
        def mutate(d):
            for v in d.get("variants", []):
                if v["id"] == variant_id:
                    if d1_affinity is not None: v["d1_affinity"] = d1_affinity
                    if d2_affinity is not None: v["d2_affinity"] = d2_affinity
        self._update_manifest(mutate)

    def set_membrane_physics(self, variant_id: int, leak_rate: int = None, homeostasis_penalty: int = None, homeostasis_decay: int = None):
        """Zero-Downtime patch of GLIF physics and homeostasis for a specific neuron type."""
        def mutate(d):
            for v in d.get("variants", []):
                if v["id"] == variant_id:
                    if leak_rate is not None: v["leak_rate"] = leak_rate
                    if homeostasis_penalty is not None: v["homeostasis_penalty"] = homeostasis_penalty
                    if homeostasis_decay is not None: v["homeostasis_decay"] = homeostasis_decay
        self._update_manifest(mutate)

    def set_max_sprouts(self, max_sprouts: int):
        """Changes the limit of new synapses per night (Structural Plasticity speed)."""
        def mutate(d):
            if "settings" not in d: d["settings"] = {}
            if "plasticity" not in d["settings"]: d["settings"]["plasticity"] = {}
            d["settings"]["plasticity"]["max_sprouts"] = max_sprouts
        self._update_manifest(mutate)

    def disable_all_plasticity(self):
        """Zero-Downtime patch for CRYSTALLIZED phase. Disables GSOP (STDP) completely."""
        def mutate(d):
            for v in d.get("variants", []):
                v["gsop_potentiation"] = 0
                v["gsop_depression"] = 0
        self._update_manifest(mutate)

    def set_inertia_curve(self, variant_id: int, new_curve: list[int]):
        """Zero-Downtime patch of the inertia curve (8 ranks) to control graph crystallization."""
        if len(new_curve) != 8:
            raise ValueError("Inertia curve must contain exactly 8 values.")
        def mutate(d):
            for v in d.get("variants", []):
                if v["id"] == variant_id:
                    v["inertia_curve"] = new_curve
        self._update_manifest(mutate)
