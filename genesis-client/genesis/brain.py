import os
import toml
from typing import Dict
from .control import GenesisControl
from .memory import GenesisMemory

def fnv1a_32(data: bytes) -> int:
    """Детерминированный хэш (совпадает с реализацией на Rust)."""
    hash_val = 0x811c9dc5
    for b in data:
        hash_val ^= b
        hash_val = (hash_val * 0x01000193) & 0xFFFFFFFF
    return hash_val

class Zone:
    """Представляет одну зону мозга (например, SensoryCortex)."""
    def __init__(self, name: str, baked_dir: str):
        self.name = name
        self.hash = fnv1a_32(name.encode('utf-8'))
        
        self.manifest_path = os.path.join(baked_dir, "manifest.toml")
        
        # Control Plane доступен всегда (даже если нода оффлайн)
        self.control = GenesisControl(self.manifest_path)
        self._memory = None

    @property
    def memory(self) -> GenesisMemory:
        """Ленивая инициализация Memory Plane (mmap). Упадет, если нода не запущена."""
        if self._memory is None:
            self._memory = GenesisMemory(self.hash)
        return self._memory

class GenesisClusterControl:
    """Глобальный контроллер: применяет команды ко всем зонам кластера одновременно."""
    def __init__(self, zones: Dict[str, Zone]):
        self._zones = zones

    def set_night_interval(self, ticks: int):
        for zone in self._zones.values():
            zone.control.set_night_interval(ticks)

    def set_prune_threshold(self, threshold: int):
        for zone in self._zones.values():
            zone.control.set_prune_threshold(threshold)

    def set_dopamine_receptors(self, variant_id: int, d1_affinity: int, d2_affinity: int):
        for zone in self._zones.values():
            zone.control.set_dopamine_receptors(variant_id, d1_affinity, d2_affinity)

class GenesisBrain:
    """
    Единая точка входа для управления мультизональным коннектомом.
    Автоматически читает топологию из brain.toml.
    """
    def __init__(self, brain_toml_path: str):
        self.brain_toml_path = brain_toml_path
        self.zones: Dict[str, Zone] = {}

        if not os.path.exists(brain_toml_path):
            raise FileNotFoundError(f"Brain configuration not found: {brain_toml_path}")

        with open(brain_toml_path, "r", encoding="utf-8") as f:
            data = toml.load(f)

        for zone_data in data.get("zone", []):
            name = zone_data["name"]
            baked_dir = zone_data.get("baked_dir", f"baked/{name}/")
            self.zones[name] = Zone(name, baked_dir)

        # Интерфейс кластерного управления
        self.control = GenesisClusterControl(self.zones)
