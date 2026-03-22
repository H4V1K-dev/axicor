import os
import toml
from typing import Callable, Any

class GenesisControl:
    """
    Control Plane SDK. 
    Управляет параметрами рантайма 'на лету' через атомарную перезапись manifest.toml.
    """
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        # [DOD FIX] Кешируем манифест для быстрого доступа к параметрам симуляции
        with open(self.manifest_path, "r") as f:
            self.manifest = toml.load(f)

    def _update_manifest(self, mutator_func: Callable[[dict[str, Any]], None]):
        # 1. Читаем текущее состояние
        with open(self.manifest_path, "r") as f:
            data = toml.load(f)

        # 2. Применяем мутацию
        mutator_func(data)
        self.manifest = data # Обновляем кеш

        # 3. Атомарная запись (Zero-downtime для Rust-оркестратора)
        tmp_path = self.manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            toml.dump(data, f)
        
        # os.replace гарантирует атомарную подмену inode на уровне файловой системы
        os.replace(tmp_path, self.manifest_path)

    def set_night_interval(self, ticks: int):
        """Изменяет частоту наступления фазы сна (Consolidation/Pruning). 0 = отключить сон."""
        def mutate(d):
            if "settings" not in d:
                d["settings"] = {}
            d["settings"]["night_interval_ticks"] = ticks
        self._update_manifest(mutate)

    def set_prune_threshold(self, threshold: int):
        """Изменяет порог агрессивности прунинга (удаления слабых связей)."""
        def mutate(d):
            if "settings" not in d:
                d["settings"] = {}
            if "plasticity" not in d["settings"]:
                d["settings"]["plasticity"] = {}
            d["settings"]["plasticity"]["prune_threshold"] = threshold
        self._update_manifest(mutate)

    def set_dopamine_receptors(self, variant_id: int, d1_affinity: int, d2_affinity: int):
        """Настраивает восприимчивость конкретного типа нейронов к наградам (R-STDP)."""
        def mutate(d):
            for v in d.get("variants", []):
                if v["id"] == variant_id:
                    v["d1_affinity"] = d1_affinity
                    v["d2_affinity"] = d2_affinity
        self._update_manifest(mutate)

    def set_membrane_physics(self, variant_id: int, leak_rate: int, homeostasis_penalty: int, homeostasis_decay: int):
        """Zero-Downtime патч физики GLIF и гомеостаза для конкретного типа нейронов."""
        def mutate(d):
            for v in d.get("variants", []):
                if v["id"] == variant_id:
                    v["leak_rate"] = leak_rate
                    v["homeostasis_penalty"] = homeostasis_penalty
                    v["homeostasis_decay"] = homeostasis_decay
        self._update_manifest(mutate)

    def set_max_sprouts(self, max_sprouts: int):
        """Изменяет лимит новых синапсов за одну ночь (Structural Plasticity speed)."""
        def mutate(d):
            if "settings" not in d: d["settings"] = {}
            if "plasticity" not in d["settings"]: d["settings"]["plasticity"] = {}
            d["settings"]["plasticity"]["max_sprouts"] = max_sprouts
        self._update_manifest(mutate)

    def disable_all_plasticity(self):
        """Zero-Downtime патч для фазы CRYSTALLIZED. Полностью выключает GSOP (STDP)."""
        def mutate(d):
            for v in d.get("variants", []):
                v["gsop_potentiation"] = 0
                v["gsop_depression"] = 0
        self._update_manifest(mutate)

    def set_inertia_curve(self, variant_id: int, new_curve: list[int]):
        """Zero-Downtime патч кривой инерции (16 рангов) для контроля кристаллизации графа."""
        if len(new_curve) != 16:
            raise ValueError("Inertia curve must contain exactly 16 values.")
        def mutate(d):
            for v in d.get("variants", []):
                if v["id"] == variant_id:
                    v["inertia_curve"] = new_curve
        self._update_manifest(mutate)
