import os
import time
import toml
from typing import Callable, Any

class GenesisControl:
    """
    Control Plane SDK. 
    Управляет параметрами рантайма 'на лету' через атомарную перезапись manifest.toml.
    """
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path

    def _update_manifest(self, mutator_func: Callable[[dict[str, Any]], None]):
        # 1. Читаем текущее состояние
        with open(self.manifest_path, "r") as f:
            data = toml.load(f)

        # 2. Применяем мутацию
        mutator_func(data)

        # 3. Атомарная запись (Zero-downtime для Rust-оркестратора)
        tmp_path = self.manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            toml.dump(data, f)

        # os.replace атомарен, но на Windows может дать PermissionError если файл заблокирован.
        for attempt in range(5):
            try:
                os.replace(tmp_path, self.manifest_path)
                return
            except PermissionError as e:
                if attempt < 4:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise PermissionError(
                        f"Cannot replace manifest (file locked?): {self.manifest_path}. "
                        "Kill any running genesis-node: Get-Process genesis-node -EA 0 | Stop-Process -Force. "
                        "Close editors with manifest.toml open."
                    ) from e

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

    def set_external_udp_in(self, port: int):
        """Updates the zone UDP input port used by genesis-node."""
        def mutate(d):
            if "network" not in d:
                d["network"] = {}
            d["network"]["external_udp_in"] = int(port)
        self._update_manifest(mutate)

    def set_external_udp_out_target(self, host: str, port: int):
        """Updates the UDP target used for output packets."""
        def mutate(d):
            if "network" not in d:
                d["network"] = {}
            d["network"]["external_udp_out_target"] = f"{host}:{int(port)}"
        self._update_manifest(mutate)

    def set_fast_path_udp_local(self, port: int):
        """Updates the local fast-path UDP port used to derive geometry/telemetry ports."""
        def mutate(d):
            if "network" not in d:
                d["network"] = {}
            d["network"]["fast_path_udp_local"] = int(port)
        self._update_manifest(mutate)

    def list_variant_ids(self) -> list[int]:
        with open(self.manifest_path, "r") as f:
            data = toml.load(f)
        return [int(v["id"]) for v in data.get("variants", [])]

    def patch_variant_fields(self, variant_id: int, **fields: Any):
        """Atomic patch of arbitrary variant fields exposed by manifest.toml."""
        def mutate(d):
            for v in d.get("variants", []):
                if v["id"] == variant_id:
                    for key, value in fields.items():
                        v[key] = value
        self._update_manifest(mutate)

    def set_adaptive_leak(
        self,
        variant_id: int,
        adaptive_leak_mode: int,
        dopamine_leak_gain: int,
        burst_leak_gain: int,
        leak_min: int,
        leak_max: int,
    ):
        """Configures adaptive leak fields for a single variant."""
        self.patch_variant_fields(
            variant_id,
            adaptive_leak_mode=adaptive_leak_mode,
            dopamine_leak_gain=dopamine_leak_gain,
            burst_leak_gain=burst_leak_gain,
            leak_min=leak_min,
            leak_max=leak_max,
        )
