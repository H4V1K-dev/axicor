import os
import glob
import math
import warnings
import toml
from pathlib import Path
from typing import List, Dict, Any

class NeuronBlueprint:
    """Обертка над спарсенным TOML-файлом типа нейрона из GNM-Library."""
    def __init__(self, filepath: str, data: list):
        self.filepath = filepath
        self.data_list = data
        # Берем имя первого нейрона для идентификации группы
        self.name = data[0].get("name", "Unknown") if data else "Unknown"

    def set_plasticity(self, pot: int, dep: int):
        """HFT-тюнинг: динамическая смена параметров R-STDP."""
        for n_type in self.data_list:
            n_type["gsop_potentiation"] = pot
            n_type["gsop_depression"] = dep
        return self

class IoMatrixDesigner:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.padded_pixels = ((width * height + 31) // 32) * 32
        self.bytes_per_tick = self.padded_pixels // 8

    def fragment(self, sync_batch_ticks: int, mtu: int = 65507) -> list[dict]:
        # 1. Полезная нагрузка (20 байт — ExternalIoHeader)
        max_payload = mtu - 20
        # 2. Байт на тик (целочисленное деление)
        max_bytes_per_tick = max_payload // sync_batch_ticks
        # 3. Пикселей на чанк (кратность 32 бита / 4 байта)
        max_aligned_pixels = (max_bytes_per_tick // 4) * 32

        # 4. Проверка вместимости
        if self.padded_pixels <= max_aligned_pixels:
            return [{"width": self.width, "height": self.height, "uv_rect": [0.0, 0.0, 1.0, 1.0]}]

        # 5. Нарезка (Row-based Slicing)
        chunk_height = max_aligned_pixels // self.width
        if chunk_height == 0:
            raise ValueError(f"[IoMatrix] Matrix width {self.width} exceeds MTU {mtu} capacity for {sync_batch_ticks} ticks. "
                             f"Max aligned pixels per packet: {max_aligned_pixels}")

        chunks = []
        current_y = 0
        while current_y < self.height:
            h = min(chunk_height, self.height - current_y)
            # UV Rect: [u_offset, v_offset, u_width, v_height]
            uv_rect = [0.0, current_y / self.height, 1.0, h / self.height]
            chunks.append({
                "width": self.width,
                "height": h,
                "uv_rect": uv_rect
            })
            current_y += h
        return chunks

    def get_uv_rect(self, mode: str = "Pie", index: int = 0, total: int = 1) -> list[float]:
        if mode == "Pie":
            return [0.0, 0.0, 1.0, 1.0]
        elif mode == "Canvas":
            # Заглушка под режим "Canvas" (нарезка сеткой) — реализуем на следующем шаге.
            pass
        return [0.0, 0.0, 1.0, 1.0]

class LayerDesigner:
    def __init__(self, zone: 'ZoneDesigner', name: str, height_pct: float, density: float):
        self.zone = zone
        self.name = name
        self.height_pct = height_pct
        self.density = density
        self.composition = {}

    def add_population(self, blueprint: NeuronBlueprint, fraction: float):
        self.composition[blueprint.name] = fraction
        self.zone._register_blueprint(blueprint)
        return self

class ZoneDesigner:
    def __init__(self, builder: 'BrainBuilder', name: str, x: int, y: int, z: int):
        self.builder = builder
        self.name = name
        
        # 1. Автоисправление (Clamping) под аппаратные лимиты PackedPosition (11/11/6 bit)
        self.vox_x = max(1, min(x, 2047))
        self.vox_y = max(1, min(y, 2047))
        self.vox_z = max(1, min(z, 63))
        
        if self.vox_x != x or self.vox_y != y or self.vox_z != z:
            warnings.warn(f"[Builder] ⚠️ Zone '{self.name}' dimensions corrected from "
                          f"({x}, {y}, {z}) to ({self.vox_x}, {self.vox_y}, {self.vox_z}) "
                          f"to fit 11/11/6-bit hardware constraints.")
            
        self.layers: List[LayerDesigner] = []
        self.blueprints_registry: Dict[str, dict] = {}
        
        # New I/O registry
        self.inputs: List[Dict[str, Any]] = []
        self.outputs: List[Dict[str, Any]] = []
        
    def add_input(self, name: str, width: int, height: int, target_type: str = "All", entry_z: str = "top", stride: int = 1):
        # 1. Валидация entry_z
        if entry_z not in ["top", "mid", "bottom"]:
            try:
                float(entry_z)
            except ValueError:
                raise ValueError(f"Invalid entry_z: '{entry_z}'. Must be 'top', 'mid', 'bottom' or a float string.")

        # 2. Фрагментация
        designer = IoMatrixDesigner(width, height)
        batch_ticks = self.builder.sim_params["sync_batch_ticks"]
        chunks = designer.fragment(sync_batch_ticks=batch_ticks)

        # 3. Регистрация чанков
        for i, chunk in enumerate(chunks):
            chunk_name = name if len(chunks) == 1 else f"{name}_chunk_{i}"
            self.inputs.append({
                "name": chunk_name,
                "target_zone": self.name,
                "target_type": target_type,
                "width": chunk["width"],
                "height": chunk["height"],
                "stride": stride,
                "entry_z": entry_z,
                "uv_rect": chunk["uv_rect"]
            })
        return self

    def add_output(self, name: str, width: int, height: int, target_type: str = "All", stride: int = 1):
        # 1. Фрагментация
        designer = IoMatrixDesigner(width, height)
        batch_ticks = self.builder.sim_params["sync_batch_ticks"]
        chunks = designer.fragment(sync_batch_ticks=batch_ticks)

        # 2. Регистрация чанков
        for i, chunk in enumerate(chunks):
            chunk_name = name if len(chunks) == 1 else f"{name}_chunk_{i}"
            self.outputs.append({
                "name": chunk_name,
                "source_zone": self.name,
                "target_type": target_type,
                "width": chunk["width"],
                "height": chunk["height"],
                "stride": stride,
                "uv_rect": chunk["uv_rect"]
            })
        return self
        
    def add_layer(self, name: str, height_pct: float, density: float) -> LayerDesigner:
        layer = LayerDesigner(self, name, height_pct, density)
        self.layers.append(layer)
        return layer
        
    def _register_blueprint(self, bp: NeuronBlueprint):
        # Регистрируем все типы из файла
        for n_type in bp.data_list:
            # [HFT FIX] Map period to DDS multiplier (heartbeat_m)
            if "spontaneous_firing_period_ticks" in n_type and n_type["spontaneous_firing_period_ticks"] > 0:
                period = n_type["spontaneous_firing_period_ticks"]
                # phase = (tick * m + salt) & 0xFFFF; heart = phase < m
                # Probability = m / 65536 = 1 / period  => m = 65536 / period
                n_type["heartbeat_m"] = int(65536 / period)

            n_name = n_type.get("name")
            if n_name not in self.blueprints_registry:
                # Защита от превышения лимита типов (4-битная маска = макс 16)
                if len(self.blueprints_registry) >= 16:
                    raise ValueError(f"Zone '{self.name}' exceeds the maximum of 16 neuron types!")
                self.blueprints_registry[n_name] = n_type

class BrainBuilder:
    def __init__(self, project_name: str, output_dir: str, gnm_lib_path: str = "GNM-Library"):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.gnm_lib_path = Path(gnm_lib_path)
        self.zones: List[ZoneDesigner] = []
        self.connections: List[Dict[str, Any]] = []
        
        # Дефолтные параметры симуляции
        self.sim_params = {
            "tick_duration_us": 100,
            "total_ticks": 0,
            "master_seed": "GENESIS",
            "voxel_size_um": 25.0,
            "signal_speed_m_s": 0.5,
            "sync_batch_ticks": 100,
            "segment_length_voxels": 2,
            "axon_growth_max_steps": 250
        }

    def add_zone(self, name: str, width_vox: int, depth_vox: int, height_vox: int) -> ZoneDesigner:
        zone = ZoneDesigner(self, name, width_vox, depth_vox, height_vox)
        self.zones.append(zone)
        return zone

    def connect(self, from_zone: ZoneDesigner, to_zone: ZoneDesigner, out_matrix: str, 
                in_width: int, in_height: int, entry_z: str = "top", target_type: str = "All", growth_steps: int = 1000):
        # Проверяем, существует ли такая выходная матрица в зоне-источнике
        if not any(out["name"] == out_matrix for out in from_zone.outputs):
            warnings.warn(f"[Builder] ⚠️ Output matrix '{out_matrix}' not found in zone '{from_zone.name}'!")

        self.connections.append({
            "from": from_zone.name,
            "to": to_zone.name,
            "output_matrix": out_matrix,
            "width": in_width,
            "height": in_height,
            "entry_z": entry_z,
            "target_type": target_type,
            "growth_steps": growth_steps
        })
        return self

    def gnm_lib(self, query: str) -> NeuronBlueprint:
        """
        Умный поиск по библиотеке. 
        Например query="VISp4/64" найдет "GNM-Library/Cortex/L4/spiny/VISp4/64.toml"
        """
        search_pattern = f"{self.gnm_lib_path}/**/*{query}*.toml"
        matches = glob.glob(search_pattern, recursive=True)
        
        if not matches:
            raise FileNotFoundError(f"⚠️ Blueprint matching '{query}' not found in {self.gnm_lib_path}")
            
        # Исправлено: берем первый найденный файл
        target_file = matches[0]
        with open(target_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
            
        if "neuron_type" not in data or not data["neuron_type"]:
            raise ValueError(f"Invalid blueprint format in {target_file}")
            
        return NeuronBlueprint(target_file, data["neuron_type"])

    def dry_run_stats(self) -> str:
        """
        [DOD] Strict C-ABI memory cost estimation.
        O(1) расчет потребления VRAM и /dev/shm до генерации TOML.
        """
        report = [f"📊 Genesis Memory Estimator: {self.project_name}"]
        total_vram = 0
        total_shm = 0

        for zone in self.zones:
            raw_neurons = 0
            cursor_pct = 0.0

            # Зеркальное отражение логики genesis-baker/src/bake/neuron_placement.rs
            for layer in zone.layers:
                z_start = int(cursor_pct * zone.vox_z)
                z_end = min(255, int((cursor_pct + layer.height_pct) * zone.vox_z))
                cursor_pct += layer.height_pct

                layer_vol = zone.vox_x * zone.vox_y * (z_end - z_start)
                layer_budget = int(math.floor(layer_vol * layer.density))
                raw_neurons += layer_budget

            # Warp Alignment (32 threads)
            padded_n = math.ceil(raw_neurons / 32) * 32

            virtual_axons = sum(inp["width"] * inp["height"] for inp in zone.inputs)
            incoming_pixels = sum(c.get("width", 0) * c.get("height", 0) for c in self.connections if c["to"] == zone.name)
            ghost_capacity = int(incoming_pixels * 2.0)

            raw_axons = padded_n + virtual_axons + ghost_capacity
            total_axons = math.ceil(raw_axons / 32) * 32

            # C-ABI Invariants (The 910-Byte Invariant)
            vram_bytes = (padded_n * 910) + (total_axons * 32)
            
            # SHM Night Phase IPC v4 (Header:64 + Weights + Targets + Flags + Handovers + Prunes)
            # 64 + (N*256) + (N*512) + (N*1) + (10000*20) + (10000*8)
            shm_bytes = 64 + (padded_n * 769) + 280_000

            total_vram += vram_bytes
            total_shm += shm_bytes

            report.append(f"  🔹 Zone '{zone.name}':")
            report.append(f"      Neurons: ~{raw_neurons} (Padded: {padded_n})")
            report.append(f"      Axons: {total_axons} (Local: {padded_n}, Virtual: {virtual_axons}, GhostCap: {ghost_capacity})")
            report.append(f"      VRAM: {vram_bytes / (1024**2):.2f} MB | SHM: {shm_bytes / (1024**2):.2f} MB")

        report.append(f"  🔻 TOTAL VRAM BUDGET: {total_vram / (1024**2):.2f} MB")
        report.append(f"  🔻 TOTAL SHM BUDGET:  {total_shm / (1024**2):.2f} MB")
        return "\n".join(report)

    def build(self):
        """Собирает ДНК мозга и генерирует все артефакты."""
        # [DOD FIX] Hard Physical Validation (Integer v_seg)
        # v_seg = (signal_speed_m_s * 1000 * (tick_duration_us / 1000)) / (voxel_size_um * segment_length_voxels)
        s_speed = self.sim_params["signal_speed_m_s"]
        t_dur = self.sim_params["tick_duration_us"]
        v_size = self.sim_params["voxel_size_um"]
        s_len = self.sim_params["segment_length_voxels"]
        
        v_seg_raw = (s_speed * 1000 * (t_dur / 1000)) / (v_size * s_len)
        
        if abs(v_seg_raw - round(v_seg_raw)) > 1e-5:
            # Interactive Auto-Fix
            import sys
            suggested_speed = (round(v_seg_raw) * v_size * s_len) / (1000 * (t_dur / 1000))
            error_msg = (f"\n❌ [Builder] Physical Validation Failed: v_seg must be an integer.\n"
                         f"Current v_seg: {v_seg_raw:.4f}\n"
                         f"To fix this, you can change signal_speed_m_s to {suggested_speed:.4f}")
            
            if sys.stdout.isatty():
                print(error_msg)
                val = input(f"Apply auto-fix (speed={suggested_speed:.4f})? [Y/n]: ").strip().lower()
                if val in ("", "y", "yes"):
                    self.sim_params["signal_speed_m_s"] = suggested_speed
                    print(f"✅ Auto-fix applied: signal_speed_m_s = {suggested_speed:.4f}")
                else:
                    raise ValueError("Manual fix required for v_seg integrality.")
            else:
                raise ValueError(error_msg)

        # [DOD FIX] Вывод расчетной стоимости графа перед генерацией
        print(f"\n{self.dry_run_stats()}")

        print(f"\n🧬 Generating Brain DNA: {self.project_name} ...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Генерируем simulation.toml (Законы Физики)
        max_w = max((z.vox_x for z in self.zones), default=40)
        max_d = max((z.vox_y for z in self.zones), default=40)
        max_h = max((z.vox_z for z in self.zones), default=63)
        vox_um = self.sim_params["voxel_size_um"]
        
        sim_config = {
            "world": {
                "width_um": int(max_w * vox_um),
                "depth_um": int(max_d * vox_um),
                "height_um": int(max_h * vox_um),
            },
            "simulation": self.sim_params
        }
        
        sim_path = self.output_dir / "simulation.toml"
        with open(sim_path, "w", encoding="utf-8") as f:
            toml.dump(sim_config, f)
            
        # 2. Генерируем brain.toml (Топология)
        brain_config = {
            "simulation": {"config": str(sim_path.absolute())},
            "zone": [],
            "connection": self.connections
        }
        
        
        # 3. Генерируем конфиги для каждой зоны
        for zone in self.zones:
            zone_dir = self.output_dir / zone.name
            zone_dir.mkdir(exist_ok=True)
            
            brain_config["zone"].append({
                "name": zone.name,
                "blueprints": str((zone_dir / "blueprints.toml").absolute()),
                "anatomy": str((zone_dir / "anatomy.toml").absolute()),
                "shard": str((zone_dir / "shard.toml").absolute()),
                "io": str((zone_dir / "io.toml").absolute()),
                "baked_dir": str((self.output_dir / "baked" / zone.name).absolute())
            })
            
            anatomy_data = {"layer": []}
            total_height = sum(l.height_pct for l in zone.layers)
            if abs(total_height - 1.0) > 1e-4:
                warnings.warn(f"[Builder] ⚠️ Zone '{zone.name}' layers height sum is {total_height:.2f}, not 1.0!")
            
            for layer in reversed(zone.layers):
                # Регистрация блюпринтов
                for bp_name in layer.composition.keys():
                    # Нам нужно найти блюпринт по имени в объектах (упрощено: предполагаем регистрацию через gnm_lib)
                    pass

                total_comp = sum(layer.composition.values())
                if abs(total_comp - 1.0) > 1e-4:
                    warnings.warn(f"[Builder] ⚠️ Layer '{layer.name}' composition sum is {total_comp:.2f}, not 1.0!")
                    
                anatomy_data["layer"].append({
                    "name": layer.name,
                    "height_pct": layer.height_pct,
                    "density": layer.density,
                    "composition": layer.composition
                })
                
            with open(zone_dir / "anatomy.toml", "w", encoding="utf-8") as f:
                toml.dump(anatomy_data, f)
                
            # Сборка blueprints.toml
            # Нужно гарантировать, что все блюпринты из слоев попали в registry
            # В данном коде предполагается, что пользователь сам их регистрирует или они попадают туда через слои.
            # Для надежности в коде пользователя они должны быть загружены через конструктор.
            blueprints_data = {"neuron_type": list(zone.blueprints_registry.values())}
            with open(zone_dir / "blueprints.toml", "w", encoding="utf-8") as f:
                toml.dump(blueprints_data, f)
                
            incoming_pixels = sum(c.get("width", 0) * c.get("height", 0) for c in self.connections if c["to"] == zone.name)
            ghost_capacity = int(incoming_pixels * 2.0)

            shard_data = {
                "zone_id": zone.name,
                "world_offset": {"x": 0, "y": 0, "z": 0},
                "dimensions": {"w": zone.vox_x, "d": zone.vox_y, "h": zone.vox_z},
                "neighbors": {},
                "settings": {
                    "save_checkpoints_interval_ticks": self.sim_params.get("save_checkpoints_interval_ticks", 1_000_000),
                    "night_interval_ticks": self.sim_params.get("night_interval_ticks", 0),
                    "prune_threshold": self.sim_params.get("prune_threshold", 10),
                    "ghost_capacity": ghost_capacity
                }
            }
            with open(zone_dir / "shard.toml", "w", encoding="utf-8") as f:
                toml.dump(shard_data, f)
                
            # Сборка io.toml
            io_data = {"input": zone.inputs, "output": zone.outputs}
            with open(zone_dir / "io.toml", "w", encoding="utf-8") as f:
                # Очищаем пустые списки, чтобы TOML был чистым
                clean_io = {k: v for k, v in io_data.items() if v}
                toml.dump(clean_io, f)
                
        with open(self.output_dir / "brain.toml", "w", encoding="utf-8") as f:
            toml.dump(brain_config, f)
            
        print(f"✅ DNA successfully created at '{self.output_dir}'")
