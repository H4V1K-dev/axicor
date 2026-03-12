import os
import glob
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
        
        # 1. Автоисправление (Clamping) под аппаратные лимиты PackedPosition
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
        
    def add_input(self, name: str, width: int, height: int, entry_z: str = "top", 
                  target_type: str = "All", growth_steps: int = 1500, empty_pixel: str = "skip", stride: int = 1):
        self.inputs.append({
            "name": name,
            "zone": self.name,
            "width": width,
            "height": height,
            "entry_z": entry_z,
            "target_type": target_type,
            "growth_steps": growth_steps,
            "empty_pixel": empty_pixel,
            "stride": stride
        })
        return self

    def add_output(self, name: str, width: int, height: int, target_type: str = "All", stride: int = 1):
        self.outputs.append({
            "name": name,
            "zone": self.name,
            "width": width,
            "height": height,
            "target_type": target_type,
            "stride": stride
        })
        return self
        
    def add_layer(self, name: str, height_pct: float, density: float) -> LayerDesigner:
        layer = LayerDesigner(self, name, height_pct, density)
        self.layers.append(layer)
        return layer
        
    def _register_blueprint(self, bp: NeuronBlueprint):
        # Регистрируем все типы из файла
        for n_type in bp.data_list:
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

    def build(self):
        """Собирает ДНК мозга и генерирует все артефакты."""
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
                
            shard_data = {
                "zone_id": zone.name,
                "world_offset": {"x": 0, "y": 0, "z": 0},
                "dimensions": {"w": zone.vox_x, "d": zone.vox_y, "h": zone.vox_z},
                "neighbors": {},
                "settings": {
                    "save_checkpoints_interval_ticks": 100000,
                    "night_interval_ticks": 10000,
                    "prune_threshold": 15
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
