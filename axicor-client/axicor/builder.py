import os
import glob
import math
import shutil
import warnings
import toml
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

class NeuronBlueprint:
    """Wrapper over a parsed GNM-Library neuron type TOML file."""
    def __init__(self, filepath: str, data: list):
        self.filepath = filepath
        self.data_list = data
        # Use the first neuron's name to identify the group
        self.name = data[0].get("name", "Unknown") if data else "Unknown"

    def set_plasticity(self, pot: int, dep: int):
        """HFT tuning: dynamic switching of R-STDP parameters."""
        for n_type in self.data_list:
            n_type["gsop_potentiation"] = pot
            n_type["gsop_depression"] = dep
        return self

class IoMatrixDesigner:
    def __init__(self, width: int, height: int, is_input: bool = True):
        self.width = width
        self.height = height
        self.is_input = is_input

        if self.is_input:
            # 64-bit alignment (8 bytes) for bitmasks
            self.padded_pixels = ((width * height + 63) // 64) * 64
            self.bytes_per_tick = self.padded_pixels // 8
        else:
            # 64-byte alignment (L2 Cache Line) for raw u8 arrays
            self.padded_pixels = ((width * height + 63) // 64) * 64
            self.bytes_per_tick = self.padded_pixels

    def fragment(self, sync_batch_ticks: int, mtu: int = 65507) -> list[dict]:
        # 1. Useful payload (20 bytes  ExternalIoHeader)
        max_payload = mtu - 20
        # 2. Bytes per tick (integer division)
        max_bytes_per_tick = max_payload // sync_batch_ticks
        
        # 3. Pixels per chunk (multiple of 64 bits / 8 bytes for masks or L2 cache for u8)
        if self.is_input:
            max_aligned_pixels = (max_bytes_per_tick // 8) * 64
        else:
            max_aligned_pixels = (max_bytes_per_tick // 64) * 64

        # 4. Capacity check
        if self.padded_pixels <= max_aligned_pixels:
            return [{"width": self.width, "height": self.height, "uv_rect": [0.0, 0.0, 1.0, 1.0]}]

        # 5. Row-based Slicing
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
            # Placeholder for "Canvas" mode (grid-based slicing)  to be implemented.
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
        
        # 1. Auto-correction (Clamping) to fit hardware PackedPosition limits (10/10/8 bit)
        self.vox_x = max(1, min(x, 1023))
        self.vox_y = max(1, min(y, 1023))
        self.vox_z = max(1, min(z, 255))
        
        if self.vox_x != x or self.vox_y != y or self.vox_z != z:
            warnings.warn(f"[Builder] [WARN] Zone '{self.name}' dimensions corrected from "
                          f"({x}, {y}, {z}) to ({self.vox_x}, {self.vox_y}, {self.vox_z}) "
                          f"to fit 10/10/8-bit hardware constraints.")
            
        self.layers: List[LayerDesigner] = []
        self.blueprints_registry: Dict[str, dict] = {}
        
        # New I/O registry
        self.inputs: List[Dict[str, Any]] = []
        self.outputs: List[Dict[str, Any]] = []
        
    def add_input(self, name: str, width: int, height: int, target_type: str = "All", entry_z: str = "top", stride: int = 1, growth_steps: int = 1000, layout: list[str] = None, uv_rect: list[float] = None):
        import uuid
        # 1. entry_z validation
        if entry_z not in ["top", "mid", "bottom"]:
            try:
                float(entry_z)
            except ValueError:
                raise ValueError(f"Invalid entry_z: '{entry_z}'. Must be 'top', 'mid', 'bottom' or a float string.")

        # 2. Fragmentation
        designer = IoMatrixDesigner(width, height, is_input=True)
        batch_ticks = self.builder.sim_params["sync_batch_ticks"]
        # [DOD FIX] Inputs are never fragmented (atomic bitmask requirement for GPU)
        # Force a single chunk by setting MTU to infinity
        chunks = [{"width": width, "height": height, "uv_rect": [0.0, 0.0, 1.0, 1.0]}]

        # 3. ID Generation Convention
        shard_suffix = self.name[-4:] if len(self.name) >= 4 else self.name
        matrix_uuid = uuid.uuid4().hex[:8]
        matrix_id = f"{shard_suffix}_{matrix_uuid}"
        
        matrix = {
            "matrix_id_v1": {"id": matrix_id},
            "name": name,
            "entry_z": entry_z,
            "pin": [] # Serialized as [[input.pin]] in TOML
        }

        matrix_suffix = matrix_id[-4:]
        for i, chunk in enumerate(chunks):
            pin_name = name if len(chunks) == 1 else f"{name}_chunk_{i}"
            pin_uuid = uuid.uuid4().hex[:4]
            
            uv = uv_rect if uv_rect else chunk["uv_rect"]
            
            matrix["pin"].append({
                "pin_id_v1": {"id": f"{matrix_suffix}_{pin_uuid}"},
                "name": pin_name,
                "target_type": target_type,
                "width": chunk["width"],
                "height": chunk["height"],
                "local_u": uv[0],
                "local_v": uv[1],
                "u_width": uv[2],
                "v_height": uv[3],
                "stride": stride,
                "growth_steps": growth_steps
            })
            
        self.inputs.append(matrix)
        return self

    def add_output(self, name: str, width: int, height: int, target_type: str = "All", stride: int = 1, layout: list[str] = None, uv_rect: list[float] = None):
        import uuid
        # 1. Fragmentation
        designer = IoMatrixDesigner(width, height, is_input=False)
        batch_ticks = self.builder.sim_params["sync_batch_ticks"]
        chunks = designer.fragment(sync_batch_ticks=batch_ticks)

        # 2. ID Generation Convention
        shard_suffix = self.name[-4:] if len(self.name) >= 4 else self.name
        matrix_uuid = uuid.uuid4().hex[:8]
        matrix_id = f"{shard_suffix}_{matrix_uuid}"
        
        matrix = {
            "matrix_id_v1": {"id": matrix_id},
            "name": name,
            "entry_z": "bottom", # Outputs are usually at the bottom
            "pin": []
        }

        matrix_suffix = matrix_id[-4:]
        for i, chunk in enumerate(chunks):
            pin_name = name if len(chunks) == 1 else f"{name}_chunk_{i}"
            pin_uuid = uuid.uuid4().hex[:4]
            
            uv = uv_rect if uv_rect else chunk["uv_rect"]
            
            matrix["pin"].append({
                "pin_id_v1": {"id": f"{matrix_suffix}_{pin_uuid}"},
                "name": pin_name,
                "target_type": target_type,
                "width": chunk["width"],
                "height": chunk["height"],
                "local_u": uv[0],
                "local_v": uv[1],
                "u_width": uv[2],
                "v_height": uv[3],
                "stride": stride
            })
            
        self.outputs.append(matrix)
        return self
        
    def add_layer(self, name: str, height_pct: float, density: float) -> LayerDesigner:
        layer = LayerDesigner(self, name, height_pct, density)
        self.layers.append(layer)
        return layer
        
    def _register_blueprint(self, bp: NeuronBlueprint):
        # Register all types from the file
        for n_type in bp.data_list:
            # [HFT FIX] Map period to DDS multiplier (heartbeat_m)
            if "spontaneous_firing_period_ticks" in n_type and n_type["spontaneous_firing_period_ticks"] > 0:
                period = n_type["spontaneous_firing_period_ticks"]
                # phase = (tick * m + salt) & 0xFFFF; heart = phase < m
                # Probability = m / 65536 = 1 / period  => m = 65536 / period
                n_type["heartbeat_m"] = int(65536 / period)

            n_name = n_type.get("name")
            if n_name not in self.blueprints_registry:
                # Protection against exceeding type limit (4-bit mask = max 16)
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
        
        # Default simulation parameters
        self.sim_params = {
            "tick_duration_us": 100,
            "total_ticks": 0,
            "master_seed": "AXICOR",
            "voxel_size_um": 25.0,
            "signal_speed_m_s": 0.5,
            "sync_batch_ticks": 100,
            "segment_length_voxels": 2,
            "axon_growth_max_steps": 250
        }

        # Library indexing for O(1) internal name lookup
        self._lib_index: Dict[str, str] = {}
        self._index_gnm_library()

    def _index_gnm_library(self):
        """Scans the library at startup and builds an index by internal names (O(1) search)."""
        search_pattern = f"{self.gnm_lib_path}/**/*.toml"
        for filepath in glob.glob(search_pattern, recursive=True):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = toml.load(f)
                # [DOD FIX] `[[neuron_type]]` is a list, not a dict.
                if "neuron_type" in data and isinstance(data["neuron_type"], list) and len(data["neuron_type"]) > 0:
                    # [DOD FIX] Take index 0 as [[neuron_type]] is an array of tables
                    name = data["neuron_type"][0].get("name")
                    if name:
                        self._lib_index[name] = filepath
            except Exception as e:
                # Explicit logging. Struct error suppression is prohibited.
                print(f"[Indexer Warning] Failed to parse {filepath}: {e}")

    def add_zone(self, name: str, width_vox: int, depth_vox: int, height_vox: int) -> ZoneDesigner:
        zone = ZoneDesigner(self, name, width_vox, depth_vox, height_vox)
        self.zones.append(zone)
        return zone

    def connect(self, from_zone: ZoneDesigner, to_zone: ZoneDesigner, out_matrix: str, 
                in_width: int, in_height: int, entry_z: str = "top", target_type: str = "All", growth_steps: int = 1000):
        # Verify if the output matrix exists in the source zone
        if not any(out["name"] == out_matrix for out in from_zone.outputs):
            warnings.warn(f"[Builder] [WARN] Output matrix '{out_matrix}' not found in zone '{from_zone.name}'!")

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
        Smart library search.
        First looks for an exact internal name (name) match within TOML.
        If not found, searches for a partial filename match.
        """
        # 1. Search by internal name (O(1))
        if query in self._lib_index:
            target_file = self._lib_index[query]
        else:
            # 2. Fallback: search by partial filename
            search_pattern = f"{self.gnm_lib_path}/**/*{query}*.toml"
            matches = glob.glob(search_pattern, recursive=True)

            if not matches:
                raise FileNotFoundError(f"[WARN] Blueprint matching '{query}' not found in {self.gnm_lib_path}")
            
            # [DOD FIX] Extract first element from glob array
            target_file = matches[0]

        with open(target_file, "r", encoding="utf-8") as f:
            data = toml.load(f)

        if "neuron_type" not in data or not data["neuron_type"]:
            raise ValueError(f"Invalid blueprint format in {target_file}")

        return NeuronBlueprint(target_file, data["neuron_type"])

    def dry_run_stats(self) -> str:
        """
        [DOD] Strict C-ABI memory cost estimation.
        O(1) calculation of VRAM and /dev/shm consumption prior to TOML generation.
        """
        report = [f" Axicor Memory Estimator: {self.project_name}"]
        total_vram = 0
        total_shm = 0

        for zone in self.zones:
            raw_neurons = 0
            cursor_pct = 0.0

            # Reflects logic in axicor-baker/src/bake/neuron_placement.rs
            for layer in zone.layers:
                z_start = int(cursor_pct * zone.vox_z)
                z_end = min(255, int((cursor_pct + layer.height_pct) * zone.vox_z))
                cursor_pct += layer.height_pct

                layer_vol = zone.vox_x * zone.vox_y * (z_end - z_start)
                layer_budget = int(math.floor(layer_vol * layer.density))
                raw_neurons += layer_budget

            # Warp Alignment (64 threads / AMD Wavefront + L2 Cache Line)
            padded_n = math.ceil(raw_neurons / 64) * 64

            virtual_axons = sum(pin["width"] * pin["height"] for matrix in zone.inputs for pin in matrix["pin"])
            incoming_pixels = sum(c.get("width", 0) * c.get("height", 0) for c in self.connections if c["to"] == zone.name)
            ghost_capacity = int(incoming_pixels * 2.0)

            raw_axons = padded_n + virtual_axons + ghost_capacity
            total_axons = math.ceil(raw_axons / 64) * 64

            # [DOD FIX] The 1166-Byte Invariant (i32 weights)
            vram_bytes = (padded_n * 1166) + (total_axons * 32)
            
            # SHM Night Phase IPC v4 (Header:64 + Weights + Targets + Flags + Handovers + Prunes)
            # 64 + (N*512) + (N*512) + (N*1) + (10000*20) + (10000*8)
            shm_bytes = 64 + (padded_n * 1025) + 280_000

            total_vram += vram_bytes
            total_shm += shm_bytes

            report.append(f"   Zone '{zone.name}':")
            report.append(f"      Neurons: ~{raw_neurons} (Padded: {padded_n})")
            report.append(f"      Axons: {total_axons} (Local: {padded_n}, Virtual: {virtual_axons}, GhostCap: {ghost_capacity})")
            report.append(f"      VRAM: {vram_bytes / (1024**2):.2f} MB | SHM: {shm_bytes / (1024**2):.2f} MB")

        report.append(f"   TOTAL VRAM BUDGET: {total_vram / (1024**2):.2f} MB")
        report.append(f"   TOTAL SHM BUDGET:  {total_shm / (1024**2):.2f} MB")
        return "\n".join(report)

    def build(self):
        """Assembles Brain DNA and generates all artifacts."""
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
            error_msg = (f"\n[ERROR] [Builder] Physical Validation Failed: v_seg must be an integer.\n"
                         f"Current v_seg: {v_seg_raw:.4f}\n"
                         f"To fix this, you can change signal_speed_m_s to {suggested_speed:.4f}")
            
            if sys.stdout.isatty():
                print(error_msg)
                val = input(f"Apply auto-fix (speed={suggested_speed:.4f})? [Y/n]: ").strip().lower()
                if val in ("", "y", "yes"):
                    self.sim_params["signal_speed_m_s"] = suggested_speed
                    print(f"[OK] Auto-fix applied: signal_speed_m_s = {suggested_speed:.4f}")
                else:
                    raise ValueError("Manual fix required for v_seg integrality.")
            else:
                raise ValueError(error_msg)

        # [DOD FIX] Output estimated graph cost prior to generation
        print(f"\n{self.dry_run_stats()}")

        print(f"\n Generating Brain DNA: {self.project_name} ...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Generate simulation.toml (Physics Laws)
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
            
        # 2. Generate brain.toml (Topology)
        brain_config = {
            "simulation": {"config": "simulation.toml"},
            "zone": [],
            "connection": self.connections
        }
        
        
        # 3. Generate configs for each zone
        for zone in self.zones:
            zone_dir = self.output_dir / zone.name
            zone_dir.mkdir(exist_ok=True)
            
            # DOD FIX: Strict relative paths for .axic archive portability!
            brain_config["zone"].append({
                "name": zone.name,
                "blueprints": f"{zone.name}/blueprints.toml",
                "anatomy": f"{zone.name}/anatomy.toml",
                "shard": f"{zone.name}/shard.toml",
                "io": f"{zone.name}/io.toml",
                "baked_dir": f"baked/{zone.name}/"
            })
            
            anatomy_data = {"layer": []}
            total_height = sum(l.height_pct for l in zone.layers)
            if abs(total_height - 1.0) > 1e-4:
                warnings.warn(f"[Builder] [WARN] Zone '{zone.name}' layers height sum is {total_height:.2f}, not 1.0!")
            
            for layer in reversed(zone.layers):
                # Blueprint registration
                for bp_name in layer.composition.keys():
                    # Need to find blueprint by name in objects (simplified: assume gnm_lib registration)
                    pass

                total_comp = sum(layer.composition.values())
                if abs(total_comp - 1.0) > 1e-4:
                    warnings.warn(f"[Builder] [WARN] Layer '{layer.name}' composition sum is {total_comp:.2f}, not 1.0!")
                    
                anatomy_data["layer"].append({
                    "name": layer.name,
                    "height_pct": layer.height_pct,
                    "density": layer.density,
                    "composition": layer.composition
                })
                
            with open(zone_dir / "anatomy.toml", "w", encoding="utf-8") as f:
                toml.dump(anatomy_data, f)
                
            # blueprints.toml assembly
            # Ensure all blueprints from layers are in the registry
            # Here it is assumed they are registered manually or via layers.
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
                
            # io.toml assembly
            io_data = {"input": zone.inputs, "output": zone.outputs}
            with open(zone_dir / "io.toml", "w", encoding="utf-8") as f:
                # Clean up empty lists
                clean_io = {k: v for k, v in io_data.items() if v}
                toml.dump(clean_io, f)
                
        with open(self.output_dir / "brain.toml", "w", encoding="utf-8") as f:
            toml.dump(brain_config, f)

        print(f"[OK] DNA successfully created at '{self.output_dir}'")
        return self  # [DOD FIX] Method chaining support

    def bake(self, clean: bool = False):
        """
        Invokes the axicor-baker Rust compiler to generate binary VRAM dumps.
        """
        print("\n Starting Axicor Baker (CPU Compiler)...")
        brain_toml_path = self.output_dir / "brain.toml"

        cmd = ["cargo", "run", "--release", "-p", "axicor-baker"]

        has_cuda = shutil.which("nvcc") is not None
        has_rocm = shutil.which("hipcc") is not None

        if has_rocm:
            cmd.extend(["--features", "amd"])
        elif not has_cuda:
            cmd.extend(["--features", "mock-gpu"])
        cmd.extend([
            "--bin", "axicor-baker", "--",
            "--brain", str(brain_toml_path.absolute()),
            "--yes"
        ])
        
        if clean:
            cmd.append("--clean")
            
        # Run process. Assume script is called from workspace root.
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("\n[OK] Model successfully baked and ready for GPU loading.")
        else:
            print("\n[ERROR] Connectome compilation failed. Check Rust compiler logs.")
            sys.exit(1)
