import sys
import os

# --- Backend Selection ---
# Force interactive backend if --show is requested
if "--show" in sys.argv:
    import matplotlib
    # Try common interactive backends
    found_backend = False
    for backend in ["QtAgg", "TkAgg", "GTK3Agg"]:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            found_backend = True
            break
        except:
            continue
    
    if not found_backend:
        print("\n⚠️  WARNING: Could not find an interactive backend (PyQt/Tkinter).")
        print("Falling back to image generation mode.\n")
        matplotlib.use("Agg")

import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import toml

VOXEL_SIZE_UM = 25.0
VIS_GAP_UM = 500.0

class Shard:
    def __init__(self, name, base_path):
        self.name = name
        self.path = base_path
        self.dna_path = f"{base_path}/BrainDNA"
        
        # Load config
        with open(f"{self.dna_path}/shard.toml", "r") as f:
            cfg = toml.load(f)
            self.world_offset = cfg["world_offset"]
            self.dimensions = cfg["dimensions"]

        # Load manifest for memory layout (padded_n)
        with open(f"{base_path}/manifest.toml", "r") as f:
            self.manifest = toml.load(f)
            self.padded_n = self.manifest["memory"]["padded_n"]
            
        # Raw data
        self.pos_data = np.fromfile(f"{base_path}/shard.pos", dtype=np.uint32)
        
        # Load state for soma_to_axon mapping
        state_raw = np.fromfile(f"{base_path}/shard.state", dtype=np.uint8)
        s2a_off = self.padded_n * 10
        self.soma_to_axon = np.frombuffer(state_raw, dtype=np.uint32, count=self.padded_n, offset=s2a_off)

        # Load paths
        try:
            paths_raw = np.fromfile(f"{base_path}/shard.paths", dtype=np.uint8)
            magic, version, total_axons, max_segments = struct.unpack_from('<IIII', paths_raw, 0)
            self.axon_lengths = paths_raw[16 : 16 + total_axons]
            padding = (64 - ((16 + total_axons) % 64)) % 64
            matrix_offset = 16 + total_axons + padding
            self.paths_matrix = np.frombuffer(
                paths_raw, dtype=np.uint32, count=total_axons * 256, offset=matrix_offset
            ).reshape(total_axons, 256)
        except:
            self.paths_matrix = None
            self.axon_lengths = None

    def get_global_microns(self, packed):
        x = (packed & 0x7FF) * VOXEL_SIZE_UM
        y = ((packed >> 11) & 0x7FF) * VOXEL_SIZE_UM
        z = ((packed >> 22) & 0x3F) * VOXEL_SIZE_UM
        
        # PackedPosition coordinates are already global
        gx = x + (self.world_offset['x'] / 240.0) * VIS_GAP_UM
        return gx, y, z

def load_ghosts(src_name, dst_name):
    path = f"baked/{dst_name}/atlas/{src_name}.ghosts"
    if not os.path.exists(path): return []
    
    with open(path, "rb") as f:
        header = f.read(16)
        magic, version, count, _ = struct.unpack("<4sIII", header)
        if magic not in [b"GHST", b"TSHG"]:
            return []
            
        conns = []
        for _ in range(count):
            data = f.read(8)
            src_soma, dst_ghost = struct.unpack("<II", data)
            conns.append((src_soma, dst_ghost))
            
    return conns

def draw_cube(ax, shard, color):
    w = shard.dimensions['w'] * VOXEL_SIZE_UM
    d = shard.dimensions['d'] * VOXEL_SIZE_UM
    h = shard.dimensions['h'] * VOXEL_SIZE_UM
    
    x0, y0, z0 = shard.get_global_microns(0)
    
    v = np.array([
        [x0, y0, z0], [x0+w, y0, z0], [x0+w, y0+d, z0], [x0, y0+d, z0],
        [x0, y0, z0+h], [x0+w, y0, z0+h], [x0+w, y0+d, z0+h], [x0, y0+d, z0+h]
    ])
    
    edges = [
        [v[0],v[1],v[2],v[3],v[0]],
        [v[4],v[5],v[6],v[7],v[4]],
        [v[0],v[4]], [v[1],v[5]], [v[2],v[6]], [v[3],v[7]]
    ]
    for edge in edges:
        pts = np.array(edge)
        ax.plot3D(pts[:,0], pts[:,1], pts[:,2], color=color, alpha=0.3)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render 3D Ghost Axon Network")
    parser.add_argument("--show", action="store_true", help="Open interactive window")
    parser.add_argument("--save", action="store_true", help="Save to PNG (default if --show not specified)")
    args = parser.parse_args()

    do_save = args.save or not args.show
    do_show = args.show

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 8), dpi=200 if do_save else 100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050505')

    shards_to_load = ["SensoryCortex", "HiddenCortex", "MotorCortex"]
    loaded_shards = {}
    colors = ["#ff5500", "#00ff55", "#0055ff"]

    for name, color in zip(shards_to_load, colors):
        print(f"Loading {name}...")
        s = Shard(name, f"baked/{name}")
        loaded_shards[name] = s
        draw_cube(ax, s, color)
        
        soma_points = []
        for i in range(0, len(s.pos_data), 500):
            if s.pos_data[i] != 0:
                soma_points.append(s.get_global_microns(s.pos_data[i]))
        
        if soma_points:
            pts = np.array(soma_points)
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], color=color, s=0.5, alpha=0.1)

    motor_shard = loaded_shards["MotorCortex"]
    print("Drawing local axons for MotorCortex...")
    for soma_id in range(0, motor_shard.padded_n, 400):
        axon_id = motor_shard.soma_to_axon[soma_id]
        if axon_id == 0xFFFFFFFF: continue
        
        l_len = motor_shard.axon_lengths[axon_id]
        l_path = motor_shard.paths_matrix[axon_id, :l_len]
        l_points = [motor_shard.get_global_microns(p) for p in l_path if p != 0]
        
        if len(l_points) > 1:
            pts = np.array(l_points)
            ax.plot3D(pts[:,0], pts[:,1], pts[:,2], color="#00aaff", alpha=0.2, linewidth=0.5)

    links = [
        ("SensoryCortex", "HiddenCortex", "#ffaa00", "Sensory → Hidden"),
        ("HiddenCortex", "MotorCortex", "#00ffaa", "Hidden → Motor")
    ]

    for src_name, dst_name, color, label in links:
        conns = load_ghosts(src_name, dst_name)
        print(f"Drawing connections for {src_name} -> {dst_name} (found {len(conns)})")
        
        src_shard = loaded_shards[src_name]
        dst_shard = loaded_shards[dst_name]
        
        virtual_axons = dst_shard.manifest["memory"].get("virtual_axons", 0)
        ghost_start_idx = dst_shard.padded_n + virtual_axons

        for src_soma_id, target_ghost_id in conns[:5000]:
            try:
                if src_soma_id >= len(src_shard.soma_to_axon): continue
                src_axon_id = src_shard.soma_to_axon[src_soma_id]
                if src_axon_id == 0xFFFFFFFF: continue
                if src_shard.paths_matrix is None: continue
                
                s_len = src_shard.axon_lengths[src_axon_id]
                s_path = src_shard.paths_matrix[src_axon_id, :s_len]
                s_points = [src_shard.get_global_microns(p) for p in s_path if p != 0]

                ghost_axon_id = ghost_start_idx + target_ghost_id
                if dst_shard.paths_matrix is None: continue
                if ghost_axon_id >= dst_shard.paths_matrix.shape[0]: continue

                g_len = dst_shard.axon_lengths[ghost_axon_id]
                g_path = dst_shard.paths_matrix[ghost_axon_id, :g_len]
                g_points = [dst_shard.get_global_microns(p) for p in g_path if p != 0]

                if len(s_points) > 1:
                    pts = np.array(s_points)
                    ax.plot3D(pts[:,0], pts[:,1], pts[:,2], color=color, alpha=0.3, linewidth=0.5)
                
                if len(g_points) > 1:
                    pts = np.array(g_points)
                    ax.plot3D(pts[:,0], pts[:,1], pts[:,2], color=color, alpha=0.5, linewidth=1.0)

                if s_points and g_points:
                    bridge = np.array([s_points[-1], g_points[0]])
                    ax.plot3D(bridge[:,0], bridge[:,1], bridge[:,2], color="#ffffff", alpha=0.6, linewidth=0.5)

            except Exception as e:
                continue

    total_w_vox = 720
    max_d_vox = 240
    max_h_vox = 64
    
    total_w_um = total_w_vox * VOXEL_SIZE_UM + 2 * VIS_GAP_UM
    max_d_um = max_d_vox * VOXEL_SIZE_UM
    max_h_um = max_h_vox * VOXEL_SIZE_UM
    
    ax.set_xlim(0, total_w_um)
    ax.set_ylim(0, max_d_um)
    ax.set_zlim(0, max_d_um)
    ax.set_box_aspect((total_w_um, max_d_um, max_h_um * 2)) 

    ax.grid(False)
    ax.set_axis_off()
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#ff8800', lw=2, label='Sensory Cortex'),
        Line2D([0], [0], color='#00ff88', lw=2, label='Hidden Cortex'),
        Line2D([0], [0], color='#00aaff', lw=2, label='Motor Cortex'),
        Line2D([0], [0], color='#ffffff', lw=1, ls='--', label='Inter-shard Link'),
        Line2D([0], [0], color='gray', lw=1, marker='o', ls='', label='Neuron (Soma)')
    ]
    leg = ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
                    facecolor='#111111', edgecolor='#333333', labelcolor='white', fontsize=10)
    leg.get_frame().set_alpha(0.8)

    info_text = (
        f"GENESIS SIMULATION ATLAS\n"
        f"------------------------\n"
        f"Total Neurons:   {sum(s.padded_n for s in loaded_shards.values()):,}\n"
        f"Voxel Size:      {VOXEL_SIZE_UM} um\n"
        f"Chain Length:    {total_w_um/1000:.1f} mm\n"
        f"Packed Format:   11-11-6-4"
    )
    ax.text2D(0.98, 0.05, info_text, transform=ax.transAxes, color='white', 
              fontsize=9, family='monospace', ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#111111', alpha=0.7, edgecolor='#444444'))

    plt.title("Genesis: Neural Pipeline Chain (Ghost Network)", color="white", pad=30, fontsize=18, fontweight='bold')
    plt.suptitle("3D Structural Visualization • Global Coordinates", color="#aaaaaa", y=0.88, fontsize=12)
    
    if do_show:
        print("💡 Opening interactive window. Use mouse to rotate/zoom.")
        plt.show()
    
    if do_save:
        plt.savefig("ghost_network_3d_fixed.png", dpi=200, bbox_inches='tight', facecolor='#050505')
        print("✅ Visualization saved with metadata!")

if __name__ == "__main__":
    main()
