import sys
import os

# --- Backend Selection ---
import matplotlib
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# Force interactive backend unless --save is requested
if "--save" not in sys.argv:
    # Try common interactive backends
    found_backend = False
    for backend in ["QtAgg", "TkAgg", "GTK3Agg"]:
        try:
            plt.switch_backend(backend)
            found_backend = True
            break
        except Exception:
            continue
    
    if not found_backend:
        print("\n⚠️  WARNING: Could not find an interactive backend (PyQt/Tkinter).")
        print("Falling back to image generation mode (enforcing --save).\n")
        plt.switch_backend("Agg")
        sys.argv.append("--save")

import struct
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

VOXEL_SIZE_UM = 25.0
MAX_DENDRITES = 128

def unpack_position(packed):
    """Zero-cost распаковка 32-битного PackedPosition [Type(4) | Z(6) | Y(11) | X(11)]"""
    x = (packed & 0x7FF) * VOXEL_SIZE_UM
    y = ((packed >> 11) & 0x7FF) * VOXEL_SIZE_UM
    z = ((packed >> 22) & 0x3F) * VOXEL_SIZE_UM
    return x, y, z

def load_neuron_topology(baked_dir, target_soma_id):
    # 1. Читаем позицию сомы (shard.pos)
    pos_data = np.fromfile(f"{baked_dir}/shard.pos", dtype=np.uint32)
    soma_packed = pos_data[target_soma_id]
    if soma_packed == 0:
        print("FATAL: Выбран мертвый нейрон (Warp Padding). Выбери другой ID.")
        sys.exit(1)
    soma_xyz = unpack_position(soma_packed)

    # 2. Читаем матрицу путей аксонов (shard.paths)
    paths_raw = np.fromfile(f"{baked_dir}/shard.paths", dtype=np.uint8)
    magic, version, total_axons, max_segments = struct.unpack_from('<IIII', paths_raw, 0)
    
    lengths = paths_raw[16 : 16 + total_axons]
    padding = (64 - ((16 + total_axons) % 64)) % 64
    matrix_offset = 16 + total_axons + padding
    
    # Плоская матрица [total_axons, 256]
    paths_matrix = np.frombuffer(
        paths_raw, dtype=np.uint32, count=total_axons * 256, offset=matrix_offset
    ).reshape(total_axons, 256)

    # 3. Читаем состояние дендритов и личный аксон (shard.state)
    state_data = np.fromfile(f"{baked_dir}/shard.state", dtype=np.uint8)
    padded_n = len(state_data) // 910  # 910 байт на нейрон (C-ABI контракт)
    
    # Смещения (Offsets)
    s2a_off = padded_n * 10            # soma_to_axon
    tgt_off = padded_n * 14            # dendrite_targets
    w_off   = padded_n * 526           # dendrite_weights

    s2a = np.frombuffer(state_data, dtype=np.uint32, count=padded_n, offset=s2a_off)
    my_axon_id = s2a[target_soma_id]

    targets = np.frombuffer(state_data, dtype=np.uint32, count=MAX_DENDRITES * padded_n, offset=tgt_off).reshape(MAX_DENDRITES, padded_n)
    weights = np.frombuffer(state_data, dtype=np.int16, count=MAX_DENDRITES * padded_n, offset=w_off).reshape(MAX_DENDRITES, padded_n)

    # --- Сборка Геометрии ---
    
    # А) Ствол аксона целевого нейрона
    axon_lines = []
    if my_axon_id != 0xFFFFFFFF:
        ax_len = lengths[my_axon_id]
        path = paths_matrix[my_axon_id, :ax_len]
        ax_points = [unpack_position(p) for p in path if p != 0]
        
        # Первая линия - от сомы до начала аксона
        if ax_points:
            axon_lines.append([soma_xyz, ax_points[0]])
            for i in range(len(ax_points)-1):
                axon_lines.append([ax_points[i], ax_points[i+1]])

    # Б) Дендритные отростки (тянутся к чужим аксонам)
    dendrite_lines = []
    dendrite_colors = []
    
    for slot in range(MAX_DENDRITES):
        tgt = targets[slot, target_soma_id]
        if tgt == 0:
            continue  # Пустой слот (Early Exit)
            
        # Zero-Index Trap Reverse
        target_axon_id = (tgt & 0x00FFFFFF) - 1
        target_seg_idx = tgt >> 24
        w = weights[slot, target_soma_id]

        if target_axon_id < total_axons and target_seg_idx < lengths[target_axon_id]:
            synapse_packed = paths_matrix[target_axon_id, target_seg_idx]
            if synapse_packed != 0:
                synapse_xyz = unpack_position(synapse_packed)
                dendrite_lines.append([soma_xyz, synapse_xyz])
                
                # Dale's Law color coding: Зеленый = Excitatory (+), Красный = Inhibitory (-)
                color = (0.1, 0.9, 0.4, 0.7) if w > 0 else (1.0, 0.2, 0.2, 0.7)
                dendrite_colors.append(color)

    return soma_xyz, axon_lines, dendrite_lines, dendrite_colors

def render_arxiv_plot(soma_xyz, axon_lines, dendrite_lines, dendrite_colors, output_file, target_id, save=False):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 10), dpi=300 if save else 100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#050505')

    # Soma
    ax.scatter(*soma_xyz, color='#ffffff', s=100, edgecolor='#00ffff', linewidth=2, zorder=10, label="Soma")

    # Axon
    if axon_lines:
        ax_coll = Line3DCollection(axon_lines, colors='#ff8800', linewidths=2.5, alpha=0.9, zorder=5)
        ax.add_collection3d(ax_coll)
        ax.scatter(*axon_lines[-1][1], color='#ffaa00', s=30, marker='X', zorder=11, label="Axon Tip")

    # Dendrites
    if dendrite_lines:
        dend_coll = Line3DCollection(dendrite_lines, colors=dendrite_colors, linewidths=0.8, linestyle='--')
        ax.add_collection3d(dend_coll)
        
        syn_x = [line[1][0] for line in dendrite_lines]
        syn_y = [line[1][1] for line in dendrite_lines]
        syn_z = [line[1][2] for line in dendrite_lines]
        ax.scatter(syn_x, syn_y, syn_z, c=dendrite_colors, s=15, alpha=0.8, label="En Passant Synapses")

    all_points = [soma_xyz] + [p for line in axon_lines + dendrite_lines for p in line]
    all_points = np.array(all_points)
    
    if len(all_points) > 1:
        # Устанавливаем честный 1:1:1 физический масштаб
        max_range = np.array([
            all_points[:,0].max() - all_points[:,0].min(),
            all_points[:,1].max() - all_points[:,1].min(),
            all_points[:,2].max() - all_points[:,2].min()
        ]).max() / 2.0
        
        mid_x = (all_points[:,0].max() + all_points[:,0].min()) * 0.5
        mid_y = (all_points[:,1].max() + all_points[:,1].min()) * 0.5
        mid_z = (all_points[:,2].max() + all_points[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range - 50, mid_x + max_range + 50)
        ax.set_ylim(mid_y - max_range - 50, mid_y + max_range + 50)
        ax.set_zlim(mid_z - max_range - 50, mid_z + max_range + 50)

    ax.axis('off')
    plt.title(f"Neuron Morphology (ID: {target_id})\nGenesis HFT Engine", color='white', pad=20, fontsize=14)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#fff', markersize=10, label='Soma'),
        Line2D([0], [0], color='#ff8800', lw=2.5, label='Efferent Axon Path'),
        Line2D([0], [0], color='#1aff66', lw=1, linestyle='--', label='Excitatory Dendrite (+W)'),
        Line2D([0], [0], color='#ff3333', lw=1, linestyle='--', label='Inhibitory Dendrite (-W)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='#111', edgecolor='#333', fontsize=9)

    plt.tight_layout()
    
    if not save:
        print("💡 Opening interactive window. Use mouse to rotate/zoom.")
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches='tight', facecolor='#050505')
        print(f"✅ Успешно отрендерено в {output_file}")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Render 3D Spiking Neuron Morphology")
    parser.add_argument("zone_name", nargs="?", help="Zone name (e.g., MotorCortex)")
    parser.add_argument("-id", "--id", type=int, required=True, help="Target Soma ID")
    parser.add_argument("--save", action="store_true", help="Save to PNG instead of opening a window")
    args, unknown = parser.parse_known_args()
    
    zone_name = args.zone_name
    if not zone_name:
        for u in unknown:
            if u.startswith("--") and u not in ["--save", "-h", "--help"]:
                zone_name = u[2:]
                break
                
    if not zone_name:
        print("❌ ERROR: Zone name not provided. Use positional (MotorCortex) or flag (--MotorCortex).")
        sys.exit(1)
        
    def find_baked(start_path):
        for root, dirs, files in os.walk(start_path):
            if ".venv" in dirs: dirs.remove(".venv")
            if ".git" in dirs: dirs.remove(".git")
            if "baked" in dirs:
                target = os.path.join(root, "baked", zone_name)
                if os.path.exists(target) and os.path.isdir(target):
                    return target
        return None

    baked_dir = find_baked('.')
    if not baked_dir:
        baked_dir = find_baked(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
    if not baked_dir:
        print(f"❌ ERROR: Could not find 'baked/{zone_name}' directory.")
        sys.exit(1)
        
    print(f"📁 Found baked data in: {baked_dir}")
        
    target_soma_id = args.id
    output_file = f"neuron_{target_soma_id}_morphology.png"
    
    # Range check
    pos_data = np.fromfile(f"{baked_dir}/shard.pos", dtype=np.uint32)
    if target_soma_id >= len(pos_data):
        print(f"❌ ERROR: ID {target_soma_id} is out of bounds. Max ID for this shard is {len(pos_data)-1}.")
        sys.exit(1)
        
    soma_xyz, axon_lines, dendrite_lines, dendrite_colors = load_neuron_topology(baked_dir, target_soma_id)
    render_arxiv_plot(soma_xyz, axon_lines, dendrite_lines, dendrite_colors, output_file, target_soma_id, save=args.save)