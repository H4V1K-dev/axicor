#!/usr/bin/env python3
"""
Genesis Ghost Axon 3D Visualizer
Два куба (Node A / Node B) с ghost-аксонами и линиями связей.
"""
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Безголовый рендер
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

GHOSTS_FILE = "/home/alex/Workflow/Genesis/baked/MotorCortex/SensoryCortex_MotorCortex.ghosts"
PADDED_N_A = 145152
PADDED_N_B = 145152

def load_ghosts(path):
    with open(path, 'rb') as f:
        data = f.read()
    count = struct.unpack_from('<I', data, 0)[0]
    src = np.frombuffer(data, dtype=np.uint32, count=count, offset=4)
    dst = np.frombuffer(data, dtype=np.uint32, count=count, offset=4 + count * 4)
    return src.copy(), dst.copy()

def axon_to_3d(axon_id, padded_n, cube_size=20.0):
    """Axon ID -> 3D позиция в кубе через раскладку по решётке."""
    side = int(np.ceil(padded_n ** (1/3)))
    idx = axon_id % padded_n  # clamp
    x = (idx % side) / side * cube_size
    y = ((idx // side) % side) / side * cube_size
    z = (idx // (side * side)) / side * cube_size
    return x, y, z

def main():
    print("Loading ghost connections...")
    src_axons, dst_ghosts = load_ghosts(GHOSTS_FILE)
    count = len(src_axons)
    print(f"  {count} ghost links")
    print(f"  src range: [{src_axons.min()}, {src_axons.max()}]")
    print(f"  dst range: [{dst_ghosts.min()}, {dst_ghosts.max()}]")
    
    CUBE = 20.0
    GAP = 35.0
    
    # Позиции src аксонов в кубе A (слева)
    src_pos = np.array([axon_to_3d(a, PADDED_N_A, CUBE) for a in src_axons])
    
    # Позиции dst ghosts в кубе B (справа, со смещением по X)
    # Используем относительный индекс внутри ghost-массива для 3D-раскладки
    dst_pos = np.array([axon_to_3d(g - dst_ghosts.min(), count, CUBE) for g in dst_ghosts])
    dst_pos[:, 0] += GAP  # Сдвигаем куб B вправо
    
    # Фоновые нейроны (подвыборка)
    N_BG = 1500
    step_a = max(1, PADDED_N_A // N_BG)
    step_b = max(1, PADDED_N_B // N_BG)
    bg_a = np.array([axon_to_3d(i, PADDED_N_A, CUBE) for i in range(0, PADDED_N_A, step_a)])
    bg_b = np.array([axon_to_3d(i, PADDED_N_B, CUBE) for i in range(0, PADDED_N_B, step_b)])
    bg_b[:, 0] += GAP
    
    # ─── Рендер ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#08080c')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#08080c')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Фон — все нейроны
    ax.scatter(bg_a[:, 0], bg_a[:, 1], bg_a[:, 2],
               c='#16162a', s=0.5, alpha=0.4)
    ax.scatter(bg_b[:, 0], bg_b[:, 1], bg_b[:, 2],
               c='#162a16', s=0.5, alpha=0.4)
    
    # Ghost-аксоны (яркие)
    ax.scatter(src_pos[:, 0], src_pos[:, 1], src_pos[:, 2],
               c='#ff6633', s=25, alpha=0.85, edgecolors='#ffaa77', linewidths=0.3,
               label=f'Src Axons ({count})', zorder=5)
    ax.scatter(dst_pos[:, 0], dst_pos[:, 1], dst_pos[:, 2],
               c='#00ccff', s=25, alpha=0.85, edgecolors='#77ddff', linewidths=0.3,
               label=f'Dst Ghosts ({count})', zorder=5)
    
    # Линии связей (с градиентом прозрачности)
    lines = []
    colors = []
    for i in range(count):
        lines.append([src_pos[i], dst_pos[i]])
        t = i / max(count - 1, 1)
        r = 1.0 - 0.6*t
        g = 0.3 + 0.4*t
        b = 0.2 + 0.8*t
        colors.append((r, g, b, 0.12))
    
    lc = Line3DCollection(lines, colors=colors, linewidths=0.4)
    ax.add_collection3d(lc)
    
    # Wireframe кубов
    def cube_edges(ox, oy, oz, s):
        c = [(ox,oy,oz),(ox+s,oy,oz),(ox+s,oy+s,oz),(ox,oy+s,oz),
             (ox,oy,oz+s),(ox+s,oy,oz+s),(ox+s,oy+s,oz+s),(ox,oy+s,oz+s)]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        return [(c[a], c[b]) for a, b in edges]
    
    for e in cube_edges(0, 0, 0, CUBE):
        ax.plot3D(*zip(*e), color='#ff6633', alpha=0.35, linewidth=0.8)
    for e in cube_edges(GAP, 0, 0, CUBE):
        ax.plot3D(*zip(*e), color='#00ccff', alpha=0.35, linewidth=0.8)
    
    # Подписи
    ax.text(CUBE/2, -3, CUBE+3, 'SensoryCortex\n(Node A)', color='#ff8855',
            fontsize=11, ha='center', fontweight='bold')
    ax.text(GAP+CUBE/2, -3, CUBE+3, 'MotorCortex\n(Node B)', color='#55ccff',
            fontsize=11, ha='center', fontweight='bold')
    
    ax.set_xlabel('X', color='#555', fontsize=7)
    ax.set_ylabel('Y', color='#555', fontsize=7)
    ax.set_zlabel('Z', color='#555', fontsize=7)
    ax.tick_params(colors='#333', labelsize=5)
    
    ax.set_title(f'Genesis Ghost Axon Network — {count} Inter-Node Connections\n'
                 f'src_axon:[{src_axons.min()}–{src_axons.max()}] → dst_ghost:[{dst_ghosts.min()}–{dst_ghosts.max()}]',
                 color='white', fontsize=13, fontweight='bold', pad=15)
    
    stats = (f"Pairs: {count}\n"
             f"Src axon IDs: {src_axons.min()}..{src_axons.max()}\n"
             f"Dst ghost IDs: {dst_ghosts.min()}..{dst_ghosts.max()}\n"
             f"Node A padded_n: {PADDED_N_A}\n"
             f"Node B padded_n: {PADDED_N_B}\n"
             f"Ghost offset: {dst_ghosts.min()}")
    ax.text2D(0.01, 0.96, stats, transform=ax.transAxes, color='#999',
              fontsize=7, fontfamily='monospace', va='top',
              bbox=dict(boxstyle='round,pad=0.4', fc='#111', alpha=0.85))
    
    ax.legend(loc='upper right', fontsize=8, framealpha=0.6,
              facecolor='#111', labelcolor='white')
    ax.view_init(elev=22, azim=-55)
    
    plt.tight_layout()
    out = "/home/alex/Workflow/Genesis/ghost_network_3d.png"
    plt.savefig(out, dpi=200, facecolor='#08080c', bbox_inches='tight')
    print(f"\n✅ Saved: {out}")

if __name__ == "__main__":
    main()
