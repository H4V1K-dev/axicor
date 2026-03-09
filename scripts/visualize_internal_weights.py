#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import sys
from pathlib import Path

MAX_DENDRITES = 128

def compute_padded_n(file_size):
    bytes_per_neuron = 14 + MAX_DENDRITES * 7
    return file_size // bytes_per_neuron

def axon_to_3d(idx, padded_n, cube_size=100.0):
    side = int(np.ceil(padded_n ** (1/3)))
    x = (idx % side) / side * cube_size
    y = ((idx // side) % side) / side * cube_size
    z = (idx // (side * side)) / side * cube_size
    return x, y, z

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_internal_weights.py <shard.state>")
        return

    path = sys.argv[1]
    data = np.fromfile(path, dtype=np.uint8)
    n = compute_padded_n(len(data))
    print(f"File: {path}, Padded N: {n}")

    # Layout offsets
    off_tgt = n * 14
    off_w = off_tgt + n * MAX_DENDRITES * 4
    
    # Read targets and weights
    tgt = np.frombuffer(data[off_tgt : off_tgt + n*MAX_DENDRITES*4], dtype=np.uint32).reshape(MAX_DENDRITES, n)
    w = np.frombuffer(data[off_w : off_w + n*MAX_DENDRITES*2], dtype=np.int16).reshape(MAX_DENDRITES, n)

    # Sample synapses
    SAMPLES = 2000
    active_mask = (tgt != 0) & (w != 0)
    slots, neurons = np.where(active_mask)
    
    if len(neurons) == 0:
        print("No active synapses found!")
        return

    idx = np.random.choice(len(neurons), min(SAMPLES, len(neurons)), replace=False)
    sample_slots = slots[idx]
    sample_neurons = neurons[idx]

    lines = []
    colors = []
    
    CUBE = 100.0
    for i in range(len(sample_neurons)):
        nid = sample_neurons[i]
        sid = sample_slots[i]
        
        target_packed = tgt[sid, nid]
        # target_packed: [31..24] Segment_Offset | [23..0] Axon_ID + 1
        target_axon_id = (target_packed & 0x00FFFFFF) - 1
        target_seg = target_packed >> 24
        
        weight = w[sid, nid]
        
        # Source (Dense ID of neuron having the dendrite)
        p_src = axon_to_3d(nid, n, CUBE)
        # Target (Axon ID it connects to)
        p_tgt = axon_to_3d(target_axon_id, n, CUBE)
        
        lines.append([p_src, p_tgt])
        
        # Color: Green for Exc (+), Red for Inh (-)
        if weight > 0:
            colors.append((0.0, 1.0, 0.5, 0.6)) # Green-ish
        else:
            colors.append((1.0, 0.2, 0.2, 0.4)) # Red

    # Render
    fig = plt.figure(figsize=(12, 12))
    fig.patch.set_facecolor('#0a0a12')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0a0a12')
    
    lc = Line3DCollection(lines, colors=colors, linewidths=0.5)
    ax.add_collection3d(lc)
    
    ax.set_xlim(0, CUBE); ax.set_ylim(0, CUBE); ax.set_zlim(0, CUBE)
    ax.axis('off')
    
    plt.title(f"Genesis Weight Landscape — {path.split('/')[-2]}\nGreen: Excitatory | Red: Inhibitory (Sample: {len(lines)} synapses)", 
              color='white', fontsize=14)
    
    out_name = f"weights_{path.split('/')[-2]}.png"
    plt.savefig(out_name, dpi=150, facecolor='#0a0a12')
    print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
