#!/usr/bin/env python3
"""
Genesis Brain State Debugger — парсит бинарные .state блобы и выводит
полную статистику сомы + дендритных весов.

.state binary layout (Zero-Copy SoA, Little-Endian):
  soma_voltage      [padded_n]       × i32  (4B)
  soma_flags        [padded_n]       × u8   (1B)
  threshold_offset  [padded_n]       × i32  (4B)
  timers            [padded_n]       × u8   (1B)
  soma_to_axon      [padded_n]       × u32  (4B)
  dendrite_targets  [padded_n × 128] × u32  (4B)
  dendrite_weights  [padded_n × 128] × i16  (2B)
  dendrite_timers   [padded_n × 128] × u8   (1B)

Использование:
  python3 scripts/brain_debugger.py {name_model}
"""

import sys
import numpy as np
from pathlib import Path

MAX_DENDRITES = 128

def compute_padded_n(file_size):
    """Обратная формула: file_size = padded_n * (4+1+4+1+4 + 128*(4+2+1))"""
    bytes_per_neuron = 4 + 1 + 4 + 1 + 4 + MAX_DENDRITES * (4 + 2 + 1)
    # = 14 + 128*7 = 14 + 896 = 910
    padded_n = file_size // bytes_per_neuron
    assert padded_n * bytes_per_neuron == file_size, \
        f"File size {file_size} not aligned to {bytes_per_neuron} bytes/neuron"
    assert padded_n % 32 == 0, f"padded_n={padded_n} not warp-aligned!"
    return padded_n


def parse_state(path):
    """Парсит .state в dict numpy массивов."""
    data = np.fromfile(path, dtype=np.uint8)
    n = compute_padded_n(len(data))
    
    off = 0
    voltage = np.frombuffer(data[off:off + n*4], dtype=np.int32); off += n*4
    flags   = np.frombuffer(data[off:off + n],   dtype=np.uint8); off += n
    thresh  = np.frombuffer(data[off:off + n*4], dtype=np.int32); off += n*4
    timers  = np.frombuffer(data[off:off + n],   dtype=np.uint8); off += n
    s2a     = np.frombuffer(data[off:off + n*4], dtype=np.uint32); off += n*4
    
    dend_tgt = np.frombuffer(data[off:off + n*MAX_DENDRITES*4], dtype=np.uint32).reshape(MAX_DENDRITES, n); off += n*MAX_DENDRITES*4
    dend_w   = np.frombuffer(data[off:off + n*MAX_DENDRITES*2], dtype=np.int16).reshape(MAX_DENDRITES, n); off += n*MAX_DENDRITES*2
    dend_t   = np.frombuffer(data[off:off + n*MAX_DENDRITES],   dtype=np.uint8).reshape(MAX_DENDRITES, n); off += n*MAX_DENDRITES
    
    assert off == len(data), f"Parse error: consumed {off}, total {len(data)}"
    
    return {
        'padded_n': n,
        'voltage': voltage,
        'flags': flags,
        'threshold_offset': thresh,
        'timers': timers,
        'soma_to_axon': s2a,
        'dendrite_targets': dend_tgt,
        'dendrite_weights': dend_w,
        'dendrite_timers': dend_t,
    }


def report_soma(s, name):
    """Выводит статистику сомы."""
    n = s['padded_n']
    v = s['voltage']
    flags = s['flags']
    thresh = s['threshold_offset']
    timers = s['timers']
    
    # Variant ID — верхние 4 бита flags
    variant_ids = (flags >> 4) & 0x0F
    is_spiking = flags & 0x01
    
    # Только «живые» нейроны (v != 0 или флаг != 0)
    alive_mask = (v != 0) | (flags != 0)
    alive = np.sum(alive_mask)
    
    print(f"\n{'='*70}")
    print(f"  🧠 ZONE: {name}")
    print(f"  Padded N: {n} | Alive: {alive} | Padding: {n - alive}")
    print(f"{'='*70}")
    
    print(f"\n  📊 SOMA VOLTAGE (i32)")
    print(f"    Min:    {v.min():>10}")
    print(f"    Max:    {v.max():>10}")
    print(f"    Mean:   {v[alive_mask].mean():>10.1f}")
    print(f"    Std:    {v[alive_mask].std():>10.1f}")
    
    # Гистограмма вольтажей
    hist_edges = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 42001]
    hist, _ = np.histogram(v[alive_mask], bins=hist_edges)
    print(f"    Voltage Distribution:")
    for i in range(len(hist)):
        bar = '█' * min(50, int(hist[i] / max(alive,1) * 50))
        print(f"      [{hist_edges[i]:>6}..{hist_edges[i+1]:>6}) {hist[i]:>5}  {bar}")
    
    print(f"\n  🔥 SPIKING & REFRACTORY")
    print(f"    Currently spiking: {np.sum(is_spiking)}")
    print(f"    In refractory:     {np.sum(timers > 0)}")
    
    print(f"\n  🎯 THRESHOLD OFFSET (Homeostasis)")
    print(f"    Min:    {thresh.min():>10}")
    print(f"    Max:    {thresh.max():>10}")
    print(f"    Mean:   {thresh[alive_mask].mean():>10.1f}")
    print(f"    Non-zero: {np.sum(thresh != 0)}")
    
    print(f"\n  🏷️  VARIANT DISTRIBUTION")
    for vid in sorted(set(variant_ids)):
        cnt = np.sum(variant_ids == vid)
        if cnt > 0:
            print(f"    Variant {vid}: {cnt:>6} neurons")


def report_dendrites(s, name):
    """Анализирует дендритные веса."""
    n = s['padded_n']
    w = s['dendrite_weights']   # shape: (128, padded_n)
    tgt = s['dendrite_targets'] # shape: (128, padded_n)
    
    # Считаем заполненные слоты (target != 0)
    connected = tgt != 0
    slots_per_neuron = connected.sum(axis=0)  # (padded_n,)
    
    alive_mask = slots_per_neuron > 0
    alive_neurons = np.sum(alive_mask)
    
    # Все активные веса
    active_weights = w[connected]
    
    print(f"\n  🔗 DENDRITE WEIGHTS (i16)")
    if len(active_weights) == 0:
        print(f"    ⚠ NO CONNECTED SYNAPSES")
        return
    
    print(f"    Total synapses:  {len(active_weights):>10}")
    print(f"    Connected neurons: {alive_neurons:>8}")
    print(f"    Avg slots/neuron:  {slots_per_neuron[alive_mask].mean():>8.1f}")
    print(f"    Max slots/neuron:  {slots_per_neuron.max():>8}")
    
    print(f"\n    Weight Statistics:")
    print(f"      Min:     {active_weights.min():>8}")
    print(f"      Max:     {active_weights.max():>8}")
    print(f"      Mean:    {active_weights.mean():>8.1f}")
    print(f"      Std:     {active_weights.std():>8.1f}")
    print(f"      Median:  {np.median(active_weights):>8.1f}")
    
    # Polarity
    excitatory = active_weights[active_weights > 0]
    inhibitory = active_weights[active_weights < 0]
    dead = active_weights[active_weights == 0]
    
    print(f"\n    Polarity:")
    print(f"      Excitatory (+): {len(excitatory):>8}  ({100*len(excitatory)/len(active_weights):.1f}%)")
    print(f"      Inhibitory (-): {len(inhibitory):>8}  ({100*len(inhibitory)/len(active_weights):.1f}%)")
    print(f"      Dead (0):       {len(dead):>8}  ({100*len(dead)/len(active_weights):.1f}%)")
    
    if len(excitatory) > 0:
        print(f"      Exc mean: {excitatory.mean():.1f}, max: {excitatory.max()}")
    if len(inhibitory) > 0:
        print(f"      Inh mean: {inhibitory.mean():.1f}, min: {inhibitory.min()}")
    
    # Распределение весов (гистограмма)
    abs_w = np.abs(active_weights.astype(np.int32))
    hist_edges = [0, 100, 500, 1000, 2000, 3500, 5000, 10000, 32768]
    hist, _ = np.histogram(abs_w, bins=hist_edges)
    print(f"\n    |Weight| Distribution:")
    for i in range(len(hist)):
        bar = '█' * min(50, int(hist[i] / max(len(active_weights),1) * 50))
        print(f"      [{hist_edges[i]:>6}..{hist_edges[i+1]:>6}) {hist[i]:>8}  {bar}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 brain_debugger.py <ModelName|shard.state> [shard.state ...]")
        print("Example 1: python3 brain_debugger.py HumanoidAgent")
        print("Example 2: python3 brain_debugger.py baked/*/shard.state")
        sys.exit(1)
    
    paths_to_check = []
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    models_dir = project_root / "Genesis-Models"
    
    potential_model_dir = models_dir / sys.argv[1]
    
    # If a single argument is passed and it matches a folder in Genesis-Models
    if len(sys.argv) == 2 and potential_model_dir.is_dir():
        print(f"🔍 Auto-discovering shards for model: {sys.argv[1]}")
        baked_dir = potential_model_dir / "baked"
        if not baked_dir.exists():
            print(f"❌ Error: 'baked' directory not found in {potential_model_dir}")
            sys.exit(1)
            
        for shard_file in baked_dir.rglob("shard.state"):
            paths_to_check.append(shard_file)
            
        if not paths_to_check:
            print(f"❌ Error: No shard.state files found in {baked_dir}")
            sys.exit(1)
            
        paths_to_check.sort()
    else:
        paths_to_check = [Path(p) for p in sys.argv[1:]]
        
    for path in paths_to_check:
        if not path.exists():
            print(f"❌ File not found: {path}")
            continue
        
        zone_name = path.parent.name
        print(f"🔍 Analyzing: {path}")
        s = parse_state(str(path))
        report_soma(s, zone_name)
        report_dendrites(s, zone_name)
    
    print(f"\n{'='*70}")
    print(f"  ✅ Debug complete")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
