#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy==1.26.4",
# ]
# ///
"""
Axicor Brain State Debugger - parses binary .state blobs.

Usage:
  python3 brain_debugger.py <file.state>                - General stats (Voltage, Synapses)
  python3 brain_debugger.py <file.state> --id <N>       - Full dump of neuron N (Soma + Dendrites)
  python3 brain_debugger.py <f1.state> match <f2.state> - Compare two states (global)
  python3 brain_debugger.py <f1.state> match <f2.state> --id <N> - Compare neuron N between states
  python3 brain_debugger.py <f1.state> search_diff <f2.state> [--from <N>] - Find first diverging neuron
"""

import sys
import struct
import numpy as np
from pathlib import Path

# Strict version enforcement
# Relaxed version check for environment compatibility
if not (np.__version__.startswith("1.") or np.__version__.startswith("2.")):
    print(f"[FATAL] Unsupported numpy version: {np.__version__}")
    sys.exit(1)

MAX_DENDRITES = 128

def compute_padded_n(file_size):
    bytes_per_neuron = 4 + 1 + 4 + 1 + 4 + MAX_DENDRITES * (4 + 4 + 1)
    padded_n = file_size // bytes_per_neuron
    assert padded_n * bytes_per_neuron == file_size, f"File size {file_size} not aligned"
    return padded_n

def parse_state(path):
    data = np.fromfile(path, dtype=np.uint8)
    if len(data) > 4 and struct.unpack_from("<I", data, 0)[0] == 0x41584943:
        h = struct.unpack_from("<IBBHIIIIQIIIIIIIIIII13I", data, 0)
        n = h[4]
        off_w, off_tgt, off_f, off_v, off_th, off_tm = h[6], h[7], h[16], h[17], h[18], h[19]
        voltage = np.frombuffer(data[off_v:off_v + n*4], dtype=np.int32)
        flags   = np.frombuffer(data[off_f:off_f + n],   dtype=np.uint8)
        thresh  = np.frombuffer(data[off_th:off_th + n*4], dtype=np.int32)
        timers  = np.frombuffer(data[off_tm:off_tm + n],   dtype=np.uint8)
        s2a     = np.zeros(n, dtype=np.uint32)
        dend_tgt = np.frombuffer(data[off_tgt:off_tgt + n*MAX_DENDRITES*4], dtype=np.uint32).reshape(MAX_DENDRITES, n)
        dend_w   = np.frombuffer(data[off_w:off_w + n*MAX_DENDRITES*4], dtype=np.int32).reshape(MAX_DENDRITES, n)
        dend_t   = np.zeros((MAX_DENDRITES, n), dtype=np.uint8)
    else:
        n = compute_padded_n(len(data))
        off = 0
        voltage = np.frombuffer(data[off:off + n*4], dtype=np.int32); off += n*4
        flags   = np.frombuffer(data[off:off + n],   dtype=np.uint8); off += n
        thresh  = np.frombuffer(data[off:off + n*4], dtype=np.int32); off += n*4
        timers  = np.frombuffer(data[off:off + n],   dtype=np.uint8); off += n
        s2a     = np.frombuffer(data[off:off + n*4], dtype=np.uint32); off += n*4
        dend_tgt = np.frombuffer(data[off:off + n*MAX_DENDRITES*4], dtype=np.uint32).reshape(MAX_DENDRITES, n); off += n*MAX_DENDRITES*4
        dend_w   = np.frombuffer(data[off:off + n*MAX_DENDRITES*4], dtype=np.int32).reshape(MAX_DENDRITES, n); off += n*MAX_DENDRITES*4
        dend_t   = np.frombuffer(data[off:off + n*MAX_DENDRITES],   dtype=np.uint8).reshape(MAX_DENDRITES, n); off += n*MAX_DENDRITES
    return {'padded_n': n, 'voltage': voltage, 'flags': flags, 'threshold_offset': thresh, 'timers': timers, 'soma_to_axon': s2a, 'dendrite_targets': dend_tgt, 'dendrite_weights': dend_w, 'dendrite_timers': dend_t}

def report_stats(s, name):
    n = s['padded_n']
    v = s['voltage']
    print(f"\n ZONE: {name} | Padded N: {n}")
    print(f" Voltage Min: {v.min()}, Max: {v.max()}, Mean: {v.mean():.1f}")
    active = np.sum(s['dendrite_targets'] != 0)
    print(f" Total Synapses: {active}")

def dump_neuron(s, n_id):
    print(f"\n--- NEURON ID: {n_id} ---")
    print(f"  Voltage: {s['voltage'][n_id]} | Flags: 0x{s['flags'][n_id]:02x}")
    print(f"  Dendrites (Slot: Tgt, Weight):")
    for i in range(MAX_DENDRITES):
        if s['dendrite_targets'][i, n_id] != 0:
            print(f"    [{i:3}] Tgt: {s['dendrite_targets'][i, n_id]:10} | W: {s['dendrite_weights'][i, n_id]:10}")

def compare_neurons(s1, s2, n_id):
    def green(val): return f"\033[32m{val}\033[0m"
    print(f"\n{'--- STATIC (Left) ---':<45} | {'--- MEMORY (Right) ---':<45}")
    v1, v2 = s1['voltage'][n_id], s2['voltage'][n_id]
    f1, f2 = s1['flags'][n_id], s2['flags'][n_id]
    th1, th2 = s1['threshold_offset'][n_id], s2['threshold_offset'][n_id]
    tm1, tm2 = s1['timers'][n_id], s2['timers'][n_id]
    
    rv = green(v2) if v1 != v2 else str(v2)
    rf = green(f"0x{f2:02x}") if f1 != f2 else f"0x{f2:02x}"
    rth = green(th2) if th1 != th2 else str(th2)
    rtm = green(tm2) if tm1 != tm2 else str(tm2)

    print(f"  Voltage:    {v1:<35} | Voltage:    {rv}")
    print(f"  Flags:      0x{f1:02x} {' ':<28} | Flags:      {rf}")
    print(f"  Thresh Off: {th1:<35} | Thresh Off: {rth}")
    print(f"  Timer:      {tm1:<35} | Timer:      {rtm}")
    
    print(f"\n  Dendrites Comparison:")
    for i in range(MAX_DENDRITES):
        t1, t2 = s1['dendrite_targets'][i, n_id], s2['dendrite_targets'][i, n_id]
        w1, w2 = s1['dendrite_weights'][i, n_id], s2['dendrite_weights'][i, n_id]
        if t1 == 0 and t2 == 0: continue
        left = f"[{i:3}] Tgt:{t1:<10} W:{w1:<10}"
        rt = green(f"{t2:<10}") if t1 != t2 else f"{t2:<10}"
        rw = green(f"{w2:<10}") if w1 != w2 else f"{w2:<10}"
        print(f"  {left:<45} | [{i:3}] Tgt:{rt} W:{rw}")

def find_first_diff(s1, s2, start_id=0):
    n = s1['padded_n']
    print(f" Searching for first difference from ID {start_id}...")
    for i in range(start_id, n):
        # Quick check soma
        if s1['voltage'][i] != s2['voltage'][i] or \
           s1['flags'][i] != s2['flags'][i] or \
           s1['threshold_offset'][i] != s2['threshold_offset'][i] or \
           s1['timers'][i] != s2['timers'][i]:
            return i
        # Check dendrites
        if not np.array_equal(s1['dendrite_targets'][:, i], s2['dendrite_targets'][:, i]) or \
           not np.array_equal(s1['dendrite_weights'][:, i], s2['dendrite_weights'][:, i]):
            return i
        if i > 0 and i % 10000 == 0:
            print(f"  ...scanned {i} neurons")
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/brain_debugger.py <path1> [match <path2> | search_diff <path2> [--from <id>]] [--id <id>]")
        sys.exit(1)
        
    path1 = Path(sys.argv[1])
    s1 = parse_state(str(path1))
    
    if "search_diff" in sys.argv:
        idx = sys.argv.index("search_diff")
        path2 = Path(sys.argv[idx + 1])
        start_id = 0
        if "--from" in sys.argv:
            start_id = int(sys.argv[sys.argv.index("--from") + 1])
        
        s2 = parse_state(str(path2))
        diff_id = find_first_diff(s1, s2, start_id)
        if diff_id is not None:
            print(f" [OK] Found first difference at Neuron ID: {diff_id}")
            compare_neurons(s1, s2, diff_id)
        else:
            print(" [INFO] No differences found in all neurons.")
    elif "match" in sys.argv:
        idx = sys.argv.index("match")
        path2 = Path(sys.argv[idx + 1])
        n_id = 0
        if "--id" in sys.argv:
            n_id = int(sys.argv[sys.argv.index("--id") + 1])
        s2 = parse_state(str(path2))
        compare_neurons(s1, s2, n_id)
    elif "--id" in sys.argv:
        idx = sys.argv.index("--id")
        n_id = int(sys.argv[idx + 1])
        dump_neuron(s1, n_id)
    else:
        report_stats(s1, path1.parent.name)
    
    print("\n [OK] Debug complete\n")

if __name__ == "__main__":
    main()
