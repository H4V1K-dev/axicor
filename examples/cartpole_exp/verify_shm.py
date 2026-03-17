#!/usr/bin/env python3
"""
Верификация чтения SHM и соответствия baked state.
Запускать при работающем genesis-node (и baker daemon).

Проверяет:
- zone_hash, padded_n, offsets
- Количество активных синапсов в SHM vs shard.state
- Layout (column-major [128, padded_n])
- R-STDP: меняются ли веса между сэмплами
"""

import os
import sys
import time

# Add genesis-client to path (same as agent.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "genesis-client")))

from genesis.brain import fnv1a_32
from genesis.memory import GenesisMemory
from genesis.control import GenesisControl

# Reuse agent's load_baked_state_stats
from agent import load_baked_state_stats, default_manifest_path


def main() -> None:
    zone_hash = fnv1a_32(b"SensoryCortex")
    manifest_path = default_manifest_path()

    print("=" * 60)
    print("SHM Verification (genesis-node must be running)")
    print("=" * 60)
    print(f"zone_hash: 0x{zone_hash:08X} (SensoryCortex)")
    print(f"manifest:  {manifest_path}")
    print()

    # 1. Load baked state from disk
    try:
        baked = load_baked_state_stats(manifest_path)
        print("[Baked State] from shard.state")
        print(f"  padded_n:        {baked['padded_n']}")
        print(f"  active_synapses: {baked['active_synapses']}")
        print(f"  avg_weight:      {baked['avg_weight']:.1f}")
        print(f"  max_weight:      {baked['max_weight']}")
    except Exception as e:
        print(f"[Baked State] ERROR: {e}")
        baked = None

    # 2. Connect to SHM
    try:
        memory = GenesisMemory(zone_hash, read_only=True)
    except Exception as e:
        print(f"\n[SHM] ERROR: {e}")
        print("  Ensure genesis-node is running: cargo run --release -p genesis-node -- --brain CartPole-example")
        return 1

    print("\n[SHM] Header (from mmap)")
    print(f"  padded_n:        {memory.padded_n}")
    print(f"  dendrite_slots:  {memory.dendrite_slots}")
    print(f"  weights_offset:  {memory.weights_offset}")
    print(f"  targets_offset:  {memory.targets_offset}")
    print(f"  flags_offset:    {memory.flags_offset}")

    # 3. Compare
    if baked and memory.padded_n != baked["padded_n"]:
        print(f"\n⚠️  MISMATCH: SHM padded_n={memory.padded_n} vs baked padded_n={baked['padded_n']}")
    else:
        print(f"\n✓ padded_n match: {memory.padded_n}")

    # 4. Network stats from SHM
    stats = memory.get_network_stats()
    print("\n[SHM] get_network_stats()")
    print(f"  active_synapses: {stats['active_synapses']}")
    print(f"  avg_weight:      {stats['avg_weight']:.1f}")
    print(f"  max_weight:      {stats['max_weight']}")
    print(f"  spike_rate:      {stats['spike_rate']:.6f}")

    if baked and stats["active_synapses"] != baked["active_synapses"]:
        print(f"\n⚠️  SYNAPSE COUNT MISMATCH:")
        print(f"     SHM:   {stats['active_synapses']}")
        print(f"     Baked: {baked['active_synapses']}")
        print("     Possible causes:")
        print("     - Node loaded checkpoint from different build")
        print("     - SHM not yet updated (first Night Phase not run)")
        print("     - Layout/offset bug in agent or node")

    # 5. Layout sanity: sample first column
    targets = memory.targets
    weights = memory.weights
    valid = targets != 0
    n_valid = int(valid.sum())
    print(f"\n[Layout] targets.shape={targets.shape}, valid count={n_valid}")

    if n_valid > 0:
        # First few non-zero targets
        first_col = targets[:, 0]
        nz = (first_col != 0).sum()
        print(f"  Neuron 0: {nz} non-zero slots (of 128)")
        if nz > 0:
            idx = (first_col != 0).nonzero()[0]
            sample_t = first_col[idx[:5]]
            sample_w = weights[idx[:5], 0]
            print(f"  Sample targets: {sample_t.tolist()}")
            print(f"  Sample weights: {sample_w.tolist()}")

    # 6. R-STDP check: sample twice, see if weights change
    print("\n[R-STDP] Sampling weights twice (2s apart)...")
    w1 = memory.weights.copy()
    time.sleep(2.0)
    w2 = memory.weights.copy()
    diff = (w1 != w2).sum()
    print(f"  Changed elements: {diff} (of {w1.size})")
    if diff > 0:
        print("  ✓ Weights are changing (R-STDP active)")
    else:
        print("  ⚠ Weights static — R-STDP may not be updating, or no spikes")

    memory.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
