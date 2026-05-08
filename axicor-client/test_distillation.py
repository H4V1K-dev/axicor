import os
import struct
import mmap
import time
import numpy as np
from axicor.memory import AxicorMemory
from axicor.platform import get_shm_path

def test_distillation():
    ZONE_HASH = 0xDEADBEEF
    PADDED_N = 100_000
    
    # VRAM shard emulation for 100k neurons
    # 64 (Header) + Weights (100k * 128 * 2) + Targets (100k * 128 * 4) + Handovers...
    WEIGHTS_SIZE = PADDED_N * 128 * 4
    TARGETS_SIZE = PADDED_N * 128 * 4
    # [DOD FIX] Strict C-ABI v3 Header requirements
    # 128 (Header) + Weights + Targets + Axons + Handovers + Prunes + Flags + Voltages + Thresholds + Timers
    # PADDED_N = 100,000
    WEIGHTS_SIZE = PADDED_N * 128 * 4
    TARGETS_SIZE = PADDED_N * 128 * 4
    FLAGS_SIZE = (PADDED_N + 63) & ~63
    VOLTAGE_SIZE = PADDED_N * 4
    THRESHOLD_SIZE = PADDED_N * 4
    TIMERS_SIZE = (PADDED_N + 63) & ~63

    SHM_SIZE = 128 + WEIGHTS_SIZE + TARGETS_SIZE + (10000 * 20) + (10000 * 8) + (10000 * 4) + FLAGS_SIZE + VOLTAGE_SIZE + THRESHOLD_SIZE + TIMERS_SIZE

    shm_path = get_shm_path(ZONE_HASH)

    # 1. Create fake VRAM dump
    with open(shm_path, "wb") as f:
        f.truncate(SHM_SIZE)

    with open(shm_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)

        weights_off = 128
        targets_off = 128 + WEIGHTS_SIZE
        axons_off = targets_off + TARGETS_SIZE
        handovers_off = axons_off + (10000 * 4)
        prunes_off = handovers_off + (10000 * 20)
        flags_off = prunes_off + (10000 * 8)
        voltage_off = flags_off + FLAGS_SIZE
        threshold_off = voltage_off + VOLTAGE_SIZE
        timers_off = threshold_off + THRESHOLD_SIZE

        # Write strict C-ABI Header v3 (128 bytes)
        # Format string: <IBBHIIIIQIIIIIIIIIII13I (33 items, 128 bytes)
        struct.pack_into("<IBBHIIIIQIIIIIIIIIII13I", mm, 0,
                         0x41584943, 3, 0, 0,
                         PADDED_N, 128, weights_off, targets_off,
                         0, # epoch
                         PADDED_N, # total_axons
                         handovers_off, 0, ZONE_HASH, prunes_off, 0, 0, flags_off,
                         voltage_off, threshold_off, timers_off,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # 33 items
        mm.close()

    # 2. Connect Data-Oriented SDK
    # Default is now read_only=False, which is what we need.
    mem = AxicorMemory(ZONE_HASH)
    
    # 3. Inject test data (create 100,000 strong and 100,000 weak connections)
    # Use strict C-ABI Packer (Zero-Index Trap Protection)
    mem.targets[0, :] = AxicorMemory.pack_targets(np.full(PADDED_N, 4), np.zeros(PADDED_N)) 
    mem.weights[0, :] = 100 << 16 # Strong connection
    
    mem.targets[1, :] = AxicorMemory.pack_targets(np.full(PADDED_N, 9), np.zeros(PADDED_N))
    mem.weights[1, :] = 10 << 16 # Weak connection (must be pruned)
    
    # 4. Start distillation
    print(f" Starting distillation of {PADDED_N} neurons (Threshold = 15)...")
    start = time.perf_counter()
    
    killed = mem.distill_graph(prune_threshold=15)
    
    duration_ms = (time.perf_counter() - start) * 1000
    
    print(f" Distillation time: {duration_ms:.3f} ms")
    print(f" Connections burned: {killed}")
    
    # Invariant checks
    assert killed == PADDED_N, "All connections in slot 2 should have died!"
    assert np.all(mem.targets[1, :] == 0), "Weak targets were not zeroed out!"
    
    # Verify by unpacking strong targets
    strong_axon_ids, strong_seg_offsets = AxicorMemory.unpack_targets(mem.targets[0, :])
    assert np.all(strong_axon_ids == 4), f"Strong axon_ids were corrupted! Received {strong_axon_ids[0]}"
    assert np.all(strong_seg_offsets == 0), "Strong segment offsets were corrupted!"
    
    print("[OK] Strong connections verified via Unpacker.")
    
    mem.close()
    os.remove(shm_path)
    print("[OK] Zero-Copy distillation completed flawlessly.\n")

if __name__ == '__main__':
    test_distillation()
