import os
import struct
import mmap
import numpy as np
from axicor.memory import AxicorMemory
from axicor.platform import get_shm_path

def test_checkpointing():
    ZONE_HASH = 0xCAFEBABE
    PADDED_N = 10_000 # 10k is sufficient for testing
    
    # Size calculation
    WEIGHTS_SIZE = PADDED_N * 128 * 4
    TARGETS_SIZE = PADDED_N * 128 * 4
    FLAGS_SIZE = PADDED_N * 1
    
    # [DOD FIX] Strict C-ABI v3 Header requirements
    # 128 (Header) + Weights + Targets + Axons + Handovers + Prunes + Incoming + Flags + Voltage + Threshold + Timers
    # flags_offset is at 60, but let's re-calculate offsets starting from 128
    axons_off = 128 + WEIGHTS_SIZE + TARGETS_SIZE
    handovers_off = axons_off + (PADDED_N * 4)
    prunes_off = handovers_off + (10000 * 20)
    inc_prunes_off = prunes_off + (10000 * 8)
    flags_off = inc_prunes_off + (10000 * 4)
    voltage_off = flags_off + FLAGS_SIZE
    threshold_off = voltage_off + (PADDED_N * 4)
    timers_off = threshold_off + (PADDED_N * 4)
    
    SHM_SIZE = timers_off + ((PADDED_N + 63) & ~63)
    
    shm_path = get_shm_path(ZONE_HASH)
    
    # 1. Create fake VRAM dump
    with open(shm_path, "wb") as f:
        f.truncate(SHM_SIZE)
        
    with open(shm_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        # C-ABI Header v3 (128 bytes)
        struct.pack_into("<IBBHIIIIQIIIIIIIIIII13I", mm, 0,
                         0x41584943, 3, 0, 0,
                         PADDED_N, 128, 128, 128 + WEIGHTS_SIZE,
                         0, # epoch
                         PADDED_N, 
                         handovers_off, 0, ZONE_HASH, prunes_off, 0, 0, flags_off,
                         voltage_off, threshold_off, timers_off,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # 33 items
        mm.close()

    # 2. Initialize memory
    mem = AxicorMemory(ZONE_HASH)
    
    # 3. Write marker values
    print("Writing marker values to memory...")
    mem.weights[0, 0] = 777
    mem.weights[127, PADDED_N-1] = -999
    
    mem.targets[0, 0] = 555
    
    mem.flags[0] = 123
    mem.flags[PADDED_N-1] = 255
    
    # 4. Save checkpoint
    checkpoint_file = "test_brain.npz"
    print(f"Saving checkpoint to {checkpoint_file}...")
    mem.save_checkpoint(checkpoint_file)
    
    # 5. Reset memory
    print("Clearing memory (Zeroing weights, targets, flags)...")
    mem.clear_weights()
    mem.targets.fill(0)
    mem.flags.fill(0)
    
    assert mem.weights[0, 0] == 0
    assert mem.flags[0] == 0
    
    # 6. Load checkpoint
    print(f"Loading checkpoint from {checkpoint_file}...")
    mem.load_checkpoint(checkpoint_file)
    
    # 7. Verify data restoration
    print("Validating restored values...")
    assert mem.weights[0, 0] == 777, f"Weight restoration failed: {mem.weights[0,0]}"
    assert mem.weights[127, PADDED_N-1] == -999
    assert mem.targets[0, 0] == 555
    assert mem.flags[0] == 123
    assert mem.flags[PADDED_N-1] == 255
    
    print("[OK] Zero-Copy Checkpointing confirmed!")
    
    # Cleanup
    mem.close()
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if os.path.exists(shm_path):
        os.remove(shm_path)

if __name__ == "__main__":
    test_checkpointing()
