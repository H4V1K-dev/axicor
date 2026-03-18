#!/usr/bin/env python3
import os
import sys
import numpy as np
import struct

# [DOD] Genesis-Lite: Distillation Pipeline for ESP32-S3
# Desktop .state format (910 bytes per neuron):
# - voltage: i32 (4)
# - flags: u8 (1)
# - threshold_offset: i32 (4)
# - refractory_timer: u8 (1)
# - soma_to_axon: u32 (4)
# - targets: 128 * u32 (512)
# - weights: 128 * i16 (256)
# - timers: 128 * u8 (128)
# Total = 4+1+4+1+4 + 512+256+128 = 14 + 896 = 910 bytes.

def distill(state_path, axons_path):
    print(f"[*] Analyzing shard: {state_path}")
    
    file_size = os.path.getsize(state_path)
    padded_n = file_size // 910
    if file_size % 910 != 0:
        print(f"(!) WARNING: File size {file_size} is not a multiple of 910. padded_n={padded_n}")

    # 1. Zero-Copy Extraction (NumPy)
    with open(state_path, "rb") as f:
        blob = f.read()
    
    # Offsets calculation
    off_v = 0
    off_f = off_v + padded_n * 4
    off_to = off_f + padded_n * 1
    off_rt = off_to + padded_n * 4
    off_sa = off_rt + padded_n * 1
    off_target = off_sa + padded_n * 4
    off_weight = off_target + padded_n * 128 * 4
    off_timer = off_weight + padded_n * 128 * 2

    voltage = np.frombuffer(blob, dtype=np.int32, count=padded_n, offset=off_v)
    flags = np.frombuffer(blob, dtype=np.uint8, count=padded_n, offset=off_f)
    threshold_offset = np.frombuffer(blob, dtype=np.int32, count=padded_n, offset=off_to)
    refractory_timer = np.frombuffer(blob, dtype=np.uint8, count=padded_n, offset=off_rt)
    soma_to_axon = np.frombuffer(blob, dtype=np.uint32, count=padded_n, offset=off_sa)
    
    # Targets/Weights/Timers are stored in SoA format: [slot][neuron]
    targets = np.frombuffer(blob, dtype=np.uint32, count=padded_n * 128, offset=off_target).reshape(128, padded_n)
    weights = np.frombuffer(blob, dtype=np.int16, count=padded_n * 128, offset=off_weight).reshape(128, padded_n)
    timers = np.frombuffer(blob, dtype=np.uint8, count=padded_n * 128, offset=off_timer).reshape(128, padded_n)

    # 2. Winner-Takes-All (WTA) Distillation
    print("[*] Performing WTA Distillation (128 -> 32 slots)...")
    abs_weights = np.abs(weights.astype(np.int32))
    # Select top 32 indices per neuron (highest absolute weight)
    top_indices = np.argsort(abs_weights, axis=0)[-32:, :]
    
    col_indices = np.arange(padded_n)
    esp_targets = targets[top_indices, col_indices]
    esp_weights = weights[top_indices, col_indices]
    esp_timers = timers[top_indices, col_indices]

    # 3. Dual-Memory Split
    print("[*] Generating hardware binaries...")
    
    # SRAM State (Hot State)
    with open(axons_path, "rb") as f:
        axons_data = f.read()

    sram_data = (
        voltage.tobytes() +
        flags.tobytes() +
        threshold_offset.tobytes() +
        refractory_timer.tobytes() +
        esp_weights.tobytes() +
        esp_timers.tobytes() +
        axons_data
    )
    
    with open("shard.sram", "wb") as f:
        f.write(sram_data)
    print(f"[+] Saved shard.sram ({len(sram_data)} bytes)")

    # Flash State (ReadOnly Topology)
    # [DOD FIX] 64-byte Cache-Aligned Header: Magic "TOPO" (0x4F504F54), padded_n, 56 bytes padding
    header = struct.pack("<II56x", 0x4F504F54, padded_n)
    flash_data = header + esp_targets.tobytes() + soma_to_axon.tobytes()
    
    # [КРИТИЧЕСКИЙ ИНВАРИАНТ] Pad to 64KB for MMU
    remainder = len(flash_data) % 65536
    if remainder != 0:
        pad_size = 65536 - remainder
        print(f"[*] Padding Flash blob with {pad_size} bytes for 64KB alignment...")
        flash_data += b'\x00' * pad_size

    with open("shard.flash", "wb") as f:
        f.write(flash_data)
    print(f"[+] Saved shard.flash ({len(flash_data)} bytes, 64KB aligned)")

    print("\n[✔] Distillation Pipeline Complete.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 distill_esp32.py <shard.state> <shard.axons>")
        sys.exit(1)
    
    distill(sys.argv[1], sys.argv[2])
