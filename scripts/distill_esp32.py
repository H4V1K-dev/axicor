#!/usr/bin/env python3
import os
import mmap
import struct
import sys
import numpy as np

# [DOD] Genesis SHM/Snapshot Distillation Script v1.0
MAX_DENDRITE_SLOTS = 128
HEADER_FMT = "<IBBHIIIIQIIIIIIII"
HEADER_SIZE = 64

def step2_distill_and_pack(state_path: str, out_path: str):
    print(f"\n[*] Starting Zero-Copy WTA Distillation: 128 -> 32 slots")
    ESP_SLOTS = 32
    file_size = os.path.getsize(state_path)
    
    with open(state_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        padded_n, weights_off, targets_off, flags_off, magic = get_blob_metadata(mm, file_size)
        
        # 1. Zero-Copy View (SoA Layout: [Slot, Neuron])
        weights_128 = np.frombuffer(mm, dtype=np.int16, count=padded_n * MAX_DENDRITE_SLOTS, offset=weights_off).reshape(MAX_DENDRITE_SLOTS, padded_n)
        targets_128 = np.frombuffer(mm, dtype=np.uint32, count=padded_n * MAX_DENDRITE_SLOTS, offset=targets_off).reshape(MAX_DENDRITE_SLOTS, padded_n)
        
        # Dendrite Timers follow Targets and Weights in SoA layout
        timers_off = targets_off + (padded_n * MAX_DENDRITE_SLOTS * 4) + (padded_n * MAX_DENDRITE_SLOTS * 2)
        timers_128 = np.frombuffer(mm, dtype=np.uint8, count=padded_n * MAX_DENDRITE_SLOTS, offset=timers_off).reshape(MAX_DENDRITE_SLOTS, padded_n)
        
        # 2. Transpose for neuron-wise processing: [128, N] -> [N, 128]
        w_t = weights_128.T
        t_t = targets_128.T
        tim_t = timers_128.T
        
        # 3. Vectorized Winner-Takes-All (WTA) Sorting by Absolute Weight
        print("[+] Executing Branchless SIMD Sort...")
        abs_w = np.abs(w_t.astype(np.float32))
        # Get indices of top-32 connections by descending strength
        top_indices = np.argsort(-abs_w, axis=1)[:, :ESP_SLOTS]
        
        # 4. Vectorized Extraction (Advanced Indexing)
        row_indices = np.arange(padded_n)[:, None]
        new_w = w_t[row_indices, top_indices].T  # Back to SoA [32, N]
        new_t = t_t[row_indices, top_indices].T
        new_tim = tim_t[row_indices, top_indices].T
        
        # Clean up dead slots
        invalid_mask = new_t == 0
        new_w[invalid_mask] = 0
        new_tim[invalid_mask] = 0
        
        active_survivors = np.sum(new_t != 0)
        print(f"[+] Distillation complete. Surviving synapses: {active_survivors} (Max 32 per soma)")
        
        # 5. Flat Serialization (Re-packing C-ABI Blob)
        print(f"[*] Packing binary blob to {out_path}")
        with open(out_path, "wb") as out_f:
            # Header + Static Arrays (Voltage, Flags, Thresholds, Timers, SomaToAxon)
            # These are everything from the start up to targets_off
            static_data = mm[0:targets_off]
            out_f.write(static_data)
            
            # Patch dendrite_slots in header IF it's a SHM header
            if magic == 0x47454E53:
                out_f.seek(12)
                out_f.write(struct.pack("<I", ESP_SLOTS))
                out_f.seek(targets_off)
            
            out_f.write(new_t.tobytes()) # Target [32, N]
            out_f.write(new_w.tobytes()) # Weights [32, N]
            out_f.write(new_tim.tobytes()) # Timers [32, N]
            
            final_size = out_f.tell()
            print(f"[+] ESP32 State Blob Generated: {out_path} ({final_size / 1024:.2f} KB)")
        
        # Cleanup views before closing mmap
        del weights_128, targets_128, timers_128, w_t, t_t, tim_t
        mm.close()

def get_blob_metadata(mm, file_size):
    magic = struct.unpack("<I", mm[0:4])[0]
    
    if magic == 0x50414E53: # "SNAP"
        header = struct.unpack("<IIIIQQ", mm[0:32])
        payload_size = header[4]
        padded_n = payload_size // 910
        blob_offset = 32
    elif magic == 0x47454E53: # "GENS"
        header = struct.unpack(HEADER_FMT, mm[0:64])
        padded_n = header[4]
        blob_offset = 0
        return padded_n, header[6], header[7], header[16], 0x47454E53
    else:
        # Raw SoA blob
        padded_n = file_size // 910
        blob_offset = 0

    # Offsets for SNAP/RAW
    base = blob_offset
    flags_off = base + padded_n * 4
    targets_off = base + padded_n * (4 + 1 + 4 + 1 + 4)
    weights_off = targets_off + padded_n * 128 * 4
    return padded_n, weights_off, targets_off, flags_off, magic

def step1_parse_source(state_path: str):
    print(f"[*] Analyzing State: {state_path}")
    file_size = os.path.getsize(state_path)
    with open(state_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        padded_n, weights_off, targets_off, flags_off, magic = get_blob_metadata(mm, file_size)
        
        if magic == 0x50414E53: print("[+] Format: ShardStateHeader (Snapshot)")
        elif magic == 0x47454E53: print("[+] Format: ShmHeader (Live SHM)")
        else: print("[+] Format: Raw SoA Blob")

        print(f"[+] Padded_n: {padded_n}")
        print(f"[+] Memory Offsets: flags={flags_off}, weights={weights_off}, targets={targets_off}")

        # Zero-copy NumPy views
        try:
            # We need to be careful with byte strings from mmap
            # np.frombuffer works on the mmap object directly
            
            flags = np.frombuffer(mm, dtype=np.uint8, count=padded_n, offset=flags_off)
            
            # Shape (padded_n, 128) -> neuron index is first
            targets = np.frombuffer(mm, dtype=np.uint32, count=padded_n * MAX_DENDRITE_SLOTS, offset=targets_off).reshape(padded_n, MAX_DENDRITE_SLOTS)
            weights = np.frombuffer(mm, dtype=np.int16, count=padded_n * MAX_DENDRITE_SLOTS, offset=weights_off).reshape(padded_n, MAX_DENDRITE_SLOTS)
            
            active_mask = targets != 0
            # Total active synapses
            active_count = np.sum(active_mask)
            
            print(f"[+] Topo Scan: {active_count} active synapses out of {padded_n * MAX_DENDRITE_SLOTS} allocated slots.")
            
            # Connections per neuron (sum over Dendrite Slots axis)
            conns_per_neuron = np.sum(active_mask, axis=1)
            max_connections = np.max(conns_per_neuron)
            print(f"[+] Max connections on a single soma: {max_connections} (ESP32 limit is 32)")
            
            if max_connections > 32:
                print("(!) WARNING: Model exceeds ESP32 connectivity limits!")
                
        except Exception as e:
            print(f"FATAL: Read error - {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Explicitly delete views to release reference to mmap
            if 'flags' in locals(): del flags
            if 'targets' in locals(): del targets
            if 'weights' in locals(): del weights
            mm.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/distill_esp32.py <path_to_shard.state> [output_esp32.blob]")
        sys.exit(1)
    
    src_path = sys.argv[1]
    step1_parse_source(src_path)
    
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
        step2_distill_and_pack(src_path, out_path)
