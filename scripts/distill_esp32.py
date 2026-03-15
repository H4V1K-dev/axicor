#!/usr/bin/env python3
import os
import mmap
import struct
import sys
import numpy as np

# [DOD] Genesis SHM/Snapshot Distillation Script v1.0
MAX_DENDRITE_SLOTS = 128

def step1_parse_source(state_path: str):
    print(f"[*] Analyzing State: {state_path}")
    
    file_size = os.path.getsize(state_path)
    with open(state_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read magic (4 bytes)
        magic = struct.unpack("<I", mm[0:4])[0]
        
        if magic == 0x50414E53: # "SNAP" (ShardStateHeader)
            print("[+] Format: ShardStateHeader (Snapshot with SNAP prefix)")
            header = struct.unpack("<IIIIQQ", mm[0:32])
            payload_size = header[4]
            # payload_size for .state blob is n * 910
            padded_n = payload_size // 910
            blob_offset = 32
        elif magic == 0x47454E53: # "GENS" (ShmHeader)
            print("[+] Format: ShmHeader (Live SHM)")
            header = struct.unpack("<IBBHIIIIQIIIIIIII", mm[0:64])
            padded_n = header[4]
            # SHM layout includes handovers (200,000 bytes)
            # weights_off = 64, targets_off = 64 + n*128*2, flags_off = weights_off + n*128*2 + n*128*4 + 200000
            weights_off = header[6]
            targets_off = header[7]
            flags_off = header[16]
            # We'll use these directly
            blob_offset = 0
        else:
            # Raw SoA blob (no header)
            print("[+] Format: Raw SoA Blob (No Header)")
            # size = n * 910
            padded_n = file_size // 910
            blob_offset = 0
            
            # Offsets for raw blob based on memory.rs compute_state_offsets
            # soma_voltage (4), soma_flags (1), threshold (4), timers (1), soma_to_axon (4)
            # targets (128*4), weights (128*2), timers (128*1)
            flags_off = padded_n * 4
            targets_off = padded_n * (4 + 1 + 4 + 1 + 4)
            weights_off = targets_off + padded_n * 128 * 4

        if magic != 0x47454E53:
            # For SNAP or RAW, re-calculate based on blob_offset
            # Note: targets and weights are AFTER the initial soma arrays (14*n bytes)
            base = blob_offset
            flags_off = base + padded_n * 4
            targets_off = base + padded_n * (4 + 1 + 4 + 1 + 4)
            weights_off = targets_off + padded_n * 128 * 4

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
        print("Usage: python3 scripts/distill_esp32.py <path_to_shard.state>")
        sys.exit(1)
    
    step1_parse_source(sys.argv[1])
