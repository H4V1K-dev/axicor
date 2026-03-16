#!/usr/bin/env python3
import sys
import os
import glob

# Добавляем путь к SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "genesis-client")))
import genesis

def tabula_rasa():
    print("🧹 Starting Tabula Rasa (Blank Slate) weight reset...")
    
    shm_files = glob.glob("/dev/shm/genesis_shard_*")
    if not shm_files:
        print("❌ No active Genesis shards found in /dev/shm/")
        return

    for path in shm_files:
        zone_hash_str = path.split("_")[-1]
        zone_hash = int(zone_hash_str, 16)
        
        try:
            mem = genesis.GenesisMemory(zone_hash)
            stats_before = mem.get_network_stats()
            
            mem.clear_weights()
            
            stats_after = mem.get_network_stats()
            print(f"✅ Zone {zone_hash_str}: Weights cleared. Avg Weight: {stats_before['avg_weight']:.2f} -> {stats_after['avg_weight']:.2f}")
            mem.close()
        except Exception as e:
            print(f"⚠️  Failed to reset zone {zone_hash_str}: {e}")

    print("\n✨ Done. All weights are now zero. Spikes will have to build their own paths now.")

if __name__ == "__main__":
    tabula_rasa()
