import numpy as np
import os
import sys

# Add genesis-client to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from genesis import GenesisMemory, GenesisSurgeon

def test_surgeon():
    print("--- Testing GenesisSurgeon ---")
    
    # 1. Setup mock memory shards
    shard_path = "/dev/shm/genesis_shard_test"
    if not os.path.exists(shard_path):
        # Create a dummy shard if it doesn't exist for testing
        # In a real scenario, we'd use live shards
        print(f"Note: Shard {shard_path} not found. Manual verification required on live shards.")
        # We can't easily create a valid SHM shard with correct header here without more effort,
        # but the logic is vector-based on numpy arrays.
        
    # Find a real shard to test on
    shards = [f for f in os.listdir("/dev/shm") if f.startswith("genesis_shard_")]
    if not shards:
        print("No active shards found in /dev/shm. Please start genesis-node.")
        return

    zone_hash = int(shards[0].split("_")[-1], 16)
    print(f"Testing on shard: {shards[0]} (Hash: {zone_hash:08X})")
    
    memory = GenesisMemory(zone_hash)
    surgeon = GenesisSurgeon(memory)
    
    # 1. Test GABA incubation
    print("Testing incubate_gaba...")
    count = surgeon.incubate_gaba(baseline_weight=-30000)
    print(f"Incubated {count} inhibitory synapses with weight -30000.")
    
    # 2. Test Graft Extraction
    print("Testing extract_graft...")
    offsets, signs = surgeon.extract_graft(threshold=1000)
    print(f"Extracted graft with {len(offsets)} synapses.")
    
    if len(offsets) > 0:
        print(f"First 5 offsets: {offsets[:5]}")
        print(f"First 5 signs: {signs[:5]}")
        
        # 3. Test Graft Injection
        print("Testing inject_graft...")
        surgeon.inject_graft(offsets, signs)
        print("Graft injected (weights set to +/- 32767).")
        
        # Verify weight update
        sample_idx = offsets[0]
        new_weight = memory.weights.ravel()[sample_idx]
        expected = int(signs[0]) * 32767
        print(f"Verification: Index {sample_idx}, Weight: {new_weight}, Expected: {expected}")
        
    print("--- Test Completed ---")

if __name__ == "__main__":
    test_surgeon()
