import numpy as np
import os
import sys

# Add axicor-client to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from axicor import AxicorMemory, AxicorSurgeon

def test_surgeon():
    print("--- Testing AxicorSurgeon ---")
    
    # Find a real shard to test on
    import tempfile
    shm_dir = "/dev/shm" if sys.platform != "win32" else tempfile.gettempdir()
    if not os.path.exists(shm_dir):
        print(f"No active shards found in {shm_dir}.")
        return
        
    shards = [f for f in os.listdir(shm_dir) if f.startswith("axicor_shard_")]
    if not shards:
        print(f"No active shards found in {shm_dir}. Please start axicor-node.")
        return

    zone_hash = int(shards[0].split("_")[-1], 16)
    print(f"Testing on shard: {shards[0]} (Hash: {zone_hash:08X})")
    
    memory = AxicorMemory(zone_hash)
    surgeon = AxicorSurgeon(memory)
    
    # 1. Test GABA incubation
    print("Testing incubate_gaba...")
    count = surgeon.incubate_gaba(baseline_weight=-30000)
    print(f"Incubated {count} inhibitory synapses with weight -30000.")
    
    # 2. Test Graft Extraction (Reflex Path)
    print("Testing extract_reflex_path...")
    # Use dummy root soma IDs for testing
    payload = surgeon.extract_reflex_path(root_soma_ids=np.array([0, 1, 2], dtype=np.int32))
    print(f"Extracted graft with {np.sum(payload['soma_mask'])} somas.")
    
    if np.any(payload['soma_mask']):
        # 3. Test Graft Injection (Subgraph)
        print("Testing inject_subgraph...")
        surgeon.inject_subgraph(payload)
        print("Graft injected.")
        
    print("--- Test Completed ---")

if __name__ == "__main__":
    test_surgeon()
