import os
import sys
import numpy as np

# Добавляем путь к SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "genesis-client")))

from genesis.brain import fnv1a_32
from genesis.memory import GenesisMemory

def analyze():
    zones = [
        "SensoryCortex", "ProprioceptiveHub", "ThoracicGanglion", 
        "CerebellumAnalog", "SpinalReflexArc", "MotorCortex"
    ]
    
    print(f"{'Zone Name':<20} | {'Synapses':<10} | {'Avg Weight':<10} | {'Max Weight':<10}")
    print("-" * 60)
    
    for z_name in zones:
        z_hash = fnv1a_32(z_name.encode('utf-8'))
        try:
            mem = GenesisMemory(z_hash, read_only=True)
            stats = mem.get_network_stats()
            print(f"{z_name:<20} | {stats['active_synapses']:<10} | {stats['avg_weight']:<10.1f} | {stats['max_weight']:<10}")
            mem.close()
        except Exception as e:
            print(f"{z_name:<20} | ERROR: {e}")

if __name__ == "__main__":
    analyze()
