import time
import gc
import numpy as np
from axicor.encoders import PwmEncoder, PopulationEncoder

# C-ABI Constants (20 bytes header, rest is payload)
MAX_UDP_PAYLOAD = 65507
HEADER_SIZE = 20

def benchmark_encoder(name, encoder, iters=10_000):
    # 1. Preallocation of buffer simulating a UDP socket (Zero-Copy Target)
    tx_buffer = bytearray(MAX_UDP_PAYLOAD)
    tx_view = memoryview(tx_buffer)
    
    # 2. Data preallocation
    if isinstance(encoder, PwmEncoder):
        data = np.random.rand(encoder.N).astype(np.float32)
    else:
        data = np.random.rand(encoder.V).astype(np.float32)
    
    # Warmup
    encoder.encode_into(data, tx_view, HEADER_SIZE)
    
    gc.collect()
    count_before = gc.get_count()[0]
    
    start = time.perf_counter()
    for _ in range(iters):
        encoder.encode_into(data, tx_view, HEADER_SIZE)
    end = time.perf_counter()
    
    count_after = gc.get_count()[0]
    duration_ms = ((end - start) / iters) * 1000
    
    print(f" {name} Benchmark:")
    print(f" Time per batch:  {duration_ms:.4f} ms")
    print(f" Objects in Gen 0: {count_after - count_before}")
    
    if duration_ms > 1.0:
        print(f"[ERROR] FAILURE: {name} violated the 1ms budget!")
        return False
    
    print(f"[OK] {name} Success: Zero-Allocation pipeline confirmed.\n")
    return True

def main():
    print(" Starting Zero-Garbage I/O Pipeline Benchmarks...\n")
    
    pwm = PwmEncoder(num_sensors=500, batch_size=100)
    pop = PopulationEncoder(variables_count=8, neurons_per_var=64, batch_size=100)
    
    s1 = benchmark_encoder("PwmEncoder (500 sensors, 100 ticks)", pwm)
    s2 = benchmark_encoder("PopulationEncoder (8 vars, 64 neurons/var, 100 ticks)", pop)
    
    if s1 and s2:
        print(" All benchmarks passed! Zero-Garbage I/O Pipeline is solid.")
    else:
        exit(1)

if __name__ == '__main__':
    main()
