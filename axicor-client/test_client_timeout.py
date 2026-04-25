import time
import os
import sys
from axicor.client import AxicorMultiClient

def test_client_timeout():
    print("Testing AxicorMultiClient Timeout (Biological Amnesia)...")

    # Configuration: dead address and port
    dead_addr = ("127.0.0.1", 9999)
    # One dummy matrix
    matrices = [{'zone_hash': 0x1, 'matrix_hash': 0x2, 'payload_size': 128}]

    # Set timeout to 0.01 seconds (10ms Rule)
    # [DOD FIX] Must provide rx_layout to actually enter the recvfrom wait state for timeout testing
    rx_layout = [{'matrix_hash': 0x3, 'size': 128}]
    client = AxicorMultiClient(dead_addr, matrices, rx_layout=rx_layout, timeout=0.01)

    start_time = time.perf_counter()

    # Attempting to execute a step. The node will not respond, triggering the timeout.
    print("Executing step (expecting timeout in 0.01s)...")
    rx = client.step(reward=0)

    elapsed = time.perf_counter() - start_time

    print(f"Elapsed time: {elapsed:.44f}s")
    print(f"Received buffer size: {len(rx)}")

    # 1. Verify that an empty memoryview is returned
    assert len(rx) == 0, f"Expected empty buffer on timeout, got size {len(rx)}"
    assert isinstance(rx, memoryview), "Result must be a memoryview"

    # 2. Verify execution time (Windows granularity allows slight jitter)
    # [DOD FIX] On Windows, ConnectionResetError can trigger instantly for dead local ports.
    # We accept 0.0s to 0.1s.
    assert 0.0 <= elapsed <= 0.1, f"Timeout duration out of range: {elapsed:.4f}s"

    print("[OK] AxicorMultiClient timeout handled correctly.")

if __name__ == "__main__":
    test_client_timeout()
