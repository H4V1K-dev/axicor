import importlib.util
import sys

# Load builder.py directly
import os
spec = importlib.util.spec_from_file_location("builder", os.path.join(os.path.dirname(__file__), "axicor/builder.py"))
builder = importlib.util.module_from_spec(spec)
sys.modules["builder"] = builder
spec.loader.exec_module(builder)

IoMatrixDesigner = builder.IoMatrixDesigner

def test_io_matrix():
    print("Testing IoMatrixDesigner(10, 10) [Integer Physics]...")
    designer_1 = IoMatrixDesigner(10, 10)
    assert designer_1.padded_pixels == 128, f"Expected 128, got {designer_1.padded_pixels}"
    assert designer_1.bytes_per_tick == 16, f"Expected 16, got {designer_1.bytes_per_tick}"
    print("Passed.")

def test_fragmentation():
    print("Testing Fragmentation: Designer(256, 256), sync_batch_ticks=20...")
    designer = IoMatrixDesigner(256, 256)
    chunks = designer.fragment(sync_batch_ticks=20)
    
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert chunks[0]["height"] == 102, f"Expected height 102, got {chunks[0]['height']}"
    assert chunks[1]["height"] == 102
    assert chunks[2]["height"] == 52
    
    expected_uv_0 = [0.0, 0.0, 1.0, 102/256]
    assert chunks[0]["uv_rect"] == expected_uv_0, f"Expected {expected_uv_0}, got {chunks[0]['uv_rect']}"
    
    total_height = sum(c["height"] for c in chunks)
    assert total_height == 256, f"Expected total height 256, got {total_height}"
    print("Standard Fragmentation Passed.")

def test_torture_scenarios():
    print("Torture Test: Matrix too wide for MTU...")
    # 20000 px width. 
    # mtu=1500 -> max_aligned_pixels=11840. 
    # 11840 // 20000 = 0 -> ValueError!
    designer_wide = IoMatrixDesigner(20000, 10)
    try:
        designer_wide.fragment(sync_batch_ticks=1, mtu=1500)
        assert False, "Should have raised ValueError for width 20000"
    except ValueError as e:
        print(f"Caught expected ValueError (Wide Matrix): {e}")

    print("Torture Test: Extreme fragmentation (many small chunks)...")
    designer_small = IoMatrixDesigner(100, 100)
    # mtu=500, payload=480. 
    # sync_batch_ticks=200 -> bytes_per_tick = 480 // 200 = 2 bytes.
    # max_aligned_pixels = (2 // 4) * 32 = 0.
    # chunk_height = 0 // 100 = 0 -> ValueError!
    try:
        designer_small.fragment(sync_batch_ticks=200, mtu=500)
        assert False, "Should have raised ValueError (zero aligned pixels)"
    except ValueError as e:
        print(f"Caught expected ValueError (Zero capacity): {e}")

    print("Torture Test: Edge case - exact fit...")
    designer_exact = IoMatrixDesigner(32, 1)
    # [DOD FIX] Minimum payload for 64-bit alignment is 8 bytes. Header is 20 bytes. MTU = 28.
    chunks = designer_exact.fragment(sync_batch_ticks=1, mtu=28)
    assert len(chunks) == 1
    assert chunks[0]["uv_rect"] == [0.0, 0.0, 1.0, 1.0]
    print("Exact fit test passed.")

if __name__ == "__main__":
    test_io_matrix()
    test_fragmentation()
    test_torture_scenarios()
    print("\n--- ALL TESTS PASSED (Stage 2) ---")
