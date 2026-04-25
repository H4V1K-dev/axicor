import importlib.util
import sys
import os

# Load builder.py directly
spec = importlib.util.spec_from_file_location("builder", "axicor/builder.py")
builder_mod = importlib.util.module_from_spec(spec)
sys.modules["builder"] = builder_mod
spec.loader.exec_module(builder_mod)

BrainBuilder = builder_mod.BrainBuilder

def test_builder_io_integration():
    print("Testing BrainBuilder I/O integration...")
    builder = BrainBuilder("TestProject", "out")
    zone = builder.add_zone("V1", 64, 64, 64)
    
    print("Adding input 'retina' (256x256)...")
    zone.add_input("retina", 256, 256)

    # [DOD FIX] Inputs are never fragmented (atomic bitmask requirement for GPU)
    assert len(zone.inputs) == 1, f"Expected exactly 1 matrix for inputs, got {len(zone.inputs)}"
    
    first_matrix = zone.inputs[0]
    print(f"First matrix name: {first_matrix['name']}")
    assert first_matrix["name"] == "retina"
    
    # Assertions on the first pin (chunk) of the matrix
    first_pin = first_matrix["pin"][0]
    assert first_pin["name"] == "retina"
    assert first_pin["target_type"] == "All"
    assert first_pin["stride"] == 1
    assert "u_width" in first_pin
    
    print("Adding non-fragmented output 'motor' (10, 10)...")
    zone.add_output("motor", 10, 10)
    
    assert len(zone.outputs) == 1
    motor_out = zone.outputs[0]
    assert motor_out["name"] == "motor"
    # [DOD FIX] Attributes are now inside pins
    assert motor_out["pin"][0]["target_type"] == "All"
    assert motor_out["pin"][0]["stride"] == 1
    assert "u_width" in motor_out["pin"][0]
    
    print("Testing entry_z validation...")
    # Valid float string
    zone.add_input("sensor", 10, 10, entry_z="50.0")
    assert zone.inputs[-1]["entry_z"] == "50.0"
    
    # Invalid value
    try:
        zone.add_input("broken", 10, 10, entry_z="invalid")
        assert False, "Should have raised ValueError for invalid entry_z"
    except ValueError as e:
        print(f"Caught expected validation error: {e}")

    print("\n--- ALL BUILDER I/O TESTS PASSED ---")

if __name__ == "__main__":
    test_builder_io_integration()
