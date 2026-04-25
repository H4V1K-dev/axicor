import os
import sys
import subprocess
import time
import numpy as np
from axicor.client import AxicorMultiClient
from axicor.utils import fnv1a_32

def wait_for_node_ready(process, timeout=300):
    """Wait for all 3 zones to signal voltage stabilized."""
    start_time = time.time()
    stabilized_zones = set()
    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                raise RuntimeError(f"Node exited prematurely with code {process.returncode}")
            continue
        
        line_str = line.strip()
        print(f"[node] {line_str}")
        if "Voltage stabilized" in line_str:
            # Format usually: INFO axicor_node::node::shard_thread: Warmup complete for 0xD6660067. Voltage stabilized.
            parts = line_str.split("for ")
            if len(parts) > 1:
                zone_hash = parts[1].split(".")[0]
                stabilized_zones.add(zone_hash)
                print(f"[*] Zone {zone_hash} stabilized ({len(stabilized_zones)}/3)")
        
        if len(stabilized_zones) >= 3:
            print("[*] All zones stabilized.")
            return True
    return False

def consume_stdout(process):
    for line in iter(process.stdout.readline, ''):
        if not line: break
        # print(f"[node-bg] {line.strip()}")

def run_simulation(root_dir, axic_archive, ticks=1000):
    """Runs a single simulation and returns the concatenated payload."""
    # Clear stale SRAM cache to avoid size mismatch
    import shutil
    from pathlib import Path
    mem_dir = Path(root_dir) / f"{axic_archive}.mem"
    if mem_dir.exists():
        print(f"[*] Clearing stale SRAM: {mem_dir}")
        shutil.rmtree(mem_dir)

    print("\n--- Connecting Client (Pre-Bind) ---")
    # AntConnectome hashes
    z_motor_hash = fnv1a_32(b"MotorCortex")
    m_motor_hash = fnv1a_32(b"motor_out")
    
    client = AxicorMultiClient(
        addr=("127.0.0.1", 8081),
        matrices=[{
            'zone_hash': fnv1a_32(b"SensoryCortex"),
            'matrix_hash': fnv1a_32(b"ant_sensors"),
            'payload_size': 5600 # 56 bytes * 100 ticks
        }],
        rx_layout=[{
            'matrix_hash': m_motor_hash,
            'size': 12800 # 128 bytes * 100 ticks
        }],
        timeout=10.0,
        # [DOD FIX] Node sends output to 127.0.0.1:8092 by default
        bind_addr=("127.0.0.1", 8092)
    )

    print(f"\n--- Launching Node (Archive: {axic_archive}) ---")
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"
    env["AXICOR_NUM_THREADS"] = "1"
    
    node_proc = subprocess.Popen([
        "cargo", "run", "--release", "-p", "axicor-node", 
        "--features", "axicor-compute/mock-gpu", "--",
        axic_archive
    ], cwd=root_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

    all_payloads = []
    try:
        if not wait_for_node_ready(node_proc):
            raise RuntimeError("Node failed to reach ready state within timeout")

        import threading
        threading.Thread(target=consume_stdout, args=(node_proc,), daemon=True).start()

        print(f"\n--- Hot Loop ({ticks} ticks) ---")
        num_batches = ticks // 100
        for i in range(num_batches):
            for pv in client.payload_views:
                pv.fill(0) # No external input, rely on DDS noise
            rx_view = client.step(reward=0)
            
            # [DOD FIX] Strict Assertion. Silent drops break determinism tests.
            assert len(rx_view) == 12800, f"FATAL: UDP Drop detected at batch {i}. Expected 12800 bytes, got {len(rx_view)}."
            
            all_payloads.append(rx_view.tobytes())
            if i % 2 == 0:
                print(f"Batch {i}/{num_batches} processed")

        print("\n--- Simulation Run Complete ---")
    finally:
        print("\n--- Teardown Node & Client ---")
        if 'client' in locals():
            client.sock.close()
        node_proc.terminate()
        try:
            node_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            node_proc.kill()
        print("Node terminated.")
    
    return b"".join(all_payloads)

def test_live_1k_ticks_determinism():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.join(root_dir, "axicor-client"))
    from axicor.builder import BrainBuilder

    # 1. Build Brain DNA with active spontaneous firing
    print("\n--- Phase 1: Building Deterministic Brain DNA ---")
    gnm_path = os.path.join(root_dir, "Axicor_Neuron-Lib")
    out_dir = os.path.join(root_dir, "Axicor-Models", "DeterminismTest")
    
    if os.path.exists(out_dir):
        print(f"[*] Cleaning stale project dir: {out_dir}")
        import shutil
        shutil.rmtree(out_dir)

    builder = BrainBuilder(project_name="DeterminismTest", output_dir=out_dir, gnm_lib_path=gnm_path)
    builder.sim_params["master_seed"] = "DETERMINISM_TEST"
    builder.sim_params["sync_batch_ticks"] = 100
    builder.sim_params["tick_duration_us"] = 100
    
    # [DOD FIX] Dynamically resolve types to prevent test fragility
    available_types = sorted(list(builder._lib_index.keys()))
    if len(available_types) < 2:
        raise ValueError(f"Neuron library must contain at least 2 profiles. Found: {len(available_types)}")
    
    exc_type = builder.gnm_lib(available_types[0])
    inh_type = builder.gnm_lib(available_types[1])
    
    # Zone 1: Sensory
    sensory = builder.add_zone("SensoryCortex", width_vox=16, depth_vox=16, height_vox=8)
    sensory.add_layer("L4", height_pct=1.0, density=0.5).add_population(exc_type, 1.0)
    sensory.add_input("ant_sensors", width=28, height=16)
    sensory.add_output("to_thoracic", width=16, height=16)
    
    # Zone 2: Thoracic
    thoracic = builder.add_zone("ThoracicGanglion", width_vox=16, depth_vox=16, height_vox=8)
    thoracic.add_layer("L_Main", height_pct=1.0, density=0.5).add_population(exc_type, 0.7).add_population(inh_type, 0.3)
    thoracic.add_output("to_motor", width=16, height=16)
    
    # Zone 3: Motor
    motor = builder.add_zone("MotorCortex", width_vox=16, depth_vox=16, height_vox=8)
    motor.add_layer("L5", height_pct=1.0, density=0.5).add_population(exc_type, 1.0)
    motor.add_output("motor_out", width=16, height=8)
    
    builder.connect(sensory, thoracic, "to_thoracic", 16, 16)
    builder.connect(thoracic, motor, "to_motor", 16, 16)
    
    builder.build()
    
    # 2. Bake Model
    print("\n--- Phase 2: Baking Model ---")
    brain_toml = os.path.join(out_dir, "brain.toml")
    res = subprocess.run([
        "cargo", "run", "--release", "-p", "axicor-baker", "--bin", "axicor-baker", 
        "--features", "axicor-compute/mock-gpu", "--",
        "--brain", brain_toml, "--clean", "--yes"
    ], cwd=root_dir, capture_output=True, text=True)
    assert res.returncode == 0, f"Baker failed: {res.stderr}"
    
    axic_archive = os.path.join(root_dir, "Axicor-Models", "DeterminismTest.axic")
    assert os.path.exists(axic_archive), f"Archive not found: {axic_archive}"

    # 3. Run 1
    print("\n--- Phase 3: Run 1 (Capture) ---")
    run1_data = run_simulation(root_dir, axic_archive, ticks=1000)
    
    # 4. Run 2
    print("\n--- Phase 4: Run 2 (Validate) ---")
    run2_data = run_simulation(root_dir, axic_archive, ticks=1000)
    
    # 5. Assertions
    print("\n--- Phase 5: Result Analysis ---")
    
    # Non-Zero Activity Check
    total_bytes_run1 = sum(run1_data)
    print(f"Total Activity (Run 1): {total_bytes_run1} bytes")
    assert total_bytes_run1 > 0, "ERROR: Zero activity detected! DDS Heartbeat failed to trigger spikes."
    
    # Absolute Determinism Check
    assert len(run1_data) == len(run2_data), "ERROR: Payload length mismatch!"
    if run1_data == run2_data:
        print("[SUCCESS] Absolute Determinism Verified (Run 1 == Run 2)")
    else:
        print("[FAILED] Determinism Mismatch Detected!")
        # Find first difference
        for i in range(len(run1_data)):
            if run1_data[i] != run2_data[i]:
                print(f"First mismatch at byte {i}: {run1_data[i]} != {run2_data[i]}")
                break
        assert run1_data == run2_data

if __name__ == "__main__":
    test_live_1k_ticks_determinism()
