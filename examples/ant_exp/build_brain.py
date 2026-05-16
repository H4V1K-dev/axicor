#!/usr/bin/env python3
import os
import sys
import subprocess

if not (sys.prefix != sys.base_prefix or 'VIRTUAL_ENV' in os.environ):
    print("❌ ERROR: Virtual environment not active!")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "axicor-client")))
from axicor.builder import BrainBuilder

def build_ant_connectome():
    print("Initializing architect (6-Zone Feedforward WTA)...")
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # [DOD FIX] Use the new regenerated catalog
    gnm_path = os.path.join(base_path, "Axicor_Neuron-Lib")
    out_dir = os.path.join(base_path, "Axicor-Models/AntConnectome")

    builder = BrainBuilder(project_name="AntConnectome", output_dir=out_dir, gnm_lib_path=gnm_path)
    
    # [BATCH_SIZE FIX] 2ms HFT Batch for ultra-fast response
    builder.sim_params["sync_batch_ticks"] = 20
    builder.sim_params["tick_duration_us"] = 100

    # ============================================================
    # EPIC 11: INTEGRATION HARDWARE PROFILES (Validated Canonical Cells)
    # ============================================================
    # SENSORY: Integrators and Fast-Spiking
    sens_exc_1 = builder.gnm_lib("Cortex/L4/spiny/VISp4/1") # Integrator
    sens_exc_2 = builder.gnm_lib("Cortex/L4/spiny/VISp4/1") # Fallback mapping
    sens_inh   = builder.gnm_lib("Cortex/L4/aspiny/VISp4/1") # Fast-Spiking Inhibitory

    # THALAMUS (CPG): Pacemakers and Relays
    thal_pace_exc  = builder.gnm_lib("Cortex/L4/spiny/VISp4/2") # Pacemaker
    thal_relay_exc = builder.gnm_lib("Cortex/L4/spiny/VISpm4/1") # Relay
    thal_pace_inh  = builder.gnm_lib("Cortex/L4/aspiny/VISp4/1") # Fast-Spiking (Brake)
    thal_mod_inh   = builder.gnm_lib("Cortex/L4/aspiny/VISp4/2") # Martinotti Modulator

    # SPINAL GANGLIA: Motor outputs
    motor_pyr_1 = builder.gnm_lib("Cortex/L4/spiny/VISp4/1") # Integrator Pyramid
    motor_pyr_2 = builder.gnm_lib("Cortex/L4/spiny/VISpm4/1") # Relay Pyramid
    motor_inh   = builder.gnm_lib("Cortex/L4/aspiny/VISp4/2") # Martinotti

    # ============================================================
    # SHARD 1: SENSORY CORTEX
    # ============================================================
    sensory = builder.add_zone("SensoryCortex", width_vox=64, depth_vox=64, height_vox=16)
    sensory.add_layer("L4_Input", height_pct=1.0, density=0.8)\
           .add_population(sens_exc_1, fraction=0.8)\
           .add_population(sens_inh, fraction=0.2)
    
    sensory.add_input("ant_sensors", width=28, height=16, entry_z="top")
    sensory.add_output("to_thalamus", width=16, height=16)

    # ============================================================
    # SHARD 2: THALAMUS (Central Pattern Generator Hub)
    # ============================================================
    thalamus = builder.add_zone("Thalamus", width_vox=64, depth_vox=64, height_vox=32)
    thalamus.add_layer("L_Hub", height_pct=1.0, density=0.25)\
            .add_population(thal_pace_exc, fraction=0.3)\
            .add_population(thal_relay_exc, fraction=0.2)\
            .add_population(thal_pace_inh, fraction=0.3)\
            .add_population(thal_mod_inh, fraction=0.2)
            
    thalamus.add_output("to_fl", width=16, height=8)
    thalamus.add_output("to_fr", width=16, height=8)
    thalamus.add_output("to_bl", width=16, height=8)
    thalamus.add_output("to_br", width=16, height=8)

    # ============================================================
    # SHARDS 3-6: SPINAL GANGLIA (Independent Legs)
    # ============================================================
    legs = {}
    for leg in ["FL", "FR", "BL", "BR"]:
        ganglion = builder.add_zone(f"{leg}_Ganglion", width_vox=32, depth_vox=32, height_vox=16)
        ganglion.add_layer("L_Motor", height_pct=1.0, density=0.2)\
                .add_population(motor_pyr_1, fraction=0.5)\
                .add_population(motor_pyr_2, fraction=0.2)\
                .add_population(motor_inh, fraction=0.3)
                
        # [DOD FIX] Strict targeting for motor output
        ganglion.add_output(f"motor_out_{leg}", width=8, height=4, target_type="All")
        ganglion.add_output(f"proprio_{leg}", width=8, height=8)
        legs[leg] = ganglion

    # ============================================================
    # ПРОВОДКА (Ghost Axons L2 Routing)
    # ============================================================
    print("[*] Laying axons (Zero-Copy L2 Ghost Sync)...")
    builder.connect(sensory, thalamus, out_matrix="to_thalamus", 
                    in_width=16, in_height=16, entry_z="top", growth_steps=1200)

    for leg, ganglion in legs.items():
        builder.connect(thalamus, ganglion, out_matrix=f"to_{leg.lower()}", 
                        in_width=16, in_height=8, entry_z="top", growth_steps=1000)
        
        # [DOD FIX] Проприоцепция жестко бьет в гиперактивный тормоз Таламуса, обрывая фазу шага
        builder.connect(ganglion, thalamus, out_matrix=f"proprio_{leg}", 
                        in_width=8, in_height=8, entry_z="bottom", 
                        target_type="Cortex/L4/aspiny/VISp4/1", growth_steps=1500)

    builder.build()

    print("\nStarting Axicor Baker...")
    brain_toml_path = os.path.join(out_dir, "brain.toml")

    cargo_cmd = ["cargo", "run", "--release", "-p", "axicor-baker", "--bin", "axicor-baker"]
    
    # [DOD FIX] Zero-Config Hardware Detection. Никаких ручных флагов.
    import shutil
    has_cuda = shutil.which("nvcc") is not None
    has_rocm = shutil.which("hipcc") is not None

    if has_rocm:
        cargo_cmd.extend(["--features", "amd"])
    elif not has_cuda:
        print(" [HW] GPU compilers not found. Forcing CPU Fallback (mock-gpu)...")
        # Передаем feature для axicor-baker, который пробросит его в axicor-compute
        cargo_cmd.extend(["--features", "mock-gpu"])

    cargo_cmd.extend(["--", "--brain", brain_toml_path, "--clean", "--yes"])

    result = subprocess.run(cargo_cmd, cwd=base_path)

    if result.returncode != 0:
        print("\n[ERROR] Connectome compilation failed.")
        sys.exit(1)

if __name__ == '__main__':
    build_ant_connectome()