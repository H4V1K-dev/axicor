#!/usr/bin/env python3
import os
import sys
import subprocess

if not (sys.prefix != sys.base_prefix or 'VIRTUAL_ENV' in os.environ):
    print("❌ ERROR: Virtual environment not active!")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "genesis-client")))
from genesis.builder import BrainBuilder

def build_ant_connectome():
    print("🧠 Инициализация архитектора (3-Zone Feedforward WTA)...")
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    gnm_path = os.path.join(base_path, "GNM-Library")
    out_dir = os.path.join(base_path, "Genesis-Models/AntConnectome")

    builder = BrainBuilder(project_name="AntConnectome", output_dir=out_dir, gnm_lib_path=gnm_path)
    
    # [BATCH_SIZE FIX] 2ms HFT Batch for ultra-fast response
    builder.sim_params["sync_batch_ticks"] = 20
    builder.sim_params["tick_duration_us"] = 100

    # ============================================================
    # АППАРАТНЫЕ ПРОФИЛИ (Strict Reward-Gated Plasticity)
    # ============================================================
    # pot=0 (рост только от дофамина), dep=2 (постоянное выжигание мусора)
    exc_type = builder.gnm_lib("VISp4/141").set_plasticity(pot=0, dep=2)
    inh_type = builder.gnm_lib("VISp4/114").set_plasticity(pot=0, dep=2)
    
    motor_type = builder.gnm_lib("VISp4/141").set_plasticity(pot=0, dep=2)
    motor_type.name = "Motor_Pyramidal"
    for d in motor_type.data_list:
        d["name"] = "Motor_Pyramidal"
        d["initial_synapse_weight"] = 12000 # Форсируем сильный стартовый контакт
        d["dendrite_radius_um"] = 500.0     # Широкий захват дендритов

    # ============================================================
    # SHARD 1: SENSORY CORTEX (Входной шлюз)
    # ============================================================
    sensory = builder.add_zone("SensoryCortex", width_vox=64, depth_vox=64, height_vox=16)
    
    sensory.add_layer("L4_Input", height_pct=1.0, density=0.15) \
           .add_population(exc_type, fraction=0.9) \
           .add_population(inh_type, fraction=0.1)
           
    # Strict VRAM Stride Alignment: 28 * 16 = 448 pixels
    sensory.add_input("ant_sensors", width=28, height=16, entry_z="top")
    sensory.add_output("to_thoracic", width=16, height=16)

    # ============================================================
    # SHARD 2: THORACIC GANGLION (Генератор Паттернов / Хаб)
    # ============================================================
    thoracic = builder.add_zone("ThoracicGanglion", width_vox=64, depth_vox=64, height_vox=32)
    
    thoracic.add_layer("L_Lower", height_pct=0.5, density=0.2) \
            .add_population(exc_type, fraction=0.7) \
            .add_population(inh_type, fraction=0.3)

    thoracic.add_layer("L_Upper", height_pct=0.5, density=0.2) \
            .add_population(exc_type, fraction=0.6) \
            .add_population(inh_type, fraction=0.4)

    thoracic.add_output("to_motor", width=16, height=16)

    # ============================================================
    # SHARD 3: MOTOR CORTEX (Выходной шлюз + Winner-Takes-All)
    # ============================================================
    motor = builder.add_zone("MotorCortex", width_vox=64, depth_vox=64, height_vox=32)
    
    motor.add_layer("L5_Lower", height_pct=0.6, density=0.25) \
         .add_population(motor_type, fraction=0.4) \
         .add_population(inh_type, fraction=0.6) # Winner-Takes-All Inhibition

    motor.add_layer("L5_Upper", height_pct=0.4, density=0.15) \
         .add_population(exc_type, fraction=1.0)

    # Выход строго с пирамид
    motor.add_output("motor_out", width=16, height=8, target_type="Motor_Pyramidal")

    # ============================================================
    # ПРОВОДКА (Ghost Axons Routing)
    # ============================================================
    print("[*] Прокладка аксонов...")
    builder.connect(sensory, thoracic, out_matrix="to_thoracic", 
                    in_width=16, in_height=16, entry_z="bottom", growth_steps=1200)

    builder.connect(thoracic, motor, out_matrix="to_motor", 
                    in_width=16, in_height=16, entry_z="bottom", target_type="Motor_Pyramidal", growth_steps=1500)

    builder.build()

    print("\n🔥 Запускаем Genesis Baker (CPU Compiler)...")
    brain_toml_path = os.path.join(out_dir, "brain.toml")

    result = subprocess.run([
        "cargo", "run", "--release", "-p", "genesis-baker", "--bin", "baker", "--",
        "--brain", brain_toml_path, "--clean"
    ], cwd=base_path)

    if result.returncode != 0:
        print("\n❌ Ошибка компиляции коннектома.")
        sys.exit(1)

if __name__ == '__main__':
    build_ant_connectome()
