#!/usr/bin/env python3
import os
import sys
import subprocess

if not (sys.prefix != sys.base_prefix or 'VIRTUAL_ENV' in os.environ):
    print("❌ ERROR: Virtual environment not active!")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "genesis-client")))
from genesis.builder import BrainBuilder

def build_humanoid_connectome():
    print("🧠 Architect: 5-Zone Humanoid Connectome Init...")
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(base_path, "Genesis-Models/HumanoidAgent")

    builder = BrainBuilder(project_name="HumanoidAgent", output_dir=out_dir, gnm_lib_path=os.path.join(base_path, "GNM-Library"))
    
    # [ANT PATTERN] 80 ticks = 8ms. Input fits in ONE UDP packet (61,460 bytes < 65,507).
    builder.sim_params["sync_batch_ticks"] = 80
    builder.sim_params["tick_duration_us"] = 100

    # --- HARDWARE PROFILES ---
    # Hippocampus pyramidal: THE gold standard for LTP. Native pot=115, dep=85.
    # Strong Hebbian learning built-in, no set_plasticity override needed.
    exc_type = builder.gnm_lib("Hippocampus/Mouse/pyramidal/775")
    
    # Hippocampus GABAergic interneuron: for inhibition
    inh_type = builder.gnm_lib("Hippocampus/Mouse/gabaergic/9")
    
    # Cerebellum specialists (unchanged — they already work)
    purkinje_type = builder.gnm_lib("Cerebellum/Mouse/purkinje/141")
    granule_type = builder.gnm_lib("Cerebellum/Mouse/gabaergic/476")
    
    # Thalamic oscillator for CPG rhythm
    tc_oscillator = builder.gnm_lib("Thalamus/Mouse/relay/141")
    
    # Motor pyramidal: L5 cortex — actual motor cortex neurons
    motor_pyramidal = builder.gnm_lib("Cortex/L5/spiny/VISp5/1")
    motor_pyramidal.name = "Motor_Pyramidal"
    
    # Boost excitatory connection strengths for initial activity
    for bp in [exc_type, purkinje_type, tc_oscillator, motor_pyramidal]:
        for d in bp.data_list:
            d["name"] = bp.name
            d["dendrite_radius_um"] = 800.0 

    for bp in [inh_type, granule_type]:
        for d in bp.data_list:
            d["dendrite_radius_um"] = 200.0

    # 1. SENSORY CORTEX (S1/V1 Analog) - 6 Layers for Z-Gradient
    sensory = builder.add_zone("SensoryCortex", width_vox=80, depth_vox=80, height_vox=40)
    # L1: Молекулярный слой (Только торможение, горизонтальный роутинг)
    sensory.add_layer("L1_Molecular", height_pct=0.1, density=0.05).add_population(inh_type, fraction=1.0)
    # L2/3: Обработка признаков
    sensory.add_layer("L23_Hidden", height_pct=0.3, density=0.20).add_population(exc_type, fraction=0.8).add_population(inh_type, fraction=0.2)
    # L4: Входной гранулярный слой (Сюда ударят сенсоры среды)
    sensory.add_layer("L4_Granular", height_pct=0.3, density=0.30).add_population(exc_type, fraction=0.85).add_population(inh_type, fraction=0.15)
    # L5: Проекционный слой (Выход на моторы)
    sensory.add_layer("L5_Output", height_pct=0.2, density=0.15).add_population(exc_type, fraction=0.7).add_population(inh_type, fraction=0.3)
    # L6: Полиморфный слой (Фидбек)
    sensory.add_layer("L6_Feedback", height_pct=0.1, density=0.10).add_population(exc_type, fraction=0.6).add_population(inh_type, fraction=0.4)

    # IO: 384 sensor vars * 16 neurons = 6144 virtual axons
    sensory.add_input("humanoid_sensors", width=384, height=16, entry_z="top", target_type="All")
    sensory.add_output("sensory_to_proprio", width=12, height=12)
    sensory.add_output("sensory_to_thoracic", width=12, height=12)
    sensory.add_output("sensory_to_motor_reflex", width=12, height=12)

    # 2. PROPRIOCEPTIVE HUB (Thalamus Relay) - Ядерная структура
    proprio = builder.add_zone("ProprioceptiveHub", width_vox=40, depth_vox=40, height_vox=20)
    proprio.add_layer("Relay_Core", height_pct=0.7, density=0.25).add_population(exc_type, fraction=0.8).add_population(inh_type, fraction=0.2)
    proprio.add_layer("Inhibitory_Shell", height_pct=0.3, density=0.15).add_population(inh_type, fraction=1.0)

    proprio.add_output("proprio_to_thoracic", width=12, height=12)
    proprio.add_output("proprio_to_cerebellum", width=12, height=12)

    # 3. THORACIC GANGLION (CPG / Spinal Cord)
    thoracic = builder.add_zone("ThoracicGanglion", width_vox=40, depth_vox=40, height_vox=30)
    # Дорсальные рога (Прием сенсорики)
    thoracic.add_layer("Dorsal_Horn", height_pct=0.3, density=0.20).add_population(exc_type, fraction=0.7).add_population(inh_type, fraction=0.3)
    # Промежуточная зона (Генераторы паттернов ходьбы)
    thoracic.add_layer("Intermediate_Oscillator", height_pct=0.4, density=0.25).add_population(tc_oscillator, fraction=0.6).add_population(inh_type, fraction=0.4)
    # Вентральные рога (Выдача мото-сигналов)
    thoracic.add_layer("Ventral_Horn", height_pct=0.3, density=0.20).add_population(exc_type, fraction=0.8).add_population(inh_type, fraction=0.2)

    thoracic.add_output("thoracic_to_motor", width=12, height=12)
    thoracic.add_output("thoracic_to_cerebellum", width=12, height=12)

    # 4. CEREBELLUM ANALOG (Timing & Homeostasis)
    cerebellum = builder.add_zone("CerebellumAnalog", width_vox=50, depth_vox=50, height_vox=20)
    cerebellum.add_layer("Molecular", height_pct=0.3, density=0.10).add_population(inh_type, fraction=1.0)
    cerebellum.add_layer("Purkinje", height_pct=0.2, density=0.15).add_population(purkinje_type, fraction=0.8).add_population(inh_type, fraction=0.2)
    cerebellum.add_layer("Granular", height_pct=0.5, density=0.35).add_population(granule_type, fraction=0.9).add_population(inh_type, fraction=0.1)

    cerebellum.add_output("cerebellum_to_motor", width=34, height=8)

    # 5. MOTOR CORTEX (M1 Analog)
    motor = builder.add_zone("MotorCortex", width_vox=60, depth_vox=60, height_vox=40)
    motor.add_layer("L23_Integration", height_pct=0.4, density=0.20).add_population(exc_type, fraction=0.8).add_population(inh_type, fraction=0.2)
    motor.add_layer("L5_Upper", height_pct=0.2, density=0.15).add_population(exc_type, fraction=0.7).add_population(inh_type, fraction=0.3)
    # Строгий Winner-Takes-All слой. Сюда запекается профиль Motor_Pyramidal, к которому привязан output.
    motor.add_layer("L5_Lower_Pyramidal", height_pct=0.3, density=0.25).add_population(motor_pyramidal, fraction=0.4).add_population(inh_type, fraction=0.6)
    motor.add_layer("L6_Polymorphic", height_pct=0.1, density=0.10).add_population(exc_type, fraction=0.5).add_population(inh_type, fraction=0.5)
    
    # 17 DOF: 34 muscles. Height 8 provides vertical integration.
    motor.add_output("motor_out", width=34, height=8, target_type="Motor_Pyramidal")
    # MotorCortex: 272 + 144 = 416 pixels. SAFE.
    motor.add_output("motor_to_proprio", width=12, height=12)

    # --- ROUTING (Strict Unique Hashes) ---
    print("[*] Protruding Inter-Zone Ghost Axons...")
    builder.connect(sensory, proprio, out_matrix="sensory_to_proprio", in_width=12, in_height=12, growth_steps=2000)
    builder.connect(sensory, thoracic, out_matrix="sensory_to_thoracic", in_width=12, in_height=12, growth_steps=2000)
    builder.connect(proprio, thoracic, out_matrix="proprio_to_thoracic", in_width=12, in_height=12, growth_steps=2000)
    
    # Direct Reflex Path
    builder.connect(sensory, motor, out_matrix="sensory_to_motor_reflex", in_width=12, in_height=12, 
                    entry_z="bottom", target_type="Motor_Pyramidal", growth_steps=2000)
    
    # Motor Drive
    builder.connect(thoracic, motor, out_matrix="thoracic_to_motor", in_width=12, in_height=12, 
                    entry_z="bottom", target_type="Motor_Pyramidal", growth_steps=2000)
    builder.connect(cerebellum, motor, out_matrix="cerebellum_to_motor", in_width=34, in_height=8, 
                    entry_z="bottom", target_type="Motor_Pyramidal", growth_steps=2000)

    builder.connect(thoracic, cerebellum, out_matrix="thoracic_to_cerebellum", in_width=12, in_height=12, growth_steps=1500)
    builder.connect(proprio, cerebellum, out_matrix="proprio_to_cerebellum", in_width=12, in_height=12, growth_steps=1200)
    
    builder.connect(motor, proprio, out_matrix="motor_to_proprio", in_width=12, in_height=12, growth_steps=1000)

    builder.build()

    print("\n🔥 Baking VRAM Blobs via genesis-baker...")
    res = subprocess.run(["cargo", "run", "--release", "-p", "genesis-baker", "--bin", "baker", "--", "--brain", os.path.join(out_dir, "brain.toml"), "--clean"], cwd=base_path, input=b"y\n")
    if res.returncode != 0:
        print("❌ Baker Compilation Failed!")
        sys.exit(1)

if __name__ == '__main__':
    build_humanoid_connectome()