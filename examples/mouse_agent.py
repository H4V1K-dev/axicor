#!/usr/bin/env python3
import os
import sys
import subprocess

# Добавляем путь к SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "genesis-client")))
from genesis import BrainBuilder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    print("🐁 Assembling Mouse Embodied Agent (7 Zones)...")
    # Инициализация конструктора
    b = BrainBuilder(project_name="MouseAgent", output_dir="Genesis_Models/mouse_agent", gnm_lib_path="GNM-Library")

    print("🧬 Loading Allen Brain Genetics...")
    # Точные пути из GNM-Library
    thalamus_relay = b.gnm_lib("Thalamus/Mouse/relay/141")
    v1_l4 = b.gnm_lib("Cortex/L4/spiny/VISp4/117")
    v1_l23 = b.gnm_lib("Cortex/L23/spiny/VISp23/11")
    hippo_teml = b.gnm_lib("Cortex/L5/spiny/TemL/1")         # Височная доля (Память)
    pfc_frol = b.gnm_lib("Cortex/L3/spiny/FroL/10")          # Лобная доля (Логика)
    striatum_inh = b.gnm_lib("Cortex/L4/aspiny/VISp4/100")   # Тормозной фильтр
    motor_l5 = b.gnm_lib("Cortex/L5/spiny/MTG/1")            # Гигантские пирамиды
    purkinje = b.gnm_lib("Cerebellum/Mouse/purkinje/141")    # Мозжечок

    print("🧠 Designing Zones and Layers...")
    # 1. Thalamus (Входной хаб)
    lgn = b.add_zone("LGN_Thalamus", 80, 80, 35)
    lgn.add_layer("Relay", 1.0, 0.08).add_population(thalamus_relay, 1.0)

    # 2. V1 Cortex (Зрение)
    v1 = b.add_zone("V1_Cortex", 80, 80, 35)
    v1.add_layer("L23_Hidden", 0.6, 0.05).add_population(v1_l23, 1.0)
    v1.add_layer("L4_Input", 0.4, 0.08).add_population(v1_l4, 1.0)

    # 3. Hippocampus (Память)
    hippo = b.add_zone("Hippocampus", 80, 80, 35)
    hippo.add_layer("CA1", 1.0, 0.06).add_population(hippo_teml, 1.0)

    # 4. PFC (Принятие решений)
    pfc = b.add_zone("PFC_Cortex", 80, 80, 55)
    pfc.add_layer("Executive", 1.0, 0.05).add_population(pfc_frol, 1.0)

    # 5. Striatum (Фильтр действий)
    striatum = b.add_zone("Striatum", 80, 80, 35)
    striatum.add_layer("Matrix", 1.0, 0.08).add_population(striatum_inh, 1.0)

    # 6. Motor Cortex (Исполнение)
    motor = b.add_zone("Motor_Cortex", 120, 120, 60)
    motor.add_layer("L5_Output", 1.0, 0.05).add_population(motor_l5, 1.0)

    # 7. Cerebellum (Координация и баланс)
    cerebellum = b.add_zone("Cerebellum", 120, 120, 20)
    cerebellum.add_layer("Purkinje_Layer", 1.0, 0.03).add_population(purkinje, 1.0)

    print("🔌 Configuring I/O Matrices...")
    # Внешний вход в Таламус
    lgn.add_input("retina_rgb", 32, 32, target_type=thalamus_relay.name)
    # Внешний выход из Моторной коры и Мозжечка
    motor.add_output("raw_motor_cmd", 8, 8, target_type=motor_l5.name)
    cerebellum.add_output("smoothed_motor_cmd", 8, 8, target_type=purkinje.name)

    print("🕸️ Weaving Inter-Zone Connectome (Ghost Axons)...")
    # Глаза -> Таламус -> V1
    lgn.add_output("to_v1", 16, 16)
    b.connect(lgn, v1, "to_v1", 16, 16, entry_z="top", target_type=v1_l4.name)

    # Ассоциации: V1 -> Память(Hippo) и Логика(PFC)
    v1.add_output("to_hippo", 8, 8)
    b.connect(v1, hippo, "to_hippo", 8, 8)

    v1.add_output("to_pfc", 8, 8)
    b.connect(v1, pfc, "to_pfc", 8, 8)

    # Память -> Логика
    hippo.add_output("to_pfc", 8, 8)
    b.connect(hippo, pfc, "to_pfc", 8, 8)

    # Логика -> Стриатум (Фильтр)
    pfc.add_output("to_striatum", 16, 16)
    b.connect(pfc, striatum, "to_striatum", 16, 16)

    # Стриатум -> Моторная кора
    striatum.add_output("to_motor", 16, 16)
    b.connect(striatum, motor, "to_motor", 16, 16, target_type=motor_l5.name)

    # Моторная кора -> Мозжечок (Копия команды для балансировки)
    motor.add_output("to_cerebellum", 16, 16)
    b.connect(motor, cerebellum, "to_cerebellum", 16, 16, target_type=purkinje.name)

    print("🔨 Generating Brain DNA...")
    b.build()

    print("\n🔥 Firing up genesis-baker on the massive cluster...")
    brain_path = "Genesis_Models/mouse_agent/brain.toml"
    
    # Запускаем компилятор напрямую из скрипта
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "genesis-baker", "--bin", "baker", "--", "--brain", brain_path],
        cwd=PROJECT_ROOT,
    )

    if result.returncode == 0:
        print("\n✅ MASSIVE BRAIN BAKE SUCCESSFUL! We just grew 7 lobes of a mouse brain.")
    else:
        print("\n❌ BAKER PANICKED.")

if __name__ == "__main__":
    main()