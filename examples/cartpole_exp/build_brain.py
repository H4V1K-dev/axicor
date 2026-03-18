#!/usr/bin/env python3
import os
import sys

# Проверка активации виртуального окружения
if not (sys.prefix != sys.base_prefix or 'VIRTUAL_ENV' in os.environ):
    print("❌ ERROR: Virtual environment not active!")
    print("Please run: source .venv/bin/activate")
    sys.exit(1)
import subprocess

# Добавляем путь к SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../genesis-client")))
from genesis.builder import BrainBuilder

def build_cartpole_brain():
    print("🧠 Инициализация архитектора коннектома CartPole...")
    
    gnm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../GNM-Library"))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Genesis-Models/CartPole-example"))
    
    builder = BrainBuilder(project_name="CartPoleAgent", output_dir=out_dir, gnm_lib_path=gnm_path)
    
    # Тонкая настройка физики под HFT-цикл (100 ticks = 10 ms)
    # Тонкая настройка физики под HFT-цикл (20 ticks = 2 ms)
    builder.sim_params["sync_batch_ticks"] = 20
    builder.sim_params["tick_duration_us"] = 100
    
    # Компактный резервуар: 16x16x16 вокселей (400x400x400 мкм)
    # Максимальное время полета сигнала от края до края = 16 тиков.
    cortex = builder.add_zone("SensoryCortex", width_vox=16, depth_vox=16, height_vox=16)

    try:
        exc_type = builder.gnm_lib("VISp4/141").set_plasticity(pot=0, dep=2)

        inh_type = builder.gnm_lib("VISp4/114").set_plasticity(pot=0, dep=2)
        
        # [DOD FIX] Изолированный тип для моторного слоя
        motor_type = builder.gnm_lib("VISp4/141").set_plasticity(pot=0, dep=2)
        motor_type.name = "Motor_Pyramidal" # Уникальное имя для GXO маппинга
        for d in motor_type.data_list:
            d["name"] = "Motor_Pyramidal"   # Жесткая перезапись для blueprints.toml
        
        # [DOD FIX] Принудительная мутация параметров для всех типов
        for bp in [exc_type, inh_type, motor_type]:
            for d in bp.data_list:
                d["initial_synapse_weight"] = 8000
                d["dendrite_radius_um"] = 400.0               # Охватывает весь Nuclear куб
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)
    
    # Единый ядерный слой (Basal Ganglia / Cerebellum style)
    # Плотность 0.2 даст нам около 819 нейронов (идеально для CartPole)
    cortex.add_layer("Nuclear", height_pct=1.0, density=0.2)\
          .add_population(exc_type, fraction=0.5)\
          .add_population(inh_type, fraction=0.2)\
          .add_population(motor_type, fraction=0.3)
          
    # I/O Матрицы
    cortex.add_input("cartpole_sensors", width=8, height=8, entry_z="bottom")
    cortex.add_output("motor_out", width=16, height=8, target_type="Motor_Pyramidal")
    
    # 1. Генерируем TOML-ДНК
    builder.build()
    
    # 2. АВТОМАТИЧЕСКОЕ ЗАПЕКАНИЕ (Genesis Baker)
    print("\n🔥 Запускаем Genesis Baker (CPU Compiler)...")
    brain_toml_path = os.path.join(out_dir, "brain.toml")
    
    result = subprocess.run([
        "cargo", "run", "--release", "-p", "genesis-baker", "--bin", "baker", "--", 
        "--brain", brain_toml_path
    ])
    
    if result.returncode == 0:
        print("\n✅ Модель успешно запечена и готова к загрузке на GPU.")
    else:
        print("\n❌ Ошибка компиляции коннектома. Проверьте логи Rust-компилятора.")
        sys.exit(1)

if __name__ == '__main__':
    build_cartpole_brain()
