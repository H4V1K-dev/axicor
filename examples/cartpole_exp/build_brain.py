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
    # Тонкая настройка физики под HFT-цикл (2ms шаг)
    # v_seg = (0.5 * 1000 * 0.1) / (25 * 2) = 1.0 (Strict Integer)
    builder.sim_params["sync_batch_ticks"] = 20
    builder.sim_params["tick_duration_us"] = 100
    builder.sim_params["signal_speed_m_s"] = 0.5
    builder.sim_params["segment_length_voxels"] = 2
    
    # Компактный резервуар: 16x16x30 вокселей (растягиваем по Z для лучшей изоляции слоев)
    cortex = builder.add_zone("SensoryCortex", width_vox=30, depth_vox=30, height_vox=30)

    try:
        # pot=0 (рост только от дофамина), dep=2 (постоянное выжигание мусора)
        exc_type = builder.gnm_lib("VISp4/141").set_plasticity(pot=0, dep=2)
        inh_type = builder.gnm_lib("VISp4/114").set_plasticity(pot=0, dep=2)
        
        motor_type = builder.gnm_lib("VISp4/141").set_plasticity(pot=0, dep=2)
        motor_type.name = "Motor_Pyramidal"
        for d in motor_type.data_list:
            d["name"] = "Motor_Pyramidal"
            d["initial_synapse_weight"] = 10000
            d["dendrite_radius_um"] = 200.0 # Локальный захват в верхнем слое
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)
    
    # ============================================================
    # СЛОИ (Архитектура AntV4 Middle Layer + Isolation)
    # ============================================================
    # 1. Слой входов (Нижняя треть: 0-10 вокселей)
    cortex.add_layer("L4_Sensory", height_pct=0.33, density=0.45)\
          .add_population(exc_type, fraction=0.7) \
            .add_population(inh_type, fraction=0.3)

    # 2. Слой процессинга (Средняя треть: 10-20 вокселей) - Стиль AntV4
    cortex.add_layer("L23_Middle", height_pct=0.34, density=0.45)\
          .add_population(exc_type, fraction=0.6) \
            .add_population(inh_type, fraction=0.4)

    # 3. Слой выходов (Верхняя треть: 20-30 вокселей) - Winner-Takes-All
    cortex.add_layer("L5_Motor", height_pct=0.33, density=0.45)\
          .add_population(motor_type, fraction=0.2)\
          .add_population(inh_type, fraction=0.8)
          
    # ============================================================
    # I/O МАТРИЦЫ (SDK v2 Features)
    # ============================================================
    # Вход: прорастает снизу вверх, застревает в L4
    cortex.add_input("cartpole_sensors", width=8, height=8, entry_z="top")
    
    # Выход: мапится строго на верхнюю треть зоны (uv_rect), забирает сигнал только с Motor_Pyramidal
    # Мы используем новый функционал uv_rect для аппаратной фильтрации соматических выходов
    cortex.add_output("motor_out", width=16, height=8, 
                      target_type="Motor_Pyramidal")
    
    # 1. Генерируем TOML-ДНК (Автоматически вызовет dry_run_stats)
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
