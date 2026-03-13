#!/usr/bin/env python3
import os
import sys
import time
import numpy as np

# Добавляем путь к SDK ( genesis-client/ )
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "genesis-client")))

try:
    import gymnasium as gym
except ImportError:
    print("Установите gymnasium: pip install gymnasium")
    exit(1)

from genesis.client import GenesisMultiClient
from genesis.encoders import PopulationEncoder
from genesis.decoders import PwmDecoder
from genesis.control import GenesisControl
from genesis.tuner import GenesisAutoTuner
from genesis.brain import fnv1a_32

def run_cartpole():
    env = gym.make("CartPole-v1")
    
    # 1. Hashing C-ABI (совпадает с Rust FNV-1a)
    zone_hash = fnv1a_32(b"SensoryCortex")
    matrix_hash = fnv1a_32(b"cartpole_sensors")
    
    # 64 сенсора (4 переменных * 16 нейронов) * 100 тиков / 8 бит = 800 байт
    input_payload_size = (64 * 100) // 8 

    # 2. Инициализация HFT Транспорта
    client = GenesisMultiClient(
        addr=("127.0.0.1", 8081),
        matrices=[{
            'zone_hash': zone_hash,
            'matrix_hash': matrix_hash,
            'payload_size': input_payload_size
        }]
    )

    # ЖЕСТКАЯ ПРИВЯЗКА К ПОРТУ ОТВЕТОВ: Нода шлет GSOO пакеты на этот порт
    try:
        client.sock.bind(("0.0.0.0", 8092))
    except OSError as e:
        print(f"⚠️  Could not bind to port 8092: {e}. If the node is not running, this is fine.")

    # 3. DOD Энкодеры и Декодеры (Без аллокаций)
    encoder = PopulationEncoder(variables_count=4, neurons_per_var=16, batch_size=100)
    # Выход MotorCortex: 128 моторных нейронов (64 на лево, 64 на право)
    decoder = PwmDecoder(num_outputs=128, batch_size=100)

    # 4. Векторизованная нормализация (Матрица диапазонов среды)
    bounds = np.array([
        [-2.4, 2.4],     # Cart Position
        [-3.0, 3.0],     # Cart Velocity
        [-0.41, 0.41],   # Pole Angle
        [-2.0, 2.0]      # Pole Velocity At Tip
    ], dtype=np.float16)
    range_diff = bounds[:, 1] - bounds[:, 0]

    # 5. Гормональная панель (Control Plane)
    # Используем fallback-путь для тестов
    control = GenesisControl("baked/SensoryCortex/manifest.toml") if __import__('os').path.exists("baked/SensoryCortex/manifest.toml") else None
    tuner = GenesisAutoTuner(control, target_score=500.0) if control else None

    state, _ = env.reset()
    episodes = 0
    dopamine = 0
    score = 0

    print("🚀 Starting Genesis HFT CartPole Loop...")
    
    while episodes < 2000:
        # --- HOT LOOP (10ms Budget) ---
        
        # 1. Zero-Cost Normalization [0.0, 1.0]
        norm_state = np.clip((state - bounds[:, 0]) / range_diff, 0.0, 1.0)
        
        # 2. Population Encoding прямо в сетевой буфер (смещение 0, так как payload_views уже сдвинут за хидер)
        encoder.encode_into(norm_state, client.payload_views, 0)
        
        # 3. Lockstep Barrier (Блокирующий пинг-понг с нодой)
        rx_view = client.step(dopamine)
        
        # 4. Zero-Copy Decoding (WTA - Winner Takes All)
        motor_out = decoder.decode_from(rx_view, 0)
        left_force = np.sum(motor_out[:64])
        right_force = np.sum(motor_out[64:])
        action = 0 if left_force > right_force else 1
        
        # 5. Среда
        state, reward, terminated, truncated, _ = env.step(action)
        score += 1
        
        # 6. R-STDP Dopamine Injection
        if terminated or truncated:
            dopamine = -255 # Смертельный сигнал (выжигает ошибки)
            phase_str = "N/A"
            if tuner:
                phase = tuner.step(score)
                phase_str = phase.name
                
            print(f"Episode {episodes:4d} | Score: {score:3d} | Phase: {phase_str}")
            
            state, _ = env.reset()
            score = 0
            episodes += 1
        else:
            dopamine = 10 # Фоновый тонус (поддерживает живые связи)

if __name__ == '__main__':
    run_cartpole()
