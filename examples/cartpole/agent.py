#!/usr/bin/env python3
import os
import sys

# Проверка активации виртуального окружения
if not (sys.prefix != sys.base_prefix or 'VIRTUAL_ENV' in os.environ):
    print("❌ ERROR: Virtual environment not active!")
    print("Please run: source .venv/bin/activate")
    sys.exit(1)
import time
import numpy as np
import gymnasium as gym

# Добавляем путь к SDK ( genesis-client/ ) если скрипт запущен напрямую из примера
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "genesis-client")))

from genesis.client import GenesisMultiClient
from genesis.encoders import PopulationEncoder
from genesis.decoders import PwmDecoder
from genesis.control import GenesisControl
from genesis.tuner import GenesisAutoTuner
from genesis.memory import GenesisMemory
from genesis.surgeon import GenesisSurgeon
from genesis.brain import fnv1a_32

#============================================================
#                   CONFIGURATION ЕКСПЕРИМЕНТАЛЬНО!
#============================================================
EPISODES = 20_000_000         # Количество эпизодов до остановки обучения
BATCH_SIZE = 20               # HFT-цикл: 1 пакет = 20 тиков (Должно быть равно tick_duration_us в build_brain.py)
NIGHT_INTERVAL = 50_000       # Периодичность сна (50к тиков = 100с физики)
PRUNE_THRESHOLD = 10          # Порог удаления слабых синапсов (Аналог квантования fp32->fp16 когда срезается хвост, но здесь не критично, дает возможность перестроить связь)

# Баланс R-STDP (Near-Zero Economy)
DOPAMINE_PULSE = 2            # Околонулевая эрозия
DOPAMINE_REWARD = -2          # Микро-награда для защиты от LTP Runaway
DOPAMINE_PUNISHMENT = +20     # Death Signal (Максимальное наказание)

# Гиперпараметры Физики (GLIF & Receptors)
D1_AFFINITY = 172             # Аффинность D1
D2_AFFINITY = 252             # Аффинность D2
LEAK_RATE = 1086              # Коэффициент утечки
HOMEOS_PENALTY = 5960         # Штраф за homeostasis
HOMEOS_DECAY = 49             # Декремент homeostasis

# Тюнинг Градиента (Error Gradient)
ERROR_ANGLE_WEIGHT = 0.8      # Вес ошибки угла
ERROR_VEL_WEIGHT = 0.2        # Вес ошибки скорости
ANGLE_LIMIT = 0.2094          # 12 градусов
VELOCITY_LIMIT = 2.0          # Максимальная скорость

# Тюнинг Болевого Шока (Kinetic Amplifier)
SHOCK_BASE = 10               # Базовый шок
SHOCK_SCORE_BITSHIFT = 5      # score >> 5 (каждые 32 тика +1 батч)
SHOCK_VEL_MULT = 5            # Штраф за скорость удара
SHOCK_MAX_BATCHES = 100       # Максимальное количество батчей

# Архитектурные параметры
ENCODER_SIGMA = 0.2           # Сигма энкодера
TARGET_SCORE = 50000.0        # Целевая оценка
#============================================================
#               END OF CONFIGURATION
#============================================================
def run_cartpole():
    # Синхронизация времени Вселенной и Мозга (1 шаг = 2 мс = 20 тиков)
    env = gym.make("CartPole-v1").unwrapped
    env.tau = 0.002
    state, _ = env.reset()
    
    zone_hash = fnv1a_32(b"SensoryCortex")
    matrix_hash = fnv1a_32(b"cartpole_sensors")    
    
    # 64 сенсора (4 переменных * 16 нейронов) * BATCH_SIZE тиков / 8 бит
    input_payload_size = (64 * BATCH_SIZE) // 8 

    # 2. Инициализация HFT Транспорта
    client = GenesisMultiClient(
        addr=("127.0.0.1", 8081),
        matrices=[{'zone_hash': zone_hash, 'matrix_hash': matrix_hash, 'payload_size': input_payload_size}]
    )
    
    # ЖЕСТКАЯ ПРИВЯЗКА К ПОРТУ ОТВЕТОВ: Нода шлет GSOO пакеты на этот порт
    try:
        client.sock.bind(("0.0.0.0", 8092))
    except OSError as e:
        print(f"❌ FATAL: Port 8092 is busy! Kill zombie agents before running. Error: {e}")
        sys.exit(1)

    # 3. DOD Энкодеры и Декодеры (Без аллокаций)
    encoder = PopulationEncoder(variables_count=4, neurons_per_var=16, batch_size=BATCH_SIZE, sigma=ENCODER_SIGMA)
    # Выход MotorCortex: 128 моторных нейронов (64 на лево, 64 на право)
    decoder = PwmDecoder(num_outputs=128, batch_size=BATCH_SIZE)

    # 4. Векторизованная нормализация
    bounds = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.41, 0.41], [-2.0, 2.0]], dtype=np.float16)
    range_diff = bounds[:, 1] - bounds[:, 0]

    # 5. Контроль и телеметрия прямо через маппинг памяти ОС (Zero-Copy)
    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Genesis-Models/CartPole-example/baked/SensoryCortex/manifest.toml"))
    
    if not os.path.exists(manifest_path):
        print(f"❌ FATAL: Control Plane manifest NOT FOUND at {manifest_path}")
        sys.exit(1)
        
    control = GenesisControl(manifest_path)
    # [DOD FIX] Вечная жизнь. Сеть никогда не заморозит пластичность.
    tuner = GenesisAutoTuner(control, target_score=TARGET_SCORE)
    
    # [DOD FIX] Принудительная установка интервалов
    control.set_night_interval(NIGHT_INTERVAL)
    control.set_prune_threshold(PRUNE_THRESHOLD)

    # [DOD FIX] Внедрение HFT-физики, найденной Optuna
    print("🧬 Injecting Optuna GLIF & Receptor parameters via Zero-Downtime Control Plane...")
    control.set_dopamine_receptors(0, d1_affinity=D1_AFFINITY, d2_affinity=D2_AFFINITY)
    control.set_dopamine_receptors(1, d1_affinity=D1_AFFINITY, d2_affinity=D2_AFFINITY)
    
    # Регуляция мембранной физики
    control.set_membrane_physics(0, LEAK_RATE, HOMEOS_PENALTY, HOMEOS_DECAY)
    control.set_membrane_physics(1, int(LEAK_RATE * 1.5), int(HOMEOS_PENALTY * 0.8), HOMEOS_DECAY)
    
    
    # Подключаем Memory Plane для аналитики графа
    print("⏳ Ожидание инициализации Genesis Node (Shared Memory)...")
    memory = None
    for i in range(20):
        try:
            # Открываем не в read_only для Surgeon и Distillation
            memory = GenesisMemory(zone_hash, read_only=False)
            surgeon = GenesisSurgeon(memory)
            print("✅ Telemetry Plane (Zero-Copy mmap) подключен!")
            break
        except (FileNotFoundError, AssertionError) as e:
            if i % 5 == 0:
                print(f"   [Retry {i}/20] SHM not ready: {e}")
            time.sleep(1)
    
    if not memory:
        print("❌ FATAL: Could not connect to Shared Memory! Is Genesis Node and Baker Daemon running?")
        sys.exit(1)
    


    episodes, score = 0, 0
    synapses, avg_w = 0, 0.0
    terminated, truncated = False, False
    
    norm_state = np.zeros(4, dtype=np.float16)
    print(f"🚀 Starting Genesis DOD CartPole Loop (Lockstep BATCH_SIZE={BATCH_SIZE})...")
    
    while episodes < EPISODES:
        # 1. EPISODE TERMINATION (Pain Shock & Recovery)
        if terminated or truncated:
            # [DOD FIX] Жестко транслируем кадр ошибки в VRAM, иначе выжигание идет "вслепую"
            encoder.encode_into(norm_state, client.payload_views[0], 0)
            
            # [DOD FIX] Non-Linear Death Signal & Kinetic Amplifier
            # Базовый шок + экспоненциальный штраф за долгий экстаз
            shock_batches = SHOCK_BASE + (score >> SHOCK_SCORE_BITSHIFT)
            
            # Кинетический штраф за угловую скорость падения (state[1])
            pole_velocity = abs(state[1])
            kinetic_penalty = int(pole_velocity * SHOCK_VEL_MULT)
            
            # Хард-лимит на батчи боли, чтобы не усыпить сеть навсегда
            total_shock = min(SHOCK_MAX_BATCHES, shock_batches + kinetic_penalty)
            
            # Пролонгированное выжигание виновных синапсов (LTD)
            for _ in range(total_shock):
                client.step(DOPAMINE_PUNISHMENT)

            # Извлекаем метрики после падения
            if memory:
                stats = memory.get_network_stats()
                synapses = stats["active_synapses"]
                avg_w = stats["avg_weight"]
            
            phase_str = tuner.step(score).name if tuner else "N/A"
            
            # Если синапсов 0, значит Ночная фаза еще не прошла
            if synapses == 0:
                print(f"Ep {episodes:04d} | Score: {score:3d} | Phase: {phase_str:<12} | [Awaiting Night Phase DMA...]")
            else:
                print(f"Ep {episodes:04d} | Score: {score:3d} | Phase: {phase_str:<12} | Synapses: {synapses:5d} | Avg W: {avg_w:.1f}")
            
            state, _ = env.reset()
            score, episodes = 0, episodes + 1
            terminated, truncated = False, False
            continue

        # Нормализация состояния
        norm_state = (np.clip(state, bounds[:, 0], bounds[:, 1]) - bounds[:, 0]) / range_diff

        # [DOD FIX] Continuous Error Gradient (Zero Branches)
        pole_angle = abs(state[2])
        pole_velocity = abs(state[3])

        # 1. Нормализация ошибки (0.0 = идеал, 1.0 = крах)
        angle_error = min(1.0, pole_angle / ANGLE_LIMIT)
        vel_error = min(1.0, pole_velocity / VELOCITY_LIMIT)
 
        # 2. Взвешенная ошибка
        error = min(1.0, angle_error * ERROR_ANGLE_WEIGHT + vel_error * ERROR_VEL_WEIGHT)
 
        # 3. Линейная алгебра дофамина (без if/else)
        dopamine_signal = int(DOPAMINE_REWARD * (1.0 - error) + DOPAMINE_PULSE * error)
        
        # --- SINGLE BATCH HFT (2 ms) ---
        # Теперь 20 тиков ровно хватает на сквойной пролет сигнала через Nuclear Layer
        encoder.encode_into(norm_state, client.payload_views[0], 0)
        rx = client.step(dopamine_signal)
        
        total_motor = decoder.decode_from(rx, 0)
        # Winner-Takes-All: Суммируем спайки по левой (0-63) и правой (64-127) группам
        action = 0 if np.sum(total_motor[:64]) > np.sum(total_motor[64:]) else 1

        state, reward, terminated, truncated, _ = env.step(action)
        score += 1

if __name__ == '__main__':
    run_cartpole()
