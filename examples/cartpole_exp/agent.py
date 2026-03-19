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

# ЕКСПЕРИМЕНТАЛЬНЫЕ ПАРАМЕТРЫ ЛИШЬ ДЛЯ ДЕМОНСТРАЦИИ
#          СБОРКИ СКРИПТА СРЕДЫ ОБУЧЕНИЯ
# ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ ТРЕБУЕТ ИССЛЕДОВАНИЯ

#============================================================
#       CLIENT & ENVIRONMENT SETTINGS
#============================================================
EPISODES = 20_000_000         # Количество эпизодов до остановки обучения
BATCH_SIZE = 20               # HFT-цикл: 1 пакет = 20 тиков (Должно быть равно tick_duration_us в build_brain.py)
ENCODER_SIGMA = 0.2           # Сигма энкодера (разброс признаков)

#============================================================
#       TRAINING: EXPLORATION (Base State)
#============================================================
# Целевой показатель (SMA за окно) для перехода к Дистилляции
EXPLORATION_TARGET_SCORE = 400

EXPLORE_NIGHT_INTERVAL = 10_000       # Периодичность сна
EXPLORE_PRUNE_THRESHOLD = 5           # «фильтр выживания» для синапсов
EXPLORE_MAX_SPROUTS = 128             # Максимальное количество новых связей
# Баланс R-STDP (Near-Zero Economy)
EXPLORE_DOPAMINE_PULSE = 0          # Околонулевая эрозия
EXPLORE_DOPAMINE_REWARD = 3          # Микро-награда
EXPLORE_DOPAMINE_PUNISHMENT = -5     # Death Signal
# Гиперпараметры Физики (GLIF & Receptors)
EXPLORE_D1_AFFINITY = 150             # Аффинность D1
EXPLORE_D2_AFFINITY = 220             # Аффинность D2
EXPLORE_LEAK_RATE = 850               # Коэффициент утечки
EXPLORE_HOMEOS_PENALTY = 2500         # Штраф за homeostasis
EXPLORE_HOMEOS_DECAY = 100             # Декремент homeostasis
# Тюнинг Градиента (Error Gradient)
EXPLORE_ERROR_ANGLE_WEIGHT = 0.8      # Вес ошибки угла
EXPLORE_ERROR_VEL_WEIGHT = 0.2        # Вес ошибки скорости
EXPLORE_ANGLE_LIMIT = 0.2094          # 12 градусов
EXPLORE_VELOCITY_LIMIT = 2.0          # Максимальная скорость
# Тюнинг Болевого Шока (Kinetic & Emotional Amplifier)
EXPLORE_SHOCK_BASE = 0                # Базовое кол-во батчей боли (минимум при любом падении)
EXPLORE_SHOCK_SCORE_BITSHIFT = 0      # Штраф за "обидное" падение: чем выше счет, тем дольше боль (score >> 5) Но стоит ли наказывать тех выдержал почти до конца?
EXPLORE_SHOCK_VEL_MULT = 5            # Кинетический штраф: сильнее наказывает за падение на высокой скорости
EXPLORE_SHOCK_MAX_BATCHES = 5         # Предохранитель: макс. кол-во батчей боли, чтобы не выжечь мозг в ноль

#============================================================
#               Этап DISTILLATION
#============================================================
# Целевой показатель (SMA за окно) для перехода к Кристаллизации
DISTILLATION_TARGET_SCORE = 600

DISTILLATION_NIGHT_INTERVAL = 20_000       # Периодичность сна
DISTILLATION_PRUNE_THRESHOLD = 0           # «фильтр выживания» для синапсов
DISTILLATION_MAX_SPROUTS = 32             # Максимальное количество новых связей
# Баланс R-STDP (Near-Zero Economy)
DISTILLATION_DOPAMINE_PULSE = 0          # Околонулевая эрозия
DISTILLATION_DOPAMINE_REWARD = 4          # Микро-награда
DISTILLATION_DOPAMINE_PUNISHMENT = -5     # Death Signal
# Гиперпараметры Физики (GLIF & Receptors)
DISTILLATION_D1_AFFINITY = 150             # Аффинность D1
DISTILLATION_D2_AFFINITY = 220             # Аффинность D2
DISTILLATION_LEAK_RATE = 8500               # Коэффициент утечки
DISTILLATION_HOMEOS_PENALTY = 2500         # Штраф за homeostasis
DISTILLATION_HOMEOS_DECAY = 100             # Декремент homeostasis
# Тюнинг Градиента (Error Gradient)
DISTILLATION_ERROR_ANGLE_WEIGHT = 0.8      # Вес ошибки угла
DISTILLATION_ERROR_VEL_WEIGHT = 0.2        # Вес ошибки скорости
DISTILLATION_ANGLE_LIMIT = 0.2094          # 12 градусов
DISTILLATION_VELOCITY_LIMIT = 2.0          # Максимальная скорость
# Тюнинг Болевого Шока (Kinetic & Emotional Amplifier)
DISTILLATION_SHOCK_BASE = 0                # Базовое кол-во батчей боли (минимум при любом падении)
DISTILLATION_SHOCK_SCORE_BITSHIFT = 0      # Штраф за "обидное" падение: чем выше счет, тем дольше боль (score >> 5) Но стоит ли наказывать тех выдержал почти до конца?
DISTILLATION_SHOCK_VEL_MULT = 1            # Кинетический штраф: сильнее наказывает за падение на высокой скорости
DISTILLATION_SHOCK_MAX_BATCHES = 3         # Предохранитель: макс. кол-во батчей боли, чтобы не выжечь мозг в ноль

#============================================================
#               Этап CRYSTALLIZED
#============================================================
# Целевой показатель (SMA за окно) для перехода к Кристаллизации
CRYSTALLIZATION_TARGET_SCORE = 800

CRYSTALLIZATION_NIGHT_INTERVAL = 500_000       # Периодичность сна
CRYSTALLIZATION_PRUNE_THRESHOLD = 500           # «фильтр выживания» для синапсов
CRYSTALLIZATION_MAX_SPROUTS = 128             # Максимальное количество новых связей
# Баланс R-STDP (Near-Zero Economy)
CRYSTALLIZATION_DOPAMINE_PULSE = 0          # Околонулевая эрозия
CRYSTALLIZATION_DOPAMINE_REWARD = 3          # Микро-награда
CRYSTALLIZATION_DOPAMINE_PUNISHMENT = 0     # Death Signal
# Гиперпараметры Физики (GLIF & Receptors)
CRYSTALLIZATION_D1_AFFINITY = 172             # Аффинность D1
CRYSTALLIZATION_D2_AFFINITY = 252             # Аффинность D2
CRYSTALLIZATION_LEAK_RATE = 850               # Коэффициент утечки
CRYSTALLIZATION_HOMEOS_PENALTY = 5560         # Штраф за homeostasis
CRYSTALLIZATION_HOMEOS_DECAY = 49             # Декремент homeostasis
# Тюнинг Градиента (Error Gradient)
CRYSTALLIZATION_ERROR_ANGLE_WEIGHT = 0.8      # Вес ошибки угла
CRYSTALLIZATION_ERROR_VEL_WEIGHT = 0.2        # Вес ошибки скорости
CRYSTALLIZATION_ANGLE_LIMIT = 0.2094          # 12 градусов
CRYSTALLIZATION_VELOCITY_LIMIT = 2.0          # Максимальная скорость
# Тюнинг Болевого Шока (Kinetic & Emotional Amplifier)
CRYSTALLIZATION_SHOCK_BASE = 0                # Базовое кол-во батчей боли (минимум при любом падении)
CRYSTALLIZATION_SHOCK_SCORE_BITSHIFT = 0      # Штраф за "обидное" падение: чем выше счет, тем дольше боль (score >> 5) Но стоит ли наказывать тех выдержал почти до конца?
CRYSTALLIZATION_SHOCK_VEL_MULT = 5            # Кинетический штраф: сильнее наказывает за падение на высокой скорости
CRYSTALLIZATION_SHOCK_MAX_BATCHES = 5         # Предохранитель: макс. кол-во батчей боли, чтобы не выжечь мозг в ноль

#============================================================
#               END OF CONFIGURATION
#============================================================

def run_cartpole():
    global BATCH_SIZE
    # 1. Загрузка манифеста для синхронизации физики
    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Genesis-Models/CartPole-example/baked/SensoryCortex/manifest.toml"))
    if not os.path.exists(manifest_path):
        print(f"❌ FATAL: Control Plane manifest NOT FOUND at {manifest_path}")
        sys.exit(1)
        
    control = GenesisControl(manifest_path)
    
    # [DOD FIX] Синхронизация BATCH_SIZE с реальностью ноды
    actual_batch_size = control.manifest.get("simulation", {}).get("sync_batch_ticks", BATCH_SIZE)
    if actual_batch_size != BATCH_SIZE:
        print(f"⚠️ Warning: BATCH_SIZE adjusted from {BATCH_SIZE} to {actual_batch_size} (from manifest)")
        BATCH_SIZE = actual_batch_size

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
    # (control уже инициализирован выше для синхронизации BATCH_SIZE)

    tuner = GenesisAutoTuner(
        control, 
        # Exploration
        explore_target_score=EXPLORATION_TARGET_SCORE,
        explore_prune=EXPLORE_PRUNE_THRESHOLD,
        explore_night=EXPLORE_NIGHT_INTERVAL,
        explore_sprouts=EXPLORE_MAX_SPROUTS,
        explore_dopamine_pulse=EXPLORE_DOPAMINE_PULSE,
        explore_dopamine_reward=EXPLORE_DOPAMINE_REWARD,
        explore_dopamine_punish=EXPLORE_DOPAMINE_PUNISHMENT,
        explore_leak=EXPLORE_LEAK_RATE,
        explore_homeos_penalty=EXPLORE_HOMEOS_PENALTY,
        explore_homeos_decay=EXPLORE_HOMEOS_DECAY,
        explore_d1=EXPLORE_D1_AFFINITY,
        explore_d2=EXPLORE_D2_AFFINITY,
        explore_err_angle=EXPLORE_ERROR_ANGLE_WEIGHT,
        explore_err_vel=EXPLORE_ERROR_VEL_WEIGHT,
        explore_angle_limit=EXPLORE_ANGLE_LIMIT,
        explore_vel_limit=EXPLORE_VELOCITY_LIMIT,
        explore_shock_base=EXPLORE_SHOCK_BASE,
        explore_shock_bitshift=EXPLORE_SHOCK_SCORE_BITSHIFT,
        explore_shock_vel_mult=EXPLORE_SHOCK_VEL_MULT,
        explore_shock_max_batches=EXPLORE_SHOCK_MAX_BATCHES,

        # Distillation
        distill_target_score=DISTILLATION_TARGET_SCORE,
        distill_prune=DISTILLATION_PRUNE_THRESHOLD,
        distill_night=DISTILLATION_NIGHT_INTERVAL,
        distill_sprouts=DISTILLATION_MAX_SPROUTS,
        distill_dopamine_pulse=DISTILLATION_DOPAMINE_PULSE,
        distill_dopamine_reward=DISTILLATION_DOPAMINE_REWARD,
        distill_dopamine_punish=DISTILLATION_DOPAMINE_PUNISHMENT,
        distill_leak=DISTILLATION_LEAK_RATE,
        distill_homeos_penalty=DISTILLATION_HOMEOS_PENALTY,
        distill_homeos_decay=DISTILLATION_HOMEOS_DECAY,
        distill_d1=DISTILLATION_D1_AFFINITY,
        distill_d2=DISTILLATION_D2_AFFINITY,
        distill_err_angle=DISTILLATION_ERROR_ANGLE_WEIGHT,
        distill_err_vel=DISTILLATION_ERROR_VEL_WEIGHT,
        distill_angle_limit=DISTILLATION_ANGLE_LIMIT,
        distill_vel_limit=DISTILLATION_VELOCITY_LIMIT,
        distill_shock_base=DISTILLATION_SHOCK_BASE,
        distill_shock_bitshift=DISTILLATION_SHOCK_SCORE_BITSHIFT,
        distill_shock_vel_mult=DISTILLATION_SHOCK_VEL_MULT,
        distill_shock_max_batches=DISTILLATION_SHOCK_MAX_BATCHES,
        
        # Crystallized
        crystallized_target_score=CRYSTALLIZATION_TARGET_SCORE,
        crystallized_prune=CRYSTALLIZATION_PRUNE_THRESHOLD,
        crystallized_night=CRYSTALLIZATION_NIGHT_INTERVAL,
        crystallized_sprouts=CRYSTALLIZATION_MAX_SPROUTS,
        crystallized_dopamine_pulse=CRYSTALLIZATION_DOPAMINE_PULSE,
        crystallized_dopamine_reward=CRYSTALLIZATION_DOPAMINE_REWARD,
        crystallized_dopamine_punish=CRYSTALLIZATION_DOPAMINE_PUNISHMENT,
        crystallized_leak=CRYSTALLIZATION_LEAK_RATE,
        crystallized_homeos_penalty=CRYSTALLIZATION_HOMEOS_PENALTY,
        crystallized_homeos_decay=CRYSTALLIZATION_HOMEOS_DECAY,
        crystallized_d1=CRYSTALLIZATION_D1_AFFINITY,
        crystallized_d2=CRYSTALLIZATION_D2_AFFINITY,
        crystallized_err_angle=CRYSTALLIZATION_ERROR_ANGLE_WEIGHT,
        crystallized_err_vel=CRYSTALLIZATION_ERROR_VEL_WEIGHT,
        crystallized_angle_limit=CRYSTALLIZATION_ANGLE_LIMIT,
        crystallized_vel_limit=CRYSTALLIZATION_VELOCITY_LIMIT,
        crystallized_shock_base=CRYSTALLIZATION_SHOCK_BASE,
        crystallized_shock_bitshift=CRYSTALLIZATION_SHOCK_SCORE_BITSHIFT,
        crystallized_shock_vel_mult=CRYSTALLIZATION_SHOCK_VEL_MULT,
        crystallized_shock_max_batches=CRYSTALLIZATION_SHOCK_MAX_BATCHES
    )
    
    # [DOD FIX] Принудительная установка интервалов
    control.set_night_interval(EXPLORE_NIGHT_INTERVAL)
    control.set_prune_threshold(EXPLORE_PRUNE_THRESHOLD)
    control.set_max_sprouts(EXPLORE_MAX_SPROUTS)
    
    # Регуляция мембранной физики
    control.set_membrane_physics(0, EXPLORE_LEAK_RATE, EXPLORE_HOMEOS_PENALTY, EXPLORE_HOMEOS_DECAY)
    control.set_membrane_physics(1, int(EXPLORE_LEAK_RATE * 1.5), int(EXPLORE_HOMEOS_PENALTY * 0.8), EXPLORE_HOMEOS_DECAY)
    
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
    
    # [DOD FIX] Zero-Garbage Buffers для нормализации
    norm_state = np.zeros(4, dtype=np.float16)
    temp_buffer = np.zeros(4, dtype=np.float16)
    
    print(f"🚀 Starting Genesis DOD CartPole Loop (Lockstep BATCH_SIZE={BATCH_SIZE})...")
    
    while episodes < EPISODES:
        # 1. EPISODE TERMINATION (Pain Shock & Recovery)
        if terminated or truncated:
            # [DOD FIX] Жестко транслируем кадр ошибки в VRAM, иначе выжигание идет "вслепую"
            encoder.encode_into(norm_state, client.payload_views[0], 0)
            
            # [DOD FIX] Non-Linear Death Signal & Kinetic Amplifier
            # Базовый шок + экспоненциальный штраф за долгий экстаз
            shock_batches = tuner.shock_base + (score >> tuner.shock_bitshift)
            
            # Кинетический штраф за угловую скорость падения (state[1])
            pole_velocity = abs(state[1])
            kinetic_penalty = int(pole_velocity * tuner.shock_vel_mult)
            
            # Хард-лимит на батчи боли, чтобы не усыпить сеть навсегда
            total_shock = min(tuner.shock_max_batches, shock_batches + kinetic_penalty)
            
            # Пролонгированное выжигание виновных синапсов (LTD)
            for _ in range(total_shock):
                client.step(tuner.dopamine_punish)

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

        # [DOD FIX] Zero-Garbage In-Place Normalization
        # Никаких временных массивов в Hot Loop
        np.clip(state, bounds[:, 0], bounds[:, 1], out=temp_buffer)
        np.subtract(temp_buffer, bounds[:, 0], out=temp_buffer)
        np.divide(temp_buffer, range_diff, out=norm_state)

        # [DOD FIX] Continuous Error Gradient (Zero Branches)
        pole_angle = abs(state[2])
        pole_velocity = abs(state[3])

        # 1. Нормализация ошибки (0.0 = идеал, 1.0 = крах)
        angle_error = min(1.0, pole_angle / tuner.angle_limit)
        vel_error = min(1.0, pole_velocity / tuner.vel_limit)
 
        # 2. Взвешенная ошибка
        error = min(1.0, angle_error * tuner.err_angle_weight + vel_error * tuner.err_vel_weight)
 
        # 3. Линейная алгебра дофамина (без if/else)
        dopamine_signal = int(tuner.dopamine_reward * (1.0 - error) + tuner.dopamine_pulse * error)
        
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
