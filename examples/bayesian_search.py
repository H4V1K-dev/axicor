#!/usr/bin/env python3
import os
import sys
import numpy as np

# Проверка активации виртуального окружения
if not (sys.prefix != sys.base_prefix or 'VIRTUAL_ENV' in os.environ):
    print("❌ ERROR: Virtual environment not active!")
    sys.exit(1)

try:
    import optuna
    import gymnasium as gym
except ImportError:
    print("❌ ERROR: optuna or gymnasium not installed. Run: pip install optuna gymnasium")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "genesis-client")))
from genesis.client import GenesisMultiClient
from genesis.encoders import PopulationEncoder
from genesis.decoders import PwmDecoder
from genesis.control import GenesisControl
from genesis.memory import GenesisMemory
from genesis.brain import fnv1a_32

# ==========================================
# 1. Глобальная инициализация (Zero-Downtime)
# ==========================================
BATCH_SIZE = 20  # [DOD FIX] Strict BSP Sync: 20 тиков (2 мс)
zone_hash = fnv1a_32(b"SensoryCortex")
matrix_hash = fnv1a_32(b"cartpole_sensors")
input_payload_size = (64 * BATCH_SIZE) // 8

print("🔌 Connecting to Genesis Node (Data & Memory Planes)...")
client = GenesisMultiClient(
    addr=("127.0.0.1", 8081),
    matrices=[{'zone_hash': zone_hash, 'matrix_hash': matrix_hash, 'payload_size': input_payload_size}]
)

try:
    client.sock.bind(("0.0.0.0", 8092))
except OSError as e:
    print(f"❌ FATAL: Port 8092 is busy! Kill zombie agents before running. Error: {e}")
    sys.exit(1)

encoder = PopulationEncoder(variables_count=4, neurons_per_var=16, batch_size=BATCH_SIZE, sigma=0.2)
decoder = PwmDecoder(num_outputs=128, batch_size=BATCH_SIZE)

manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Genesis-Models/CartPole-example/baked/SensoryCortex/manifest.toml"))
if not os.path.exists(manifest_path):
    print(f"❌ FATAL: Control Plane manifest NOT FOUND at {manifest_path}")
    sys.exit(1)

control = GenesisControl(manifest_path)
memory = GenesisMemory(zone_hash, read_only=False)

bounds = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.41, 0.41], [-2.0, 2.0]], dtype=np.float16)
range_diff = bounds[:, 1] - bounds[:, 0]

# ==========================================
# 2. Optuna Objective
# ==========================================
def objective(trial):
    # 1. Сэмплируем гиперпараметры R-STDP (Полный спектр)
    dopamine_pulse = trial.suggest_int("dopamine_pulse", -255, -1)
    dopamine_reward = trial.suggest_int("dopamine_reward", 1, 255)
    prune_threshold = trial.suggest_int("prune_threshold", 2, 30)
    d1_affinity = trial.suggest_int("d1_affinity", 64, 255)
    d2_affinity = trial.suggest_int("d2_affinity", 64, 255)

    # Физика мембран (GLIF)
    # leak_rate: как быстро нейрон забывает входящий потенциал
    leak_rate = trial.suggest_int("leak_rate", 100, 3000)
    # homeostasis_penalty: насколько сильно нейрон "устает" после спайка
    homeostasis_penalty = trial.suggest_int("homeostasis_penalty", 500, 10000)
    # homeostasis_decay: как быстро нейрон "отдыхает"
    homeostasis_decay = trial.suggest_int("homeostasis_decay", 1, 100)

    # 2. Hot-Patching Control Plane
    control.set_prune_threshold(prune_threshold)
    control.set_night_interval(2000) # Жесткая фиксация частоты сна для быстрых мутаций
    
    # Применяем рецепторы к типам (0: Excitatory, 1: Inhibitory)
    control.set_dopamine_receptors(0, d1_affinity, d2_affinity)
    control.set_dopamine_receptors(1, d1_affinity, d2_affinity)
    
    # Применяем физику мембран (Hot-Patching VRAM LUT)
    control.set_membrane_physics(0, leak_rate, homeostasis_penalty, homeostasis_decay)
    control.set_membrane_physics(1, int(leak_rate * 1.5), int(homeostasis_penalty * 0.8), homeostasis_decay)

    # 3. Tabula Rasa (Хирургическое стирание VRAM)
    memory.clear_weights()

    # [DOD FIX] Синхронизация времени Вселенной и Мозга (1 шаг = 2 мс = 20 тиков)
    env = gym.make("CartPole-v1").unwrapped
    env.tau = 0.002
    state, _ = env.reset()
    norm_state = np.zeros(4, dtype=np.float16)
    
    # [DOD FIX] Deep Survival Metrics (400 seconds window)
    global_steps = 0
    max_score = 0
    MAX_STEPS = 200000
    score = 0
    terminated, truncated = False, False

    # Гоняем сеть, пока она не умрет от эпилепсии или не упрется в лимит времени
    while global_steps < MAX_STEPS:

        if terminated or truncated:
            max_score = max(max_score, score)
            
            # Жестко транслируем кадр ошибки в VRAM
            encoder.encode_into(norm_state, client.payload_views[0], 0)
            
            # Болевой шок на 10 батчей (20 мс биологического времени)
            for _ in range(10):
                client.step(-255)

            stats = memory.get_network_stats()
            avg_w = stats["avg_weight"]
            synapses = stats["active_synapses"]

            # [DOD FIX] Graceful Fatality: Сеть сгорела. Фиксируем время жизни.
            if synapses > 0 and (avg_w > 25000 or avg_w < 10):
                break

            state, _ = env.reset()
            score = 0
            terminated, truncated = False, False
            continue

        # Zero-Cost Normalization (In-Place)
        np.subtract(state, bounds[:, 0], out=norm_state, casting='unsafe')
        np.divide(norm_state, range_diff, out=norm_state, casting='unsafe')
        np.clip(norm_state, 0.0, 1.0, out=norm_state, casting='unsafe')

        # [DOD FIX] Continuous Error Gradient (Zero Branches)
        pole_angle = abs(state[2])
        pole_velocity = abs(state[3])

        # 1. Нормализация ошибки (0.0 = идеал, 1.0 = крах)
        angle_error = min(1.0, pole_angle / 0.2094)
        vel_error = min(1.0, pole_velocity / 2.0)

        # 2. Взвешенная ошибка (удерживаем фокус на угле, но гасим раскачку)
        error = min(1.0, angle_error * 0.8 + vel_error * 0.2)

        # 3. Линейная алгебра дофамина (без if/else)
        dop_sig = int(dopamine_reward * (1.0 - error) + dopamine_pulse * error)

        # --- SINGLE BATCH HFT (2 ms) ---
        encoder.encode_into(norm_state, client.payload_views[0], 0)
        rx = client.step(dop_sig)
        
        total_motor = decoder.decode_from(rx, 0)
        action = 0 if np.sum(total_motor[:64]) > np.sum(total_motor[64:]) else 1

        state, reward, terminated, truncated, _ = env.step(action)
        score += 1
        global_steps += 1

    env.close()
    
    # Сохраняем побочную аналитику для дашборда
    max_score = max(max_score, score)
    trial.set_user_attr("max_score", max_score)
    
    # [DOD FIX] Composite Metric: Выживаемость + Навык (с жестким приоритетом навыка для бессмертных)
    return float(global_steps + (max_score * 1000))

if __name__ == '__main__':
    # Снижаем шум от логов Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING) # [DOD FIX] Глушим дефолтный мусор

    # [DOD FIX] Zero-Cost Telemetry Hook
    def hft_telemetry_callback(study, trial):
        max_score = trial.user_attrs.get("max_score", 0)
        print(f"🏁 Trial {trial.number:03d} | Survived: {trial.value:5.0f} steps | MAX SCORE: {max_score:5.0f} | "
              f"Pulse: {trial.params.get('dopamine_pulse')}, Reward: {trial.params.get('dopamine_reward')}")

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    print("🚀 Starting Zero-Downtime Bayesian Search...")
    
    try:
        # Гоняем 200 триалов
        study.optimize(objective, n_trials=200, callbacks=[hft_telemetry_callback])
        print("\n🏆 Best Hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        print(f"🏅 Best Score: {study.best_value:.1f}")
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")