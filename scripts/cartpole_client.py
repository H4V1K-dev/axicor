import numpy as np
import gymnasium as gym
import sys
import os

DOPAMINE_REWARD = 10
DEATH_PENALTY = -255

# Добавляем путь к genesis-client если не установлен системно
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "genesis-client")))

import genesis

# --- Хэширование для C-ABI UDP ---
def fnv1a_32(data: bytes) -> int:
    hash_val = 0x811c9dc5
    for b in data:
        hash_val ^= b
        hash_val = (hash_val * 0x01000193) & 0xFFFFFFFF
    return hash_val

ZONE_SENSORY = fnv1a_32(b"SensoryCortex")
MATRIX_SENSORS = fnv1a_32(b"cartpole_sensors")

# 4 переменные среды * 16 нейронов = 64 виртуальных аксона = 8 байт
PAYLOAD_SIZE = 8

# --- DOD Population Coding Arrays ---
# Преаллокация границ для векторизованного вычисления без циклов for
BOUNDS = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.41, 0.41], [-2.0, 2.0]], dtype=np.float32)
CENTERS = np.linspace(0, 1, 16, dtype=np.float32)

def encode_state_in_place(state: np.ndarray, out_view: np.ndarray):
    """
    Zero-Loop кодирование 4-х float в 64-битную маску спайков.
    Использует Гауссовы рецептивные поля (tuning width σ = 0.15).
    """
    # Нормализация в диапазон [0.0, 1.0]
    norm = np.clip((state - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0]), 0.0, 1.0)
    # Вычисление квадрата дистанции до 16 центров (Shape: 4 x 16)
    dist = norm[:, None] - CENTERS
    prob = np.exp(-(dist * dist) / (2.0 * 0.15 * 0.15))
    # Запись битов напрямую в память пакета
    packed = np.packbits((prob > 0.5).ravel(), bitorder='little')
    out_view[:] = packed

def sum_spikes_wta(memory_view) -> tuple[int, int]:
    """
    O(1) парсинг C-ABI матрицы Output_History [tick][channel].
    Моторы: 128 каналов (0..63 = Left, 64..127 = Right).
    """
    if len(memory_view) == 0:
        return 0, 0
    # Натягиваем NumPy без копирования
    arr = np.frombuffer(memory_view, dtype=np.uint8).reshape(-1, 128)
    return int(arr[:, :64].sum()), int(arr[:, 64:].sum())

def main():
    print("🚀 Booting Genesis E2E CartPole Client...")

    # 1. Инициализация HFT-Транспорта (Data Plane)
    client = genesis.GenesisMultiClient(
        addr=("127.0.0.1", 8081),
        matrices=[{'zone_hash': ZONE_SENSORY, 'matrix_hash': MATRIX_SENSORS, 'payload_size': PAYLOAD_SIZE}]
    )
    # ЖЕСТКАЯ ПРИВЯЗКА К ПОРТУ ОТВЕТОВ: Нода шлет GSOO пакеты на этот порт
    try:
        client.sock.bind(("0.0.0.0", 8092))
    except OSError as e:
        print(f"⚠️  Could not bind to port 8092: {e}. If the node is not running, this is fine.")
        
    input_view = client.payload_views[0] # Исправлено: payload_views это список

    # 2. Инициализация Управления (Control Plane)
    try:
        manifest_path = "examples/cartpole/baked/SensoryCortex/manifest.toml"
        ctrl = genesis.GenesisControl(manifest_path)
        # Инициализируем AutoTuner (цель: 500 очков)
        tuner = genesis.GenesisAutoTuner(ctrl, target_score=500, window_size=15)
        # Подключаем Memory Plane для аналитики
        memory = genesis.GenesisMemory(ZONE_SENSORY)
    except FileNotFoundError:
        print("⚠️ manifest.toml не найден. Control Plane отключен.")
        ctrl = None
        tuner = None
        memory = None

    # 3. RL Среда
    env = gym.make("CartPole-v1", render_mode="human" if "--render" in sys.argv else None)
    state, _ = env.reset()

    score = 0
    episodes = 0
    is_inferencing = False

    # ctrl.set_prune_threshold(150)   # 150 - это порог для "выжигания" синапсов

    print("🧠 Connected. Beginning Lockstep Loop...")
    try:
        while True:
            # --- ИНВАРИАНТ: SINGLE-TICK PULSE ---
            # Батч 1: Впрыск состояния среды + Дофамин за выживание
            reward = DOPAMINE_REWARD if score > 0 else 0
            encode_state_in_place(state, input_view)
            out1 = client.step(reward=reward)

            # Батч 2 & 3: Тишина (Продвижение волны по коннектому)
            # Если держать сенсоры "вкл", сеть уйдет в эпилепсию и гомеостаз ослепит её
            input_view.fill(0)
            out2 = client.step(reward=0)
            out3 = client.step(reward=0)

            # --- POPULATION DECODING (Winner-Takes-All) ---
            # Собираем активность за всё окно (3 батча)
            l1, r1 = sum_spikes_wta(out1)
            l2, r2 = sum_spikes_wta(out2)
            l3, r3 = sum_spikes_wta(out3)

            total_left = l1 + l2 + l3
            total_right = r1 + r2 + r3

            # Эксплорация: если сеть молчит, делаем случайное движение
            if total_left == 0 and total_right == 0:
                action = env.action_space.sample()
            else:
                action = 0 if total_left > total_right else 1

            # --- ЭВОЛЮЦИЯ СРЕДЫ ---
            state, env_reward, terminated, truncated, _ = env.step(action)
            score += 1

            if terminated or truncated:
                episodes += 1
                
                # Передаем скор в тюнер и получаем текущую фазу
                current_phase = tuner.step(score) if tuner else "N/A"
                phase_name = current_phase.name if tuner else "MANUAL"
                
                # Читаем аналитику графа напрямую из VRAM (Zero-Copy)
                stats = memory.get_network_stats() if memory else {"active_synapses": 0, "avg_weight": 0.0}
                synapses = stats["active_synapses"]
                avg_w = stats["avg_weight"]

                print(f"Ep {episodes:04d} | Score: {score:3d} | Spikes L/R: {total_left:3d}/{total_right:3d} | Phase: {phase_name:<12} | Synapses: {synapses:5d} | Avg W: {avg_w:.1f}")

                # --- ИНВАРИАНТ: THE DEATH SIGNAL ---
                # Выжигаем (LTD) синапсы, приведшие к падению (только если мы не в инференсе)
                if tuner and current_phase != genesis.Phase.CRYSTALLIZED:
                    client.step(reward= DEATH_PENALTY)

                state, _ = env.reset()
                score = 0

    except KeyboardInterrupt:
        print("\nHalting Lockstep...")
        env.close()

if __name__ == "__main__":
    main()
