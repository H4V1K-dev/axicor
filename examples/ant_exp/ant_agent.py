#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import gymnasium as gym

# Добавляем путь к SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "genesis-client")))

from genesis.client import GenesisMultiClient
from genesis.encoders import PopulationEncoder
from genesis.brain import fnv1a_32

# CONFIG
EPISODES = 20_000_000
BATCH_SIZE = 20
DOPAMINE_REWARD = -2
DOPAMINE_PUNISHMENT = 255       # Death Signal
TARGET_SCORE = 50_000           # 50k score
TARGET_TIME = 10_000            # 10k steps

def run_ant():
    env = gym.make("Ant-v4", render_mode="human", max_episode_steps=TARGET_TIME if TARGET_TIME > 0 else 1000)
    state, _ = env.reset()
    
    # Hashing
    zone_hash = fnv1a_32(b"SensoryCortex")
    matrix_hash = fnv1a_32(b"ant_sensors")
    
    # [DOD FIX] Stride Alignment: 28 vars * 16 neurons = 448 pixels
    padded_sensors = 448
    input_payload_size = (padded_sensors * BATCH_SIZE) // 8
    
    # HFT Transport
    client = GenesisMultiClient(
        addr=("127.0.0.1", 8081),
        matrices=[{'zone_hash': zone_hash, 'matrix_hash': matrix_hash, 'payload_size': input_payload_size}]
    )
    
    try:
        client.sock.bind(("0.0.0.0", 8092))
    except OSError:
        sys.exit(1)
    
    # Encoder
    encoder = PopulationEncoder(variables_count=27, neurons_per_var=16, batch_size=BATCH_SIZE)
    
    # [DOD FIX] RX Buffer Calibration (MotorCortex only)
    TOTAL_OUTPUT_PIXELS = 128  # motor_out 16x8
    EXPECTED_RX_BYTES = TOTAL_OUTPUT_PIXELS * BATCH_SIZE
    
    # --- MEMORY PREALLOCATION (Hot Loop Safety) ---
    total_motor = np.zeros(128, dtype=np.float16)
    action = np.empty(8, dtype=np.float32)
    inv_b = np.float16(1.0 / BATCH_SIZE)
    
    # Bounds for fast normalization
    bounds = np.tile([-5.0, 5.0], (27, 1)).astype(np.float16)
    range_diff = bounds[:, 1] - bounds[:, 0]
    
    episodes, score, steps = 0, 0.0, 0
    terminated, truncated = False, False
    
    print(f"🚀 Minimalist DOD Hot Loop Started (Batch={BATCH_SIZE}, TargetScore={TARGET_SCORE}, TargetTime={TARGET_TIME})...")
    
    try:
        while episodes < EPISODES:
            # Check for Target Time/Score Termination
            time_reached = (TARGET_TIME > 0 and steps >= TARGET_TIME)
            score_reached = (TARGET_SCORE >= 1000 and score >= TARGET_SCORE)
            
            if terminated or truncated or time_reached or score_reached:
                # [DOD FIX] Death Signal: 15 LTD Batches
                for _ in range(15):
                    client.step(DOPAMINE_PUNISHMENT)
                
                if time_reached: reason = "Time Reached"
                elif score_reached: reason = "Score Reached"
                else: reason = "Terminated"
                
                print(f"Ep {episodes:04d} | Score: {score:6.1f} | Steps: {steps:5d} | [{reason}]")
                
                episodes += 1
                state, _ = env.reset()
                score = 0.0
                steps = 0
                terminated, truncated = False, False
                continue

            # 1. Zero-Allocation Normalization
            norm_state = np.clip((state - bounds[:, 0]) / range_diff, 0.0, 1.0).astype(np.float16)
            
            # 2. Encoder with C-ABI Header Offset (20 bytes)
            # [DOD FIX] Передаем _tx_packets[0], чтобы offset=20 не выходил за границы payload
            encoder.encode_into(norm_state, client._tx_packets[0], offset=20)
            
            # 3. Genesis Step
            rx = client.step(DOPAMINE_REWARD, expected_rx_hash=fnv1a_32(b"motor_out"))
            
            if len(rx) != EXPECTED_RX_BYTES:
                print(f"FATAL: RX mismatch {len(rx)} vs {EXPECTED_RX_BYTES}")
                break
                
            # 4. Zero-Copy Decoding
            raw_bytes = np.frombuffer(rx, dtype=np.uint8)
            spikes_2d = raw_bytes.reshape((BATCH_SIZE, TOTAL_OUTPUT_PIXELS))
            
            # 5. Fast Response Resolution
            np.sum(spikes_2d, axis=0, dtype=np.float16, out=total_motor)
            np.multiply(total_motor, inv_b, out=total_motor)
            
            # Biceps/Triceps resolution
            motor_view = total_motor.reshape(8, 16)
            action[:] = np.sum(motor_view[:, :8], axis=1) - np.sum(motor_view[:, 8:], axis=1)
            
            # 6. Physical Step
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            steps += 1
                    
    finally:
        env.close()

if __name__ == '__main__':
    run_ant()