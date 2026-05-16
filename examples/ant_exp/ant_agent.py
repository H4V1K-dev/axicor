#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import gymnasium as gym

# Добавляем путь к SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "axicor-client")))

from axicor.client import AxicorMultiClient
from axicor.encoders import PopulationEncoder
from axicor.decoders import PwmDecoder
from axicor.contract import AxicorIoContract
from axicor.brain import fnv1a_32

# CONFIG
EPISODES = 20_000_000
BATCH_SIZE = 20
DOPAMINE_REWARD = -2
DOPAMINE_PUNISHMENT = -255      # Death Signal
TARGET_SCORE = 50_000           # 50k score
TARGET_TIME = 10_000            # 10k steps

def run_ant():
    env = gym.make("Ant-v4", render_mode="human", max_episode_steps=TARGET_TIME if TARGET_TIME > 0 else 1000)
    state, _ = env.reset()

    # --- 1. Load Zero-Cost Contracts ---
    print("Loading C-ABI Contracts for 6 zones...")
    model_path = "Axicor-Models/AntConnectome.axic"

    c_sensory = AxicorIoContract(model_path, "SensoryCortex")
    c_fl = AxicorIoContract(model_path, "FL_Ganglion")
    c_fr = AxicorIoContract(model_path, "FR_Ganglion")
    c_bl = AxicorIoContract(model_path, "BL_Ganglion")
    c_br = AxicorIoContract(model_path, "BR_Ganglion")

    # Build combined layout
    cfg_sens = c_sensory.get_client_config(BATCH_SIZE)
    cfg_fl = c_fl.get_client_config(BATCH_SIZE)
    cfg_fr = c_fr.get_client_config(BATCH_SIZE)
    cfg_bl = c_bl.get_client_config(BATCH_SIZE)
    cfg_br = c_br.get_client_config(BATCH_SIZE)

    matrices = cfg_sens["matrices"] # Only Sensory has external inputs
    rx_layout = (cfg_fl["rx_layout"] + cfg_fr["rx_layout"] +
                 cfg_bl["rx_layout"] + cfg_br["rx_layout"])

    # HFT Transport
    client = AxicorMultiClient(
        addr=("127.0.0.1", 8081),
        matrices=matrices,
        rx_layout=rx_layout
    )

    try:
        client.sock.bind(("0.0.0.0", 8092))
    except OSError:
        pass

    # --- 2. Zero-Garbage Encoders/Decoders ---
    encoder = c_sensory.create_population_encoder("ant_sensors", vars_count=27, batch_size=BATCH_SIZE)

    dec_fl = c_fl.create_pwm_decoder("motor_out_FL", BATCH_SIZE)
    dec_fr = c_fr.create_pwm_decoder("motor_out_FR", BATCH_SIZE)
    dec_bl = c_bl.create_pwm_decoder("motor_out_BL", BATCH_SIZE)
    dec_br = c_br.create_pwm_decoder("motor_out_BR", BATCH_SIZE)

    # --- MEMORY PREALLOCATION (Hot Loop Safety) ---
    action = np.zeros(8, dtype=np.float32)
    norm_state = np.zeros(27, dtype=np.float16)

    bounds = np.tile([-5.0, 5.0], (27, 1)).astype(np.float16)
    range_diff = bounds[:, 1] - bounds[:, 0]

    episodes, score, steps = 0, 0.0, 0
    terminated, truncated = False, False

    print(f"🚀 Thalamo-Cortical-Spinal Hot Loop Started (Batch={BATCH_SIZE})...")

    try:
        while episodes < EPISODES:
            time_reached = (TARGET_TIME > 0 and steps >= TARGET_TIME)
            score_reached = (TARGET_SCORE >= 1000 and score >= TARGET_SCORE)

            if terminated or truncated or time_reached or score_reached:
                # Death Signal: 15 LTD Batches
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
            np.subtract(state, bounds[:, 0], out=norm_state, dtype=np.float16)
            np.divide(norm_state, range_diff, out=norm_state, dtype=np.float16)
            np.clip(norm_state, 0.0, 1.0, out=norm_state)

            # 2. Encode to Network Buffer
            encoder.encode_into(norm_state, client.payload_views[0])

            # 3. Axicor Step (Network I/O)
            rx_raw = client.step(DOPAMINE_REWARD)
            rx_view = memoryview(rx_raw)

            # 4. Zero-Copy Decoding for each Leg
            off = 0
            fl_out = dec_fl.decode_from(rx_view[off : off + dec_fl.payload_size]); off += dec_fl.payload_size
            fr_out = dec_fr.decode_from(rx_view[off : off + dec_fr.payload_size]); off += dec_fr.payload_size
            bl_out = dec_bl.decode_from(rx_view[off : off + dec_bl.payload_size]); off += dec_bl.payload_size
            br_out = dec_br.decode_from(rx_view[off : off + dec_br.payload_size]); off += dec_br.payload_size

            # 5. Fast Response Resolution (Biceps/Triceps per joint)
            # Ant-v4 Action Space:
            # 0: Hip FL, 1: Ankle FL
            action[0] = np.sum(fl_out[0:8]) - np.sum(fl_out[8:16])
            action[1] = np.sum(fl_out[16:24]) - np.sum(fl_out[24:32])

            # 2: Hip FR, 3: Ankle FR
            action[2] = np.sum(fr_out[0:8]) - np.sum(fr_out[8:16])
            action[3] = np.sum(fr_out[16:24]) - np.sum(fr_out[24:32])

            # 4: Hip BL, 5: Ankle BL
            action[4] = np.sum(bl_out[0:8]) - np.sum(bl_out[8:16])
            action[5] = np.sum(bl_out[16:24]) - np.sum(bl_out[24:32])

            # 6: Hip BR, 7: Ankle BR
            action[6] = np.sum(br_out[0:8]) - np.sum(br_out[8:16])
            action[7] = np.sum(br_out[16:24]) - np.sum(br_out[24:32])

            # 6. Physical Step
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            steps += 1

    finally:
        env.close()

if __name__ == '__main__':
    run_ant()