#!/usr/bin/env python3
import sys
import os
import numpy as np
import gymnasium as gym

if not (sys.prefix != sys.base_prefix or 'VIRTUAL_ENV' in os.environ):
    print("❌ ERROR: Virtual environment not active!")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "genesis-client")))
from genesis.client import GenesisMultiClient
from genesis.encoders import PopulationEncoder
from genesis.brain import fnv1a_32
from genesis.control import GenesisControl
from genesis.tuner import GenesisAutoTuner
from genesis.memory import GenesisMemory
from genesis.surgeon import GenesisSurgeon

# ============================================================
# CONFIGURATION (All tunable parameters in one place)
# ============================================================
BATCH_SIZE = 80                  # 80 ticks = 8ms cycle (fits in one UDP packet)
HEADER_OFFSET = 20               # C-ABI ExternalIoHeader size

# Dopamine / R-STDP
DOPAMINE_BASELINE = -2           # Resting dopamine (slight depression to prune noise)
PUNISHMENT_FULL = 255            # Death signal strength (max LTD)
PUNISHMENT_SOFT = -128            # Timeout signal (half LTD)
PUNISHMENT_BATCHES = -255          # Number of LTD batches on death
REWARD_BATCHES = 80              # Number of reward batches on success

# Physics
ACTION_SCALE = 0.05               # Empirical: raw spikes -> Mujoco torque range
TARGET_SCORE = 50_000
TARGET_TIME = 10_000

# VRAM Layout: Node sends ALL zone outputs aggregated.
# MotorCortex: motor_out(34x8=272) + motor_to_proprio(12x12=144) = 416 pixels total.
TOTAL_ZONE_OUTPUT_PIXELS = 416
MOTOR_OUT_PIXELS = 272           # Only first 272 are motor_out
EXPECTED_RX_BYTES = TOTAL_ZONE_OUTPUT_PIXELS * BATCH_SIZE  # 33280 bytes

# Encoder
NUM_SENSORS = 384                # 376 obs + 8 padding
NEURONS_PER_VAR = 16
SIGMA = 0.15

# Diagnostics
DIAG_INTERVAL = 200              # Print diagnostics every N steps

def run_humanoid():
    env = gym.make("Humanoid-v4", render_mode="human", max_episode_steps=TARGET_TIME if TARGET_TIME > 0 else 1000)
    state, _ = env.reset()

    # Hashing
    zone_hash = fnv1a_32(b"SensoryCortex")
    matrix_hash = fnv1a_32(b"humanoid_sensors")
    motor_rx_hash = fnv1a_32(b"motor_out")

    # [ANT PATTERN] ONE packet. 384 * 16 = 6144 axons → 61,440 bytes payload.
    padded_sensors = NUM_SENSORS * NEURONS_PER_VAR  # 6144
    input_payload_size = (padded_sensors * BATCH_SIZE) // 8
    
    client = GenesisMultiClient(
        addr=("127.0.0.1", 8081),
        matrices=[{'zone_hash': zone_hash, 'matrix_hash': matrix_hash, 'payload_size': input_payload_size}]
    )
    
    try:
        client.sock.bind(("0.0.0.0", 8092))
    except OSError:
        print("❌ Port 8092 already in use!")
        sys.exit(1)
    
    encoder = PopulationEncoder(variables_count=NUM_SENSORS, neurons_per_var=NEURONS_PER_VAR, 
                                batch_size=BATCH_SIZE, sigma=SIGMA)
    
    # AutoTuner
    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Genesis-Models", "HumanoidAgent", "baked", "ThoracicGanglion", "manifest.toml"))
    control = GenesisControl(manifest_path) if os.path.exists(manifest_path) else None
    tuner = GenesisAutoTuner(control=control, target_score=6000.0, window_size=15) if control else None

    # --- AUTO-BOUNDS from Gymnasium observation_space ---
    obs_low_raw = env.observation_space.low.astype(np.float32)
    obs_high_raw = env.observation_space.high.astype(np.float32)
    # Clip infinities to reasonable range
    obs_low_clip = np.clip(obs_low_raw, -200.0, 200.0)
    obs_high_clip = np.clip(obs_high_raw, -200.0, 200.0)
    
    num_obs = len(obs_low_clip)  # 376 for Humanoid-v4
    bounds_low = np.zeros(NUM_SENSORS, dtype=np.float16)
    bounds_high = np.ones(NUM_SENSORS, dtype=np.float16)
    bounds_low[:num_obs] = obs_low_clip.astype(np.float16)
    bounds_high[:num_obs] = obs_high_clip.astype(np.float16)
    
    range_diff = (bounds_high - bounds_low).astype(np.float16)
    # Prevent division by zero for constant observations
    range_diff[range_diff == 0] = np.float16(1.0)

    # --- VRAM DIAGNOSTICS FUNCTION ---
    zone_names = ["SensoryCortex", "ProprioceptiveHub", "ThoracicGanglion", "CerebellumAnalog", "MotorCortex"]
    def print_vram_stats(label=""):
        print(f"\n📊 VRAM Network Stats {label}")
        print(f"  {'Zone':<22} {'Neurons':>8} {'Synapses':>12} {'Avg|W|':>10} {'Max|W|':>10}")
        print(f"  {'─'*22} {'─'*8} {'─'*12} {'─'*10} {'─'*10}")
        for zname in zone_names:
            try:
                zh = fnv1a_32(zname.encode())
                mem = GenesisMemory(zh, read_only=True)
                stats = mem.get_network_stats()
                print(f"  {zname:<22} {mem.padded_n:>8,} {stats['active_synapses']:>12,} "
                      f"{stats['avg_weight']:>10.1f} {stats['max_weight']:>10,}")
                mem.close()
            except Exception as e:
                print(f"  {zname:<22} ❌ {e}")
        print()
    
    print_vram_stats("(Startup)")

    # --- ZERO-ALLOCATION PREALLOCATION ---
    norm_state = np.zeros(NUM_SENSORS, dtype=np.float16)
    total_motor = np.zeros(34, dtype=np.float32)
    action = np.zeros(17, dtype=np.float32)

    score, steps = 0.0, 0
    episodes = 0
    terminated, truncated = False, False

    print(f"🚀 Bipedal HFT Loop Started (Batch={BATCH_SIZE}, Freq={1000//(BATCH_SIZE//10)}Hz, "
          f"ActionScale={ACTION_SCALE:.6f}, Ant Pattern Lockstep)...")

    try:
        while episodes < 1_000_000:
            # [ANT PATTERN] Check termination FIRST
            time_reached = (TARGET_TIME > 0 and steps >= TARGET_TIME)
            score_reached = (TARGET_SCORE >= 1000 and score >= TARGET_SCORE)
            
            if terminated or truncated or time_reached or score_reached:
                # Differentiated death/reward signals
                if terminated:
                    # Full punishment — fell down
                    for _ in range(PUNISHMENT_BATCHES):
                        client.step(PUNISHMENT_FULL)
                elif truncated or time_reached:
                    # Soft punishment — ran out of time but didn't fall
                    for _ in range(5):
                        client.step(PUNISHMENT_SOFT)
                elif score_reached:
                    # Reward — excellent performance
                    for _ in range(REWARD_BATCHES):
                        client.step(0)
                
                if terminated: reason = "💀 Fell"
                elif time_reached: reason = "⏰ Time"
                elif score_reached: reason = "🏆 Score"
                else: reason = "📋 Truncated"
                
                print(f"[Ep {episodes:04d}] {reason} | Score: {score:7.1f} | Steps: {steps:5d}")
                
                if tuner: tuner.step(score)
                episodes += 1
                
                # Periodic VRAM stats every 15 episodes
                if episodes % 15 == 0:
                    print_vram_stats(f"(Ep {episodes})")
                
                state, _ = env.reset()
                score, steps = 0.0, 0
                terminated, truncated = False, False
                continue

            # 1. State Normalization (Zero-Allocation, In-Place)
            np.subtract(state[:num_obs], bounds_low[:num_obs], out=norm_state[:num_obs])
            np.divide(norm_state[:num_obs], range_diff[:num_obs], out=norm_state[:num_obs])
            np.clip(norm_state[:num_obs], 0.0, 1.0, out=norm_state[:num_obs])
            
            # 2. [ANT PATTERN] Encode into the SINGLE tx_packet
            encoder.encode_into(norm_state, client._tx_packets[0], offset=HEADER_OFFSET)
            
            # 3. Dynamic Dopamine (balance + velocity reward)
            height = state.item(0)                           # Torso height
            forward_vel = state.item(22) if num_obs > 22 else 0.0  # Forward velocity
            
            balance_signal = np.clip(height * 50.0, -128, 127)
            velocity_signal = np.clip(forward_vel * 30.0, -128, 127)
            dopamine = int(np.clip(balance_signal + velocity_signal, -128, 127))
            
            # 4. [ANT PATTERN] Atomic send→recv. ONE call. Perfect lockstep.
            rx = client.step(dopamine, expected_rx_hash=motor_rx_hash)
            
            if len(rx) != EXPECTED_RX_BYTES:
                # Mismatch — step with zero action to keep physics running
                state, reward, terminated, truncated, _ = env.step(action)
                score += reward
                steps += 1
                continue
            
            # 5. Zero-Copy Z-Integration Decoding (34 Muscles x 8 Pyramid Layers)
            raw_bytes = np.frombuffer(rx, dtype=np.uint8)
            zone_output = raw_bytes.reshape(BATCH_SIZE, TOTAL_ZONE_OUTPUT_PIXELS)
            motor_output = zone_output[:, :MOTOR_OUT_PIXELS].reshape(BATCH_SIZE, 8, 34)
            
            # Integrate activity across time and layers
            motor_output.sum(axis=(0, 1), out=total_motor)
            
            # action = (agonist - antagonist) * physics_scale
            np.subtract(total_motor[0::2], total_motor[1::2], out=action)
            action *= ACTION_SCALE
            
            # 6. Physical Step
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            steps += 1
            
            # 7. Diagnostics
            if steps % DIAG_INTERVAL == 0:
                max_act = np.max(np.abs(action))
                dom_joint = np.argmax(np.abs(action))
                spikes = int(np.sum(raw_bytes[:MOTOR_OUT_PIXELS * BATCH_SIZE]))
                print(f"  Step {steps:5d} | Score: {score:7.1f} | "
                      f"H: {state.item(0):.2f} | MaxAct: {max_act:.4f} | "
                      f"Joint: {dom_joint:2d} | Spikes: {spikes:6d} | "
                      f"Dopa: {dopamine:4d}")
                    
    finally:
        env.close()

if __name__ == '__main__':
    run_humanoid()