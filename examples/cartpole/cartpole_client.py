#!/usr/bin/env python3
"""
CartPole ↔ Genesis Neural Node Client
Population Coding: 4 float → 64 virtual axons (Gaussian Receptive Fields)
Motor Readout:     popcount(motor_left) vs popcount(motor_right) → WTA
"""
import socket
import struct
import gymnasium as gym
import numpy as np
import math

# ─── Memory Contracts (Spec 08) ──────────────────────────────────
GSIO_MAGIC  = 0x4F495347  # "GSIO"
GSOO_MAGIC  = 0x4F4F5347  # "GSOO"
# magic(4) + zone_hash(4) + matrix_hash(4) + payload_size(4) + global_reward(2) + _pad(2) = 20 bytes
HEADER_FMT  = "<IIIIhH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 20

GENESIS_IP  = "127.0.0.1"
PORT_OUT    = 8081   # Node receives input here
PORT_IN     = 8092   # Node sends output here (MotorCortex)
TELEMETRY_PORT = 8100 # Dashboard listens here
BATCH_TICKS = 100
NUM_NEURONS = 64     # neurons per variable


def fnv1a_32(data: bytes) -> int:
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


ZONE_IN  = fnv1a_32(b"SensoryCortex")
MAT_IN   = fnv1a_32(b"env_sensors")
ZONE_OUT = fnv1a_32(b"MotorCortex")


# ─── Population Coding ───────────────────────────────────────────
def encode_population(value: float, min_val: float, max_val: float,
                      n: int = NUM_NEURONS, sigma: int = 2) -> int:
    """
    Float → N-bit bitmask via Gaussian Receptive Field.
    sigma=2 → center±2 (5 neurons active), smoother encoding.
    """
    norm = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    center = int(norm * (n - 1))
    mask = 0
    for i in range(max(0, center - sigma), min(n, center + sigma + 1)):
        mask |= (1 << i)
    return mask


def build_input_batch(cart_x: float, cart_v: float,
                      pole_a: float, pole_av: float) -> bytes:
    """
    Encode 4 variables → 64 bits = 2×u32 words.
    Tile the same bitmask across all BATCH_TICKS ticks.
    Returns DMA-ready bytes: BATCH_TICKS × 8 bytes = 800 bytes.
    """
    v0 = encode_population(cart_x,  -2.4, 2.4)
    v1 = encode_population(cart_v,  -3.0, 3.0)
    v2 = encode_population(pole_a,  -0.41, 0.41)
    v3 = encode_population(pole_av, -2.0, 2.0)

    word0 = (v0 & 0xFFFF) | ((v1 & 0xFFFF) << 16)
    word1 = (v2 & 0xFFFF) | ((v3 & 0xFFFF) << 16)

    # One tick = 8 bytes. Tile × BATCH_TICKS without Python loop.
    tick = struct.pack("<II", word0, word1)
    return tick * BATCH_TICKS  # 800 bytes total


def decode_output(payload: bytes) -> tuple[int, int]:
    """
    Parse 12800 bytes (BATCH_TICKS × 128 neurons) → (left_spikes, right_spikes).
    numpy.sum over C-contiguous array — no Python loop.
    """
    spikes = np.frombuffer(payload, dtype=np.uint8).reshape((BATCH_TICKS, 128))
    total  = np.sum(spikes, axis=0)   # shape: (128,)
    return int(np.sum(total[0:64])), int(np.sum(total[64:128]))


def main() -> None:
    env  = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT_IN))
    sock.settimeout(0.5)

    print("🧠 [Genesis] CartPole I/O Loop Started")
    print(f"   TX → {GENESIS_IP}:{PORT_OUT}  |  RX ← 0.0.0.0:{PORT_IN}")
    print(f"   Batch: {BATCH_TICKS} ticks | Input: 256 virt-axons | Output: 64 neurons\n")

    action       = 0
    episode      = 1
    total_reward = 0.0
    left_power   = 0
    right_power  = 0

    import time
    start_time = time.time()
    total_ticks = 0

    while True:
        # ... stay in loop ...
        # (Actually I need to place this inside the loop properly)
        # ── 1. Environment step ──────────────────────────────────
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        cart_x, cart_v, pole_a, pole_av = obs

        # ── 2. Dopamine signal (R-STDP steering) ─────────────────
        # Upright pole → positive. Terminal → softer depression (was -30000).
        # Added velocity penalty to discourage fast swings.
        if terminated:
            dopamine = -20000
        else:
            dopamine = int((0.03 - abs(pole_a)) * 25000 - abs(pole_av) * 5000)
        dopamine = max(-32768, min(32767, dopamine))

        # ── 3. Encode & send ─────────────────────────────────────
        # 4 variables * 64 bits = 256 bits = 32 bytes per tick
        v0 = encode_population(cart_x,  -2.4, 2.4, n=64)
        v1 = encode_population(cart_v,  -3.0, 3.0, n=64)
        v2 = encode_population(pole_a,  -0.41, 0.41, n=64)
        v3 = encode_population(pole_av, -2.0, 2.0, n=64)
        
        tick = struct.pack("<QQQQ", v0, v1, v2, v3)
        payload = tick * BATCH_TICKS  # 3200 bytes total

        header  = struct.pack(HEADER_FMT,
                              GSIO_MAGIC, ZONE_IN, MAT_IN,
                              len(payload), dopamine, 0)
        sock.sendto(header + payload, (GENESIS_IP, PORT_OUT))

        # ── 4. Receive motor readout ─────────────────────────────
        try:
            data, _ = sock.recvfrom(65535)
            if len(data) >= HEADER_SIZE:
                magic, z_hash, _, p_size, _, _ = struct.unpack(
                    HEADER_FMT, data[:HEADER_SIZE])

                if magic == GSOO_MAGIC and z_hash == ZONE_OUT:
                    out_payload = data[HEADER_SIZE : HEADER_SIZE + p_size]
                    if len(out_payload) == BATCH_TICKS * 64:
                        # Decode 64 neurons (1 byte per neuron)
                        spikes = np.frombuffer(out_payload, dtype=np.uint8).reshape((BATCH_TICKS, 64))
                        total  = np.sum(spikes, axis=0) # shape: (64,)
                        left_power  = int(np.sum(total[0:32]))
                        right_power = int(np.sum(total[32:64]))
                        if left_power > right_power:
                            action = 0
                        elif right_power > left_power:
                            action = 1
                        # If left_power == right_power, action remains unchanged (persistence)

        except socket.timeout:
            print("⚠️  [Genesis] RX Timeout — waiting for node...")

        # ── 5. Episode reset ─────────────────────────────────────
        total_ticks += BATCH_TICKS
        elapsed = time.time() - start_time
        tps = total_ticks / elapsed if elapsed > 0 else 0

        # ── 5. Live Telemetry (Heartbeat) ────────────────────────
        tps = total_ticks / elapsed if elapsed > 0 else 0
        try:
            # Payload: Episode(I), Score(f), TPS(f), IsDone(f)
            is_done = 1.0 if (terminated or truncated) else 0.0
            tel_msg = struct.pack("<Ifff", episode, total_reward, tps, is_done)
            sock.sendto(tel_msg, ("127.0.0.1", TELEMETRY_PORT))
        except: pass

        # ── 6. Episode reset logic ───────────────────────────────
        if terminated or truncated:
            print(f"🔄  Episode {episode:4d} | Score: {int(total_reward):5d} | "
                  f"L={left_power:6d}  R={right_power:6d} | "
                  f"dopamine={dopamine:+7d} | TPS: {int(tps)}")
            
            obs, _ = env.reset()
            episode      += 1
            total_reward  = 0.0


if __name__ == "__main__":
    main()
