#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import json
import os
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import toml

# Добавляем путь к SDK ( genesis-client/ ) если скрипт запущен напрямую из примера
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "genesis-client")))

from genesis.brain import fnv1a_32
from genesis.client import GenesisMultiClient
from genesis.control import GenesisControl
from genesis.decoders import PwmDecoder
from genesis.encoders import PopulationEncoder
from genesis.memory import GenesisMemory
from genesis.tuner import GenesisAutoTuner, Phase


class CartPole3ActionWrapper(gym.Wrapper):
    """
    Wraps CartPole-v1 to support 3 actions: 0=left, 1=wait (force=0), 2=right.
    CartPole natively has 2 actions; wait is simulated by advancing physics with force=0.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def step(self, action: int):
        if action == 0:
            return self.env.step(0)
        if action == 2:
            return self.env.step(1)
        # action == 1: wait (force=0)
        base = self.env.unwrapped
        x, x_dot, theta, theta_dot = base.state
        force = 0.0
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + base.polemass_length * theta_dot**2 * sintheta) / base.total_mass
        thetaacc = (base.gravity * sintheta - costheta * temp) / (
            base.length * (4.0 / 3.0 - base.masspole * costheta**2 / base.total_mass)
        )
        xacc = temp - base.polemass_length * thetaacc * costheta / base.total_mass
        if base.kinematics_integrator == "euler":
            x = x + base.tau * x_dot
            x_dot = x_dot + base.tau * xacc
            theta = theta + base.tau * theta_dot
            theta_dot = theta_dot + base.tau * thetaacc
        else:
            x_dot = x_dot + base.tau * xacc
            x = x + base.tau * x_dot
            theta_dot = theta_dot + base.tau * thetaacc
            theta = theta + base.tau * theta_dot
        base.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)
        terminated = bool(
            x < -base.x_threshold or x > base.x_threshold
            or theta < -base.theta_threshold_radians or theta > base.theta_threshold_radians
        )
        reward = 1.0 if not terminated else (1.0 if base.steps_beyond_terminated is None else 0.0)
        if terminated and base.steps_beyond_terminated is None:
            base.steps_beyond_terminated = 0
        elif terminated:
            base.steps_beyond_terminated += 1
        truncated = False
        if hasattr(self.env, "_elapsed_steps") and hasattr(self.env, "_max_episode_steps"):
            self.env._elapsed_steps = (self.env._elapsed_steps or 0) + 1
            if self.env._elapsed_steps >= self.env._max_episode_steps:
                truncated = True
        obs = np.array(base.state, dtype=np.float32)
        if getattr(base, "render_mode", None) == "human":
            base.render()
        return obs, reward, terminated, truncated, {}


def ensure_virtualenv() -> None:
    if sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ:
        return
    print("ERROR: Virtual environment not active.")
    print("Activate the project venv before running the CartPole agent.")
    sys.exit(1)

EPISODES = 20_000_000
BATCH_SIZE = 20
NIGHT_INTERVAL = 100_000
PRUNE_THRESHOLD = 1
MAX_SPROUTS_PER_NIGHT = 128

DOPAMINE_PULSE = 0
DOPAMINE_REWARD = 10
DOPAMINE_PUNISHMENT = -210

LEAK_RATE = 850
HOMEOS_PENALTY = 5560
HOMEOS_DECAY = 49

ERROR_ANGLE_WEIGHT = 0.8
ERROR_VEL_WEIGHT = 0.2
ANGLE_LIMIT = 0.2094
VELOCITY_LIMIT = 2.0

SHOCK_BASE = 0
SHOCK_SCORE_BITSHIFT = 5
SHOCK_VEL_MULT = 5
SHOCK_MAX_BATCHES = 2

ENCODER_SIGMA = 0.2
TARGET_SCORE = 700

# Action space: 0=left, 1=right (2-action) or 0=left, 1=wait, 2=right (3-action)
MOTOR_SPLIT = 64  # motor_out: neurons [0:64]=left, [64:128]=right
WAIT_BALANCE_THRESHOLD = 0.05  # |left_sum - right_sum| < this → wait (3-action mode)

ADAPTIVE_LEAK_MODE_DISABLED = 0
ADAPTIVE_LEAK_MODE_CONTINUOUS = 1
ADAPTIVE_LEAK_MODE_DISCRETE = 2
MEMBRANE_MODE_STABLE = 0
MEMBRANE_MODE_RESPONSIVE = 1
MEMBRANE_MODE_EXCITED = 2
MEMBRANE_MODE_RECOVERY = 3
RECOVERY_BURST_THRESHOLD = 4
WEIGHT_SATURATION_THRESHOLD = 31_129
MAX_DENDRITES = 128
STATE_FILE_HEADER_SIZE = 16
STATE_FILE_MAGIC = b"GSNS"
GXO_HEADER_SIZE = 12
GXO_DESCRIPTOR_SIZE = 16
MOTOR_OUTPUT_NAME = "motor_out"


@dataclass(frozen=True)
class AdaptiveLeakConfig:
    adaptive_leak_mode: int = 0
    dopamine_leak_gain: int = 0
    burst_leak_gain: int = 0
    leak_min: int = 0
    leak_max: int = 0
    variant_ids: list[int] | None = None


@dataclass(frozen=True)
class CartPoleRunConfig:
    scenario_name: str = "manual"
    scenario_label: str = "Manual CartPole"
    episodes: int = EPISODES
    batch_size: int = BATCH_SIZE
    seed: int = 123
    response_port: int = 8092
    node_addr: str = "127.0.0.1"
    node_port: int = 8081
    manifest_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../Genesis-Models/CartPole-example/baked/SensoryCortex/manifest.toml")
    )
    night_interval: int = NIGHT_INTERVAL
    prune_threshold: int = PRUNE_THRESHOLD
    max_sprouts: int = MAX_SPROUTS_PER_NIGHT
    threshold_score: int = TARGET_SCORE
    use_autotuner: bool = True
    noise_std: float = 0.0
    stats_sample_stride: int = 5
    hot_reload_wait_s: float = 0.5
    output_path: str | None = None
    log_episodes: bool = True
    render: bool = False
    render_delay_s: float = 0.0
    render_fps: float = 60.0
    adaptive_leak: AdaptiveLeakConfig = field(default_factory=AdaptiveLeakConfig)
    use_3_actions: bool = False  # Left(0), Wait(1), Right(2) instead of Left(0), Right(1)
    max_episode_steps: int = 5000  # CartPole episode length limit (default 500 in Gymnasium)
    # Epsilon-greedy в EXPLORATION: высокая случайность на старте, затухание к инференсу
    exploration_epsilon_start: float = 0.8
    exploration_epsilon_end: float = 0.05
    exploration_decay_episodes: int = 2000


@dataclass
class RuntimeSample:
    spike_rate: float
    saturated_weight_share: float
    effective_leak_mean: float
    mean_burst_count: float
    active_synapses: int
    avg_weight: float
    max_weight: int
    structure_metrics_source: str
    mode_counts: dict[str, int]


def compute_padded_n_from_state_blob(file_size: int, header_size: int = 0) -> int:
    bytes_per_neuron = 4 + 1 + 4 + 1 + 4 + MAX_DENDRITES * (4 + 2 + 1)
    payload_size = file_size - header_size
    if payload_size < 0 or payload_size % bytes_per_neuron != 0:
        raise ValueError(
            f"State blob size {file_size} is not aligned to {bytes_per_neuron} bytes per neuron "
            f"after the {header_size}-byte header"
        )
    return payload_size // bytes_per_neuron


def count_variant_ids(flags: np.ndarray) -> dict[int, int]:
    variant_ids = flags >> 4
    return {
        int(variant_id): int(np.sum(variant_ids == variant_id))
        for variant_id in np.unique(variant_ids)
    }


def load_baked_state_payload(manifest_path: str) -> tuple[np.ndarray, int]:
    state_path = os.path.join(os.path.dirname(manifest_path), "shard.state")
    blob = np.fromfile(state_path, dtype=np.uint8)
    if blob.size == 0:
        raise ValueError(f"State blob is empty: {state_path}")

    if blob.size >= STATE_FILE_HEADER_SIZE and bytes(blob[:4]) == STATE_FILE_MAGIC:
        magic, version, padded_n, _total_axons = struct.unpack_from("<4sIII", blob, 0)
        if version != 1:
            raise ValueError(f"Unsupported state version {version} in {state_path}")

        padded_n_from_size = compute_padded_n_from_state_blob(int(blob.size), header_size=STATE_FILE_HEADER_SIZE)
        if padded_n != padded_n_from_size:
            raise ValueError(
                f"State header padded_n={padded_n} does not match payload-derived padded_n={padded_n_from_size}"
            )
        return blob[STATE_FILE_HEADER_SIZE:], int(padded_n)

    padded_n = compute_padded_n_from_state_blob(int(blob.size))
    return blob, int(padded_n)


def default_manifest_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../Genesis-Models/CartPole-example/baked/SensoryCortex/manifest.toml")
    )


def load_manifest(manifest_path: str) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return toml.load(handle)


def load_output_soma_ids(manifest_path: str, matrix_name: str = MOTOR_OUTPUT_NAME) -> list[int]:
    gxo_path = os.path.join(os.path.dirname(manifest_path), "shard.gxo")
    with open(gxo_path, "rb") as handle:
        blob = handle.read()

    if len(blob) < GXO_HEADER_SIZE:
        raise ValueError(f"Output mapping blob is too small: {gxo_path}")

    num_matrices = struct.unpack_from("<H", blob, 6)[0]
    total_pixels = struct.unpack_from("<I", blob, 8)[0]
    payload_offset = GXO_HEADER_SIZE + num_matrices * GXO_DESCRIPTOR_SIZE
    payload_size = total_pixels * 4
    payload_end = payload_offset + payload_size
    if len(blob) < payload_end:
        raise ValueError(f"Output mapping payload is truncated: {gxo_path}")

    target_hash = fnv1a_32(matrix_name.encode("utf-8"))
    soma_ids: np.ndarray | None = None
    for matrix_idx in range(num_matrices):
        desc_offset = GXO_HEADER_SIZE + matrix_idx * GXO_DESCRIPTOR_SIZE
        name_hash, offset, width, height, _stride = struct.unpack_from("<IIHHB3x", blob, desc_offset)
        if name_hash != target_hash:
            continue

        matrix_pixels = int(width) * int(height)
        matrix_start = payload_offset + offset * 4
        matrix_end = matrix_start + matrix_pixels * 4
        if matrix_end > payload_end:
            raise ValueError(f"Output mapping for '{matrix_name}' is truncated in {gxo_path}")

        soma_ids = np.frombuffer(blob[matrix_start:matrix_end], dtype=np.uint32)
        break

    if soma_ids is None:
        raise ValueError(f"Output matrix '{matrix_name}' not found in {gxo_path}")

    return [int(soma_id) for soma_id in soma_ids if int(soma_id) != 0xFFFF_FFFF]


def load_baked_state_stats(manifest_path: str, saturation_threshold: int = WEIGHT_SATURATION_THRESHOLD) -> dict[str, Any]:
    data, padded_n = load_baked_state_payload(manifest_path)

    off = 0
    off += padded_n * 4  # soma_voltage
    flags = np.frombuffer(data[off:off + padded_n], dtype=np.uint8)
    off += padded_n
    off += padded_n * 4  # threshold_offset
    off += padded_n      # timers
    off += padded_n * 4  # soma_to_axon

    dendrite_targets = np.frombuffer(
        data[off:off + padded_n * MAX_DENDRITES * 4],
        dtype=np.uint32,
    ).reshape(MAX_DENDRITES, padded_n)
    off += padded_n * MAX_DENDRITES * 4

    dendrite_weights = np.frombuffer(
        data[off:off + padded_n * MAX_DENDRITES * 2],
        dtype=np.int16,
    ).reshape(MAX_DENDRITES, padded_n)

    connected = dendrite_targets != 0
    active_weights = dendrite_weights[connected]
    abs_weights = np.abs(active_weights)
    variant_counts = count_variant_ids(flags)

    if active_weights.size == 0:
        return {
            "padded_n": padded_n,
            "active_synapses": 0,
            "avg_weight": 0.0,
            "max_weight": 0,
            "saturated_weight_share": 0.0,
            "variant_counts": variant_counts,
        }

    return {
        "padded_n": padded_n,
        "active_synapses": int(np.sum(connected)),
        "avg_weight": float(np.mean(abs_weights)),
        "max_weight": int(np.max(abs_weights)),
        "saturated_weight_share": float(np.mean(abs_weights >= saturation_threshold)),
        "variant_counts": variant_counts,
    }


def load_baked_output_variant_counts(manifest_path: str, output_soma_ids: list[int]) -> dict[int, int]:
    if not output_soma_ids:
        return {}

    data, padded_n = load_baked_state_payload(manifest_path)
    flags_offset = padded_n * 4
    flags = np.frombuffer(data[flags_offset:flags_offset + padded_n], dtype=np.uint8)
    soma_idx = np.asarray(output_soma_ids, dtype=np.intp)
    return count_variant_ids(flags[soma_idx])


def resolve_variant_ids(manifest_data: dict[str, Any], requested: list[int] | None) -> list[int]:
    if requested:
        return list(requested)
    return [int(variant["id"]) for variant in manifest_data.get("variants", [])]


def configure_runtime(control: GenesisControl, config: CartPoleRunConfig) -> list[dict[str, Any]]:
    manifest = load_manifest(config.manifest_path)
    variant_ids = resolve_variant_ids(manifest, config.adaptive_leak.variant_ids)

    control.set_night_interval(config.night_interval)
    control.set_prune_threshold(config.prune_threshold)
    control.set_max_sprouts(config.max_sprouts)
    control.set_membrane_physics(0, LEAK_RATE, HOMEOS_PENALTY, HOMEOS_DECAY)
    control.set_membrane_physics(1, int(LEAK_RATE * 1.5), int(HOMEOS_PENALTY * 0.8), HOMEOS_DECAY)

    for variant_id in variant_ids:
        control.set_adaptive_leak(
            variant_id,
            adaptive_leak_mode=config.adaptive_leak.adaptive_leak_mode,
            dopamine_leak_gain=config.adaptive_leak.dopamine_leak_gain,
            burst_leak_gain=config.adaptive_leak.burst_leak_gain,
            leak_min=config.adaptive_leak.leak_min,
            leak_max=config.adaptive_leak.leak_max,
        )

    time.sleep(config.hot_reload_wait_s)
    return load_manifest(config.manifest_path).get("variants", [])


def build_variant_lookup(variants: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(variant["id"]): variant for variant in variants}


def clamp_array(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.minimum(np.maximum(values, lo), hi)


def compute_effective_leak_array(variant: dict[str, Any], dopamine: int, burst_count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_leak = int(variant["leak_rate"])
    adaptive_mode = int(variant.get("adaptive_leak_mode", 0))
    dopamine_gain = int(variant.get("dopamine_leak_gain", 0))
    burst_gain = int(variant.get("burst_leak_gain", 0))
    leak_min = int(variant.get("leak_min", 0))
    leak_max = int(variant.get("leak_max", 0))

    if adaptive_mode == ADAPTIVE_LEAK_MODE_DISABLED or leak_min >= leak_max:
        leak = np.full_like(burst_count, max(base_leak, 1), dtype=np.int32)
        return leak, np.full_like(burst_count, MEMBRANE_MODE_STABLE, dtype=np.uint8)

    dopamine_mod = (int(dopamine) * dopamine_gain) >> 7
    burst_mod = burst_count.astype(np.int32) * burst_gain

    if adaptive_mode == ADAPTIVE_LEAK_MODE_CONTINUOUS:
        leak = clamp_array(base_leak + dopamine_mod + burst_mod, leak_min, leak_max)
        return np.maximum(leak, 1), np.full_like(burst_count, MEMBRANE_MODE_STABLE, dtype=np.uint8)

    band = max((leak_max - leak_min) >> 2, 1)
    combined = dopamine_mod + burst_mod
    base = int(np.clip(base_leak, leak_min, leak_max))
    responsive = int(np.clip((base + leak_min) >> 1, leak_min, leak_max))
    excited = int(np.clip((base + leak_max + 1) >> 1, leak_min, leak_max))

    mode = np.full_like(burst_count, MEMBRANE_MODE_STABLE, dtype=np.uint8)
    recovery_mask = (burst_count >= RECOVERY_BURST_THRESHOLD) & (burst_mod > 0)
    responsive_mask = combined <= -band
    excited_mask = combined >= band

    mode = np.where(responsive_mask, MEMBRANE_MODE_RESPONSIVE, mode)
    mode = np.where(excited_mask, MEMBRANE_MODE_EXCITED, mode)
    mode = np.where(recovery_mask, MEMBRANE_MODE_RECOVERY, mode).astype(np.uint8)

    leak = np.full_like(burst_count, base, dtype=np.int32)
    leak = np.where(mode == MEMBRANE_MODE_RESPONSIVE, responsive, leak)
    leak = np.where(mode == MEMBRANE_MODE_EXCITED, excited, leak)
    leak = np.where(mode == MEMBRANE_MODE_RECOVERY, leak_max, leak)
    return np.maximum(leak, 1), mode


def sample_runtime_metrics(
    memory: GenesisMemory,
    variant_lookup: dict[int, dict[str, Any]],
    dopamine: int,
    baked_state_stats: dict[str, Any] | None = None,
) -> RuntimeSample:
    stats = memory.get_network_stats(saturation_threshold=WEIGHT_SATURATION_THRESHOLD)
    flags = memory.soma_flags
    variant_ids = flags >> 4
    burst_count = (flags >> 1) & 0x07
    structure_metrics_source = "night_shm_snapshot"

    if baked_state_stats is not None and stats["active_synapses"] == 0 and stats["avg_weight"] == 0.0 and stats["max_weight"] == 0:
        stats = {
            **stats,
            "active_synapses": int(baked_state_stats["active_synapses"]),
            "avg_weight": float(baked_state_stats["avg_weight"]),
            "max_weight": int(baked_state_stats["max_weight"]),
            "saturated_weight_share": float(baked_state_stats["saturated_weight_share"]),
        }
        structure_metrics_source = "baked_state_fallback"

    effective_leak = np.zeros(memory.padded_n, dtype=np.int32)
    mode_counts = {
        "stable": 0,
        "responsive": 0,
        "excited": 0,
        "recovery": 0,
    }

    for variant_id, variant in variant_lookup.items():
        mask = variant_ids == variant_id
        if not np.any(mask):
            continue
        leak_values, modes = compute_effective_leak_array(variant, dopamine, burst_count[mask])
        effective_leak[mask] = leak_values
        mode_counts["stable"] += int(np.sum(modes == MEMBRANE_MODE_STABLE))
        mode_counts["responsive"] += int(np.sum(modes == MEMBRANE_MODE_RESPONSIVE))
        mode_counts["excited"] += int(np.sum(modes == MEMBRANE_MODE_EXCITED))
        mode_counts["recovery"] += int(np.sum(modes == MEMBRANE_MODE_RECOVERY))

    return RuntimeSample(
        spike_rate=float(stats["spike_rate"]),
        saturated_weight_share=float(stats["saturated_weight_share"]),
        effective_leak_mean=float(np.mean(effective_leak)) if effective_leak.size else 0.0,
        mean_burst_count=float(stats["mean_burst_count"]),
        active_synapses=int(stats["active_synapses"]),
        avg_weight=float(stats["avg_weight"]),
        max_weight=int(stats["max_weight"]),
        structure_metrics_source=structure_metrics_source,
        mode_counts=mode_counts,
    )


def sample_output_readout(
    memory: GenesisMemory,
    output_soma_ids: list[int],
    variant_lookup: dict[int, dict[str, Any]],
    dopamine: int,
) -> dict[str, Any]:
    if not output_soma_ids:
        return {
            "mapped_output_count": 0,
            "variant_counts": {},
            "spiking_count": 0,
            "left_spiking_count": 0,
            "right_spiking_count": 0,
            "burst_mean": 0.0,
            "voltage_mean": 0.0,
            "left_voltage_mean": 0.0,
            "right_voltage_mean": 0.0,
            "threshold_offset_mean": 0.0,
            "activation_margin_mean": 0.0,
            "left_activation_margin_mean": 0.0,
            "right_activation_margin_mean": 0.0,
            "effective_leak_mean": 0.0,
            "left_effective_leak_mean": 0.0,
            "right_effective_leak_mean": 0.0,
            "flags_hash": "",
            "voltage_hash": "",
            "threshold_hash": "",
            "timers_hash": "",
            "mode_counts": {"stable": 0, "responsive": 0, "excited": 0, "recovery": 0},
        }

    soma_idx = np.asarray(output_soma_ids, dtype=np.intp)
    flags = np.ascontiguousarray(memory.soma_flags[soma_idx]).copy()
    burst_count = (flags >> 1) & 0x07
    variant_ids = flags >> 4
    spiking_mask = (flags & 0x01) != 0
    effective_leak = np.zeros(flags.shape[0], dtype=np.int32)
    mode_counts = {"stable": 0, "responsive": 0, "excited": 0, "recovery": 0}

    for variant_id in np.unique(variant_ids):
        mask = variant_ids == variant_id
        variant = variant_lookup.get(int(variant_id))
        if variant is None:
            continue
        leak_values, modes = compute_effective_leak_array(variant, dopamine, burst_count[mask])
        effective_leak[mask] = leak_values
        mode_counts["stable"] += int(np.sum(modes == MEMBRANE_MODE_STABLE))
        mode_counts["responsive"] += int(np.sum(modes == MEMBRANE_MODE_RESPONSIVE))
        mode_counts["excited"] += int(np.sum(modes == MEMBRANE_MODE_EXCITED))
        mode_counts["recovery"] += int(np.sum(modes == MEMBRANE_MODE_RECOVERY))

    split = len(output_soma_ids) // 2
    left = slice(0, split)
    right = slice(split, None)
    variant_counts = {
        int(variant_id): int(np.sum(variant_ids == variant_id))
        for variant_id in np.unique(variant_ids)
    }

    # Runtime SHM exposes flags/weights/targets, but not voltage/threshold/timers.
    # Keep these fields neutral in CartPole diagnostics instead of reading garbage bytes.
    zero_hash = hashlib.sha256(b"").hexdigest()

    return {
        "mapped_output_count": int(len(output_soma_ids)),
        "variant_counts": variant_counts,
        "spiking_count": int(np.sum(spiking_mask)),
        "left_spiking_count": int(np.sum(spiking_mask[left])),
        "right_spiking_count": int(np.sum(spiking_mask[right])),
        "burst_mean": float(np.mean(burst_count)),
        "voltage_mean": 0.0,
        "left_voltage_mean": 0.0,
        "right_voltage_mean": 0.0,
        "threshold_offset_mean": 0.0,
        "activation_margin_mean": 0.0,
        "left_activation_margin_mean": 0.0,
        "right_activation_margin_mean": 0.0,
        "effective_leak_mean": float(np.mean(effective_leak)),
        "left_effective_leak_mean": float(np.mean(effective_leak[left])),
        "right_effective_leak_mean": float(np.mean(effective_leak[right])),
        "flags_hash": hashlib.sha256(flags.tobytes()).hexdigest(),
        "voltage_hash": zero_hash,
        "threshold_hash": zero_hash,
        "timers_hash": zero_hash,
        "mode_counts": mode_counts,
    }


def rolling_threshold_episode(scores: list[int], threshold_score: int, window: int = 10) -> int | None:
    if len(scores) < window:
        return None
    for idx in range(window, len(scores) + 1):
        if float(np.mean(scores[idx - window:idx])) >= threshold_score:
            return idx
    return None


def _render_loop(
    env: gym.Env[Any, Any],
    lock: threading.Lock,
    stop_event: threading.Event,
    fps: float,
) -> None:
    interval = 1.0 / fps if fps > 0 else 1.0 / 60.0
    while not stop_event.is_set():
        with lock:
            try:
                env.render()
            except Exception:
                pass
        stop_event.wait(timeout=interval)


def wait_for_memory(zone_hash: int, read_only: bool) -> GenesisMemory:
    print("Waiting for Genesis Node shared memory...")
    for attempt in range(20):
        try:
            memory = GenesisMemory(zone_hash, read_only=read_only)
            print("Telemetry plane connected.")
            return memory
        except (FileNotFoundError, AssertionError) as exc:
            if attempt % 5 == 0:
                print(f"  [retry {attempt}/20] SHM not ready: {exc}")
            time.sleep(1)
    raise RuntimeError("Could not connect to shared memory. Is genesis-node running?")


def _step_with_warmup_retry(client: GenesisMultiClient, dopamine: int, log_episodes: bool) -> memoryview:
    """First step may timeout while node is in warmup (CUDA JIT + 2000 ticks). Retry with longer timeout."""
    for attempt in range(4):
        extended = attempt > 0
        if extended:
            old_timeout = client.sock.gettimeout()
            client.sock.settimeout(90.0)
        try:
            try:
                return client.step(dopamine)
            except TimeoutError:
                if attempt < 3:
                    if log_episodes:
                        print(f"  Node warmup timeout (attempt {attempt + 1}/4), retrying in 5s...")
                    time.sleep(5)
                else:
                    raise
        finally:
            if extended:
                client.sock.settimeout(old_timeout)


def run_cartpole_experiment(config: CartPoleRunConfig) -> dict[str, Any]:
    manifest_path = os.path.abspath(config.manifest_path or default_manifest_path())
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Control plane manifest not found at {manifest_path}")

    zone_hash = fnv1a_32(b"SensoryCortex")
    matrix_hash = fnv1a_32(b"cartpole_sensors")
    input_payload_size = (64 * config.batch_size) // 8

    render_mode = "human" if config.render else None
    env = gym.make("CartPole-v1", max_episode_steps=config.max_episode_steps, render_mode=render_mode)
    env.unwrapped.tau = 0.002
    if config.use_3_actions:
        env = CartPole3ActionWrapper(env)

    # 30s timeout: first CUDA kernel launch triggers JIT (10–30s on Windows)
    client = GenesisMultiClient(
        addr=(config.node_addr, config.node_port),
        matrices=[{"zone_hash": zone_hash, "matrix_hash": matrix_hash, "payload_size": input_payload_size}],
        rx_timeout_s=30.0,
    )
    control = GenesisControl(manifest_path)
    memory = None

    try:
        client.sock.bind(("0.0.0.0", config.response_port))
    except OSError as exc:
        client.sock.close()
        raise RuntimeError(f"Response port {config.response_port} is busy: {exc}") from exc

    try:
        encoder = PopulationEncoder(variables_count=4, neurons_per_var=16, batch_size=config.batch_size, sigma=ENCODER_SIGMA)
        decoder = PwmDecoder(num_outputs=128, batch_size=config.batch_size)

        bounds = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.41, 0.41], [-2.0, 2.0]], dtype=np.float32)
        range_diff = bounds[:, 1] - bounds[:, 0]

        variants = configure_runtime(control, config)
        variant_lookup = build_variant_lookup(variants)
        tuner = GenesisAutoTuner(control, target_score=config.threshold_score) if config.use_autotuner else None
        if tuner is not None:
            control.set_night_interval(config.night_interval)
            control.set_prune_threshold(config.prune_threshold)
            control.set_max_sprouts(config.max_sprouts)

        baked_state_stats = load_baked_state_stats(manifest_path)
        output_soma_ids = load_output_soma_ids(manifest_path, MOTOR_OUTPUT_NAME)
        baked_output_variant_counts = load_baked_output_variant_counts(manifest_path, output_soma_ids)
        memory = wait_for_memory(zone_hash, read_only=False)
        rng = np.random.default_rng(config.seed)

        episodes = 0
        total_steps = 0
        episode_scores: list[int] = []
        per_episode: list[dict[str, Any]] = []
        run_action_trace: list[int] = []
        run_output_state_hashes: list[str] = []
        start_time = time.perf_counter()
        first_step_done = False

        if config.log_episodes:
            print(f"Starting CartPole scenario '{config.scenario_name}' (episodes={config.episodes}, seed={config.seed})...")

        render_lock = threading.Lock() if config.render else None
        render_stop = threading.Event() if config.render else None
        render_thread: threading.Thread | None = None
        if config.render and render_lock and render_stop:
            render_thread = threading.Thread(
                target=_render_loop,
                args=(env, render_lock, render_stop, config.render_fps),
                daemon=True,
            )
            render_thread.start()

        try:
            env_ctx = render_lock if render_lock else contextlib.nullcontext()
            while episodes < config.episodes:
                with env_ctx:
                    state, _ = env.reset(seed=config.seed + episodes)
                terminated = False
                truncated = False
                score = 0
                norm_state = np.zeros(4, dtype=np.float32)
                runtime_samples: list[RuntimeSample] = []
                last_dopamine = 0
                action_trace: list[int] = []
                motor_trace_chunks: list[bytes] = []
                motor_balance_trace: list[int] = []
                motor_prefix: list[dict[str, float]] = []
                raw_readout_trace = hashlib.sha256()
                raw_readout_prefix: list[dict[str, Any]] = []
                output_samples: list[dict[str, Any]] = []
                output_prefix: list[dict[str, Any]] = []
                dopamine_sum = 0
                dopamine_min = DOPAMINE_REWARD
                dopamine_max = DOPAMINE_REWARD

                while not (terminated or truncated):
                    norm_state = (np.clip(state, bounds[:, 0], bounds[:, 1]) - bounds[:, 0]) / range_diff
                    if config.noise_std > 0.0:
                        norm_state = np.clip(norm_state + rng.normal(0.0, config.noise_std, size=norm_state.shape), 0.0, 1.0)

                    pole_angle = abs(state[2])
                    pole_velocity = abs(state[3])
                    angle_error = min(1.0, pole_angle / ANGLE_LIMIT)
                    vel_error = min(1.0, pole_velocity / VELOCITY_LIMIT)
                    error = min(1.0, angle_error * ERROR_ANGLE_WEIGHT + vel_error * ERROR_VEL_WEIGHT)
                    dopamine_signal = int(DOPAMINE_REWARD * (1.0 - error) + DOPAMINE_PULSE * error)
                    last_dopamine = dopamine_signal
                    dopamine_sum += dopamine_signal
                    dopamine_min = min(dopamine_min, dopamine_signal)
                    dopamine_max = max(dopamine_max, dopamine_signal)

                    encoder.encode_into(norm_state.astype(np.float16), client.payload_views[0], 0)
                    # First step may timeout while node is in warmup (CUDA JIT + 2000 ticks). Retry with patience.
                    if first_step_done:
                        rx = client.step(dopamine_signal)
                    else:
                        rx = _step_with_warmup_retry(client, dopamine_signal, config.log_episodes)
                        first_step_done = True
                    raw_spikes = np.frombuffer(rx, dtype=np.uint8, count=decoder.payload_size)
                    raw_readout_trace.update(raw_spikes.tobytes())
                    if len(raw_readout_prefix) < 8:
                        spikes_2d = raw_spikes.reshape((config.batch_size, decoder.N))
                        raw_readout_prefix.append(
                            {
                                "step": score + 1,
                                "left_spike_sum": int(np.sum(spikes_2d[:, :MOTOR_SPLIT], dtype=np.int32)),
                                "right_spike_sum": int(np.sum(spikes_2d[:, MOTOR_SPLIT:], dtype=np.int32)),
                                "payload_hash": hashlib.sha256(raw_spikes.tobytes()).hexdigest(),
                            }
                        )

                    total_motor = decoder.decode_from(rx, 0)
                    left_sum = float(np.sum(total_motor[:MOTOR_SPLIT], dtype=np.float32))
                    right_sum = float(np.sum(total_motor[MOTOR_SPLIT:], dtype=np.float32))
                    motor_trace_chunks.append(total_motor.astype(np.float16, copy=True).tobytes())
                    motor_balance_trace.append(int(round((left_sum - right_sum) * 1024.0)))
                    if len(motor_prefix) < 32:
                        motor_prefix.append(
                            {
                                "left_sum": left_sum,
                                "right_sum": right_sum,
                                "balance": left_sum - right_sum,
                            }
                        )
                    balance = left_sum - right_sum
                    # Epsilon-greedy в EXPLORATION: высокая случайность на старте → затухание к инференсу
                    use_random = False
                    if tuner and tuner.phase == Phase.EXPLORATION and config.exploration_decay_episodes > 0:
                        eps = config.exploration_epsilon_end + (
                            config.exploration_epsilon_start - config.exploration_epsilon_end
                        ) * max(0.0, 1.0 - episodes / config.exploration_decay_episodes)
                        use_random = rng.random() < eps
                    if use_random:
                        n_actions = 3 if config.use_3_actions else 2
                        action = int(rng.integers(0, n_actions))
                    elif config.use_3_actions:
                        if abs(balance) < WAIT_BALANCE_THRESHOLD:
                            action = 1
                        elif balance > 0:
                            action = 0
                        else:
                            action = 2
                    else:
                        if balance > 0:
                            action = 0
                        elif balance < 0:
                            action = 1
                        else:
                            action = int(rng.integers(0, 2))
                    action_trace.append(action)
                    run_action_trace.append(action)

                    with env_ctx:
                        state, _, terminated, truncated, _ = env.step(action)
                    score += 1
                    total_steps += 1

                    if memory and (score % max(config.stats_sample_stride, 1) == 0):
                        runtime_samples.append(sample_runtime_metrics(memory, variant_lookup, dopamine_signal, baked_state_stats))
                        output_sample = sample_output_readout(memory, output_soma_ids, variant_lookup, dopamine_signal)
                        output_samples.append(output_sample)
                        if len(output_prefix) < 8:
                            output_prefix.append(
                                {
                                    "step": score,
                                    "spiking_count": output_sample["spiking_count"],
                                    "left_spiking_count": output_sample["left_spiking_count"],
                                    "right_spiking_count": output_sample["right_spiking_count"],
                                    "voltage_mean": output_sample["voltage_mean"],
                                    "activation_margin_mean": output_sample["activation_margin_mean"],
                                    "effective_leak_mean": output_sample["effective_leak_mean"],
                                    "flags_hash": output_sample["flags_hash"],
                                    "voltage_hash": output_sample["voltage_hash"],
                                    "threshold_hash": output_sample["threshold_hash"],
                                }
                            )

                shock_batches = SHOCK_BASE + (score >> SHOCK_SCORE_BITSHIFT)
                kinetic_penalty = int(abs(state[1]) * SHOCK_VEL_MULT)
                total_shock = min(SHOCK_MAX_BATCHES, shock_batches + kinetic_penalty)
                # Смягчаем punishment при забывании (DISTILLATION + падающий SMA),
                # чтобы не усугублять порочный круг: забывание → reward↓ → LTD → забывание↑
                punishment_modifier_used = tuner.get_punishment_modifier() if tuner else 1.0
                if tuner:
                    total_shock = max(1, int(total_shock * punishment_modifier_used))
                encoder.encode_into(norm_state.astype(np.float16), client.payload_views[0], 0)
                for _ in range(total_shock):
                    client.step(DOPAMINE_PUNISHMENT)

                if memory:
                    runtime_samples.append(sample_runtime_metrics(memory, variant_lookup, last_dopamine, baked_state_stats))
                    output_samples.append(sample_output_readout(memory, output_soma_ids, variant_lookup, last_dopamine))

                action_trace_hash = hashlib.sha256(bytes(action_trace)).hexdigest()
                motor_trace_hash = hashlib.sha256(b"".join(motor_trace_chunks)).hexdigest()
                motor_balance_hash = hashlib.sha256(
                    np.asarray(motor_balance_trace, dtype=np.int32).tobytes()
                ).hexdigest()
                raw_readout_trace_hash = raw_readout_trace.hexdigest()
                output_state_hash = hashlib.sha256(
                    b"".join(
                        (
                            sample["flags_hash"]
                            + sample["voltage_hash"]
                            + sample["threshold_hash"]
                            + sample["timers_hash"]
                        ).encode("ascii")
                        for sample in output_samples
                    )
                ).hexdigest()
                run_action_trace.append(0xFF)
                run_output_state_hashes.append(output_state_hash)

                phase = tuner.step(score).name if tuner else "FIXED"
                summary_sample = runtime_samples[-1] if runtime_samples else RuntimeSample(
                    spike_rate=0.0,
                    saturated_weight_share=0.0,
                    effective_leak_mean=0.0,
                    mean_burst_count=0.0,
                    active_synapses=0,
                    avg_weight=0.0,
                    max_weight=0,
                    structure_metrics_source="no_runtime_samples",
                    mode_counts={"stable": 0, "responsive": 0, "excited": 0, "recovery": 0},
                )

                left_out_mean = float(np.mean([s["left_spiking_count"] for s in output_samples])) if output_samples else 0.0
                right_out_mean = float(np.mean([s["right_spiking_count"] for s in output_samples])) if output_samples else 0.0

                sma = tuner.last_sma if tuner else 0.0
                thresh_distill = config.threshold_score * 0.7 if tuner else 0
                tuner_state = tuner.get_state_for_logging() if tuner else {}
                exploration_epsilon = (
                    config.exploration_epsilon_end
                    + (config.exploration_epsilon_start - config.exploration_epsilon_end)
                    * max(0.0, 1.0 - episodes / max(1, config.exploration_decay_episodes))
                ) if (tuner and tuner.phase == Phase.EXPLORATION and config.exploration_decay_episodes > 0) else 0.0
                episode_record = {
                    "episode_index": episodes,
                    "score": score,
                    "phase": phase,
                    "sma": sma,
                    "punishment_modifier": punishment_modifier_used,
                    "dopamine_sum": dopamine_sum,
                    "dopamine_min": dopamine_min,
                    "dopamine_max": dopamine_max,
                    # Tuner/structural params (для анализа потери навыков)
                    "prune_threshold": tuner_state.get("prune_threshold"),
                    "night_interval": tuner_state.get("night_interval"),
                    "rollback_threshold": tuner_state.get("rollback_threshold"),
                    "distillation_enter_threshold": tuner_state.get("distillation_enter_threshold"),
                    "exploration_epsilon": exploration_epsilon,
                    # Structural (Night Phase)
                    "structure_active_synapses": summary_sample.active_synapses,
                    "structure_avg_weight": summary_sample.avg_weight,
                    "structure_max_weight": summary_sample.max_weight,
                    "structure_metrics_source": summary_sample.structure_metrics_source,
                    # Online Dynamics
                    "spike_rate": float(np.mean([sample.spike_rate for sample in runtime_samples])) if runtime_samples else 0.0,
                    "saturated_weight_share": float(np.mean([sample.saturated_weight_share for sample in runtime_samples])) if runtime_samples else 0.0,
                    "effective_leak_mean": float(np.mean([sample.effective_leak_mean for sample in runtime_samples])) if runtime_samples else 0.0,
                    "mean_burst_count": float(np.mean([sample.mean_burst_count for sample in runtime_samples])) if runtime_samples else 0.0,
                    # Output Policy
                    "output_spiking_mean": float(np.mean([sample["spiking_count"] for sample in output_samples])) if output_samples else 0.0,
                    "left_output_spiking_mean": left_out_mean,
                    "right_output_spiking_mean": right_out_mean,
                    "output_balance_mean": left_out_mean - right_out_mean,
                    # Mode Distribution
                    "mode_counts": {
                        "stable": int(np.sum([sample.mode_counts["stable"] for sample in runtime_samples])),
                        "responsive": int(np.sum([sample.mode_counts["responsive"] for sample in runtime_samples])),
                        "excited": int(np.sum([sample.mode_counts["excited"] for sample in runtime_samples])),
                        "recovery": int(np.sum([sample.mode_counts["recovery"] for sample in runtime_samples])),
                    },
                    # Compatibility/Hashes
                    "action_trace_hash": action_trace_hash,
                    "motor_trace_hash": motor_trace_hash,
                    "motor_balance_hash": motor_balance_hash,
                    "output_state_hash": output_state_hash,
                    "active_synapses": summary_sample.active_synapses,
                    "avg_weight": summary_sample.avg_weight,
                    "max_weight": summary_sample.max_weight,
                }
                per_episode.append(episode_record)
                episode_scores.append(score)

                rolling5 = float(np.mean(episode_scores[-5:])) if len(episode_scores) >= 5 else float(np.mean(episode_scores))
                rolling10 = float(np.mean(episode_scores[-10:])) if len(episode_scores) >= 10 else float(np.mean(episode_scores))

                if config.log_episodes:
                    m_counts = episode_record["mode_counts"]
                    mode_str = f"S:{m_counts['stable']} R:{m_counts['responsive']} E:{m_counts['excited']} Rec:{m_counts['recovery']}"
                    prune_str = f"prune:{episode_record['prune_threshold']}" if episode_record.get("prune_threshold") is not None else ""
                    night_str = f"night:{episode_record['night_interval']}" if episode_record.get("night_interval") is not None else ""
                    mod_str = f" shock×{punishment_modifier_used:.2f}" if punishment_modifier_used < 1.0 else ""
                    tuner_str = f" | {prune_str} {night_str}{mod_str}" if prune_str or night_str or mod_str else ""
                    print(
                        f"Ep {episodes:04d} | Score: {score:3d} | Phase: {phase:<12} | SMA: {sma:.0f} | "
                        f"Spike: {episode_record['spike_rate']:.3f} | Burst: {episode_record['mean_burst_count']:.2f} | "
                        f"OutBal: {episode_record['output_balance_mean']:.1f} | Leak: {episode_record['effective_leak_mean']:.1f} | "
                        f"Syn: {episode_record['structure_active_synapses']} | "
                        f"Modes: {mode_str}{tuner_str}"
                    )

                episodes += 1

            elapsed = time.perf_counter() - start_time
            episodes_to_threshold = rolling_threshold_episode(episode_scores, config.threshold_score)

            mean_score = float(np.mean(episode_scores)) if episode_scores else 0.0
            result = {
                "scenario": config.scenario_name,
                "scenario_label": config.scenario_label,
                "scenario_name": config.scenario_name,
                "seed": config.seed,
                "episodes_completed": episodes,
                "episodes": config.episodes,
                "batch_size": config.batch_size,
                "total_steps": total_steps,
                "elapsed_sec": elapsed,
                "ticks_per_second": (total_steps * config.batch_size) / elapsed if elapsed > 0 else 0.0,
                "mean_score": mean_score,
                "mean_episode_length": mean_score,
                "reward_variance": float(np.var(episode_scores)) if episode_scores else 0.0,
                "max_score": int(np.max(episode_scores)) if episode_scores else 0,
                "episodes_to_threshold": episodes_to_threshold,
                "padded_n": int(baked_state_stats["padded_n"]),
                # Neural Dashboard Summary
                "mean_spike_rate": float(np.mean([ep["spike_rate"] for ep in per_episode])) if per_episode else 0.0,
                "mean_burst_count": float(np.mean([ep["mean_burst_count"] for ep in per_episode])) if per_episode else 0.0,
                "mean_output_spiking": float(np.mean([ep["output_spiking_mean"] for ep in per_episode])) if per_episode else 0.0,
                "mean_output_balance": float(np.mean([ep["output_balance_mean"] for ep in per_episode])) if per_episode else 0.0,
                "mean_effective_leak": float(np.mean([ep["effective_leak_mean"] for ep in per_episode])) if per_episode else 0.0,
                "mean_saturated_weight_share": float(np.mean([ep["saturated_weight_share"] for ep in per_episode])) if per_episode else 0.0,
                # Structural Summary
                "mean_structure_active_synapses": float(np.mean([ep["structure_active_synapses"] for ep in per_episode])) if per_episode else 0.0,
                "mean_structure_avg_weight": float(np.mean([ep["structure_avg_weight"] for ep in per_episode])) if per_episode else 0.0,
                "structure_metrics_sources_seen": sorted({ep["structure_metrics_source"] for ep in per_episode}),
                # Mode Distribution Summary
                "mode_share_summary": {
                    mode: float(np.mean([ep["mode_counts"][mode] for ep in per_episode]))
                    for mode in ["stable", "responsive", "excited", "recovery"]
                } if per_episode else {},
                # Tuner/structural analysis (потеря навыков)
                "tuner_summary": {
                    "phases_seen": sorted({ep["phase"] for ep in per_episode if ep.get("phase")}),
                    "prune_thresholds_seen": sorted({ep["prune_threshold"] for ep in per_episode if ep.get("prune_threshold") is not None}),
                    "night_intervals_seen": sorted({ep["night_interval"] for ep in per_episode if ep.get("night_interval") is not None}),
                    "punishment_softened_episodes": sum(1 for ep in per_episode if ep.get("punishment_modifier", 1.0) < 1.0),
                } if per_episode else {},
                "per_episode": per_episode,
                "baked_structure_stats": baked_state_stats,
                "run_action_trace_hash": hashlib.sha256(bytes(run_action_trace)).hexdigest(),
                "adaptive_leak": asdict(config.adaptive_leak),
            }

            if config.output_path:
                os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
                with open(config.output_path, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2)

            return result
        finally:
            if render_stop is not None and render_thread is not None:
                render_stop.set()
                render_thread.join(timeout=2.0)
    finally:
        if memory is not None:
            memory.close()
        client.sock.close()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CartPole Genesis agent.")
    parser.add_argument("--episodes", type=int, default=EPISODES, help="Number of CartPole episodes to run.")
    parser.add_argument("--seed", type=int, default=123, help="Environment seed.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Stddev of Gaussian sensor noise added after normalization.")
    parser.add_argument("--threshold-score", type=int, default=TARGET_SCORE, help="Reward threshold for speed-to-threshold metric.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Number of ticks sent per environment step.")
    parser.add_argument("--node-port", type=int, default=8081, help="Genesis node external UDP input port.")
    parser.add_argument("--response-port", type=int, default=8092, help="Local UDP port for motor packets.")
    parser.add_argument("--manifest-path", default=default_manifest_path(), help="Path to the baked zone manifest.")
    parser.add_argument("--output-path", help="Optional JSON path for serialized run metrics.")
    parser.add_argument("--scenario-name", default="manual", help="Short scenario identifier stored in artifacts.")
    parser.add_argument("--scenario-label", default="Manual CartPole", help="Human readable scenario label.")
    parser.add_argument("--stats-sample-stride", type=int, default=5, help="Collect runtime stats every N environment steps.")
    parser.add_argument("--fixed-runtime", action="store_true", help="Disable the autotuner and keep benchmark settings fixed.")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-episode logging.")
    parser.add_argument("--render", action="store_true", help="Render CartPole in a live window (async, non-blocking).")
    parser.add_argument(
        "--render-fps",
        type=float,
        default=60.0,
        help="Render frame rate when --render is used (default: 60).",
    )
    parser.add_argument("--adaptive-leak-mode", type=int, default=0, help="Adaptive leak mode override for targeted variants.")
    parser.add_argument("--dopamine-leak-gain", type=int, default=0, help="Adaptive dopamine gain override.")
    parser.add_argument("--burst-leak-gain", type=int, default=0, help="Adaptive burst gain override.")
    parser.add_argument("--leak-min", type=int, default=0, help="Adaptive leak clamp minimum.")
    parser.add_argument("--leak-max", type=int, default=0, help="Adaptive leak clamp maximum.")
    parser.add_argument(
        "--variant-ids",
        type=int,
        nargs="*",
        help="Variant IDs to patch. Defaults to all manifest variants when omitted.",
    )
    parser.add_argument(
        "--use-3-actions",
        action="store_true",
        help="Use Left(0), Wait(1), Right(2) instead of Left(0), Right(1). Wait = force 0.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=5000,
        help="Max steps per episode (default: 5000). CartPole-v1 default is 500.",
    )
    parser.add_argument(
        "--exploration-epsilon-start",
        type=float,
        default=0.8,
        help="Epsilon-greedy: random action prob at start of EXPLORATION (default: 0.8).",
    )
    parser.add_argument(
        "--exploration-epsilon-end",
        type=float,
        default=0.05,
        help="Epsilon-greedy: random action prob after decay (default: 0.05).",
    )
    parser.add_argument(
        "--exploration-decay-episodes",
        type=int,
        default=2000,
        help="Episodes over which epsilon decays from start to end (default: 2000).",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        ensure_virtualenv()
        result = run_cartpole_experiment(
            CartPoleRunConfig(
                scenario_name=args.scenario_name,
                scenario_label=args.scenario_label,
                episodes=args.episodes,
                batch_size=args.batch_size,
                seed=args.seed,
                response_port=args.response_port,
                node_port=args.node_port,
                manifest_path=os.path.abspath(args.manifest_path),
                threshold_score=args.threshold_score,
                use_autotuner=not args.fixed_runtime,
                noise_std=args.noise_std,
                stats_sample_stride=args.stats_sample_stride,
                output_path=os.path.abspath(args.output_path) if args.output_path else None,
                log_episodes=not args.quiet,
                render=args.render,
                render_delay_s=0.0,
                render_fps=args.render_fps,
                adaptive_leak=AdaptiveLeakConfig(
                    adaptive_leak_mode=args.adaptive_leak_mode,
                    dopamine_leak_gain=args.dopamine_leak_gain,
                    burst_leak_gain=args.burst_leak_gain,
                    leak_min=args.leak_min,
                    leak_max=args.leak_max,
                    variant_ids=args.variant_ids,
                ),
                use_3_actions=args.use_3_actions,
                max_episode_steps=args.max_episode_steps,
                exploration_epsilon_start=args.exploration_epsilon_start,
                exploration_epsilon_end=args.exploration_epsilon_end,
                exploration_decay_episodes=args.exploration_decay_episodes,
            )
        )

        if args.quiet:
            print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}, indent=2))
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
