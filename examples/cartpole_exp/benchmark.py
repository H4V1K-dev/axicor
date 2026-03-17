#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "genesis-client")))

from genesis.brain import fnv1a_32
from genesis.control import GenesisControl
from genesis.memory import GenesisMemory

from agent import (
    AdaptiveLeakConfig,
    BATCH_SIZE,
    CartPoleRunConfig,
    LEAK_RATE,
    MAX_SPROUTS_PER_NIGHT,
    NIGHT_INTERVAL,
    PRUNE_THRESHOLD,
    TARGET_SCORE,
    ensure_virtualenv,
    run_cartpole_experiment,
)
from build_brain import BuildConfig, build_cartpole_brain


DOPAMINE_GAIN_DEFAULT = 96
DOPAMINE_GAIN_MOTOR = 1000  # Higher gain for Motor_Pyramidal to overcome saturated regime
BURST_GAIN_DEFAULT = 24
LEAK_MIN_DEFAULT = int(LEAK_RATE * 0.75)
LEAK_MAX_DEFAULT = int(LEAK_RATE * 1.25)
# Motor_Pyramidal (variant 2) has base leak_rate=223; use windows tailored to it
MOTOR_LEAK_MIN = 50
MOTOR_LEAK_MAX = 800
MOTOR_LEAK_NARROW = (100, 600)  # Tighter range around base
MOTOR_LEAK_WIDE = (30, 900)     # Wider modulation for discrete mode
MOTOR_VARIANT_ID = 2  # Motor_Pyramidal — output layer
ADAPTIVE_LEAK_MODE_CONTINUOUS = 1
ADAPTIVE_LEAK_MODE_DISCRETE = 2


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    adaptive_leak: AdaptiveLeakConfig
    noise_std: float = 0.0


def default_manifest_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../Genesis-Models/CartPole-example/baked/SensoryCortex/manifest.toml")
    )


def default_artifacts_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../artifacts/cartpole_benchmark"))


def default_fast_path_port(node_port: int) -> int:
    return int(node_port) + 1000


def default_scenarios() -> list[Scenario]:
    motor_window = {"leak_min": MOTOR_LEAK_MIN, "leak_max": MOTOR_LEAK_MAX}
    motor_narrow = {"leak_min": MOTOR_LEAK_NARROW[0], "leak_max": MOTOR_LEAK_NARROW[1]}
    motor_wide = {"leak_min": MOTOR_LEAK_WIDE[0], "leak_max": MOTOR_LEAK_WIDE[1]}
    return [
        Scenario(
            name="baseline",
            label="Baseline",
            adaptive_leak=AdaptiveLeakConfig(adaptive_leak_mode=0),
        ),
        Scenario(
            name="dopamine_only",
            label="Adaptive leak dopamine (Motor gain=1000, 50–800)",
            adaptive_leak=AdaptiveLeakConfig(
                adaptive_leak_mode=ADAPTIVE_LEAK_MODE_CONTINUOUS,
                dopamine_leak_gain=DOPAMINE_GAIN_MOTOR,
                burst_leak_gain=0,
                variant_ids=[MOTOR_VARIANT_ID],
                **motor_window,
            ),
        ),
        Scenario(
            name="dopamine_narrow",
            label="Adaptive leak dopamine (Motor gain=1000, 100–600)",
            adaptive_leak=AdaptiveLeakConfig(
                adaptive_leak_mode=ADAPTIVE_LEAK_MODE_CONTINUOUS,
                dopamine_leak_gain=DOPAMINE_GAIN_MOTOR,
                burst_leak_gain=0,
                variant_ids=[MOTOR_VARIANT_ID],
                **motor_narrow,
            ),
        ),
        Scenario(
            name="dopamine_wide",
            label="Adaptive leak dopamine (Motor gain=1000, 30–900)",
            adaptive_leak=AdaptiveLeakConfig(
                adaptive_leak_mode=ADAPTIVE_LEAK_MODE_CONTINUOUS,
                dopamine_leak_gain=DOPAMINE_GAIN_MOTOR,
                burst_leak_gain=0,
                variant_ids=[MOTOR_VARIANT_ID],
                **motor_wide,
            ),
        ),
        Scenario(
            name="burst_only",
            label="Adaptive leak burst only (Motor_Pyramidal)",
            adaptive_leak=AdaptiveLeakConfig(
                adaptive_leak_mode=ADAPTIVE_LEAK_MODE_CONTINUOUS,
                dopamine_leak_gain=0,
                burst_leak_gain=BURST_GAIN_DEFAULT,
                variant_ids=[MOTOR_VARIANT_ID],
                **motor_window,
            ),
        ),
        Scenario(
            name="combined",
            label="Combined modulation (Motor gain=1000, 50–800)",
            adaptive_leak=AdaptiveLeakConfig(
                adaptive_leak_mode=ADAPTIVE_LEAK_MODE_CONTINUOUS,
                dopamine_leak_gain=DOPAMINE_GAIN_MOTOR,
                burst_leak_gain=BURST_GAIN_DEFAULT,
                variant_ids=[MOTOR_VARIANT_ID],
                **motor_window,
            ),
        ),
        Scenario(
            name="combined_noise",
            label="Combined modulation + input noise (Motor gain=1000)",
            adaptive_leak=AdaptiveLeakConfig(
                adaptive_leak_mode=ADAPTIVE_LEAK_MODE_CONTINUOUS,
                dopamine_leak_gain=DOPAMINE_GAIN_MOTOR,
                burst_leak_gain=BURST_GAIN_DEFAULT,
                variant_ids=[MOTOR_VARIANT_ID],
                **motor_window,
            ),
            noise_std=0.05,
        ),
        Scenario(
            name="discrete_dopamine",
            label="Discrete mode dopamine (Motor gain=1000, 30–900)",
            adaptive_leak=AdaptiveLeakConfig(
                adaptive_leak_mode=ADAPTIVE_LEAK_MODE_DISCRETE,
                dopamine_leak_gain=DOPAMINE_GAIN_MOTOR,
                burst_leak_gain=0,
                variant_ids=[MOTOR_VARIANT_ID],
                **motor_wide,
            ),
        ),
        Scenario(
            name="discrete_combined",
            label="Discrete mode combined (Motor gain=1000, 30–900)",
            adaptive_leak=AdaptiveLeakConfig(
                adaptive_leak_mode=ADAPTIVE_LEAK_MODE_DISCRETE,
                dopamine_leak_gain=DOPAMINE_GAIN_MOTOR,
                burst_leak_gain=BURST_GAIN_DEFAULT,
                variant_ids=[MOTOR_VARIANT_ID],
                **motor_wide,
            ),
        ),
    ]


def apply_scenario_to_manifest(
    manifest_path: str,
    adaptive_leak: AdaptiveLeakConfig,
    node_port: int,
    response_port: int,
    fast_path_port: int,
) -> list[int]:
    control = GenesisControl(manifest_path)
    control.set_external_udp_in(node_port)
    control.set_external_udp_out_target("127.0.0.1", response_port)
    control.set_fast_path_udp_local(fast_path_port)
    variant_ids = adaptive_leak.variant_ids or control.list_variant_ids()
    for variant_id in variant_ids:
        control.set_adaptive_leak(
            variant_id,
            adaptive_leak_mode=adaptive_leak.adaptive_leak_mode,
            dopamine_leak_gain=adaptive_leak.dopamine_leak_gain,
            burst_leak_gain=adaptive_leak.burst_leak_gain,
            leak_min=adaptive_leak.leak_min,
            leak_max=adaptive_leak.leak_max,
        )
    return variant_ids


def node_command(brain_name: str) -> list[str]:
    return ["cargo", "run", "--release", "-p", "genesis-node", "--", "--brain", brain_name, "--log"]


def read_node_log_tail(log_path: str, max_lines: int = 20) -> str:
    if not log_path or not os.path.exists(log_path):
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return "".join(lines[-max_lines:]).strip()


def wait_for_node_ready(
    zone_hash: int,
    process: subprocess.Popen,
    log_path: str,
    timeout_s: float = 30.0,
) -> None:
    shm_path = GenesisMemory._resolve_path(zone_hash)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            log_tail = read_node_log_tail(log_path)
            detail = f"\n\nRecent node log:\n{log_tail}" if log_tail else ""
            raise RuntimeError(f"Genesis node exited before becoming ready.{detail}")
        if os.path.exists(shm_path):
            # SHM becomes visible slightly before the UDP IO server is fully ready on Windows.
            time.sleep(2.0)
            return
        time.sleep(0.5)
    log_tail = read_node_log_tail(log_path)
    detail = f"\n\nRecent node log:\n{log_tail}" if log_tail else ""
    raise TimeoutError(
        f"Genesis node did not expose shared memory at {shm_path} within {timeout_s:.1f}s.{detail}"
    )


def parse_node_log_metrics(log_path: str, padded_n: int) -> dict[str, float]:
    if not os.path.exists(log_path):
        return {"mean_zone_spikes": 0.0, "mean_spike_rate": 0.0}

    spike_counts: list[int] = []
    pattern = re.compile(r"\[ZONE\].*:\s+(\d+)\s+spikes")
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                spike_counts.append(int(match.group(1)))

    if not spike_counts or padded_n <= 0:
        return {"mean_zone_spikes": 0.0, "mean_spike_rate": 0.0}

    mean_zone_spikes = float(np.mean(spike_counts))
    return {
        "mean_zone_spikes": mean_zone_spikes,
        "mean_spike_rate": mean_zone_spikes / float(padded_n),
    }


def start_node_process(
    repo_root: str,
    artifacts_dir: str,
    brain_name: str,
    scenario_name: str,
    seed: int,
) -> tuple[subprocess.Popen, str, Any]:
    log_path = os.path.join(artifacts_dir, f"{scenario_name}_seed{seed}_node.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        node_command(brain_name),
        cwd=repo_root,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_path, handle


def stop_node_process(process: subprocess.Popen, handle: Any, port_release_delay_s: float = 3.0) -> None:
    """Stop genesis-node and wait for port release (Windows often holds UDP port briefly)."""
    if process.poll() is not None:
        handle.close()
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    handle.close()
    if port_release_delay_s > 0:
        time.sleep(port_release_delay_s)


def summarize_scenario_runs(scenario: Scenario, runs: list[dict]) -> dict:
    mean_episode_lengths = [run["mean_episode_length"] for run in runs]
    threshold_runs = [run["episodes_to_threshold"] for run in runs if run["episodes_to_threshold"] is not None]
    summary = {
        "scenario_name": scenario.name,
        "scenario_label": scenario.label,
        "seed_count": len(runs),
        "noise_std": scenario.noise_std,
        "adaptive_leak": asdict(scenario.adaptive_leak),
        "mean_episode_length": float(sum(mean_episode_lengths) / len(mean_episode_lengths)),
        "reward_variance_across_seeds": float(np.var(mean_episode_lengths)),
        "mean_reward_variance_within_run": float(np.mean([run["reward_variance"] for run in runs])),
        "mean_spike_rate": float(np.mean([run["mean_spike_rate"] for run in runs])),
        "mean_saturated_weight_share": float(np.mean([run["mean_saturated_weight_share"] for run in runs])),
        "mean_effective_leak": float(np.mean([run["mean_effective_leak"] for run in runs])),
        "mean_ticks_per_second": float(np.mean([run["ticks_per_second"] for run in runs])),
        "episodes_to_threshold_mean": float(np.mean(threshold_runs)) if threshold_runs else None,
        "runs": runs,
    }
    return summary


def threshold_delta_vs_baseline(candidate: dict, baseline: dict) -> float:
    candidate_eps = candidate["episodes_to_threshold_mean"]
    baseline_eps = baseline["episodes_to_threshold_mean"]
    if candidate_eps is None and baseline_eps is None:
        return 0.0
    if candidate_eps is not None and baseline_eps is None:
        return 1.0
    if candidate_eps is None and baseline_eps is not None:
        return -1.0
    return float((baseline_eps - candidate_eps) / max(baseline_eps, 1.0))


def compare_to_baseline(summary: dict, baseline: dict) -> dict:
    return {
        "episode_length_delta": float(
            (summary["mean_episode_length"] - baseline["mean_episode_length"]) / max(baseline["mean_episode_length"], 1.0)
        ),
        "reward_variance_delta": float(
            (summary["reward_variance_across_seeds"] - baseline["reward_variance_across_seeds"])
            / max(baseline["reward_variance_across_seeds"], 1.0)
        ),
        "throughput_delta": float(
            (summary["mean_ticks_per_second"] - baseline["mean_ticks_per_second"])
            / max(baseline["mean_ticks_per_second"], 1.0)
        ),
        "threshold_delta": threshold_delta_vs_baseline(summary, baseline),
    }


def compute_verdict(summary: dict, baseline: dict, combined_summary: dict | None) -> tuple[str, list[str]]:
    if summary["scenario_name"] == "baseline":
        return "baseline", []

    deltas = compare_to_baseline(summary, baseline)
    notes: list[str] = []
    if deltas["episode_length_delta"] <= -0.10:
        notes.append("episode_length_regressed")
    if deltas["reward_variance_delta"] >= 0.25:
        notes.append("reward_variance_regressed")
    if deltas["throughput_delta"] <= -0.15:
        notes.append("throughput_regressed")

    if summary["scenario_name"] == "combined_noise" and combined_summary is not None:
        noise_robustness_delta = float(
            summary["mean_episode_length"] / max(combined_summary["mean_episode_length"], 1.0)
        )
        summary["noise_robustness_delta"] = noise_robustness_delta
        if noise_robustness_delta < 0.85:
            notes.append("noise_robustness_regressed")

    if notes:
        return "regressed", notes

    if deltas["episode_length_delta"] >= 0.05 or deltas["threshold_delta"] >= 0.05:
        return "improved", notes

    return "neutral", notes


def write_summary_csv(path: str, summaries: list[dict]) -> None:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "scenario_name": summary["scenario_name"],
                "scenario_label": summary["scenario_label"],
                "seed_count": summary["seed_count"],
                "mean_episode_length": summary["mean_episode_length"],
                "reward_variance_across_seeds": summary["reward_variance_across_seeds"],
                "mean_reward_variance_within_run": summary["mean_reward_variance_within_run"],
                "mean_spike_rate": summary["mean_spike_rate"],
                "mean_saturated_weight_share": summary["mean_saturated_weight_share"],
                "mean_effective_leak": summary["mean_effective_leak"],
                "mean_ticks_per_second": summary["mean_ticks_per_second"],
                "episodes_to_threshold_mean": summary["episodes_to_threshold_mean"],
                "verdict": summary.get("verdict", ""),
                "noise_robustness_delta": summary.get("noise_robustness_delta", ""),
            }
        )

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark adaptive leak scenarios on CartPole.")
    parser.add_argument("--episodes", type=int, default=25, help="Episodes per seed and scenario (25 for stats).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303], help="Environment seeds (101 202 303 for stats).")
    parser.add_argument("--quick", action="store_true", help="Short run: 5 episodes, seed 101 only.")
    parser.add_argument("--brain-name", default="CartPole-example", help="Brain name passed to genesis-node.")
    parser.add_argument("--manifest-path", default=default_manifest_path(), help="Path to the baked CartPole manifest.")
    parser.add_argument("--artifacts-dir", default=default_artifacts_dir(), help="Directory for JSON/CSV/log artifacts.")
    parser.add_argument("--master-seed", default="GENESIS-CARTPOLE-M5", help="Topology seed used by build_brain.py.")
    parser.add_argument("--rebuild-brain", action="store_true", help="Rebuild and rebake CartPole before running the benchmark.")
    parser.add_argument("--reuse-running-node", action="store_true", help="Do not spawn/stop genesis-node between runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the scenario matrix without executing runs.")
    parser.add_argument("--response-port", type=int, default=8092, help="Local UDP response port for agent runs.")
    parser.add_argument("--node-port", type=int, default=8081, help="Genesis node external UDP input port.")
    parser.add_argument(
        "--fast-path-port",
        type=int,
        default=None,
        help="Genesis node fast-path UDP base port. Geometry/telemetry use +1/+2; defaults to node-port + 1000.",
    )
    parser.add_argument("--port-release-delay", type=float, default=3.0, help="Seconds to wait after stopping node before next start (Windows port release).")
    parser.add_argument("--stats-sample-stride", type=int, default=5, help="Collect runtime stats every N env steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.episodes = 5
        args.seeds = [101]
    scenarios = default_scenarios()

    if args.dry_run:
        print(json.dumps({"seeds": args.seeds, "scenarios": [asdict(s) for s in scenarios]}, indent=2))
        return

    ensure_virtualenv()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    manifest_path = os.path.abspath(args.manifest_path)
    artifacts_dir = os.path.abspath(args.artifacts_dir)
    fast_path_port = args.fast_path_port if args.fast_path_port is not None else default_fast_path_port(args.node_port)

    if args.rebuild_brain:
        build_cartpole_brain(
            BuildConfig(
                project_name="CartPoleAgent",
                output_dir=os.path.abspath(os.path.join(repo_root, "Genesis-Models", args.brain_name)),
                gnm_path=os.path.abspath(os.path.join(repo_root, "GNM-Library")),
                master_seed=args.master_seed,
            )
        )

    os.makedirs(artifacts_dir, exist_ok=True)
    zone_hash = fnv1a_32(b"SensoryCortex")
    benchmark_runs: dict[str, list[dict]] = {}

    for scenario in scenarios:
        benchmark_runs[scenario.name] = []
        for seed in args.seeds:
            print(f"[Benchmark] Scenario={scenario.name} Seed={seed}")
            apply_scenario_to_manifest(
                manifest_path,
                scenario.adaptive_leak,
                node_port=args.node_port,
                response_port=args.response_port,
                fast_path_port=fast_path_port,
            )

            process = None
            process_handle = None
            log_path = None
            if not args.reuse_running_node:
                process, log_path, process_handle = start_node_process(repo_root, artifacts_dir, args.brain_name, scenario.name, seed)
                try:
                    wait_for_node_ready(zone_hash, process, log_path)
                except Exception:
                    if process is not None:
                        stop_node_process(process, process_handle)
                    raise

            try:
                output_path = os.path.join(artifacts_dir, "raw_runs", f"{scenario.name}_seed{seed}.json")
                run = run_cartpole_experiment(
                    CartPoleRunConfig(
                        scenario_name=scenario.name,
                        scenario_label=scenario.label,
                        episodes=args.episodes,
                        batch_size=BATCH_SIZE,
                        seed=seed,
                        response_port=args.response_port,
                        node_port=args.node_port,
                        manifest_path=manifest_path,
                        night_interval=NIGHT_INTERVAL,
                        prune_threshold=PRUNE_THRESHOLD,
                        max_sprouts=MAX_SPROUTS_PER_NIGHT,
                        threshold_score=TARGET_SCORE,
                        use_autotuner=False,
                        noise_std=scenario.noise_std,
                        stats_sample_stride=args.stats_sample_stride,
                        output_path=output_path,
                        log_episodes=False,
                        adaptive_leak=scenario.adaptive_leak,
                    )
                )
                if log_path:
                    run.update(parse_node_log_metrics(log_path, int(run.get("padded_n", 0))))
                with open(output_path, "w", encoding="utf-8") as handle:
                    json.dump(run, handle, indent=2)
                benchmark_runs[scenario.name].append(run)
            finally:
                if process is not None:
                    stop_node_process(process, process_handle, args.port_release_delay)

    summaries = [summarize_scenario_runs(scenario, benchmark_runs[scenario.name]) for scenario in scenarios]
    baseline_summary = next(summary for summary in summaries if summary["scenario_name"] == "baseline")
    combined_summary = next((summary for summary in summaries if summary["scenario_name"] == "combined"), None)

    improved_count = 0
    critical_regressions: list[str] = []
    for summary in summaries:
        verdict, notes = compute_verdict(summary, baseline_summary, combined_summary)
        summary["verdict"] = verdict
        summary["notes"] = notes
        if verdict == "improved":
            improved_count += 1
        if verdict == "regressed":
            critical_regressions.append(summary["scenario_name"])
        if summary["scenario_name"] != "baseline":
            summary["baseline_delta"] = compare_to_baseline(summary, baseline_summary)

    acceptance = {
        "at_least_one_improved": improved_count > 0,
        "no_critical_regression": not critical_regressions,
        "hot_loop_overhead_acceptable": not any("throughput_regressed" in summary.get("notes", []) for summary in summaries),
    }
    acceptance["passes_milestone_5"] = all(acceptance.values())

    summary_json = {
        "meta": {
            "benchmark": "Milestone 5 CartPole Benchmark & Validation",
            "master_seed": args.master_seed,
            "batch_size": BATCH_SIZE,
            "night_interval": NIGHT_INTERVAL,
            "prune_threshold": PRUNE_THRESHOLD,
            "max_sprouts": MAX_SPROUTS_PER_NIGHT,
            "episodes_per_seed": args.episodes,
            "seeds": args.seeds,
            "manifest_path": manifest_path,
            "node_port": args.node_port,
            "response_port": args.response_port,
            "fast_path_port": fast_path_port,
        },
        "summaries": summaries,
        "acceptance": acceptance,
    }

    summary_json_path = os.path.join(artifacts_dir, "benchmark_summary.json")
    summary_csv_path = os.path.join(artifacts_dir, "benchmark_summary.csv")
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2)
    write_summary_csv(summary_csv_path, summaries)
    apply_scenario_to_manifest(
        manifest_path,
        scenarios[0].adaptive_leak,
        node_port=args.node_port,
        response_port=args.response_port,
        fast_path_port=fast_path_port,
    )

    print(json.dumps(summary_json, indent=2))

    if not acceptance["passes_milestone_5"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
