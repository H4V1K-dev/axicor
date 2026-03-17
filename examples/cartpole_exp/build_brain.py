#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

# Добавляем путь к SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../genesis-client")))
from genesis.builder import BrainBuilder


def ensure_virtualenv() -> None:
    if sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ:
        return
    print("ERROR: Virtual environment not active.")
    print("Activate the project venv before building CartPole artifacts.")
    sys.exit(1)


@dataclass(frozen=True)
class BuildConfig:
    project_name: str
    output_dir: str
    gnm_path: str
    master_seed: str
    sync_batch_ticks: int = 20
    tick_duration_us: int = 100
    adaptive_leak_mode: int = 0
    dopamine_leak_gain: int = 0
    burst_leak_gain: int = 0
    leak_min: int = 0
    leak_max: int = 0


def build_cartpole_brain(config: BuildConfig) -> str:
    print("Initializing CartPole connectome build...")

    builder = BrainBuilder(
        project_name=config.project_name,
        output_dir=config.output_dir,
        gnm_lib_path=config.gnm_path,
    )
    builder.sim_params["master_seed"] = config.master_seed
    builder.sim_params["sync_batch_ticks"] = config.sync_batch_ticks
    builder.sim_params["tick_duration_us"] = config.tick_duration_us

    cortex = builder.add_zone("SensoryCortex", width_vox=24, depth_vox=24, height_vox=16)

    try:
        exc_type = builder.gnm_lib("VISp4/141").set_plasticity(pot=0, dep=2)
        inh_type = builder.gnm_lib("VISp4/114").set_plasticity(pot=0, dep=2)
        motor_type = builder.gnm_lib("VISp4/141").set_plasticity(pot=0, dep=2)
        motor_type.name = "Motor_Pyramidal"
        for item in motor_type.data_list:
            item["name"] = "Motor_Pyramidal"

        for bp in [exc_type, inh_type, motor_type]:
            for item in bp.data_list:
                item["initial_synapse_weight"] = 8000
                item["dendrite_radius_um"] = 400.0
                item["adaptive_leak_mode"] = config.adaptive_leak_mode
                item["dopamine_leak_gain"] = config.dopamine_leak_gain
                item["burst_leak_gain"] = config.burst_leak_gain
                item["leak_min"] = config.leak_min
                item["leak_max"] = config.leak_max
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    cortex.add_layer("Nuclear", height_pct=1.0, density=0.4) \
        .add_population(exc_type, fraction=0.5) \
        .add_population(inh_type, fraction=0.2) \
        .add_population(motor_type, fraction=0.3)

    cortex.add_input("cartpole_sensors", width=8, height=8, entry_z="bottom")
    cortex.add_output("motor_out", width=16, height=8, target_type="Motor_Pyramidal")

    builder.build()

    print("\nRunning Genesis Baker...")
    brain_toml_path = os.path.join(config.output_dir, "brain.toml")
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "-p",
            "genesis-baker",
            "--bin",
            "baker",
            "--",
            "--brain",
            brain_toml_path,
        ],
        check=False,
    )

    if result.returncode != 0:
        print("\nERROR: CartPole brain bake failed. Inspect baker logs above.")
        sys.exit(result.returncode)

    print("\nCartPole model baked successfully.")
    return brain_toml_path


def parse_args() -> argparse.Namespace:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    default_out_dir = os.path.join(repo_root, "Genesis-Models", "CartPole-example")
    default_gnm_path = os.path.join(repo_root, "GNM-Library")

    parser = argparse.ArgumentParser(description="Build and bake the CartPole example brain.")
    parser.add_argument("--master-seed", default="GENESIS-CARTPOLE-M5", help="Deterministic topology seed.")
    parser.add_argument("--project-name", default="CartPoleAgent", help="Builder project name.")
    parser.add_argument("--output-dir", default=default_out_dir, help="Output directory for the generated brain.")
    parser.add_argument("--gnm-path", default=default_gnm_path, help="Path to the GNM library.")
    parser.add_argument("--sync-batch-ticks", type=int, default=20, help="Batch size baked into simulation.toml.")
    parser.add_argument("--tick-duration-us", type=int, default=100, help="Tick duration baked into simulation.toml.")
    parser.add_argument("--adaptive-leak-mode", type=int, default=0, help="Optional default adaptive leak mode baked into blueprints.")
    parser.add_argument("--dopamine-leak-gain", type=int, default=0, help="Optional default dopamine leak gain.")
    parser.add_argument("--burst-leak-gain", type=int, default=0, help="Optional default burst leak gain.")
    parser.add_argument("--leak-min", type=int, default=0, help="Optional default adaptive leak clamp minimum.")
    parser.add_argument("--leak-max", type=int, default=0, help="Optional default adaptive leak clamp maximum.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_virtualenv()
    build_cartpole_brain(
        BuildConfig(
            project_name=args.project_name,
            output_dir=os.path.abspath(args.output_dir),
            gnm_path=os.path.abspath(args.gnm_path),
            master_seed=args.master_seed,
            sync_batch_ticks=args.sync_batch_ticks,
            tick_duration_us=args.tick_duration_us,
            adaptive_leak_mode=args.adaptive_leak_mode,
            dopamine_leak_gain=args.dopamine_leak_gain,
            burst_leak_gain=args.burst_leak_gain,
            leak_min=args.leak_min,
            leak_max=args.leak_max,
        )
    )


if __name__ == "__main__":
    main()
