#!/usr/bin/env python3
"""
Unified CartPole launcher — build, run, benchmark, render.

Single entry point for all CartPole workflows with a consistent CLI.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

# Ensure we can import from this package
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "genesis-client"))


def _bold(text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[1m{text}\033[0m"
    return text


def _dim(text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[2m{text}\033[0m"
    return text


def _cyan(text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[36m{text}\033[0m"
    return text


def _green(text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[32m{text}\033[0m"
    return text


def _yellow(text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[33m{text}\033[0m"
    return text


def _section(title: str) -> None:
    print()
    print(_bold(f"  {title}"))
    print(_dim("  " + "─" * (len(title) + 2)))


def _cmd(args: list[str], cwd: Path | None = None) -> int:
    """Run a command and return exit code."""
    return subprocess.call(args, cwd=cwd or _REPO_ROOT)


# Module-level state for node started from interactive menu
_node_process: subprocess.Popen | None = None
_node_log_handle = None


def _stop_managed_node() -> bool:
    """Stop the node we started from the menu. Returns True if stopped."""
    global _node_process, _node_log_handle
    if _node_process is None:
        return False
    if _node_process.poll() is not None:
        _node_process = None
        if _node_log_handle is not None:
            try:
                _node_log_handle.close()
            except Exception:
                pass
            _node_log_handle = None
        return False
    _node_process.terminate()
    try:
        _node_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _node_process.kill()
        _node_process.wait(timeout=5)
    _node_process = None
    if _node_log_handle is not None:
        try:
            _node_log_handle.close()
        except Exception:
            pass
        _node_log_handle = None
    return True


def _kill_node_by_name() -> bool:
    """Try to kill genesis-node and genesis-baker-daemon. Returns True if something was killed."""
    import platform
    killed = False
    if platform.system() == "Windows":
        for proc in ["genesis-node.exe", "genesis-baker-daemon.exe"]:
            r = subprocess.run(["taskkill", "/F", "/IM", proc], capture_output=True, text=True)
            if r.returncode == 0:
                killed = True
        return killed
    r = subprocess.run(["pkill", "-f", "genesis-node"], capture_output=True)
    if r.returncode == 0:
        killed = True
    r = subprocess.run(["pkill", "-f", "genesis-baker-daemon"], capture_output=True)
    if r.returncode == 0:
        killed = True
    return killed


def cmd_start_node(args: argparse.Namespace) -> int:
    """Start genesis-node in background (for run/render)."""
    import time

    _section("Start Genesis Node")
    from genesis.brain import fnv1a_32
    from genesis.memory import GenesisMemory

    global _node_process, _node_log_handle
    _stop_managed_node()
    # Ensure no orphaned node/baker holds port 8081 or menu_node.log
    if _kill_node_by_name():
        time.sleep(1.0)

    manifest_path = str(_REPO_ROOT / "Genesis-Models" / args.brain_name / "baked" / "SensoryCortex" / "manifest.toml")
    if not os.path.exists(manifest_path):
        print(_yellow("  Brain not baked. Run 'build' first."))
        return 1

    from agent import AdaptiveLeakConfig
    from benchmark import apply_scenario_to_manifest

    fast_path_port = args.node_port + 1000
    apply_scenario_to_manifest(
        manifest_path,
        AdaptiveLeakConfig(
            adaptive_leak_mode=1,
            dopamine_leak_gain=1000,
            burst_leak_gain=24,
            leak_min=50,
            leak_max=800,
            variant_ids=[2],
        ),
        node_port=args.node_port,
        response_port=args.response_port,
        fast_path_port=fast_path_port,
    )
    time.sleep(0.5)

    log_path = _REPO_ROOT / "artifacts" / "cartpole_benchmark" / "menu_node.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _node_log_handle = open(log_path, "w", encoding="utf-8")
    _node_process = subprocess.Popen(
        ["cargo", "run", "--release", "-p", "genesis-node", "--", "--brain", args.brain_name, "--log"],
        cwd=_REPO_ROOT,
        stdout=_node_log_handle,
        stderr=subprocess.STDOUT,
    )

    zone_hash = fnv1a_32(b"SensoryCortex")
    shm_path = GenesisMemory._resolve_path(zone_hash)
    for _ in range(30):
        if _node_process.poll() is not None:
            print(_yellow("  Node exited early. Check ") + str(log_path))
            _node_process = None
            _node_log_handle.close()
            _node_log_handle = None
            return 1
        if os.path.exists(shm_path):
            time.sleep(2.0)
            break
        time.sleep(0.5)
    else:
        _stop_managed_node()
        print(_yellow("  Node did not become ready in time."))
        return 1

    print(_green("  Node ready. Use 'run' or 'render' to start the agent."))
    return 0


def cmd_stop_node(args: argparse.Namespace) -> int:
    """Stop genesis-node (started from this menu or by name)."""
    _section("Stop Genesis Node")
    if _stop_managed_node():
        print(_green("  Node stopped."))
        return 0
    if _kill_node_by_name():
        print(_green("  Node process(es) stopped."))
        return 0
    print(_dim("  No genesis-node process found."))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Build and bake the CartPole brain."""
    _section("Build CartPole Brain")
    from build_brain import BuildConfig, build_cartpole_brain, ensure_virtualenv

    ensure_virtualenv()
    output_dir = _REPO_ROOT / "Genesis-Models" / args.brain_name
    gnm_path = _REPO_ROOT / "GNM-Library"
    build_cartpole_brain(
        BuildConfig(
            project_name="CartPoleAgent",
            output_dir=str(output_dir),
            gnm_path=str(gnm_path),
            master_seed=args.master_seed,
        )
    )
    print(_green("  Done. Brain baked at: ") + str(output_dir))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the CartPole agent (node must be running in another terminal)."""
    _section("Run CartPole Agent")
    print(_dim("  Ensure genesis-node is running: cargo run --release -p genesis-node -- --brain CartPole-example --log"))
    print()

    agent_args = [
        sys.executable,
        "-u",
        str(_SCRIPT_DIR / "agent.py"),
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--scenario-name", args.scenario_name,
        "--node-port", str(args.node_port),
        "--response-port", str(args.response_port),
    ]
    if args.fixed_runtime:
        agent_args.append("--fixed-runtime")
    if args.quiet:
        agent_args.append("--quiet")
    if args.output_path:
        agent_args.append("--output-path")
        agent_args.append(args.output_path)
    if args.manifest_path:
        agent_args.append("--manifest-path")
        agent_args.append(args.manifest_path)
    if args.stats_sample_stride != 5:
        agent_args.extend(["--stats-sample-stride", str(args.stats_sample_stride)])
    if args.adaptive_leak_mode != 0:
        agent_args.extend(["--adaptive-leak-mode", str(args.adaptive_leak_mode)])
    if args.dopamine_leak_gain != 0:
        agent_args.extend(["--dopamine-leak-gain", str(args.dopamine_leak_gain)])
    if args.burst_leak_gain != 0:
        agent_args.extend(["--burst-leak-gain", str(args.burst_leak_gain)])
    if args.leak_min != 0 or args.leak_max != 0:
        agent_args.extend(["--leak-min", str(args.leak_min), "--leak-max", str(args.leak_max)])
    vids = getattr(args, "variant_ids", None)
    if vids:
        agent_args.extend(["--variant-ids"] + [str(v) for v in vids])
    if getattr(args, "use_3_actions", False):
        agent_args.append("--use-3-actions")
    agent_args.extend(["--max-episode-steps", str(getattr(args, "max_episode_steps", 5000))])

    return _cmd(agent_args, cwd=_SCRIPT_DIR)


def cmd_render(args: argparse.Namespace) -> int:
    """Run CartPole with live rendering (node must be running)."""
    _section("Run CartPole with Render")
    print(_dim("  Ensure genesis-node is running in another terminal."))
    print()

    agent_args = [
        sys.executable,
        "-u",
        str(_SCRIPT_DIR / "agent.py"),
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--render",
        "--scenario-name", args.scenario_name,
        "--node-port", str(args.node_port),
        "--response-port", str(args.response_port),
    ]
    if args.render_fps != 60.0:
        agent_args.extend(["--render-fps", str(args.render_fps)])
    if args.output_path:
        agent_args.extend(["--output-path", args.output_path])
    if args.manifest_path:
        agent_args.extend(["--manifest-path", args.manifest_path])
    agent_args.extend(["--max-episode-steps", str(getattr(args, "max_episode_steps", 5000))])

    return _cmd(agent_args, cwd=_SCRIPT_DIR)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the full Milestone 5 benchmark (spawns node, runs all scenarios)."""
    _section("CartPole Benchmark")
    print(_dim(f"  Episodes: {args.episodes}  Seeds: {args.seeds}  Artifacts: {args.artifacts_dir}"))
    print()

    manifest_path = args.manifest_path or str(_REPO_ROOT / "Genesis-Models" / args.brain_name / "baked" / "SensoryCortex" / "manifest.toml")
    benchmark_args = [
        sys.executable,
        "-u",
        str(_SCRIPT_DIR / "benchmark.py"),
        "--episodes", str(args.episodes),
        "--seeds", *map(str, args.seeds),
        "--brain-name", args.brain_name,
        "--manifest-path", manifest_path,
        "--artifacts-dir", args.artifacts_dir,
        "--master-seed", args.master_seed,
        "--node-port", str(args.node_port),
        "--response-port", str(args.response_port),
    ]
    if args.rebuild_brain:
        benchmark_args.append("--rebuild-brain")
    if args.quick:
        benchmark_args.append("--quick")
    if args.reuse_running_node:
        benchmark_args.append("--reuse-running-node")
    if args.dry_run:
        benchmark_args.append("--dry-run")

    return _cmd(benchmark_args, cwd=_SCRIPT_DIR)


def cmd_full(args: argparse.Namespace) -> int:
    """Build brain, start node, run agent with render — all in one (node in subprocess)."""
    import time

    _section("Full Run: Build + Node + Agent (Render)")
    from build_brain import BuildConfig, build_cartpole_brain, ensure_virtualenv
    from genesis.brain import fnv1a_32
    from genesis.memory import GenesisMemory

    ensure_virtualenv()
    manifest_path = str(_REPO_ROOT / "Genesis-Models" / args.brain_name / "baked" / "SensoryCortex" / "manifest.toml")
    fast_path_port = args.node_port + 1000

    # 1. Build
    print(_cyan("  [1/3] Building brain..."))
    output_dir = _REPO_ROOT / "Genesis-Models" / args.brain_name
    gnm_path = _REPO_ROOT / "GNM-Library"
    build_cartpole_brain(
        BuildConfig(
            project_name="CartPoleAgent",
            output_dir=str(output_dir),
            gnm_path=str(gnm_path),
            master_seed=args.master_seed,
        )
    )
    print(_green("  Done.\n"))

    # 2. Patch manifest (ports) and start node
    from agent import AdaptiveLeakConfig
    from benchmark import apply_scenario_to_manifest

    # Combined adaptive leak: dopamine + burst modulation for Motor_Pyramidal (variant 2)
    apply_scenario_to_manifest(
        manifest_path,
        AdaptiveLeakConfig(
            adaptive_leak_mode=1,
            dopamine_leak_gain=1000,
            burst_leak_gain=24,
            leak_min=50,
            leak_max=800,
            variant_ids=[2],
        ),
        node_port=args.node_port,
        response_port=args.response_port,
        fast_path_port=fast_path_port,
    )
    time.sleep(0.5)

    print(_cyan("  [2/3] Starting genesis-node..."))
    log_path = _REPO_ROOT / "artifacts" / "cartpole_benchmark" / "run_node.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            ["cargo", "run", "--release", "-p", "genesis-node", "--", "--brain", args.brain_name, "--log"],
            cwd=_REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

    # Wait for SHM
    zone_hash = fnv1a_32(b"SensoryCortex")
    shm_path = GenesisMemory._resolve_path(zone_hash)
    for _ in range(30):
        if proc.poll() is not None:
            print(_yellow("  Node exited early. Check ") + str(log_path))
            return 1
        if os.path.exists(shm_path):
            time.sleep(2.0)
            break
        time.sleep(0.5)
    else:
        proc.terminate()
        print(_yellow("  Node did not become ready in time."))
        return 1
    print(_green("  Node ready.\n"))

    # 3. Run agent with render
    print(_cyan("  [3/3] Running agent with render..."))
    agent_args = [
        sys.executable, "-u", str(_SCRIPT_DIR / "agent.py"),
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--render",
        "--scenario-name", "full_run",
        "--node-port", str(args.node_port),
        "--response-port", str(args.response_port),
    ]
    if args.render_fps != 60.0:
        agent_args.extend(["--render-fps", str(args.render_fps)])
    agent_args.extend(["--max-episode-steps", str(getattr(args, "max_episode_steps", 5000))])

    try:
        code = _cmd(agent_args, cwd=_SCRIPT_DIR)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return code


def _prompt(text: str, default: str | int | float = "") -> str:
    """Prompt for input with optional default. Returns string."""
    if default != "":
        prompt = f"{text} [{default}]: "
    else:
        prompt = f"{text}: "
    result = input(_cyan(prompt)).strip()
    return str(result) if result else str(default)


def _prompt_int(text: str, default: int) -> int:
    """Prompt for integer."""
    while True:
        raw = _prompt(text, default)
        try:
            return int(raw)
        except ValueError:
            print(_yellow("  Enter a number."))


def _prompt_float(text: str, default: float) -> float:
    """Prompt for float."""
    while True:
        raw = _prompt(text, default)
        try:
            return float(raw)
        except ValueError:
            print(_yellow("  Enter a number."))


def _prompt_yes_no(text: str, default: bool = True) -> bool:
    """Prompt for y/n."""
    d = "Y/n" if default else "y/N"
    raw = _prompt(text, d).lower()
    if not raw or raw == d.split("/")[0].lower():
        return default
    return raw in ("y", "yes", "1")


def _interactive_menu(parent_args: argparse.Namespace) -> int:
    """Show interactive menu and run selected command."""
    menu = [
        ("build", "Build and bake the CartPole brain", cmd_build),
        ("start-node", "Start genesis-node (background)", cmd_start_node),
        ("stop-node", "Stop genesis-node", cmd_stop_node),
        ("run", "Run agent (node must be running)", cmd_run),
        ("render", "Run with live CartPole window", cmd_render),
        ("benchmark", "Full Milestone 5 benchmark", cmd_benchmark),
        ("full", "Build + node + render (all-in-one)", cmd_full),
        ("quit", "Exit", None),
    ]

    while True:
        print()
        print(_bold("  CartPole Launcher"))
        print(_dim("  " + "─" * 40))
        for i, (name, desc, _) in enumerate(menu, 1):
            print(f"  {_cyan(str(i))}) {name:<12} {_dim(desc)}")
        print()

        choice = _prompt("Choice", "1").strip()
        if not choice:
            choice = "1"

        try:
            idx = int(choice)
        except ValueError:
            idx = next((i for i, (n, _, _) in enumerate(menu, 1) if n.startswith(choice.lower())), 0)
        else:
            if 1 <= idx <= len(menu):
                pass
            else:
                idx = 0

        if idx == 0:
            print(_yellow("  Invalid choice."))
            continue

        name, _, func = menu[idx - 1]
        if func is None:
            if _stop_managed_node():
                print(_green("  Node stopped."))
            print(_green("  Bye."))
            return 0

        # Build args namespace from parent + prompts
        class Args:
            pass

        args = Args()
        args.manifest_path = parent_args.manifest_path or ""
        args.node_port = parent_args.node_port
        args.response_port = parent_args.response_port
        args.brain_name = parent_args.brain_name
        args.master_seed = parent_args.master_seed

        if name == "build":
            args.master_seed = _prompt("Master seed", args.master_seed)

        elif name == "run":
            args.episodes = _prompt_int("Episodes", 100)
            args.seed = _prompt_int("Seed", 123)
            args.scenario_name = _prompt("Scenario name", "manual")
            args.fixed_runtime = _prompt_yes_no("Fixed runtime (no autotuner)?", False)
            args.quiet = _prompt_yes_no("Quiet (less logging)?", False)
            args.output_path = _prompt("Output JSON path (empty to skip)", "")
            args.stats_sample_stride = _prompt_int("Stats sample stride", 5)
            args.adaptive_leak_mode = _prompt_int("Adaptive leak mode (0=off, 1=continuous)", 1)
            args.dopamine_leak_gain = _prompt_int("Dopamine leak gain", 1000)
            args.burst_leak_gain = _prompt_int("Burst leak gain", 24)
            args.leak_min = _prompt_int("Leak min", 50)
            args.leak_max = _prompt_int("Leak max", 800)
            args.variant_ids = [2]
            args.use_3_actions = _prompt_yes_no("3 actions (Left-Wait-Right)?", False)
            args.max_episode_steps = _prompt_int("Max episode steps", 5000)

        elif name == "render":
            args.episodes = _prompt_int("Episodes", 100)
            args.seed = _prompt_int("Seed", 123)
            args.scenario_name = _prompt("Scenario name", "render")
            args.render_fps = _prompt_float("Render FPS", 60.0)
            args.max_episode_steps = _prompt_int("Max episode steps", 5000)
            args.output_path = _prompt("Output JSON path (empty to skip)", "")

        elif name == "benchmark":
            args.episodes = _prompt_int("Episodes per seed", 25)
            seeds_str = _prompt("Seeds (space-separated)", "101 202 303")
            try:
                args.seeds = [int(s) for s in seeds_str.split() if s.strip()] or [101, 202, 303]
            except ValueError:
                args.seeds = [101, 202, 303]
            args.quick = _prompt_yes_no("Quick (5 eps, 1 seed)?", False)
            args.rebuild_brain = _prompt_yes_no("Rebuild brain first?", False)
            args.reuse_running_node = _prompt_yes_no("Reuse running node?", False)
            args.dry_run = _prompt_yes_no("Dry run (no execute)?", False)
            args.artifacts_dir = str(_REPO_ROOT / "artifacts" / "cartpole_benchmark")

        elif name == "full":
            args.episodes = _prompt_int("Episodes", 10000)
            args.seed = _prompt_int("Seed", 123)
            args.render_fps = _prompt_float("Render FPS", 60.0)
            args.max_episode_steps = _prompt_int("Max episode steps", 5000)

        elif name == "start-node":
            args.brain_name = _prompt("Brain name", args.brain_name)

        print()
        code = func(args)
        if code != 0:
            print(_yellow(f"  Exit code: {code}"))
        if not _prompt_yes_no("\nRun another command?", True):
            return code


def _handle_sigint(signum: int, frame) -> None:  # noqa: ARG001
    """Ctrl+C: stop managed node and exit cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if _stop_managed_node():
        print(_green("\n  Node stopped."))
    print(_yellow("  Interrupted."))
    sys.exit(130)


def main() -> int:
    signal.signal(signal.SIGINT, _handle_sigint)
    parser = argparse.ArgumentParser(
        description="CartPole unified launcher — build, run, benchmark, render.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  run                          Interactive menu (no args)
  run -i                       Interactive menu
  run build                    Build and bake the brain
  run run --episodes 100       Run agent (node must be running)
  run render --episodes 50     Run with live CartPole window
  run benchmark --quick        Quick benchmark (5 eps, 1 seed)
  run full --episodes 10000    Build + node + render in one go
        """,
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive menu mode")
    parser.add_argument("--manifest-path", default="", help="Override manifest path")
    parser.add_argument("--node-port", type=int, default=8081, help="Genesis node UDP port")
    parser.add_argument("--response-port", type=int, default=8092, help="Agent response port")
    parser.add_argument("--brain-name", default="CartPole-example", help="Brain name for node/build")
    parser.add_argument("--master-seed", default="GENESIS-CARTPOLE-M5", help="Topology seed for build")

    subparsers = parser.add_subparsers(dest="command", required=False, help="Command to run")

    # build
    p_build = subparsers.add_parser("build", help="Build and bake the CartPole brain")
    p_build.set_defaults(func=cmd_build)

    # run
    p_run = subparsers.add_parser("run", help="Run agent (start node separately)")
    p_run.add_argument("--episodes", type=int, default=100)
    p_run.add_argument("--seed", type=int, default=123)
    p_run.add_argument("--scenario-name", default="manual")
    p_run.add_argument("--fixed-runtime", action="store_true")
    p_run.add_argument("--quiet", action="store_true")
    p_run.add_argument("--output-path", default="")
    p_run.add_argument("--stats-sample-stride", type=int, default=5)
    p_run.add_argument("--adaptive-leak-mode", type=int, default=1, help="0=off, 1=continuous (default for learning)")
    p_run.add_argument("--dopamine-leak-gain", type=int, default=1000)
    p_run.add_argument("--burst-leak-gain", type=int, default=24)
    p_run.add_argument("--leak-min", type=int, default=50)
    p_run.add_argument("--leak-max", type=int, default=800)
    p_run.add_argument("--variant-ids", type=int, nargs="*", default=[2], help="Variant IDs for adaptive leak (default: Motor_Pyramidal)")
    p_run.add_argument("--use-3-actions", action="store_true", help="Left(0), Wait(1), Right(2) instead of Left(0), Right(1)")
    p_run.add_argument("--max-episode-steps", type=int, default=5000, help="Max steps per episode (default: 5000)")
    p_run.set_defaults(func=cmd_run)

    # render
    p_render = subparsers.add_parser("render", help="Run agent with live CartPole window (async render)")
    p_render.add_argument("--episodes", type=int, default=100)
    p_render.add_argument("--seed", type=int, default=123)
    p_render.add_argument("--scenario-name", default="render")
    p_render.add_argument("--render-fps", type=float, default=60.0, help="Render frame rate (default: 60)")
    p_render.add_argument("--output-path", default="")
    p_render.add_argument("--max-episode-steps", type=int, default=5000, help="Max steps per episode (default: 5000)")
    p_render.set_defaults(func=cmd_render)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run full Milestone 5 benchmark")
    p_bench.add_argument("--episodes", type=int, default=25)
    p_bench.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    p_bench.add_argument("--quick", action="store_true")
    p_bench.add_argument("--rebuild-brain", action="store_true")
    p_bench.add_argument("--reuse-running-node", action="store_true")
    p_bench.add_argument("--dry-run", action="store_true")
    p_bench.add_argument("--artifacts-dir", default=str(_REPO_ROOT / "artifacts" / "cartpole_benchmark"))
    p_bench.set_defaults(func=cmd_benchmark)

    # full
    p_full = subparsers.add_parser("full", help="Build + start node + run with render (all-in-one)")
    p_full.add_argument("--episodes", type=int, default=10000)
    p_full.add_argument("--seed", type=int, default=123)
    p_full.add_argument("--render-fps", type=float, default=60.0, help="Render frame rate (default: 60)")
    p_full.add_argument("--max-episode-steps", type=int, default=5000, help="Max steps per episode (default: 5000)")
    p_full.set_defaults(func=cmd_full)

    # start-node
    p_start = subparsers.add_parser("start-node", help="Start genesis-node in background")
    p_start.set_defaults(func=cmd_start_node)

    # stop-node
    p_stop = subparsers.add_parser("stop-node", help="Stop genesis-node")
    p_stop.set_defaults(func=cmd_stop_node)

    args = parser.parse_args()

    if args.interactive or args.command is None:
        if not sys.stdin.isatty():
            print("Interactive mode requires a TTY. Use: run <command> [options]")
            return 1
        return _interactive_menu(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
