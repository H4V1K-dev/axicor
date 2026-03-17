# Genesis HFT: CartPole Benchmark Harness

`examples/cartpole_exp` is the canonical CartPole workflow for Milestone 5 benchmark and validation on Windows + CUDA.

## Unified Launcher (`run.py`)

Single entry point for all CartPole workflows. Run without arguments for an interactive menu:

```bash
# Interactive menu (choose command, then enter parameters)
python examples/cartpole_exp/run.py

# Or explicitly
python examples/cartpole_exp/run.py -i

# Build and bake the brain
python examples/cartpole_exp/run.py build

# Run agent (start node separately in another terminal)
python examples/cartpole_exp/run.py run --episodes 100

# Start/stop node (for run/render)
python examples/cartpole_exp/run.py start-node
python examples/cartpole_exp/run.py stop-node

# Run with live CartPole window (async render at 60 FPS, agent runs at full speed)
python examples/cartpole_exp/run.py render --episodes 50 --render-fps 60

# Full benchmark (spawns node, runs all scenarios)
python examples/cartpole_exp/run.py benchmark --quick
python examples/cartpole_exp/run.py benchmark --rebuild-brain --episodes 25 --seeds 101 202 303

# All-in-one: build + node + render
python examples/cartpole_exp/run.py full --episodes 200
```

## Build Once

Activate the project virtual environment, then bake a deterministic CartPole brain:

```bash
python examples/cartpole_exp/run.py build
# or: python examples/cartpole_exp/build_brain.py --master-seed GENESIS-CARTPOLE-M5
```

This generates the brain under `Genesis-Models/CartPole-example` and bakes the runtime artifacts via `genesis-baker`.

## Run The Full Milestone 5 Benchmark

The benchmark harness orchestrates the 5 roadmap scenarios:

- baseline
- dopamine-only adaptive leak
- burst-only adaptive leak
- combined adaptive leak
- combined adaptive leak with input noise

It writes per-run JSON artifacts plus a summary JSON/CSV report.

```bash
python examples/cartpole_exp/run.py benchmark --rebuild-brain --episodes 25 --seeds 101 202 303
```

Generated artifacts land in `artifacts/cartpole_benchmark/`:

- `raw_runs/*.json`: per-seed scenario results
- `benchmark_summary.json`: aggregated milestone verdict
- `benchmark_summary.csv`: spreadsheet-friendly summary
- `*_node.log`: captured `genesis-node` logs per isolated run

By default, the benchmark starts and stops `genesis-node` for every scenario/seed pair so weights do not leak across A/B runs.

## Run A Manual CartPole Session

If you want to inspect one scenario interactively, start the node first:

```bash
cargo run --release -p genesis-node -- --brain CartPole-example --log
```

Then run the Python agent in a second terminal:

```bash
python examples/cartpole_exp/run.py run --episodes 100 --scenario-name manual
```

For fixed-runtime experiments, disable the autotuner and patch adaptive leak directly from the CLI:

```bash
python examples/cartpole_exp/run.py run --episodes 40 --fixed-runtime --adaptive-leak-mode 1 --dopamine-leak-gain 96 --burst-leak-gain 24 --leak-min 637 --leak-max 1062
```

## Notes

- Interactive menu includes `start-node` and `stop-node` to manage genesis-node separately before running the agent.
- `run.py render` uses async rendering: the agent runs at full speed while a separate thread displays at `--render-fps` (default 60).
- `run.py build` / `build_brain.py` accept `--master-seed` so topology is reproducible across benchmark reruns.
- `run.py benchmark` / `benchmark.py` keep batch size and GSOP-sensitive runtime settings constant across scenarios.
- `run.py run` / `agent.py` emit machine-readable run metrics when `--output-path` is provided.
