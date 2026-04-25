<div align="center">
  <img src="./LOGO.svg" height="512" alt="Axicor Logo">
  <h1>Axicor 0.1.0 Public</h1>
  <a href="https://github.com/H4V1K-dev/axicor"><img src="https://img.shields.io/badge/last%20stable%20commit-current-red" alt="Last Stable Commit"></a>
  <p><strong>Neuromorphic Computing Engine. SNN AI Architecture base from biology.</strong></p>

  <p>
    <a href="https://github.com/H4V1K-dev/Axicor/actions/workflows/ci.yml"><img src="https://github.com/H4V1K-dev/Axicor/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/axicor-core"><img src="https://img.shields.io/crates/v/axicor-core.svg" alt="Crates.io"></a>
    <a href="https://docs.rs/axicor-core"><img src="https://img.shields.io/docsrs/axicor-core" alt="docs.rs"></a>
    <a href="https://pypi.org/project/axicor-client/"><img src="https://img.shields.io/pypi/v/axicor-client.svg" alt="PyPI"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg" alt="License"></a>
    <a href="https://github.com/H4V1K-dev/axicor/commits"><img src="https://img.shields.io/github/last-commit/H4V1K-dev/axicor" alt="Last Commit"></a>
  </p>

  <p>
    <a href="https://releases.rs/docs/1.75.0/"><img src="https://img.shields.io/badge/MSRV-1.75-informational" alt="MSRV"></a>
    <a href="https://crates.io/crates/axicor-core"><img src="https://img.shields.io/crates/d/axicor-core" alt="Crates Downloads"></a>
    <a href="https://pypi.org/project/axicor-client/"><img src="https://img.shields.io/pypi/dm/axicor-client" alt="PyPI Downloads"></a>
    <a href="https://deps.rs/repo/github/H4V1K-dev/Axicor"><img src="https://deps.rs/repo/github/H4V1K-dev/Axicor/status.svg" alt="dependency status"></a>
    <a href="https://axicor.dev"><img src="https://img.shields.io/badge/homepage-axicor.dev-blue" alt="Homepage"></a>
  </p>

  <p>
    <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/Rust-stable-orange?logo=rust" alt="Rust"></a>
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python"></a>
    <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia" alt="CUDA"></a>
    <a href="https://rocm.docs.amd.com"><img src="https://img.shields.io/badge/ROCm-HIP-ED1C24?logo=amd" alt="ROCm"></a>
    <a href="https://t.me/+zptNJAJhDe41ZTEy"><img src="https://img.shields.io/badge/Telegram-Contributors-26A5E4?logo=telegram" alt="Telegram"></a>
    <a href="https://ko-fi.com/axicor"><img src="https://img.shields.io/badge/Support-Ko--fi-FF5E5B?logo=ko-fi&logoColor=white" alt="Ko-fi"></a>
  </p>

  <p>
    <strong>Get AI Help:</strong>
    <a href="https://notebooklm.google.com/notebook/6b731a06-043c-4959-a557-f088b6c1216c"><img src="https://img.shields.io/badge/🫀_Core-Expert-red" alt="Core Expert"></a>
    <a href="https://notebooklm.google.com/notebook/b3baaac2-91a1-4988-94aa-430ba760a0a6"><img src="https://img.shields.io/badge/🔥_Compute-Expert-orange" alt="Compute Expert"></a>
    <a href="https://notebooklm.google.com/notebook/d540c34f-0ace-4ed6-b9f7-5ae9c4adbdff"><img src="https://img.shields.io/badge/📡_Node-Expert-blue" alt="Node Expert"></a>
    <a href="https://notebooklm.google.com/notebook/c412b68a-be7b-4f17-bb50-61f74667af21"><img src="https://img.shields.io/badge/🐍_SDK-Expert-green" alt="SDK Expert"></a>
    <a href="https://notebooklm.google.com/notebook/f5cf46c6-8f6c-4bdb-9cf0-979080fb5c06"><img src="https://img.shields.io/badge/🧬_Baker-Expert-purple" alt="Baker Expert"></a>
  </p>
</div>

---

A living brain for embodied AI. Learns in seconds. Runs everywhere - from ESP32 microcontrollers to GPU clusters. Axicor is a Spiking Neural Network engine designed for biological realism and absolute determinism. No backpropagation. No static computation graphs. Neurons fire spikes, axons physically grow through 3D space, weak synapses are pruned, strong ones fortified - all in real-time, without halting the simulation.

## Why Axicor

**[Watch the live demo ->](https://www.tiktok.com/@alex_0xgenesis_agi/video/7617581624011541781)** - an ant agent in `Gymnasium-AntV4`, no pre-training, walking within 30 seconds of spawn.

- **Behavior out of the box.** An ant agent spawned in `Gymnasium-AntV4` stands upright within seconds and begins exploratory locomotion within 30 seconds of instantiation - with **zero pre-training**. No gradient descent, no replay buffer, no training epochs. Behavior emerges from biological priors encoded directly in the connectome topology.
- **Bit-exact determinism across hardware.** Integer-only physics (GLIF, GSOP) guarantees identical simulation results on an RTX 4090, AMD RDNA, or an ESP32. No floating-point drift, no cross-platform divergence.
- **Real-time structural plasticity.** Axons physically grow and prune during the Night Phase while the Day Phase hot-loop keeps ticking. No "training mode" vs "inference mode" divide - the brain rewires itself while it lives.
- **Modest hardware friendly.** Reference demos run on a GTX 1080 Ti. No cloud-scale compute required.

## Core Architecture

Axicor is built on three Data-Oriented Design pillars:

1. **Integer Physics.** All membrane integration (GLIF - Generalized Leaky Integrate-and-Fire) and plasticity (GSOP - Generalized Synaptic Offset Plasticity) math is performed using 100% branchless integer arithmetic. Zero floats. This guarantees bit-exact determinism across entirely different silicon.
2. **Day / Night Cycle.** The GPU/MCU hot loop (Day Phase) is completely isolated from structural plasticity. It only computes spike propagation. The CPU (Night Phase) handles routing, axon sprouting, and memory defragmentation.
3. **Headerless Structure of Arrays (SoA).** Complete rejection of OOP. Neurons do not exist as objects - data is laid out in flat, warp-aligned SoA memory dumps (`.state`, `.axons`) loaded directly into VRAM via zero-copy DMA (`mmap`). Cache-miss rate is practically zero.

## Ecosystem

- **[axicor-core](axicor-core/)** - FFI memory contracts, SoA layouts, `bytemuck`-verified C-ABI structures, IPC abstractions.
- **[axicor-compute](axicor-compute/)** - dual-backend compute kernels (NVIDIA CUDA / AMD HIP) with CPU fallback (`mock-gpu` feature).
- **[axicor-baker](axicor-baker/)** - offline topology compiler. Reads TOML "brain DNA", grows axons via cone tracing, emits binary `.state` / `.axons` dumps.
- **[axicor-node](axicor-node/)** - BSP-lockstep orchestrator, UDP fast-path server, Night Phase daemon.
- **[axicor-client](axicor-client/)** - Python SDK for RL integration. Zero-garbage UDP client, aligned `memoryview` encoders/decoders.

## Quickstart (Windows)

Axicor requires **Rust stable (1.75+)**, **Python 3.10/3.11**, and **Visual Studio C++ Build Tools**. A CUDA-capable GPU or AMD ROCm device is recommended but not required — CPU fallback is shipped in the `mock-gpu` feature.

### 1. Prerequisite: C++ Build Tools
Before running the setup, ensure you have the MSVC compiler installed:
1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&channel=Stable&version=VS18&source=VSLandingPage&cid=2500&passive=false).
2. During installation, check **"Desktop development with C++"**.

### 2. Clone & Bootstrap
Open **PowerShell** and run the following commands. The automated bootstrap script will detect your GPU (CUDA/HIP/CPU), install missing dependencies (Python, Rust, Git) via `winget`, set up the virtual environment, and compile the engine.

```powershell
# Clone the repository
cd ~
git clone https://github.com/H4V1K-dev/axicor.git
cd axicor

# Run the automated bootstrap script
# Copy-paste this command into PowerShell:
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

### 3. Run the Embodied AI Demo

Once the bootstrap completes successfully, it will print the exact commands tailored to your hardware. Open **two terminals** in the `axicor` folder:

**Terminal 1 (Build Brain & Start Node):**
```powershell
.venv\Scripts\Activate.ps1
python examples\ant_exp\build_brain.py  # Add --cpu if you don't have a GPU

# Start the HFT Node (use the exact command provided by setup.ps1 at the end of its run)
cargo run --release -p axicor-node -- Axicor-Models\AntConnectome.axic --log
```
*Wait for `[Node] Bootstrap Successful. Hands-off to NodeRuntime.`*

**Terminal 2 (Start the RL Agent):**
```powershell
.venv\Scripts\Activate.ps1
python examples\ant_exp\ant_agent.py
```

*Note: If you are running inside a Virtual Machine without OpenGL support, you will get a GLFWError. Open `examples/ant_exp/ant_agent.py` and change `render_mode="human"` to `render_mode=None` to run in headless mode.*

## Quickstart (Linux)

Tested on **Debian 13 (Trixie)**. The automated bootstrap script handles Rust, Python (via `uv`), and C-ABI dependencies.

### 1. Install Base Dependencies
```bash
sudo apt-get update
sudo apt-get install -y git curl
```
*Note: If you just started your VM, wait 1-2 minutes before running to avoid `apt` lock errors.*

### 2. Clone & Bootstrap
```bash
git clone https://github.com/H4V1K-dev/axicor.git
cd axicor
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 3. Build & Launch Node (New Terminal)
Ensure you use a **new terminal** session to pick up the updated `$PATH` (Cargo/Rust).
```bash
cd axicor
source .venv/bin/activate
python3 examples/ant_exp/build_brain.py
cargo run --release -p axicor-node --features axicor-compute/mock-gpu -- Axicor-Models/AntConnectome.axic --log
```

### 4. Run RL Agent (Second New Terminal)
```bash
cd axicor
source .venv/bin/activate
python3 examples/ant_exp/ant_agent.py
```
*Note: If you are running inside a Virtual Machine without OpenGL support, open `examples/ant_exp/ant_agent.py` and change `render_mode="human"` to `render_mode=None` to run in headless mode.*

> **⚠️ IMPORTANT NOTICE:** All commands above MUST be executed from the root of the project directory. When you open a new terminal, always ensure you navigate to the project folder first:
> ```bash
> cd ~/axicor
> ```

## Engineering Invariants

Axicor makes a handful of deliberate design choices that differ from mainstream deep-learning frameworks. These are contracts, not preferences.

- **Little-Endian Network Law.** All network packets in the Axicor cluster transmit raw little-endian bytes, ignoring RFC Network Byte Order. This avoids `ntohl` / `htonl` CPU overhead on the hot path. All clients and nodes must adhere to this contract.
- **Dale's Law.** Synapse sign (excitatory / inhibitory) is immutable after connectome compilation. Plasticity can only modify magnitude.
- **BSP lockstep.** All shards advance in epoch-aligned barriers. No out-of-order simulation between zones.
- **Deterministic RNG.** All stochastic processes (spontaneous firing, axon growth noise) derive from `wyhash(master_seed, entity_id)`. Two runs with the same seed produce byte-identical results.

## 🤖 AI Crate Experts

Each crate has a dedicated AI assistant trained on its architecture, invariants, and C-ABI contracts. Click to open a conversation:

| | Expert | What I Know | Ask Me About |
|---|--------|-------------|---------------|
| 🫀 | [**Core**](https://notebooklm.google.com/notebook/6b731a06-043c-4959-a557-f088b6c1216c) | Axicor's binary foundation: integer physics, bitwise packing, and C-ABI memory contracts. | Struct alignment, bit packing, or synaptic arithmetic without floats. |
| 🔥 | [**Compute**](https://notebooklm.google.com/notebook/b3baaac2-91a1-4988-94aa-430ba760a0a6) | GPU simulation pipeline: CUDA kernels, VRAM management, and warp-level processing of 100,000+ neurons. | Simulation kernels, video memory, or Rust↔CUDA FFI contracts. |
| 📡 | [**Node**](https://notebooklm.google.com/notebook/d540c34f-0ace-4ed6-b9f7-5ae9c4adbdff) | Network orchestrator: UDP fast-path, BSP sync barrier, and signal routing between zones. | Data exchange protocol, InputSwapchain mechanics, or network stack debugging. |
| 🐍 | [**SDK**](https://notebooklm.google.com/notebook/c412b68a-be7b-4f17-bb50-61f74667af21) | Python client internals: zero-copy integration, memoryview, pre-allocated buffers, and allocation-free UDP. | Writing an agent, connecting an environment, or synapse surgery via AxicorSurgeon. |
| 🧬 | [**Baker**](https://notebooklm.google.com/notebook/f5cf46c6-8f6c-4bdb-9cf0-979080fb5c06) | Night Phase pipeline: genome compilation, axon growth, and packing neural networks into `.axic` binaries. | Building, rebuilding, or debugging an `.axic` file. |

## Documentation

Full architectural specification, neuron models, and C-ABI contract reference are available in the [Axicor Book](docs/src/SUMMARY.md).

**Architecture & Physics:**
- [Foundations & Philosophy](docs/src/architecture/foundations.md) - Data-Oriented Design and Integer Math constraints.
- [Neuron Model (GLIF)](docs/src/architecture/neuron-model.md) - Generalized Leaky Integrate-and-Fire math.
- [Baking Pipeline](docs/src/architecture/baking-pipeline.md) - TOML DNA -> binary memory dumps and axonal Cone Tracing.

**Integration & SDK:**
- [Python SDK](docs/src/guides/python-sdk.md) - Zero-garbage UDP client, `memoryview` encoders, and surgery APIs.
- [Brain Builder](docs/src/guides/brain-builder.md) - Python API for generating connectome topologies.
- [IO Matrix Protocol](docs/src/architecture/io-matrix.md) - UDP fast-path encoding for interacting with the cluster.

**Reference:**
- [C-ABI Contracts](docs/src/reference/c-abi-contracts.md) - `#[repr(C)]` layouts with compile-time static assertions.
- [Hardware Backends](docs/src/reference/hardware-backends.md) - CUDA vs AMD HIP implementation details.

## Community

- **Telegram:** [Contributors group](https://t.me/+zptNJAJhDe41ZTEy) with per-crate topic channels.
- **Issues:** [GitHub Issues](https://github.com/H4V1K-dev/Axicor/issues) for bug reports and feature requests.
- **Homepage:** [axicor.dev](https://axicor.dev)
- **Support development:** [Ko-fi](https://ko-fi.com/axicor)

## License

Licensed under either of:

- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) . [apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0))
- **MIT License** ([LICENSE-MIT](LICENSE-MIT) . [opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Axicor by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms.
