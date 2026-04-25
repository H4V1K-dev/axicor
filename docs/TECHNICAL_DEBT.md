# Technical Debt & Monolith Mapping

This document maps the identified monolithic components within the Axicor engine and defines the strategic refactoring path to maintain HFT performance and biological realism.

## Monolith Registry

| Component | File | Severity | Complexity (Est. LOC) | Problem Description | Refactoring Recommendation |
|-----------|------|----------|------------------------|---------------------|----------------------------|
| `Teardown Logic` | `core/layout.rs` | **HIGH** | 50+ | [DEINIT] C-ABI Teardown Race. OS destroys CUDA context before Rust's `Drop` (VramState) calls `cudaFree`, causing `0xc0000374` heap corruption on process exit. | Implement a dedicated `Teardown()` sequence to deallocate GPU resources BEFORE the main process exit handler. |
| `Neuron Library` | `scripts/Axicor_Neuron-Library/` | **HIGH** | N/A | [MISSING] Subcortical & Drosophila models lost. Neocortex-only library lacks Thalamus, Hippocampus, Basal Ganglia, Cerebellum, and Drosophila connectome support. Legacy scripts are missing. | Restore `generate_subcortical_library.py` and `normalize_drosophila.py`. Expand catalog to include full biological structures. |
| `UDP Orchestrator`| `network/io_server.rs` | **HIGH** | 100+ | [NETWORK] UDP MTU Overflow. Output matrices exceeding 65,507 bytes (`MAX_UDP_PAYLOAD`) cause buffer overflow and fatal crashes when resolution/sync_batch_ticks are high. | Implement L7 fragmentation for large payload chunks or enforce resolution limits in `io.toml`. |
| `NodeRuntime` | `node/mod.rs` | **HIGH** | 500+ | God-object managing lifecycle, compute dispatch, and network topology. | Decompose into `LifecycleManager`, `ComputeOrchestrator`, and `NetworkRegistry`. |
| `run_node_loop` | `node/mod.rs` | **HIGH** | 200+ | Procedural loop mixing dispatch, feedback, sync, and heartbeats. | Transform into a strict State Machine driving Data-Oriented pipeline stages. |
| `Bootloader` | `boot.rs` | **HIGH** | 500+ | Massive async procedural block handling ROM/SRAM unpacking and hardware flashing. | Split into discrete boot phases using a stateful pipeline. |
| `Topology Internal` | `baker/topology.rs`| **HIGH** | 300+ | Massive procedural function handling neuron placement, growth, and I/O mapping. | Separate into `PlacementEngine`, `GrowthOrchestrator`, and `IoMapper`. |
| `Ghost Listener` | `network/router.rs` | **HIGH** | 150+ | Loop mixing IO, amnesia checks, self-healing, and deduplication. | Separate the UDP Listener from the protocol decoder (`FastPathDecoder`). |
| `main()` | `main.rs` (Node) | **MED** | 150+ | Procedural bootstrapping and thread spawning. | Move bootstrap sequence into a dedicated `BootOrchestrator`. |
| `Incoming UDP` | `network/io_server.rs`| **MED** | 100+ | Procedural header validation and matrix logic. | Use a `PacketDispatcher` state machine for protocol versions. |
| `Update Neurons` | `cpu/physics.rs` | **HIGH** | 200+ | Complex Hot Loop with deeply nested branchless logic. | Decompose into discrete inline "Math Blocks" (Leak, Integrate, Threshold). |
| `ShardStateSoA` | `layout.rs` | **LOW** | 100+ | Central data structure prone to growing into a blob. | Use discrete "State Planes" to maintain SoA cache efficiency. |
| `Daemon Spawning`| `node/mod.rs` | **MED** | 50 | Orphaned `axicor-baker-daemon` processes lock resources (UDP ports, SHM) if Node crashes on Windows. | Implement Windows Job Object (`CreateJobObject` / `AssignProcessToJobObject`) to bind daemon lifetimes to the Node orchestrator. |

## Refactoring Order (API Impact Priority)

1.  **`Bootloader` & `main()`**: Decoupling the startup sequence allows for more robust integration testing and better error reporting before the reactor starts.
2.  **`NodeRuntime` & `run_node_loop`**: Decomposing the orchestrator is critical for multi-node scaling and implementing advanced BSP synchronization strategies.
3.  **`Ghost Listener` & `Incoming UDP`**: Sterilizing the networking path is essential for future protocol upgrades (e.g., V3 headers) and security hardening.
4.  **Topology Generation (`baker`)**: Separating concerns in the baker enables faster iterations on connectome algorithms (e.g., new axon guidance laws).
5.  **Hot Loop (`cpu/physics`)**: Fine-grained math blocks will improve readability without sacrificing the zero-branch performance invariant.

## Critical Constraints
- **NO ECS**: Do not use Entity-Component-Systems for runtime simulation. Stick to DOD (SoA/AoS) and C-ABI.
- **State Machines**: Emphasize explicit state transitions for network and lifecycle management.
- **DOD Pipeline**: Maintain the Compute -> Network Tx -> Network Rx Wait order in all orchestrator refactors.