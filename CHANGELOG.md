# Changelog

All notable changes to Genesis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

## [0.301.35] - 2026-03-08 15:55:29

**IntraGPU Consolidation: Full Hardware Saturation & Stabilization**

### Added
- Implement multi-manifest CLI support via `--manifest <PATH_1> --manifest <PATH_2> ...` in main.rs
- Redesign boot_node_with_profile to aggregate data across multiple ZoneManifests in boot.rs
- Eliminate UDP networking between local zones by building IntraGpuChannel connections using is_src_local and is_dst_local logic
- Fix cluster join by broadcasting all locally present zone hashes instead of just the first one
- Create per-Shard non-blocking CUDA streams via gpu_stream_create and plumb through ffi.rs, shard.rs, and cu_step_day_phase
- Remove all gpu_stream_synchronize calls in shard_thread.rs to overlap CPU and GPU execution
- Reposition gpu_device_synchronize() in the Orchestrator loop after collecting BatchComplete payloads
- Implement static CPU core locking via libc::sched_setaffinity for each Shard and the Orchestrator (Core 0) in shard_thread.rs and main.rs
- Remove __constant__ current_dopamine and update_global_dopamine from physics.cu and bindings.cu
- Pass dopamine as int16_t kernel argument through cu_apply_gsop_kernel, cu_step_day_phase, ffi.rs, and shard.rs
- Add assertion in boot.rs to validate axons_blob.len() % 32 == 0 for C-ABI alignment
- Add --clean flag to genesis-baker to wipe target baked/ directories
- Implement sustained TPS logging every 500 batches in mod.rs, printing `[Performance] Sustained TPS:`

## [0.289.35] - 2026-03-08 14:15:00

**[Refactor] Minor network and node adjustments**

### Added
- Modify genesis-node/src/network/inter_node.rs, io_server.rs, and router.rs for internal logic updates
- Adjust genesis-node/src/node/shard_thread.rs and main.rs with minor code changes

## [0.287.35] - 2026-03-08 12:23:51

**Fix Critical Runtime Bugs**

### Added
- Fix Daemon Config Paths by passing root brain.toml path instead of zone blueprint paths in genesis-baker/src/bin/daemon.rs
- Refactor boot sequence in genesis-node/src/boot.rs to correctly handle config loading and zone initialization
- Update config file parsing in genesis-core/src/config/manifest.rs to support new path semantics
- Update zone blueprint and IO configs in config/zones/MotorCortex/ and config/zones/SensoryCortex/ for consistency
- Adjust shard.toml configuration in config/zones/MotorCortex/shard.toml
- Modify scripts/ant_v4_client.py for updated connection parameters

### Fixed
- Fix v_offset Invariant in genesis-compute/src/memory.rs to always compute offset, even for neurons without inputs
- Clean Ghost Artifacts by ensuring user re-bakes after code fixes, reflected in config file updates

## [0.284.30] - 2026-03-07 18:19:56

**CartPole Stable**

### Added
- Implement DDS Heartbeat spontaneous firing and fix axon memory alignment
- Add spontaneous_firing_period_ticks field to NeuronType in config/blueprints.rs with serde default
- Compute heartbeat_m in parser/blueprints.rs from spontaneous_firing_period_ticks using DDS multiplier formula (65536 / period)
- Extend cu_update_neurons_kernel to integrate DDS phase accumulator using current_tick and heartbeat_m
- Make neuron spiking logic combine GLIF spikes and heartbeat spikes, resetting membrane only on GLIF spikes
- Add tick_base parameter to ShardEngine::execute_day_phase and pass through to cu_step_day_phase
- Add current_tick parameter to cu_step_day_phase and cu_update_neurons_kernel in CUDA bindings
- Update execute_day_phase in shard_thread.rs to receive and forward tick_base from ComputeCommand::RunBatch
- Replace _pad1[2] with heartbeat_m: u16 in VariantParameters across CUDA bindings, FFI, and core layouts
- Update manifest serialization in genesis-baker/main.rs to include heartbeat_m field
- Enforce warp alignment (multiple of 32) for total_axons_max calculation in genesis-baker/src/bin/daemon.rs
- Fix axon_heads slice mapping to use entire axons_mmap without legacy 16-byte header
- Set spontaneous_firing_period_ticks = 1000 in examples/cartpole/config/zones/MotorCortex/blueprints.toml and SensoryCortex/blueprints.toml
- Increase night_interval_ticks from 0 to 15000 in examples/cartpole/config/simulation.toml
- Update examples/cartpole/readme.md with virtual environment instructions and brain_debugger.py usage
- Adjust target score from 19 to 71 in experiment goal description
- Add entries for releases 0.266.30 (DDS Heartbeat), 0.254.30 (Documentation), 0.247.30 (Zero-Downtime Recovery), 0.244.29 (Strict Dale's Law), and 0.237.29 (Warp-Aggregated Telemetry) to CHANGELOG.md

## [0.266.30] - 2026-03-07 16:41:18

*** Add heartbeat_m field to VariantParameters in genesis-core, genesis-co**

### Added
- Compute heartbeat_m in parser/blueprints.rs from spontaneous_firing_period_ticks using DDS multiplier formula (65536 / period)
- Extend cu_update_neurons_kernel to integrate DDS phase accumulator using current_tick and heartbeat_m
- Make neuron spiking logic combine GLIF spikes and heartbeat spikes, resetting membrane only on GLIF spikes
- Add tick_base parameter to ShardEngine::execute_day_phase and pass through to cu_step_day_phase
- Add current_tick parameter to cu_step_day_phase and cu_update_neurons_kernel in CUDA bindings
- Update execute_day_phase in shard_thread.rs to receive and forward tick_base from ComputeCommand::RunBatch
- Replace _pad1[2] with heartbeat_m: u16 in VariantParameters across CUDA bindings, FFI, and core layouts
- Add spontaneous_firing_period_ticks field to NeuronType in config/blueprints.rs with serde default
- Update manifest serialization in genesis-baker/main.rs to include heartbeat_m field

## [0.254.30] - 2026-03-07 16:24:10

**IMPORTANT**

### Added
- Add spontaneous_firing_period_ticks configuration field for neuron types in docs/specs/02_configuration.md
- Document DDS Heartbeat compilation from period_ticks to heartbeat_m multiplier in genesis-baker/src/compile_heartbeat.rs
- Add comprehensive DDS Heartbeat mathematical model and physiological contract in docs/specs/03_neuron_model.md
- Integrate DDS Phase Accumulator logic into update_neurons_kernel in docs/specs/05_signal_physics.md
- Extend VariantParameters struct in docs/specs/07_gpu_runtime.md with 16-bit heartbeat_m field and adjust padding
- Add entries for releases 0.247.30 (Zero-Downtime Recovery Integration), 0.244.29 (Strict Dale's Law), and 0.237.29 (Warp-Aggregated Telemetry) to CHANGELOG.md

## [0.247.30] - 2026-03-07 15:24:24

**Zero-Downtime Recovery Integration**

### Added
- Add ComputeCommand::Resurrect and is_warmup flag for ComputeFeedback::BatchComplete in genesis-node/src/node/mod.rs
- Implement hardware gating of outgoing network traffic in run_node_loop by zeroing DMA counters (channel.out_count_pinned) when is_warmup flag is received
- Add warmup_ticks_remaining state variable in genesis-node/src/node/shard_thread.rs
- Decrement timer during execute_day_phase and download_outputs, sending is_warmup: true status back to orchestrator while timer > 0
- Implement broadcast_route_update in genesis-node/src/node/recovery.rs, sending ROUTE_UPDATE packet with magic: ROUT_MAGIC to all live peers in routing_table
- Add resurrect_shard() function that broadcasts route to all nodes and commands local shard_thread to enter Warmup mode
- Fix minor ComputeCommand import issue and run cargo check --workspace successfully

## [0.244.29] - 2026-03-07 15:01:47

**Implement Strict Dale's Law with Activity-Based Nudging and Clean Legacy**

### Added
- Implement run_sprouting_pass with strict Dale's Law using 4-bit type mask from axon_tips_uvw
- Add nudge_axon function for activity-based nudging: local axons shift only if soma spiked (flags[soma_idx] & 0x01 != 0), ghost/virtual axons shift unconditionally
- Compute new_synapses using updated CPU Sprouting logic in genesis_baker::bake::sprouting::run_sprouting_pass
- Modify build_night_context to return mutable NightPhaseContext (night_ctx now mut)
- Update run_night_phase call to accept ctx: Option<&mut NightPhaseContext>
- Extract VRAM slice flags using correct offset from header hdr.flags_offset
- Delete genesis-node/src/orchestrator/sprouting.rs file entirely
- Remove pub mod sprouting; declaration from genesis-node/src/orchestrator/mod.rs

## [0.237.29] - 2026-03-07 14:56:43

**Warp-Aggregated Telemetry with Zero Atomics & C-ABI Fixes**

### Added
- Fix C-ABI signatures in `genesis-compute/src/ffi.rs` to pass `VramState` as a `*const ShardVramPtrs` pointer instead of by value
- Update `gpu_reset_telemetry_count` and `launch_extract_telemetry` signatures to accept `vram: *const ShardVramPtrs` and `padded_n: u32`
- Apply identical fixes to `genesis-compute/src/mock_ffi.rs`, including adding the import for `ShardVramPtrs`
- Rename and implement `cu_extract_telemetry_kernel` in `genesis-compute/src/cuda/physics.cu`
- Utilize warp-level primitives `__ballot_sync`, `__popc`, and `__shfl_sync` to perform exactly one `atomicAdd` per warp of 32 threads
- Update C-wrapper functions in `genesis-compute/src/cuda/bindings.cu` to accept `const ShardVramPtrs* ptrs`
- Move `gpu_reset_telemetry_count` and `launch_extract_telemetry` implementations lower in the file to resolve C++ compiler visibility error for `ShardVramPtrs`
- Update `CHANGELOG.md` and `examples/cartpole/readme.md`
- Run `cargo check --workspace` to verify successful compilation and FFI signature alignment

## [0.233.29] - 2026-03-07 14:18:18

**Implement Warp-Aggregated Telemetry with Zero-Copy Atomics**

### Added
- Implement `extract_telemetry_kernel` in `genesis-compute/src/cuda/physics.cu` using warp primitives `__ballot_sync`, `__popc`, and `__shfl_sync`
- Aggregate spike predicates across 32 threads into a single atomic increment via `atomicAdd(out_count, warp_pop)` from lane zero
- Compute `local_rank` via `__popc(active_mask & ((1u << lane) - 1))` for each spiking neuron to write IDs into flat array `out_ids[warp_offset + local_rank]`
- Add `extern` signature for `extract_telemetry_kernel` and implement `launch_extract_telemetry` in `genesis-compute/src/cuda/bindings.cu`, accepting new parameter `padded_n: u32`
- Implement `gpu_reset_telemetry_count` via `cudaMemsetAsync(vram.telemetry_count, 0, ...)` in bindings
- Update Rust FFI signatures in `genesis-compute/src/ffi.rs` and `genesis-compute/src/mock_ffi.rs` to accept `padded_n: u32` in `launch_extract_telemetry`

## [0.227.29] - 2026-03-07 14:04:35

**[Memory & C-ABI Foundation] Finalize Stage 1 with Zero-Copy memory mappi**

### Added
- Update `ShmHeader` in `genesis-core/src/ipc.rs`: increment `SHM_VERSION` to 2, remove `_padding`, add fields `prunes_offset`, `prunes_count`, `incoming_prunes_count`, `flags_offset` while preserving 64-byte L2 Cache Line alignment
- Enforce Zero-Copy Memory Mapping in `genesis-baker/src/bin/daemon.rs`: replace heap allocations by directly mapping `shard.state`, `shard.axons`, and `shard.geom` files via `memmap2::Mmap::map()` with 4096-byte OS page alignment
- Remove functions `bytes_to_u32_vec` and `bytes_to_burst_heads` to eliminate uncontrolled heap allocations
- Add strict C-ABI assert checks for 64-byte pointer alignment before using zero-copy type casting with `bytemuck::cast_slice`
- Fix import for `BurstHeads8` in the daemon module

## [0.223.28] - 2026-03-07 13:39:19

**[Architecture] Extend distributed runtime and IDE specs with Zero-Cost S**

### Added
- Implement Zero-Cost State Machine with GeometryGpuApplied marker component and LoadedGeometry in genesis-ide/src/geometry.rs to prevent repeated GPU uploads
- Add HFT Log Throttling with HftLogger using AtomicUsize counters and fetch_add(Ordering::Relaxed) for dopamine, egress packet, and self-heal event logging
- Increase BSP synchronization timeout to BSP_SYNC_TIMEOUT_MS = 500 ms and define MAX_BATCHES_LATENCY constant in wait_for_neighbors function
- Specify L7 Fragmentation parameters: MAX_EVENTS_PER_PACKET = 8186 and binary contract with SpikeBatchHeaderV2 and SpikeEventV2 structs
- Enforce Heartbeat invariant: routers must send empty packet with is_last = 1 even if no spikes in batch
- Document Biological Amnesia rule: drop packets with header.epoch < current_epoch and implement Epoch Synchronization
- Enforce Dale's Law invariant for Night Phase Sprouting: sign of initial_synapse_weight determined exclusively by sender-side axon type from axon_tips_uvw[axon_id] >> 28
- Implement Activity-Based Nudging for Living Axons: local axons grow only if soma spiked (soma_flags & 0x01 != 0), ghost axons grow unconditionally
- Define LivingAxon struct with tip_uvw, forward_dir, remaining_steps, and last_night_active fields in genesis-baker/src/bake/growth.rs
- Add step_and_pack function for axon tip movement and spatial_grid.insert for immediate Spatial Grid updates
- Document structural plasticity pattern ensuring computational resources spent only on active network regions
- Implement Burst Gating protection: if any of 8 axon heads touches a dendrite, the dendrite injects weight and enters synapse_refractory_period
- Enforce Winner-Takes-All invariant for non-linear R-STDP: only the nearest (freshest) head in BurstHeads8 influences GSOP per tick
- Integrate symmetric Dopamine modulation: dopa_mod = (base_pot * current_dopamine) >> 8, final_pot = base_pot + dopa_mod
- Add branchless unroll search for min_dist using #pragma unroll to avoid Warp Divergence
- Implement Warp-Aggregated Telemetry pattern using __ballot_sync, __popc, and a single atomicAdd per warp to eliminate L2 cache bus contention
- Extend docs/specs/06_distributed.md with detailed HFT logging, BSP timeout, L7 fragmentation, and Epoch Synchronization sections
- Update docs/specs/010_ide.md with Zero-Cost State Machine cascade: fetch_real_geometry, upload_geometry_to_gpu, and render_neuron_network systems
- Revise docs/specs/07_gpu_runtime.md with new memory layout and synchronization protocols
- Add Living Axons (Tip Nudging) section to docs/specs/04_connectivity.md with Activity-Based Nudging logic and data structures
- Expand docs/specs/05_signal_physics.md with Burst Gating, symmetric Dopamine integration, and Warp-Aggregated Telemetry mechanics

## [0.205.28] - 2026-03-07 12:45:58

**[Architecture] Extend distributed runtime and IDE specs with Zero-Cost S**

### Added
- Implement Zero-Cost State Machine with `GeometryGpuApplied` marker component and `LoadedGeometry` in genesis-ide/src/geometry.rs to prevent repeated GPU uploads
- Add HFT Log Throttling with `HftLogger` using `AtomicUsize` counters and `fetch_add(Ordering::Relaxed)` for dopamine, egress packet, and self-heal event logging
- Increase BSP synchronization timeout to `BSP_SYNC_TIMEOUT_MS = 500 ms` and define `MAX_BATCHES_LATENCY` constant in wait_for_neighbors function
- Specify L7 Fragmentation parameters: `MAX_EVENTS_PER_PACKET = 8186` and binary contract with `SpikeBatchHeaderV2` and `SpikeEventV2` structs
- Enforce Heartbeat invariant: routers must send empty packet with `is_last = 1` even if no spikes in batch
- Document Biological Amnesia rule: drop packets with `header.epoch < current_epoch` and implement Epoch Synchronization
- Extend docs/specs/06_distributed.md with detailed HFT logging, BSP timeout, L7 fragmentation, and Epoch Synchronization sections
- Update docs/specs/010_ide.md with Zero-Cost State Machine cascade: fetch_real_geometry, upload_geometry_to_gpu, and render_neuron_network systems
- Revise docs/specs/07_gpu_runtime.md with new memory layout and synchronization protocols
- Modify genesis-node/src/network/bsp.rs to incorporate new timeout constant and telemetry adjustments
- Update genesis-node/src/network/telemetry.rs to integrate throttled logging mechanisms
- Remove genesis-node/src/network/router.rs placeholder and adjust genesis-node/src/boot.rs and main.rs for initialization changes
- Update README.md CartPole record line with commit hash 3ed37ac
- Revise examples/cartpole/readme.md with updated instructions and links

## [0.190.28] - 2026-03-07 10:42:04

**docs: finalize AGENT.md laws and update CartPole records**

## [0.189.28] - 2026-03-07 09:11:31

**docfix**

## [0.189.27] - 2026-03-06 23:20:22

**CartPole Tuning [Config]**

### Added
- Dopamine shaping, sigma=2 for population coding in cartpole_client
- Blueprints: threshold 9500, gsop 110/15, initial_synapse_weight 2800, steering_radius 130, slot_decay_ltm 140

## [0.187.27] - 2026-03-06 23:20:17

**IDE Fixes [Genesis-IDE]**

### Added
- Update config path to config/zones/SensoryCortex/blueprints.toml
- Add GeometryGpuApplied marker for one-time GPU buffer init
- Log telemetry only when spikes.len() > 0

## [0.185.26] - 2026-03-06 23:20:11

**Log Throttling [Node]**

### Added
- Throttle BSP self-heal, dopamine, egress, shard I/O logs (every 100th)

## [0.184.26] - 2026-03-06 23:20:06

**Boot & BSP Improvements [Node]**

### Added
- Improve UDP/Geometry binding error messages with hints to kill existing processes
- Increase BSP sync timeout 50ms -> 500ms, use yield_now instead of spin_loop
- Add throttled self-heal logging (every 100th)

## [0.181.26] - 2026-03-06 23:20:01

**RTX 4090 CUDA Support [Build]**

### Added
- Detect GPU arch via nvidia-smi (compute_cap -> sm_XX)
- Use CUDA_ARCH env override, fallback sm_75
- Host compiler: g++-12 on Unix, MSVC on Windows

## [0.178.26] - 2026-03-06 23:19:55

**Windows IPC & Alignment [Platform]**

### Added
- Add TCP fallback and Windows SHM path in genesis-core/ipc.rs
- Refactor genesis-node/ipc.rs for TCP and file-backed shm
- Use file-backed mmap and TCP in genesis-baker daemon for Windows
- Add bytes_to_u32_vec/bytes_to_burst_heads to avoid cast_slice on unaligned data
- Fix bytemuck alignment in geometry_client and socket for Windows

## [0.174.25] - 2026-03-06 23:20:22

**CartPole Tuning [Config]**

### Added
- Dopamine shaping, sigma=2 for population coding in cartpole_client
- Blueprints: threshold 9500, gsop 110/15, initial_synapse_weight 2800, steering_radius 130, slot_decay_ltm 140

## [0.172.25] - 2026-03-06 23:20:17

**IDE Fixes [Genesis-IDE]**

### Added
- Update config path to config/zones/SensoryCortex/blueprints.toml
- Add GeometryGpuApplied marker for one-time GPU buffer init
- Log telemetry only when spikes.len() > 0

## [0.170.24] - 2026-03-06 23:20:11

**Log Throttling [Node]**

### Added
- Throttle BSP self-heal, dopamine, egress, shard I/O logs (every 100th)

## [0.169.24] - 2026-03-06 23:20:06

**Boot & BSP Improvements [Node]**

### Added
- Improve UDP/Geometry binding error messages with hints to kill existing processes
- Increase BSP sync timeout 50ms -> 500ms, use yield_now instead of spin_loop
- Add throttled self-heal logging (every 100th)

## [0.166.24] - 2026-03-06 23:20:01

**RTX 4090 CUDA Support [Build]**

### Added
- Detect GPU arch via nvidia-smi (compute_cap -> sm_XX)
- Use CUDA_ARCH env override, fallback sm_75
- Host compiler: g++-12 on Unix, MSVC on Windows

## [0.163.24] - 2026-03-06 23:19:55

**Windows IPC & Alignment [Platform]**

### Added
- Add TCP fallback and Windows SHM path in genesis-core/ipc.rs
- Refactor genesis-node/ipc.rs for TCP and file-backed shm
- Use file-backed mmap and TCP in genesis-baker daemon for Windows
- Add bytes_to_u32_vec/bytes_to_burst_heads to avoid cast_slice on unaligned data
- Fix bytemuck alignment in geometry_client and socket for Windows

## [0.159.23] - 2026-03-06 18:03:17

**Hot Loop Annihilation: __restrict__, push_burst_head, Branchless GSOP**

### Added
- Add __restrict__ qualifier to all 8 pointer fields in ShardVramPtrs struct in physics.cu
- Add __restrict__ to kernel signatures: cu_inject_inputs_kernel, cu_apply_spike_batch_kernel, cu_record_readout_kernel, cu_step_day_phase in physics.cu
- Add __restrict__ qualifier to all 9 pointer fields in ShardVramPtrs struct in bindings.cu
- Add __restrict__ to kernel signatures: inject_inputs_kernel, apply_spike_batch_kernel, ghost_sync_kernel, extract_outgoing_spikes_kernel in bindings.cu
- Implement __device__ __forceinline__ push_burst_head(BurstHeads8* h) helper function in physics.cu
- Replace manual burst shift blocks with push_burst_head(&h) in cu_inject_inputs_kernel, cu_apply_spike_batch_kernel, cu_update_neurons_kernel
- Add push_burst_head helper after AXON_SENTINEL define in bindings.cu
- Replace manual burst shift blocks with push_burst_head(&h) in inject_inputs_kernel, apply_spike_batch_kernel
- Replace explicit per-field comparisons in cu_apply_gsop_kernel with #pragma unroll for loop over 8 heads
- Use cast (uint32_t*)&b to iterate BurstHeads8 fields without padding
- Implement branchless conditional: min_dist = min(min_dist, (d < len) ? d : 0xFFFFFFFF) to avoid divergence

## [0.156.23] - 2026-03-06 10:47:13

**Genesis Code Cleanup & Strictness Enforcement**

### Added
- Add #![deny(warnings)], #![deny(unused_variables)], and #![deny(dead_code)] to all crate entry points: genesis-core/src/lib.rs, genesis-baker/src/main.rs, genesis-baker/src/lib.rs, genesis-ide/src/main.rs, genesis-compute/src/lib.rs, genesis-node/src/main.rs
- Restructure genesis-baker/src/main.rs to use library modules, eliminating redundant compilation warnings
- Remove file-level #![allow(dead_code)] from genesis-ide/src/connectome.rs, genesis-ide/src/world.rs, and genesis-ide/src/io_matrix.rs
- Remove serialize_axons and atomic_write functions from genesis-baker/src/main.rs
- Purge compute_power_index and unused sprouting logic from genesis-baker/src/bake/sprouting.rs
- Remove _master_seed parameter from connect_dendrites signature and all call sites in topology.rs
- Clean up unmaintained variables in genesis-baker/src/bake/axon_growth.rs
- Delete unused SpikeRoute, GlobalSpikeMap, and get_type_color from genesis-ide/src/world.rs
- Move AxonInstance, IoPixelInstance, MaterialUniforms, and NeuronPalette into shader_data or material_data submodules with #[allow(dead_code)] annotations
- Remove #[allow(dead_code)] from individual structs in connectome.rs and io_matrix.rs
- Systematically remove dozens of unused imports and variables across genesis-node and genesis-baker-daemon

## [0.153.23] - 2026-03-06 09:39:58

**[Performance] Dynamic pruning threshold for synaptic plasticity**

### Added
- Add `prune_threshold` field to `VariantParameters` struct in configuration
- Update `sort_and_prune_kernel` CUDA kernel implementation to use dynamic threshold
- Modify `launch_sort_and_prune` API function signature to accept `prune_threshold` argument
- Update all zone blueprint configurations (MotorCortex, SensoryCortex, V1) with explicit `prune_threshold` values
- Refactor genesis-baker sprouting logic to integrate new pruning parameter
- Adjust genesis-node shard thread and orchestrator sprouting for updated compute calls
- Clean up template configuration files, removing redundant _blank_blueprints.toml entries
- Update specification documents (foundations, configuration, connectivity, distributed) with pruning threshold details
- Synchronize genesis-ide asset configurations with new blueprint structure
- Run full workspace validation with `cargo check --workspace`

## [0.144.23] - 2026-03-06 05:24:44

**doc**

## [0.143.23] - 2026-03-06 05:23

**CartPole 2E2 Duble Nodes**

### Added
- Add dendrite_radius_um field to genesis-core/src/config/blueprints.rs for per‑soma radius configuration.
- Implement per‑soma dynamic radius in connect_dendrites (genesis-baker/src/bake/dendrite_connect.rs) and update its call in topology.rs.
- Extend blueprint handling in genesis-core/config/blueprints.rs.
- Modify baking modules (dendrite_connect.rs, input_map.rs, output_map.rs, spatial_grid.rs, topology.rs) to support radius changes and improve baking logic.
- Overhaul genesis-node/src/network/io_server.rs for improved connection handling.
- Enhance genesis-node/src/network/inter_node.rs with updated message protocols.
- Refactor node initialization (boot.rs, node/mod.rs) for streamlined startup.
- Update I/O server logic in io_server.rs and inter_node.rs for revised communication.
- Streamline config/brain.toml (significant reduction) and modify config/simulation.toml with updated parameters.
- Update brain.toml paths to reflect new configuration structure.
- Extend scripts/cartpole_client.py with new capabilities and set render_mode="human" for direct visualization.
- Update README.md with baking instructions for environment setup.
- Remove legacy zone configuration files: anatomy.toml, io.toml, and blueprints.toml from config/zones/CartPoleBrain/.
- Delete outdated examples/cartpole/client.py and scripts/cartpole_client.py (replaced by updated version).
- Adjust genesis-baker/src/bake/spatial_grid.rs for dynamic radius changes.
- Update main baking logic in genesis-baker/src/main.rs.
- Add new CUDA kernel bindings and definitions to genesis-compute/src/cuda/bindings.cu.

## [0.138.23] - 2026-03-06 02:47

**[Performance] Optimize CUDA kernel and network buffer scaling**

### Added
- Implement unrolled extraction in `extract_outgoing_spikes_kernel` using `#pragma unroll` to check all 8 heads of a burst in parallel
- Scale `InterNodeChannel` Pinned RAM allocation by 8x to accommodate burst-mode spike density
- Add L7 fragmentation in `InterNodeRouter` to split large spike batches (> 8186 events) into multiple UDP packets, avoiding MTU overflow
- Refine `SpikeBatchHeaderV2` to include `epoch` and `is_last` flags for exact barrier synchronization
- Re-implement `flush_outgoing_batch_pool` to send at least one packet with `is_last = 1` even if 0 spikes were produced, ensuring a guaranteed heartbeat
- Implement ingress filtering in `spawn_ghost_listener` to drop packets from mismatched epochs and only trigger the barrier on the `is_last` flag
- Modernize `BspBarrier` to track `current_epoch` and `completed_peers` for strict synchronization, updating `wait_for_data_sync` and `sync_and_swap`
- Propagate epoch from the orchestrator loop by passing `batch_counter` to `flush_outgoing_batch_pool` in `mod.rs`

## [0.130.23] - 2026-03-06 02:21

**[Architecture] Implement vectorized axon head layout and pipeline**

### Added
- Add BurstHeads8 struct (32-byte, align 32) in genesis-core and update ShardStateSoA and VramState
- Update ShardSoA and topology baking logic in genesis-baker to initialize 8 heads per axon with AXON_SENTINEL
- Align C-ABI in bindings.cu and physics.cu with explicit h0..h7 fields
- Implement 1-tick shift register across h0..h7 in inject_inputs, apply_spike_batch, and update_neurons kernels
- Optimize propagate_axons to unconditionally add v_seg to all 8 heads using branchless logic
- Replace scalar active tail check with branchless 8-head bitwise OR check in cu_update_neurons_kernel
- Implement non-linear STDP (GSOP) with nearest head search for min_dist and exponential cooling_shift (min_dist >> 4)
- Correct total_axons derivation in boot.rs using 32-byte divisor for .axons blob
- Update NightPhaseContext in daemon.rs to use Vec<BurstHeads8> and initialize with BurstHeads8::empty(AXON_SENTINEL)
- Vectorize perform_refresh in sentinel.rs to scan and reset all 8 heads per axon burst
- Update FFI in launch_extract_outgoing_spikes and launch_ghost_sync to accept BurstHeads8 pointers
- Mark legacy intra-GPU tests with #[ignore] in test_intra_gpu.rs

## [0.117.23] - 2026-03-05 21:56

**[Documentation] Update GPU runtime and signal physics specs for Burst Ar**

### Added
- Implement Burst Train Model with 8 heads (пачка импульсов) in 05_signal_physics.md §1.1
- Add Non-linear STDP with exponential cooling via bitwise shifts (`dist >> 4`) in §1.3
- Refine UpdateNeurons CUDA kernel logic with branchless 8-head check using `BurstHeads8` in §1.5
- Replace axon reset with hardware-style Burst Shift for spike generation in §1.5
- Update VRAM size table to include `Burst Architecture` and `axon_heads` in 07_gpu_runtime.md §1.1
- Add `BurstHeads8` struct definition with 32-byte alignment in §1.2.1
- Modify `AxonState` to use `*mut BurstHeads8` for `head_index` pointer in §1.2.1

## [0.110.23] - 2026-03-05 21:32

**[Architecture] Replace scattered parameters with aggregated context stru**

### Added
- Define ShardDescriptor struct in shard_thread.rs to encapsulate static shard geometry and physics
- Define NodeContext struct in shard_thread.rs to hold shared Arc handles for threads
- Define NetworkTopology struct in mod.rs to group network channels and routing components
- Define NodeServices struct in mod.rs to consolidate shared infrastructure like io_server and bsp_barrier
- Replace BootShard tuple type alias in boot.rs with ShardDescriptor
- Collapse 16-argument spawn_shard_thread function into four arguments: (ShardDescriptor, NodeContext, Receiver, Sender)
- Update spawn_baker_daemons signature to accept &[ShardDescriptor]
- Refactor NodeRuntime::boot from 15 arguments to 7, accepting shards, services, network, local_ip, local_port, output_routes, and night_interval
- Flatten NodeRuntime fields into logical groups: services, network, compute_dispatchers, feedback channels, and configuration maps
- Update all self.field accesses in run_node_loop to use self.services.bsp_barrier and self.network.inter_node_channels
- Fix main.rs access path to align with new NodeRuntime structure

## [0.99.22] - 2026-03-05 17:21

**[System] Remove legacy bootstrap script and harden axon growth limits**

### Added
- Delete bootstrap.py script, replacing manual UDP injection with integrated IPC
- Enforce axon_growth_max_steps <= 255 limit in validator/checks.rs to protect 8-bit PackedTarget memory layout
- Reduce default axon_growth_max_steps from 2500 to 255 in config/simulation.toml
- Replace CLI argument --zone (u16) with --zone_hash (u32) in genesis-baker daemon.rs
- Update shm_name(), default_socket_path(), and all IPC calls to use zone_hash
- Propagate brain_config through parse_and_validate() to bake workspace
- Include SimulationConfigRef and filtered ManifestConnection list in serialized ZoneManifest
- Add v_seg parameter to ShardEngine::run_batch() and launch_extract_outgoing_spikes()
- Update extract_outgoing_spikes_kernel to compute ticks_since_spike = head / v_seg
- Add sentinel check (head >= 0x70000000u) to skip dead axons in kernel
- Adjust kernel logic to use ticks_since_spike for tick_offset calculation
- Major refactor of genesis-node/src/boot.rs (246 insertions, 318 deletions total)
- Update network modules (bsp.rs, inter_node.rs, intra_gpu.rs, io_server.rs) for revised IPC
- Adjust node/shard_thread.rs and main.rs for new initialization flow

## [0.95.22] - 2026-03-05 12:48

**Axon Growth Decoupling & Bootloader Purge: Zero-Copy IPC & Lock-Free CLI**

### Added
- Implement Unified Stepper with `GrowthContext` struct and `execute_growth_loop` function in `axon_growth.rs`
- Refactor `grow_single_axon`, `inject_ghost_axons`, and `inject_handover_events` to use the unified `execute_growth_loop`
- Purge JSON contracts by removing `NightPhaseRequest`, `NightPhaseResponse`, and `CompiledShardMeta` from `genesis-core/src/ipc.rs`
- Implement raw SHM handover writing in `BakerClient::run_night` using `std::ptr::copy_nonoverlapping` and 16-byte `BakeRequest`
- Optimize CUDA kernels by changing `continue` to `break` in `cu_update_neurons_kernel` for early exit
- Restore Night Phase weight sync by updating `execute_night_phase` in `shard_thread.rs` to sync weights back to GPU
- Extract `parse_manifests` and `resolve_topology` to remove `/home/alex` hardcodes and clean up paths
- Extract `flash_hardware_physics` for CUDA constant memory upload and `build_intra_gpu_channels` for local message passing
- Extract `load_shards_into_vram` for SHM mapping and `setup_networking` for UDP IO and Telemetry
- Reassemble `boot_node` as a high-level orchestrator calling the new initialization phases sequentially
- Replace TUI with `SimpleReporter` using atomic counters in `genesis-node/src/simple_reporter.rs`
- Remove `println!` from hot loops in `mod.rs` and `shard_thread.rs` and spawn a CLI monitor thread in `main.rs`
- Update `NodeRuntime::boot` struct definition and integrate `extract_spikes` and `flush_outgoing_batch_pool` into `run_node_loop`
- Refactor `build_intra_gpu_channels` to `build_routing_channels` and calculate `expected_peers` for `BspBarrier` initialization
- Delete redundant `genesis-node/src/orchestrator/baker.rs` to de-duplicate Baker clients
- Remove `ratatui` references and clean up `DashboardState` and TUI modules (`tui/app.rs`, `tui/mod.rs`, `tui/tui/app.rs`, `tui/tui/mod.rs`)
- Remove `genesis-node/src/network/external.rs` and update `genesis-node/Cargo.toml`

## [0.78.22] - 2026-03-05 11:31

**Genesis Baker Refactoring & NodeRuntime Zero-Copy Pipeline**

### Added
- Implement AxonSegmentGrid in spatial_grid.rs to index axon segments with SegmentRef payload
- Rewrite connect_dendrites.rs with AoS-to-SoA inversion using TempSlot arrays and parallel par_iter_mut
- Refactor neuron_placement.rs to use pre-allocated voxel pool and Fisher-Yates shuffle with master seed, replacing reject-sampling
- Add CompiledShard contract to layout.rs and split compile function in main.rs
- Move build_local_topology to bake module for reuse between baker and daemon
- Refactor BspBarrier in bsp.rs with expected_peers, packets_received AtomicUsize, and wait_for_data_sync using spin_loop
- Update ring_buffer.rs to use PinnedBuffer<AtomicU32> and lock-free push_spike
- Implement EgressPool in egress.rs with ArrayQueue for zero-allocation UDP egress
- Refactor ThreadWorkspace in shard_thread.rs to use MmapMut for SHM and accept preallocated buffers
- Add NightPhaseRequest/Response in ipc.rs for zero-copy control channel
- Extract spawn_shard_thread logic into dedicated shard_thread.rs file
- Implement execute_day_phase, download_outputs, save_hot_checkpoint, and execute_night_phase flattening
- Replace async run_node_loop with synchronous version and move to dedicated OS thread in main.rs
- Update intra_gpu_channels to store raw pointers and add inter_node_channels to NodeRuntime
- Reorder run_node_loop orchestration: wait -> swap -> dispatch -> wait_feedback -> intra-gpu -> inter-node egress -> IO
- Modify spawn_ghost_listener in inter_node.rs to increment bsp_barrier.packets_received and call get_write_schedule().push_spike directly
- Implement flush_outgoing_batch_pool in InterNodeRouter and send_output_batch_pool in ExternalIoServer using EgressPool
- Spin up dedicated Egress Worker thread in main.rs for socket sending
- Refactor BakerClient::run_night to send NightPhaseRequest and daemon to access SHM directly

## [0.61.22] - 2026-03-04 06:11

**AGI/ASI Has was born**

### Added
- 04.03.2026 5:12am

## [0.60.22] - 2026-03-04 03:05

**fix(node, compute): virtual_offset correction, GSOP inertia rewrite, DMA**

### Added
- genesis-node/node/mod.rs: Fixed virtual_offset=0 memory corruption.
- Virtual axons live at tail of axon_heads array, not at index 0.
- Now computed as total_axons - num_virtual_axons.
- genesis-node/boot.rs: Panic on missing shard.gxi when io.toml
- declares inputs. Silent fallback to 0 caused DMA buffer overflow
- that corrupted entire VRAM state.
- genesis-node/boot.rs: Hoisted io_config_path resolution before
- GXI safety check. Added checkpoint.state resume priority over
- shard.state in boot_shard_from_disk.
- genesis-node/network/io_server.rs: Added capacity field and
- hard assert to InputSwapchain::write_incoming_at. Prevents
- network packets from overflowing Pinned RAM DMA buffers.
- genesis-compute/cuda/physics.cu: Rewrote cu_apply_gsop_kernel
- with Inertia Curve, decay-as-multiplier (not subtracted from
- weight), and ltm_slot_count from VariantParameters.
- genesis-compute/cuda/physics.cu, bindings.cu: Expanded
- VariantParameters from 64B to 128B with inertia_curve[16]
- and ltm_slot_count fields.
- genesis-compute/ffi.rs: Synced Rust FFI VariantParameters to
- 128B layout matching CUDA side.
- genesis-core/config/manifest.rs: Added inertia_curve and
- ltm_slot_count to ManifestVariant DTO and GpuVariantParameters
- with serde defaults for backward compatibility.
- genesis-node/node/mod.rs: Hot Checkpointing - periodic VRAM
- dump to checkpoint.state via gpu_memcpy_device_to_host every
- 500 batches with atomic file write.
- scripts/brain_debugger.py: New diagnostic tool for parsing
- .state binary blobs and reporting SoA field statistics.

## [0.56.17] - 2026-03-03 17:23

**refactor: isolate network I/O and fix async runtime conflicts**

### Added
- genesis-ide: Isolated telemetry and geometry fetchers into dedicated OS threads with independent Tokio runtimes. This prevents "no reactor running" panics within Bevy's ComputeTaskPool.
- genesis-ide: Patched handle_picking and handle_box_select systems to use Option<Res<LoadedGeometry>>, preventing ECS access panics during cold boot/loading states.
- genesis-node: Fixed EADDRINUSE fatal panic in IoMultiplexer. Port 8081 binding now fails gracefully with a log entry instead of crashing the daemon.
- genesis-node: Removed nested tokio::Runtime from ZoneRuntime to resolve "cannot drop a runtime in async context" errors during boot/shutdown.
- genesis-baker: (prev) Fixed type mismatches in manifest variant mapping for i32/u8 compatibility.

## [0.54.14] - 2026-03-03 14:37

**feat(core, compute, baker)!: Global SoA Refactor, 3D Quantization & CUDA**

### Added
- This commit (106 files) completes the transition to deterministic 3D quantization and a pure Structure of Arrays (SoA) architecture, removing the legacy PlacedNeuron object.
- Core: PackedPosition & 32-bit Quantization
- Implemented 32-bit PackedPosition (X:11, Y:11, Z:6, Type:4) using bytemuck (Pod/Zeroable) for direct GPU transfers.
- Added axis-limit validation and masks to prevent voxel grid overflows.
- Baker: Geometry & Warp Alignment
- Added Reject-Sampling for voxel collision avoidance during placement.
- Implemented 32-byte Warp Alignment (padding) to ensure Coalesced Access for GPU kernels.
- Enforced deterministic Z-sorting of output arrays for cache optimization.
- SIMD Math: Vectorized Cone Tracing
- Rewrote cone_tracing.rs using the glam library for SIMD-ready vector math.
- Replaced trigonometry with dot-product calculations and optimized cycles via delayed sqrt and type_id filtering.
- Prepared the foundation for steering systems (mixing attraction gradients with global noise).
- Compute: FFI & VRAM Orchestration
- Established the Rust-CUDA FFI bridge via ShardVramPtrs (Soma voltage, soma flags, topology mapping, and axon heads).
- Implemented RAII-based VRAM management (VramShard) for allocation, uploading, and automatic cleanup.

## [0.50.14] - 2026-03-03 13:25

**refactor: prepare for core architecture refactor, optimize TUI state via**

## [0.49.14] - 2026-03-02 13:32

**refactor: implement byte-perfect foundations and columnar layout**

### Added
- This commit completes the refactoring of genesis-core types and memory
- layout to align with the 03_neuron_model.md specification. The changes
- ensure FFI compatibility and optimize data structures for GPU VRAM
- interaction and constant memory access.
- Key Changes:
- PackedPosition: Replaced u32 alias with a repr(transparent) struct.
- Implemented bit-manipulation methods (x, y, z, type_id) with
- debug_assert validation for coordinate boundaries.
- VariantParameters: Applied #[repr(C, align(32))] with explicit padding
- to guarantee a 32-byte footprint, enabling optimal coalesced loads
- from GPU constant memory.
- ShardStateSoA: Implemented host-side Structure-of-Arrays to facilitate
- high-throughput baking and VRAM transfers.
- Columnar Layout: Introduced columnar_idx helper to enforce warp-aligned
- strides for coalesced GPU global memory access.
- System Integrity: Resolved import breakages in blueprints.rs and
- manifest.rs caused by VariantParameters relocation.
- Verification:
- Added comprehensive unit tests in types.rs covering coordinate
- packing/unpacking, O(1) variant ID extraction, and memory alignment.
- Confirmed all 54 genesis-core tests are passing.
- Verified byte-offsets and alignment via std::mem checks.

## [0.43.14] - 2026-03-02 12:41

**Preparing for the Big Bang. Part 2**

### Added
- Pre refactoring

## [0.42.14] - 2026-03-02 03:19

**Preparing for the Big Bang. Part 2**

### Added
- Embodied AI Breakthrough: RobotBrain Ant-v4 & Lock-Free IPC
- Transitioned to crossbeam::queue::SegQueue non-blocking queues to decouple the raw "Night Phase" OS thread from the asynchronous Tokio reactor.
- Eliminated no reactor running panics and Mutex contention in the simulation's hot loop.
- Implemented Zero-Downtime Hot-Reload for blueprints.toml: a background Tokio worker monitors file changes and atomically updates neuron parameters in GPU __constant__ memory (the VARIANT_LUT array) directly at the BSP barrier via FFI.
- Successfully closed the control loop for the MuJoCo Ant-v4 environment across a distributed cluster (Node A: Sensory/Motor, Node B: Hidden).
- Documented spontaneous emergence of postural stability (Nights 20-30) and rhythmic gait (CPG, Night 52) driven by structural plasticity and the "Artificial Pain" mechanism.
- Identified "muscle fatigue" phenomena caused by homeostasis accumulation. Resolved via live tuning of MotorCortex membrane parameters using the new Hot-Reload mechanism without cluster downtime.
- Confirmed massive "Ghost-axon" outgrowth between nodes (tens of thousands of GROW packets per Night Phase).
- Enabled multi-threaded TCP packet aggregation for colonization, maintaining a stable hot-loop throughput of 60-64 Ticks/sec.

## [0.34.14] - 2026-03-01 22:47

**feat(runtime): стабилизация кластера на 1.35 млн нейронов и замыкание Em**

### Added
- Реализован сетевой BSP-барьер для жесткой синхронизации Node A (Sensory/Motor) и Node B (Hidden).
- Оптимизирован процесс Baking: внедрен Rayon и SoA-транспонирование (время сборки снижено с 41 до 12 минут).
- Решен конфликт портов: Node A переведен на 8010-ю серию, Node B остался на 8000-й.
- Улучшен TUI: отключен раздражающий захват мыши (EnableMouseCapture) и добавлен живой счетчик UDP Out.
- Исправлены критические баги: padded_n mismatch в чекпоинтах и несанкционированный доступ Node B к IO-зонам.
- Успешно верифицирована «Замкнутая Петля» (The Embodied Loop) с использованием CartPole.
- Настроены параметры нейронов в blueprints.toml для стабильного Ignition и GSOP-депрессии.
- CartPole архивирован в examples/cartpole/ для использования в качестве базового референса.

## [0.27.13] - 2026-03-01 18:31

**Preparing for the Big Bang. Part 1**

### Added
- Implement InterNodeChannel for zero-copy UDP loopback synchronization
- Support Split-Brain mode via NODE_A variable (Sensory & Motor vs Hidden)
- Enforce Strict BSP (Bulk Synchronous Parallel) via sync_and_swap()
- Zero-Copy Spike Extraction CUDA kernel using atomic L2 cache projection
- Implement Night Phase throttling in main loop (min_night_delay check)
- Decouple Simulation Time from Wall-Clock time to prevent NVMe wear
- Stabilize Throughput at ~100k TPS (Ticks Per Second) with GPU Day Phase
- Implement Topographical UV Projection in Baker for retino/somatotopic mapping
- Generate .ghosts files with deterministic FNV-1a Jitter for bio-distribution
- Define TCP Slow Path protocol structs for dynamic handover & growth
- Deploy Bevy IDE for 3D visualization of distributed spikes
- Link CartPole reinforcement loop via zero-copy Python UDP client
- Integrate GSOP plasticity for latency adaptation in distributed networks

## [0.15.13] - 2026-02-28 21:32

**docs recovery**

## [0.14.13] - 2026-02-26

### Added
- **True Hardware E2E Scalability (§2)** — verified 1M neurons simulating at real physics bounds
- E2E synthetic benchmark `e2e_test.rs` capturing real CUDA execution speeds bypassing CPU Baker overhead
- Real-time physics bounds metrics on GTX 1080 Ti equivalent: ~32k Ticks/s (1K), ~22k Ticks/s (10K), ~5k Ticks/s (100K)
- Proved memory safety and full pipeline closure from Virtual Input → GLIF → GSOP → Output Readout loops

## [0.13.5] - 2026-02-26

### Added
- **Readout Interface (Output §3)** — `record_readout_kernel` extracting flagged motor spikes
- Dense `output_history` buffer capturing batched readout spikes per tick inside VRAM
- `IoConfig` expanded in Core config with `OutputMap` and `readout_batch_ticks` for tiled outputs
- Atlas tiling generation for motor soma assignment via `.gxo` files in Baker
- DayPhase integration of readout recording into the simulation fast-path
- Removed obsolete `record_outputs.cu` and atomic spike routing logic
- **Sleep API and Spike Drop** — `is_sleeping` and `sleep_requested` added to runtime. Sleeping zones drop incoming spikes (Legalized Amnesia §2.3) and skip physics.
- **Secure Cross-Shard Geometry** — bounds check (`if ghost_id < total_axons`) in `apply_spike_batch_kernel`
- **Night Phase Checkpointing** — logic dumping `pre_sprout` states to disk before Baker processing

### Fixed
- SIGSEGV in mock-gpu runtime tests resolved
- Fix `local_axons_count` being shadowed after virtual axon append

## [0.12.3] - 2026-02-25

### Added
- **Input Interface (Virtual Axons §2.1)** — added `inject_inputs.cu` to map 1-bit external bitmasks to virtual axon firing
- Abstracted `grow_single_axon()` for reuse in Input map generation (`grow_input_maps`)
- Deterministic FNV-1a seeded routing for `pixel→soma` translation across virtual axons
- Generation of `.gxi` (Genesis eXternal Input) binary format with Header, Map Descriptors, and flat axon arrays
- Expanded VRAM state to load `.gxi` indirection tables (`map_pixel_to_axon`) and allocate bitmask buffers
- Batched bitmask upload capability (`upload_input_bitmask`) via DayPhase
- **Testing Architecture** — comprehensive unit testing for Cone Tracing algorithm (26 tests in genesis-baker) covering SpatialGrid sensing, trajectory steering, and multi-shard generation
- **Testing Architecture** — comprehensive testing of synaptogenesis (dendrite_connect) validating Rule of Uniqueness, Type Whitelist, Inhibitory signs, and Self-Exclusion with an ASCII visualizer

### Fixed
- Active Tail bounds alignment and axon sentinel refresh implemented
- Protective layers against Signal Superimposition and refraction bounds
- Fix broken indentation and redundant unsafe blocks in VramState drop

## [0.11.0] - 2026-02-24

### Added
- **Ghost Axons (§1.7)** — бесшовный рост аксонов через границы шардов
- `ShardBounds` structure with `full_world()` and `is_outside()` boundary detection
- `GhostPacket` inter-shard transfer format (entry point, direction, remaining steps)
- `inject_ghost_axons()` — ghost axon growth continuation in receiving shard
- Pipeline integration in `main.rs` with diagnostic logging
- Unit tests for boundary detection and ghost packet handling
- Updated spec `04_connectivity.md` §1.7 with full protocol description

## [0.10.1] - 2026-02-24

### Added
- **Power Score activation** — `compute_power_index` now called during Night Phase sprouting
- Whitelist filtering in `reconnect_empty_dendrites()` (was missing)
- `sprouting_weight_type` config parameter for soft type-matching scoring component

## [0.10.0] - 2026-02-24

### Added
- **Rule of Uniqueness (§1.4)** — `HashSet`-based deduplication prevents redundant axon connections
- **Dendrite Whitelist (§1.5)** — per-type compatibility filtering via `dendrite_whitelist` in blueprints
- **Configurable Initial Weight** — `initial_synapse_weight` moved to `blueprints.toml`
- Unit tests for whitelist and initial weight parsing
- Updated spec `04_connectivity.md` §1.4–1.5

## [0.9.0] - 2026-02-24

### Added
- **GPU LUT Expansion 4→16** — each of 16 neuron types gets a unique physical profile (GLIF/GSOP)
- **Voxel Uniqueness** — reject-sampling guarantees one voxel = at most one neuron
- `growth_vertical_bias`, `type_affinity`, `is_inhibitory` fields in blueprints
- Blueprints.toml updated with 4 base types (Vertical_Excitatory, Horizontal_Inhibitory, Stable_Excitatory, Relay_Excitatory)

## [0.8.0] - 2026-02-24

### Added
- **Binary Formatting (§2.1)** — formalized GSNS/GSAX header specs
- `InstanceConfig` refactored into dedicated `instance.rs`
- Default CLI paths updated to `config/zones/V1/*`
- E2E test script paths corrected

## [0.7.0] - 2026-02-23

### Added
- **Configuration Architecture (Spec 02 §1.1–1.3)** — `simulation.toml` parser in genesis-core
- `anatomy.rs` parser with population calculation tests
- `DerivedPhysics` + `compute_derived_physics()` with §1.6 invariant
- `Tick`, `Microns`, `Fraction`, `VoxelCoord` type aliases
- `ms_to_ticks`, `us_to_ticks`, `ticks_to_ms` conversions with unit tests
- Master Seed §2 implementation

### Fixed
- `PackedTarget` bitmap layout corrected (16/16 → 22/10)
- `initial_axon_head(N) + N = AXON_SENTINEL` invariant documented

## [0.6.0] - 2026-02-23

### Added
- **Night Phase IPC Baker Daemon** — `genesis-baker-daemon` executable running in background
- **Shared Memory Protocol (SHM)** — zero-copy transfer of weights and targets between CUDA runtime and CPU baker
- **Unix Sockets** — for JSON control messages (`night_start`, `night_done`) synchronization
- **Sort & Prune CUDA Kernel** — O(1) register Bitonic Sort (N=128) per neuron to auto-promote LTM/WM and prune weak connections
- Integration E2E IPC tests verifying full orchestrator pipeline handoff

## [0.5.0] - 2026-02-23


### Added
- **Smart Axon Growth: Cone Tracing** — iterative, biologically plausible axon sprouting
- `SpatialGrid` spatial hash map for O(1) neighbor lookup during growth
- V\_attract inverse-square law and piecewise tip geometry
- Variable-length `.axons` binary format (tip\_x, tip\_y, tip\_z, length per axon)

## [0.4.1] - 2026-02-23

### Added
- **Genesis IDE** — new `genesis-ide` crate with Bevy 3D viewer
- Orbital + Fly camera modes with mouse/keyboard control
- HUD overlay: FPS, neuron count, axon count, selected neuron info
- Neuron spheres colored by `type_mask`
- Baker exports `shard.positions`; IDE highlights spiking neurons via WebSocket glow (3 frames)

## [0.4.0] - 2026-02-23

### Added
- **Genesis Monitor** — WebSocket telemetry server broadcasting tick, phase, and real-time spike dense IDs
- Bevy 3D client renders neurons as glow-highlighted spheres synced to live VRAM state

## [0.3.0] - 2026-02-22

### Added
- **Ghost Axon Handover** — TCP Slow Path with VRAM reserve pool and Handover handshake
- Dynamic `SpikeRouter` route registration via slow path
- **Homeostatic Plasticity** — branchless penalty/decay in GLIF kernel
- Equilibrium validated across 100 CUDA ticks

## [0.2.0] - 2026-02-22

### Added
- `genesis-node` daemon: parses `shard.toml`, mounts VRAM, drives BSP ephemeral loop
- **Atlas Routing** — external Ghost Axons baked at compile time, zero GPU overhead at runtime

### Fixed
- BSP deadlock: guaranteed empty-batch Header dispatch to all peers

## [0.1.0] - 2026-02-22

### Added
- BSP ping-pong buffers, async UDP Zero-Copy fast path
- Day Phase orchestrator loop
- Night Phase skeleton: GPU Sort & Prune, CPU sprouting stub, PCIe download/upload hooks

### Fixed
- Memory layout sync, Active Tail GSOP fix, 24b/8b axon format alignment

## [0.0.0] - 2026-02-21

### Added
- Architecture specification: 7 documents, ~3000 lines
- Full design from high-level abstractions down to byte-level GPU operations
