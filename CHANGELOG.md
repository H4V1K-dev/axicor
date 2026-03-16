# Changelog

All notable changes to Genesis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

## [0.639.105] - 2026-03-16 22:23:03

**Dynamic Structural Plasticity & GPU Pruning Threshold**

### Added
- Replace `_padding: u16` with `max_sprouts: u16` in `BakeRequest` (`genesis-core/src/ipc.rs`) to preserve 16-byte contract
- Add `max_sprouts` field with Serde default to `ManifestPlasticity` and `ShardSettings` (`genesis-core/src/config/manifest.rs`, `instance.rs`)
- Integrate `max_sprouts` into `ShardAtomicSettings` and support Hot-Reload in `genesis-node/src/node/mod.rs`
- Update `BakerClient` and `shard_thread.rs` to read atomic settings and pass `max_sprouts` via IPC to the Baker Daemon
- Modify `run_sprouting_pass` in `genesis-baker/src/bake/sprouting.rs` to use dynamic `max_sprouts_per_night` parameter
- Expose `set_max_sprouts(max_sprouts: int)` in `GenesisControl` and `GenesisClusterControl` (`genesis-client/genesis/control.py`, `brain.py`) with atomic TOML mutation
- Refactor `sort_and_prune_kernel` in `genesis-compute/src/cuda/bindings.cu` and `amd/bindings.hip` to accept `int16_t global_prune_threshold`
- Update `launch_sort_and_prune` signatures in FFI layer (`genesis-compute/src/ffi.rs`, `mock_ffi.rs`) to include `prune_threshold: i16`
- Propagate dynamic threshold in orchestrator's `execute_night_phase` (`genesis-node/src/node/shard_thread.rs`)

## [0.630.105] - 2026-03-16 20:47:10

**Remove experimental CartPole example and associated files**

### Added
- Delete entire `examples/cartpole [EXPERIMENTAL]/` directory and all contained files
- Remove `README.md` with Zero-Magic Pipeline instructions, HFT reactor launch commands for CUDA/ROCm, and Python gateway steps
- Remove `agent.py` containing the Embodied AI agent with SNN, dopamine injections (R-STDP), GenesisMultiClient, PopulationEncoder, PwmDecoder, GenesisControl, GenesisAutoTuner, GenesisMemory, GenesisSurgeon
- Remove `benchmark.py` for performance testing with GenesisMultiClient and GenesisMemory
- Remove `build_brain.py` script for generating TOML topology and invoking Rust compiler (`genesis-baker`)

## [0.629.105] - 2026-03-16 18:54:02

**[Connectome & Metrics]**

### Added
- Restore 6-layer biological topology in build_brain.py, fixing NameError for motor_pyramidal
- Re-add missing add_input and add_output calls to ensure I/O integrity in build_brain.py
- Refactor humanoid_agent.py to use GenesisBrain for dynamic zone discovery from brain.toml
- Fix sprouting trigger in sprouting.rs to check full 3-bit burst counter using mask 0x0E instead of 0x02
- Resolve VRAM phantom leak in CUDA/HIP by clearing burst count bits [3:1] with mask 0xF1 in sort_and_prune_kernel
- Implement cu_reset_burst_counters_kernel in physics.hip for AMD/HIP backend
- Integrate burst counter reset call at start of cu_step_day_phase in physics.cu and physics.hip
- Implement every-batch burst reset in Genesis-Lite main.cpp, moving flags &= 0xF1 to top of daily loop
- Synchronize batch reset and pruning logic across CUDA, HIP, and Lite backends for consistent simulation
- Update genesis-lite main.cpp to create explicit sort_and_prune_kernel function with 0xF1 burst clear
- Add required function declarations to genesis_core.hpp
- Enhance scripts/brain_debugger.py for improved debugging capabilities

## [0.619.102] - 2026-03-16 04:12:01

**Humanoid Connectome Revival & Ant Pattern Lockstep**

### Added
- Implement Ant Pattern Lockstep with single `humanoid_sensors` (384x16) and `sync_batch_ticks = 80`
- Replace dual PopulationEncoder with single `PopulationEncoder(384 vars)` and atomic `client.step()`
- Enforce warp-aligned geometry: sensory input 192-bit width (6144 px), motor output 32-bit width (256 px)
- Fix signal routing collisions: rename `proprio_to_thoracic`, `thoracic_to_cerebellum`, and `proprio_to_cerebellum`
- Remove all manual socket handling, buffer drain, and multi-packet send logic
- Simplify motor decoder to direct mapping of 17 muscles from first 34 neurons of 32x8 motor matrix
- Implement Continuous Burst Encoding to prevent signal death in 5-zone deep brain
- Fix geometry enforcement: use `motor_2d = total_motor.reshape(8, 34)` and O(1) stride slicing for biceps/triceps
- Add Dopamine Low-Pass Filter (EMA) with `smoothed_velocity` outside loop
- Enforce rigid RX protocol with `EXPECTED_RX_BYTES = 272 * BATCH_SIZE` and BSP Barrier check
- Apply asymmetric weighting: Excitatory 16000, Inhibitory 4000; reduce MotorCortex GABA to 30%
- Add direct `SensoryCortex -> MotorCortex` reflex arc bypass connection
- Remove redundant `total_motor.reshape(17, 16)` and associated `action` calculation
- Delete spontaneous neural firing (`heartbeat_m`) and restore plasticity with `dep=2`
- Bind UDP socket to Port 8092 and implement Multi-Zone Handshake (8081 & 8121)
- Fix library-aware renaming for `Motor_Pyramidal` updating both `blueprint.name` and internal `data_list["name"]`

## [0.606.99] - 2026-03-15 22:58:01

**[Documentation] Update Ant-v4 example README with detailed setup and arc**

### Added
- Rewrite title and description to emphasize Spiking Neural Networks (SNN), 3-zone architecture (DOD/WTA), and dopamine injection learning (R-STDP)
- Replace architecture section with a "How to run (Zero-Magic Pipeline)" guide, adding virtual environment activation step
- Expand brain generation step to clarify creation of 3-zone topology (Sensory, Thoracic, Motor) with 60% inhibitory neuron density in motor cortex
- Split kernel launch step into separate commands for NVIDIA (CUDA) and AMD (ROCm / HIP) backends, specifying `--features amd` flag
- Rephrase agent launch step, emphasizing DOD Hot Loop operation without allocations and control via `TARGET_TIME` and `TARGET_SCORE`
- Remove explicit learning control parameters section, integrating key concepts into the final descriptive paragraph

## [0.600.99] - 2026-03-15 21:59:44

**Implement TARGET_TIME as primary termination driver for Ant agent**

### Added
- Add steps counter to hot loop, incrementing on every successful env.step
- Implement time_reached logic that triggers reset if steps >= TARGET_TIME
- Check TARGET_TIME alongside and prioritized over TARGET_SCORE
- Update console feedback to show step count: Ep 0000 | Score: 2000+ | Steps: 2000 | [Time Reached]
- Implement build_brain.py for 3-zone topology with ThoracicGanglion split into L_Lower/L_Upper
- Set hardware-locked plasticity with pot=0, dep=2
- Increase inhibitor density to 60% in MotorCortex L5_Lower for Winner-Takes-All competition
- Align VRAM Stride to 28x16 = 448 pixels
- Implement bottom-z motor routing with WTA dynamics
- Apply batch size reduction from 100 to 20, setting sync_batch_ticks to 20 for 2ms HFT batches
- Refactor ant_agent.py to remove OOP in hot loop and preallocate all arrays outside the loop
- Implement offset=20 for C-ABI compatibility and zero-copy spike summation logic
- Remove TARGET_FPS and sleep logic
- Implement 15-batch Death Signal
- Boost Motor_Pyramidal initial_synapse_weight to 12000 and dendrite_radius_um to 500.0
- Delete obsolete CartPole-example model files and examples/DELETE.py
- Move cartpole example to experimental directory and update genesis-client modules

## [0.596.99] - 2026-03-15 21:59:44

**Implement TARGET_TIME as primary termination driver for Ant agent**

### Added
- Add steps counter to hot loop, incrementing on every successful env.step
- Implement time_reached logic that triggers reset if steps >= TARGET_TIME
- Check TARGET_TIME alongside and prioritized over TARGET_SCORE
- Update console feedback to show step count: Ep 0000 | Score: 2000+ | Steps: 2000 | [Time Reached]
- Implement build_brain.py for 3-zone topology with ThoracicGanglion split into L_Lower/L_Upper
- Set hardware-locked plasticity with pot=0, dep=2
- Increase inhibitor density to 60% in MotorCortex L5_Lower for Winner-Takes-All competition
- Align VRAM Stride to 28x16 = 448 pixels
- Implement bottom-z motor routing with WTA dynamics
- Apply batch size reduction from 100 to 20, setting sync_batch_ticks to 20 for 2ms HFT batches
- Refactor ant_agent.py to remove OOP in hot loop and preallocate all arrays outside the loop
- Implement offset=20 for C-ABI compatibility and zero-copy spike summation logic
- Remove TARGET_FPS and sleep logic
- Implement 15-batch Death Signal
- Boost Motor_Pyramidal initial_synapse_weight to 12000 and dendrite_radius_um to 500.0
- Delete obsolete CartPole-example model files and examples/DELETE.py
- Move cartpole example to experimental directory and update genesis-client modules

## [0.592.99] - 2026-03-15 16:17:13

**ESP32 Telemetry & Desktop Bridge Pipeline**

### Added
- Implement DashboardFrame struct in genesis_core.hpp with 16-byte aligned layout for dashboard parser compatibility
- Add metrics calculation (TPS, Score) within the pro_core_task in main.cpp
- Implement esp_now_send broadcast of the DashboardFrame for zero-copy telemetry egress
- Create scripts/esp_now_bridge.py as a Serial-to-UDP bridge script for HFT telemetry routing
- Fix broadcast_mac array size to 6 bytes to prevent stack corruption
- Standardize bridge script argument parsing, using sys.argv[1] for serial port
- Update README.md with bridge usage information
- Perform final build check of genesis-lite to validate struct and FFI calls

## [0.584.98] - 2026-03-15 15:20:50

**[Documentation] Update technical specs for BDP and neuron model**

### Added
- Update the bit map in `docs/specs/03_neuron_model.md`
- Update BDP mathematical descriptions in `docs/specs/05_signal_physics.md`

## [0.582.98] - 2026-03-15 15:01:33

**Implement Burst-Dependent Plasticity (BDP) across CUDA and ESP32 stacks**

### Added
- Implement 3-bit burst counter in `soma_flags` bits [3:1], saturating at 7
- Add CUDA kernel `cu_reset_burst_counter` to clear burst bits at each sync batch
- Update `cu_update_neurons_kernel` to increment burst counter on every spike
- Modify `cu_apply_gsop_kernel` to multiply `delta_pot` and `delta_dep` by `burst_mult`
- Implement identical BDP logic in ESP32 `genesis-lite/main/main.cpp`, resetting counters every epoch (100 ticks)
- Call `cu_reset_burst_counters` from `genesis-compute/src/compute/shard.rs` at start of each batch
- Relocate `cu_reset_burst_counters` FFI wrapper from `bindings.cu` to `physics.cu` to fix circular dependency
- Add FFI signature for `cu_reset_burst_counters` in `genesis-compute/src/ffi.rs`
- Enforce branchless arithmetic for burst multiplier scaling in both CUDA and ESP32 kernels

## [0.574.97] - 2026-03-15 13:32:04

**ESP32 Blob Inflation: Bidirectional Distillation Cycle**

### Added
- Add scripts/inflate_esp32.py for high-speed expansion of 32-slot ESP32 blobs to 128-slot desktop format
- Implement zero-copy memory extraction using mmap for efficient reading of ESP32 C-ABI blobs
- Use vectorized NumPy broadcasting to move weights, targets, and timers from 32 slots into 128 slots
- Pack the resulting data back into a standard C-ABI blob format with strict size assertions
- Fix memory safety bugs (BufferError) in scripts/distill_esp32.py
- Refactor code to ensure reliable bidirectional conversion between formats
- Test conversion from shard.state (128 slots) to esp32_test.blob (32 slots, 862.75 KB) and back to desktop_reinflated.blob (128 slots, 3.22 MB)
- Verify re-inflated blob matches exact byte-size of standard desktop shard (padded_n * 910 bytes)

## [0.569.96] - 2026-03-15 12:48:16

**Fix medium sec findings: pin python deps & add destructive action barrie**

## [0.569.95] - 2026-03-15 12:45:38

**[Security] Implement Cluster Secret Zero-Cost Authentication**

### Added
- Add `cluster_secret: u64` field to `RouteUpdate` C-ABI struct in `genesis-core/src/ipc.rs`
- Enforce 24-byte size assertion for `RouteUpdate` to maintain 8-byte alignment
- Inject `cluster_secret` into `NodeRuntime` struct and `boot` method in `genesis-node/src/node/mod.rs`
- Update `broadcast_route_update` in `genesis-node/src/node/recovery.rs` to sign outgoing packets
- Add O(1) validation in `ExternalIoServer::process_incoming_udp` in `genesis-node/src/network/io_server.rs` for `ROUT_MAGIC` packets
- Derive and propagate `cluster_secret` from `master_seed` in `genesis-node/src/boot.rs`
- Inject `-ccbin=gcc-13` and `-w` flags into CUDA build configuration in `genesis-compute/build.rs`
- Add `gcc-13` dependency check to `scripts/setup.sh`

## [0.561.95] - 2026-03-15 12:37:41

**Lock NVCC host compiler to GCC-13 and suppress heuristic warnings**

## [0.561.94] - 2026-03-15 12:26:21

**[System] Add ESP32 model distillation tool with zero‑copy parsing**

### Added
- Implement scripts/distill_esp32.py for analyzing and distilling Genesis models for ESP32 deployment
- Support universal format auto‑detection: Live SHM (GENS), Snapshots (SNAP), and Raw SoA blobs
- Use mmap and numpy.frombuffer for Zero‑Copy parsing of multi‑gigabyte models without RAM overhead
- Perform topological scan to analyze synaptic density and identify neurons exceeding ESP32 hardware limits (32 connections)
- Verify tool against CartPole‑example/SensoryCortex, detecting 471,808 active synapses and warning of 128 connections/neuron

## [0.556.94] - 2026-03-15 12:16:41

**[Security] Fix unaligned memory access and secure network defaults**

### Added
- Replace unsafe pointer reference casting with std::ptr::read_unaligned for ExternalIoHeader and RouteUpdate in io_server.rs
- Replace unsafe pointer reference casting with std::ptr::read_unaligned for SpikeBatchHeader in router.rs
- Bind I/O Server UDP socket and geometry server address to 127.0.0.1 in boot.rs
- Bind Telemetry server WebSocket to 127.0.0.1 in telemetry.rs
- Bind inter-node Ghost axon listener to 127.0.0.1 in inter_node.rs

## [0.551.89] - 2026-03-15 12:05:13

**Axicor 8-Way Burst Synchronization & Kernel Hardening**

### Added
- Implement SAFE_CALLOC macro in genesis-lite/main/main.cpp for fail-fast SRAM allocation in init_brain
- Expand initialization loop to explicitly set AXON_SENTINEL across all 8 heads of BurstHeads8 structure
- Replace logical OR with branchless bitwise OR in day_phase_task GLIF hit detection (h.h0 - seg_idx) < prop
- Extend GSOP potentiation logic to poll the full 8-head shift register for min distance calculation
- Add strict CUDA Toolkit version check (>= 12.4) to scripts/setup.sh, exit 1 on older versions
- Document NVIDIA CUDA as Tier 1 backend in docs/specs/10_hardware_backends.md with Pascal+ GPU and Ubuntu 22.04/24.04 requirements
- Enforce exclusive PCIe Passthrough virtualization requirement and renumber subsequent backend tiers
- Rebase and resolve conflicts to integrate local neuromorphic kernel fixes with upstream 8-head cascade changes

## [0.544.89] - 2026-03-15 10:57:52

**Merge pull request #3 from aaaab000/fix/burst-heads-full-cascade**

### Added
- Fix duplicate BurstHeads8 shift register operations in spike path
- Implement proper refractory timer reset during spike emission
- Enforce DOD-compliant flag clearing for non-spike state
- Correct instantaneous spike accumulation logic with proper threshold comparison
- Maintain accumulator state between evaluation cycles per DOD specification
- Set spike flag (0x01) and refractory bits (0x02) atomically during firing

## [0.538.83] - 2026-03-15 10:11:56

**Resolve conflict: keep upstream flag bits, apply full 8-head cascade shi**

## [0.538.82] - 2026-03-15 03:07:34

**Public rename to Axicor**

### Added
- Update root Cargo.toml repository URL to https://github.com/H4V1K-dev/Axicor
- Update sub-package Cargo.toml descriptions for genesis-core, genesis-node, genesis-compute, and genesis-baker to use "Axicor"
- Update README.md title to "Axicor Alpha 0.0.1", marketing text, and GitHub clone URLs
- Update Credits.md project references and repository link
- Update Python SDK documentation titles and descriptions in QuickStart_SDK.md, Client_SDK.md, Brain_Builder.md, SDK_Encoders_Decoders.md, and SDK_Surgery_Dopamine.md
- Preserve internal codename GENESIS by not modifying any .rs files, Cargo.toml name fields, or internal technical identifiers

## [0.532.82] - 2026-03-15 00:22:08

**Alpha 0.0.1: HFT Synchronization & Nuclear Reservoir Refactor**

### Added
- Compress SensoryCortex zone from 64x64x63 to 16x16x16 voxels (400x400x400 μm) in build_brain.py
- Replace L4_Input, L23_Hidden, L5_Motor layers with unified Nuclear layer at 0.4 density
- Set dendrite_radius_um to 400.0 for all neuron types for universal Small-World connectivity
- Distribute excitatory (50%), inhibitory (20%), and motor (30%) neurons uniformly across zone height
- Align environment tau to 0.002 and sync_batch_ticks to 20 for 2ms lockstep in agent.py and build_brain.py
- Replace discrete dopamine logic with branchless continuous error gradient based on pole angle and velocity
- Implement Spike Accumulator in physics.cu and physics.hip using Bit 1 of soma_flags for 100-tick batch capture
- Increase DOPAMINE_PULSE to -15 and lower DOPAMINE_REWARD to 35 for aggressive R-STDP background erosion
- Implement non-linear kinetic pain shock: shock = BASE + (score >> 5) + (velocity * 5) capped at 100 batches
- Extract D1_AFFINITY, D2_AFFINITY, LEAK_RATE, HOMEOS_PENALTY, HOMEOS_DECAY, ANGLE_SCALE, VELOCITY_SCALE to global constants in agent.py
- Create benchmark.py with 10s stress test using GenesisMultiClient, compute TPS via (packets * BATCH_SIZE) / 10.0
- Add idle mode and simulated 20ms environment delay latency wall to benchmark.py
- Fix synapse counting in benchmark.py by debugging SHM read for targets to count synapses before training
- Add entry_z to InputMap DTO in genesis-core/src/config/io.rs for dynamic cable routing
- Implement flags_offset in ThreadWorkspace and flags_slice_mut in genesis-node/src/node/shard_thread.rs
- Inject soma_flags DMA at top of execute_night_phase and clear accumulator in bindings.cu and bindings.hip
- Switch to Bit 1 check in genesis-baker/src/bake/sprouting.rs for spike detection
- Remove GCC-13 hardcoding (std::env::set_var("CXX", "g++-13")) in genesis-compute/build.rs
- Add checks for nvcc and hipcc in scripts/setup.sh with mock-gpu feature prompt for CPU-only simulation
- Update README.md to clarify GPU recommendation and CPU-only mock mode availability

## [0.514.81] - 2026-03-15 00:22:08

**Alpha 0.0.1: HFT Synchronization & Nuclear Reservoir Refactor**

### Added
- Compress SensoryCortex zone from 64x64x63 to 16x16x16 voxels (400x400x400 μm) in build_brain.py
- Replace L4_Input, L23_Hidden, L5_Motor layers with unified Nuclear layer at 0.4 density
- Set dendrite_radius_um to 400.0 for all neuron types for universal Small-World connectivity
- Distribute excitatory (50%), inhibitory (20%), and motor (30%) neurons uniformly across zone height
- Align environment tau to 0.002 and sync_batch_ticks to 20 for 2ms lockstep in agent.py and build_brain.py
- Replace discrete dopamine logic with branchless continuous error gradient based on pole angle and velocity
- Implement Spike Accumulator in physics.cu and physics.hip using Bit 1 of soma_flags for 100-tick batch capture
- Increase DOPAMINE_PULSE to -15 and lower DOPAMINE_REWARD to 35 for aggressive R-STDP background erosion
- Implement non-linear kinetic pain shock: shock = BASE + (score >> 5) + (velocity * 5) capped at 100 batches
- Extract D1_AFFINITY, D2_AFFINITY, LEAK_RATE, HOMEOS_PENALTY, HOMEOS_DECAY, ANGLE_SCALE, VELOCITY_SCALE to global constants in agent.py
- Create benchmark.py with 10s stress test using GenesisMultiClient, compute TPS via (packets * BATCH_SIZE) / 10.0
- Add idle mode and simulated 20ms environment delay latency wall to benchmark.py
- Fix synapse counting in benchmark.py by debugging SHM read for targets to count synapses before training
- Add entry_z to InputMap DTO in genesis-core/src/config/io.rs for dynamic cable routing
- Implement flags_offset in ThreadWorkspace and flags_slice_mut in genesis-node/src/node/shard_thread.rs
- Inject soma_flags DMA at top of execute_night_phase and clear accumulator in bindings.cu and bindings.hip
- Switch to Bit 1 check in genesis-baker/src/bake/sprouting.rs for spike detection
- Remove GCC-13 hardcoding (std::env::set_var("CXX", "g++-13")) in genesis-compute/build.rs
- Add checks for nvcc and hipcc in scripts/setup.sh with mock-gpu feature prompt for CPU-only simulation
- Update README.md to clarify GPU recommendation and CPU-only mock mode availability

## [0.495.80] - 2026-03-13 19:01:02

**Implement HFT encoders/decoders and stabilize Mouse Agent feedback loop**

### Added
- Implement PwmEncoder and PopulationEncoder in genesis/encoders.py for high-frequency sensory encoding
- Implement PwmDecoder in genesis/decoders.py for motor signal decoding
- Add TelemetryListener in genesis/telemetry.py for real-time spike monitoring
- Add distill_graph and clear_weights functions in genesis/memory.py for zero-copy pruning and full network resets
- Fix genesis-node CLI arguments by removing deprecated --batch-size and using --brain flag with brain.toml
- Fix PYTHONPATH for cartpole_client.py by adding sys.path modification to include genesis-client directory
- Correct mouse_client.py to monitor Motor_Cortex (1,252,141 synapses) instead of LGN_Thalamus relay zone
- Reduce DOPAMINE_REWARD from 10 to 2 to prevent network epilepsy and oversaturation of spikes
- Broaden Gaussian sigma from 0.15 to 0.2 in population coding for better sensory coverage
- Create scripts/reset_weights.py tool for total network weight reset across all shards
- Add benchmark_encoders.py and test_distillation.py for component validation
- Update GenesisMultiClient and GenesisControl usage to match current SDK version in examples

## [0.485.75] - 2026-03-13 14:16:30

**Implement ESP32-S3 dual-core sensorimotor loop with lock-free motor stru**

### Added
- Add detailed visual data flow diagram in docs/specs/11_edge_bare_metal.md showing Core 0/Core 1 interaction via Lock-Free Ring Buffer
- Implement alignas(32) MotorOut struct with std::atomic<uint32_t> left and right fields for zero-lock motor output
- Include math.h for sensor emulation in genesis-lite/main/main.cpp
- Replace ESP-NOW-only Core 0 task with integrated Hardware I/O Loop featuring I2C gyroscope stub and PWM motor out
- Implement I2C gyroscope stub generating sine wave angle using sinf() and esp_timer_get_time()
- Add Population Coding encoder mapping float angle to spikes on receptor axons 0..9 via rx_queue.push(ev)
- Implement PWM Motor Out decoder reading and zeroing MotorOut counters via exchange(0, std::memory_order_relaxed)
- Modify Wi-Fi initialization to handle QEMU environment failure with Offline Mode fallback
- Keep ESP-NOW receive callback registration via esp_now_register_recv_cb(on_esp_now_recv)
- Add motor cortex readout in day_phase_task checking sram.flags[254] and sram.flags[255] for spikes
- Increment motors.left and motors.right via fetch_add(1, std::memory_order_relaxed) upon spike detection
- Maintain hot loop timing print every 100 ticks

## [0.477.75] - 2026-03-13 13:59:02

**Implement ESP-NOW swarm connectivity and dual-core spike injection with **

### Added
- Add ESP‑NOW, Wi‑Fi, NVS, and event loop headers to main.cpp for radio stack
- Implement pro_core_task on Core 0 with full ESP‑NOW initialization and fallback to mock sensor in QEMU
- Register on_esp_now_recv callback to receive spike packets and push them into LockFreeSpikeQueue
- Expose SpikeEvent struct as 8‑byte aligned network packet with ghost_id and tick_offset fields
- Implement LockFreeSpikeQueue in genesis_core.hpp with atomic head/tail separated by cache lines
- Add push and pop methods using std::atomic with memory_order_relaxed, acquire, and release semantics
- Define SPIKE_QUEUE_SIZE as 256 and declare global rx_queue instance
- Integrate queue pop loop in day_phase_task to apply incoming spikes as axon head resets (h0 = 0)
- Split day_phase_task as Core 1 HFT compute phase and pro_core_task as Core 0 network/sensor phase
- Replace Russian comments with English and remove redundant SRAM/Flash allocation comments
- Move global_dopamine declaration and add rx_queue global variable
- Adjust task creation in app_main to pin pro_core_task to Core 0 and day_phase_task to Core 1

## [0.474.75] - 2026-03-13 13:43:36

**Genesis-Lite HFT Core MVP**

### Added
- Genesis-Lite: ESP-IDF Port with GLIF Physics and GSOP Plasticity
- Create genesis-lite project structure with ESP-IDF CMakeLists.txt and sdkconfig
- Implement genesis_core.hpp with SramState, FlashTopology, and 64-byte VariantParameters
- Port main.cpp to FreeRTOS API using esp_timer.h and xTaskCreatePinnedToCore
- Bind day_phase_task strictly to Core 1, leaving Core 0 for I/O and networking
- Implement Axon Propagation and Dendritic Integration phases with 32 dendrite slots per neuron
- Add branchless GLIF leak calculations, threshold firing, and refractory period
- Fix VARIANT_LUT array initialization and access syntax in main.cpp
- Separate memory into SramState (Hot Data) and FlashTopology (read-only mapped data)
- Implement ApplyGSOP kernel immediately after UpdateNeurons for R-STDP loop
- Add branchless minimal distance calculation (check_head_dist) for BurstHeads8
- Replace standard C++ headers with ESP-IDF equivalents (freertos/FreeRTOS.h, freertos/task.h)
- Fix strict printf formatting warnings using inttypes.h macros (PRIu32, PRId64)
- Add watchdog yielding via vTaskDelay in simulation loop

## [0.463.73] - 2026-03-13 11:46:29

**[Specs] Expand connectivity model and finalize AMD backend MVP**

### Added
- Implement Terminal Arborization two-phase axon growth: Trunk (Cone Tracing) and Crown (V_noise chaos within arborization_radius_um)
- Add steering_fov_deg, arborization_target_layer, arborization_radius_um, arborization_density configuration parameters in 02_configuration.md
- Document Zero-Cost Spatial Search via Axon Segment Grid (O(K) Spatial Hashing) and En Passant synapses
- Update backend diversity status from [Planned] to [MVP] for AMD ROCm/HIP with bitwise-identical determinism
- Finalize Dual-Backend C-ABI architecture with mirror directories (cuda/, amd/) and compile-time feature selection
- Lower CUDA target architecture from sm_86 to sm_61 (NVIDIA Pascal) in genesis-compute/build.rs

## [0.457.73] - 2026-03-13 11:21:51

**docs(core): fix inertia rank comment 16 → 15 rängов**

## [0.457.72] - 2026-03-13 07:54:08

**Merge pull request #2 from aaaab000/fix/inertia-rank-oob-and-warp-assert**

### Added
- fix(core): inertia_rank OOB, warp assert, and fix broken tests

## [0.456.71] - 2026-03-12 18:45:38

**Implement robust constant memory management and suppress HIP warnings**

### Added
- AMD Radeon backend MVP
- Implement robust FFI functions cu_upload_constant_memory, gpu_load_constants, and update_constant_memory_hot_reload with multi-stage fallback logic using HIP_SYMBOL and direct symbol address lookups
- Update VARIANT_LUT declaration to __constant__ VariantParameters VARIANT_LUT[16]; to match 16 variants and satisfy linker
- Move telemetry launcher functions and constant memory management functions from bindings.hip to physics.hip to resolve undeclared identifier errors
- Simplify synaptic pruning logic by hardcoding sort_and_prune_kernel threshold to 15, decoupling from constant memory
- Add (void) casts to hipStreamSynchronize, hipDeviceSynchronize, hipMemset, hipMemsetAsync, hipFree, hipMemcpyToSymbol, and hipMemcpyToSymbolAsync calls to suppress nodiscard warnings
- Update genesis-compute/build.rs with new HIP compilation logic
- Add mock GPU functions to genesis-compute/src/mock_ffi.rs
- Add GPU context initialization to genesis-node/src/boot.rs
- amd completed successfully with reduced warnings

## [0.448.71] - 2026-03-12 16:40:31

**fix(core): inertia_rank OOB, warp assert, and fix broken tests**

## [0.448.70] - 2026-03-12 08:37:24

**Stabilize SDK and examples for MVP with unified Genesis_Models structure**

### Added
- Rename primary model output directory from config/ to Genesis_Models/
- Separate DNA and Baked artifacts, storing baked zone data in Genesis_Models/{project}/baked/{zone}/
- Add Genesis_Models/ to .gitignore for generated artifacts
- Update examples/mouse_agent.py to use new Genesis_Models/ path and correct sys.path.append
- Fix builder.py path resolution to use absolute paths for all discovery files
- Update baked_dir logic to point directly to the zone directory
- Add --brain argument support to automatically expand brain.toml into zone manifests
- Implement intelligent path resolver that accepts model names (e.g., --brain mouse_agent)
- Add default-run to Cargo.toml to eliminate --bin genesis-node requirement
- Update ZONE_SENSORY to match LGN_Thalamus zone name
- Update MATRIX_SENSORS to match retina_rgb sensor configuration
- Correct manifest path to Genesis_Models/mouse_agent/baked/LGN_Thalamus/manifest.toml

## [0.444.70] - 2026-03-12 07:08:04

**Genesis SDK procedural DNA generator**

### Added
- Fix sys.path.append in test_builder.py to correctly point to genesis-client directory
- Update builder.py ZoneDesigner with add_input() and add_output() for matrix registration
- Implement BrainBuilder connect() method for inter-zonal topological junctions
- Extend BrainBuilder build() to generate per-zone io.toml and populate brain.toml connections
- Ensure absolute path resolution for simulation.toml and zone configs in brain.toml generation
- Specify --bin baker in cargo run command within test_builder.py to resolve ambiguity

## [0.439.69] - 2026-03-12 04:27:17

**TUI Dashboard Enhancements: Interactive Focus and Scrolling**

### Added
- Add `FocusedPanel` enum and `focus` field to `DashboardState` in `state.rs`
- Implement `Tab` key for focus cycling between Per-Zone Telemetry and Event Log in `input.rs`
- Implement `Up` and `Down` arrow keys for scrolling the currently focused panel in `input.rs`
- Define static height layout for metric blocks (Header, Core, Zones, IO) totaling 14 rows in `layout.rs`
- Ensure Event Log automatically stretches to fill all remaining vertical terminal space in `layout.rs`
- Remove unused `draw_narrow_warning` function and `Rect` import in `layout.rs`
- Add focus-dependent Cyan border and `▶` title prefix to `zone_table.rs` and `event_log.rs`
- Implement vertical scrolling for Per-Zone table, displaying a slice of 7 zones in `zone_table.rs`
- Implement history scrolling with inverted logic for Event Log in `event_log.rs`
- Add activity bar (`█`) for visual spike rate tracking in the zone table in `zone_table.rs`
- Add color-coded levels and Green Dopamine highlights to the Event Log in `event_log.rs`
- Clean up `unused_mut` and `unused_variables` warnings in `cartpole_htf.rs`

## [0.427.68] - 2026-03-12 03:44:39

**sdk mvp**

## [0.426.68] - 2026-03-11 23:31:22

**SDK Docs**

### Added
- Remove high-level Python code examples and OOP abstractions from Client_SDK.md and 08_io_matrix.md
- Enforce strict 20-byte ExternalIoHeader per UDP chunk, linked to genesis-core/src/ipc.rs
- Define payload types: GSIO as 1 bit per virtual axon and GSOO as 1 byte per soma
- Replace "Host Convention" with Feature Pyramid Batching abstraction describing temporal unfolding of layers
- Clarify UDP L7-asymmetry: mega-batches are fragmented per MTU with header attached to each chunk
- Add mandatory paradigm stating engine only handles bitmasks/bytes; float/RGB/token processing is client responsibility
- Add uv_rect: [f32; 4] field to InputMap and OutputMap structs in genesis-core/src/config/io.rs
- Implement DOD-projection math in genesis-baker/src/bake/topology.rs to partition start_x/start_y based on uv_rect for GXI
- Pass matrix.uv_rect parameter to build_gxo_mapping function in genesis-baker/src/bake/output_map.rs with TODO for reverse UV projection
- Document Canvas, Chunked, and Pie spatial mapping modes in specs/08_io_matrix.md section 2.4
- Implement zero-cost state extraction from shard.state in scripts/visualize_neuron.py
- Aggregate morphology stats (Fan-In, Total Weight Mass, Synaptic Balance) using NumPy operations
- Add 2D Text HUD for neuron state metrics and 2D Inset Plot for Dendrite Weights Histogram
- Integrate matplotlib pick_event for interactive synapse weight inspection on click
- Add automatic parser for BrainDNA/blueprints.toml to display reference parameters in HUD, comparing threshold and rest_potential with real-time data

## [0.418.68] - 2026-03-11 17:14:36

**empty**

## [0.418.68] - 2026-03-11 15:10:58

**empty**

## [0.418.68] - 2026-03-11 05:02:57

**Assemble full cortical blueprints and implement Causal Delay Loop**

### Added
- Assemble SensoryCortex blueprints from GNM-Library in examples/cartpole/config/zones/SensoryCortex/blueprints.toml
- Assemble HiddenCortex blueprints from GNM-Library in examples/cartpole/config/zones/HiddenCortex/blueprints.toml
- Assemble MotorCortex blueprints from GNM-Library in examples/cartpole/config/zones/MotorCortex/blueprints.toml
- Recalibrate Plasticity Formulas across all GNM-Library TOML files (Cerebellum, Cortex, Hippocampus, Striatum, Thalamus)
- Fix SHM size hardcoding in boot.rs
- Wire incoming ACK queue in boot.rs
- Fix output fallback in config/io.rs
- Remove hardcoded batch_size
- Remove hardcoded night phase and prune limits
- Revert [[neuron_types]] and fix missing stride in IO
- Fix Baker Daemon IPC Acks and Show Logs
- Fix live_dashboard.py UDP connection in cartpole_htf.rs
- Implement Causal Delay Loop in cartpole_htf.rs

## [0.409.60] - 2026-03-11 03:38:54

**[Architecture] Assemble cortex blueprints from GNM-Library**

### Added
- Assemble SensoryCortex blueprints from GNM-Library in examples/cartpole/config/zones/SensoryCortex/blueprints.toml
- Assemble HiddenCortex blueprints from GNM-Library in examples/cartpole/config/zones/HiddenCortex/blueprints.toml
- Assemble MotorCortex blueprints from GNM-Library in examples/cartpole/config/zones/MotorCortex/blueprints.toml
- Update corresponding anatomy.toml, io.toml, and shard.toml files for SensoryCortex, HiddenCortex, and MotorCortex
- Remove hardcoded batch_size in genesis-core/src/config/instance.rs
- Remove hardcoded night phase and prune limits in examples/cartpole/config/simulation.toml
- Fix Baker Daemon IPC Acks and show logs in genesis-baker/src/bin/daemon.rs and genesis-baker/src/main.rs
- Fix output fallback in config/io.rs
- Revert [[neuron_types]] and fix missing stride in IO configuration
- Wire incoming ACK queue in boot.rs within genesis-node/src/node/mod.rs
- Fix live_dashboard.py UDP connection in cartpole_htf.rs
- Fix SHM size hardcoding in examples/cartpole/config/brain.toml

## [0.406.50] - 2026-03-11 00:24:42

**[System] Fix hardcoded parameters and wire ACK queue**

### Added
- Fix SHM size hardcoding in genesis-core/src/ipc.rs and genesis-node/src/ipc.rs
- Wire incoming ACK queue in genesis-node/src/boot.rs for proper IPC synchronization
- Fix output fallback logic in genesis-core/src/config/io.rs
- Remove hardcoded batch_size from examples/cartpole config files and genesis-baker modules
- Fix live_dashboard.py UDP connection handling in genesis-node/src/bin/cartpole_htf.rs
- Update neuron placement in genesis-baker/src/bake/neuron_placement.rs
- Refactor output map generation in genesis-baker/src/bake/output_map.rs
- Adjust sprouting logic in genesis-baker/src/bake/sprouting.rs
- Extend topology baking in genesis-baker/src/bake/topology.rs
- Update daemon boot sequence in genesis-baker/src/bin/daemon.rs
- Modify geometry client in genesis-node/src/network/geometry_client.rs
- Update slow path network handling in genesis-node/src/network/slow_path.rs
- Refactor shard thread communication in genesis-node/src/node/shard_thread.rs
- Adjust main node initialization in genesis-node/src/node/mod.rs and genesis-node/src/main.rs
- Extend CHANGELOG.md with 189 lines of updates
- Update example configurations across cartpole zone anatomy.toml, io.toml, and shard.toml files
- Revise examples/cartpole/readme.md
- Update _template/io.toml and scripts/weight_checker.py

## [0.394.45] - 2026-03-10 21:57:04

**Implement Dynamic Capacity Routing with hot-patching and swap-and-pop me**

### Added
- Mutate Channel structures in inter_node.rs and intra_gpu.rs to support dynamic capacity routing
- Add pre-allocation logic in boot.rs bootloader for initial channel capacity
- Implement hot-patching mechanics (Sprouting) in inter_node.rs and intra_gpu.rs for runtime capacity expansion
- Implement swap-and-pop mechanics (Pruning) in inter_node.rs and intra_gpu.rs for runtime capacity reduction
- Integrate Dynamic Capacity Routing mechanics with the Night Phase in shard_thread.rs
- Update node/mod.rs with new routing state management and integration points
- Extend genesis-core/src/ipc.rs with new IPC structures for capacity signaling
- Update docs/specs/06_distributed.md with Dynamic Capacity Routing protocol details

## [0.385.45] - 2026-03-10 21:16:06

**Refactor axon growth and sprouting with master seed and type-aware ghost**

### Added
- Refactor axon_growth.rs to pass neuron_types to inject_ghost_axons and use ev.remaining_length in inject_handover_events
- Refactor topology.rs to pass neuron_types to inject_ghost_axons
- Refactor sprouting.rs to add master_seed parameter to run_sprouting_pass and rewrite synaptogenesis scoring logic
- Refactor daemon.rs to pass ctx._master_seed to run_sprouting_pass call

## [0.381.45] - 2026-03-10 20:59:38

**[System] Remove default LTM slot count and update configurations**

### Added
- Remove default value for ltm_slot_count field in genesis-core/src/config/blueprints.rs
- Fix associated tests in genesis-core/src/config/test_blueprints.rs
- Map ltm_slot_count in genesis-baker/src/parser/blueprints.rs
- Update ltm_slot_count in all example cartpole zone blueprints.toml files
- Add ltm_slot_count to corresponding cartpole zone shard.toml files
- Update _template/blueprints.toml with ltm_slot_count

## [0.376.44] - 2026-03-10 20:50:50

**GNM-Library  patch - add ltm slors choise**

## [0.375.44] - 2026-03-10 20:43:51

**[Refactor] Remove default configs and enforce strict manifest validation**

### Added
- Refactor genesis-core/src/config/instance.rs to remove Optional and Default trait implementations
- Refactor genesis-core/src/config/mod.rs to eliminate global night_interval_ticks constant
- Refactor genesis-core/src/config/manifest.rs to make all fields strict and non-optional
- Refactor genesis-node/src/boot.rs to remove unwrap_or fallbacks for config loading
- Delete all template config files (_blank_anatomy.toml, _blank_blueprints.toml, etc.) and legacy zone configurations
- Update genesis-baker and genesis-node to handle strict configs without defaults

## [0.369.44] - 2026-03-10 19:52:12

**[Refactor] Replace population_pct with density, remove global_density**

### Added
- Refactor genesis-core/src/config/mod.rs to remove global_density field
- Refactor genesis-core/src/config/anatomy.rs to change population_pct to density
- Fix genesis-core/src/config/test_config.rs and test_anatomy.rs explicitly
- Refactor genesis-baker/src/validator/checks.rs to remove check_layer_populations
- Refactor genesis-baker/src/bake/neuron_placement.rs to update generate_placement_from_config
- Fix all calls to generate_placement_from_config across genesis-baker
- Update example configs (cartpole, ant_v4) to use density in anatomy.toml files
- Remove deprecated ant_v4 example files including client.py and all configs

## [0.363.42] - 2026-03-10 19:20:09

**Visualization of each individual neuron**

## [0.362.42] - 2026-03-10 18:39:23

**IMPORTANT**

### Added
- The engine has reached homeostasis. The network is physiologically stable.
- 16.96% surviving synapses with a perfect balance of 4.01 and 100.46% health.
- Cartpole Distributed Topology Migration & GNM-Library Slot Decay Patching
- Implement patch_gnm.py script to prepend `[[neuron_type]]` header to all .toml files in GNM-Library
- Add calculation and appending of slot_decay_ltm and slot_decay_wm based on is_inhibitory and dendrite_radius_um
- Apply formula: inhibitory neurons get fixed 128/128, excitatory use ltm = max(32, 128 - radius / 8) and wm = min(250, 128 + radius / 4)
- Process 1801 .toml files across Cortex, Cerebellum, Hippocampus, Striatum, and Thalamus directories
- Create SensoryCortex (L4), HiddenCortex (L2/3), and MotorCortex (L5) zones in examples/cartpole/config/zones/
- Implement zone-specific blueprints.toml using GNM-Library profiles (L4_spiny_MTG_1, L2_aspiny_MTG_1, L5_spiny_MTG_1, etc.)
- Configure anatomy.toml layers (L4_Sensory, L2_3_Hidden, L5_Motor) and shard.toml with world offsets (x=0,120,240)
- Set up io.toml with cartpole_sensors input (stride=1) for SensoryCortex and motor_actions output for MotorCortex
- Replace Sensorimotor zone with SensoryCortex, HiddenCortex, and MotorCortex zones in examples/cartpole/config/brain.toml
- Add [[connection]] from SensoryCortex to HiddenCortex and from HiddenCortex to MotorCortex (width=8, height=8)
- Update simulation.toml and remove obsolete cartpole_client.py and visualization PNGs
- Extend docs/specs/02_configuration.md with 13 lines of new content
- Revise docs/specs/05_signal_physics.md with 18 lines of updates to slot decay mechanics
- Modify docs/specs/07_gpu_runtime.md with 35 lines of GPU execution details

## [0.354.42] - 2026-03-10 03:11:46

**Lock-Free Telemetry Refactor & Baker Critical Fixes**

### Added
- Replace Mutex<DashboardState> with LockFreeTelemetry using AtomicU64, AtomicU32, AtomicI16, and crossbeam::queue::SegQueue
- Implement O(1) lock-free updates via telemetry.update_zone_spikes and push_log in genesis-node/src/tui/state.rs
- Refactor shard_thread.rs to remove all .lock().unwrap() calls, using direct atomic operations and zero-cost spike reporting
- Optimize orchestrator in node/mod.rs, replacing reporter with telemetry and using relaxed atomic stores for batch/tick counts
- Eliminate Mutex<DashboardState> from main.rs and boot.rs, passing telemetry as the only shared state during bootstrap
- Refine run_app and run_log_reporter in tui/mod.rs to own a local DashboardState and pull data from atomics
- Add dynamic zone discovery to automatically initialize ZoneMetrics based on telemetry bridge hashes
- Implement wall clock history synchronization using local_state.push_wall_ms(wall) for accurate UI sparklines
- Optimize log drainage from the SegQueue buffer to update local UI state atomically
- Fix type mask packing in axon_tips_uvw within topology.rs, implementing 11-11-6-4 packing with a 4-bit Type Mask in bits [31..28]
- Correct Dale's Law implementation in dendrite_connect.rs, ensuring the presynaptic neuron (axon segment owner) dictates synapse sign using owner_type.is_inhibitory

## [0.347.40] - 2026-03-10 01:00:43

**Implement full axon path storage and interactive 3D visualization**

### Added
- Add ShardSoA.paths field and dump_to_disk logic in genesis-baker/src/bake/layout.rs to write 16-byte header, lengths array, and A*256 u32 segments
- Update genesis-baker/src/bake/sprouting.rs to retain and provide segment data using 11-11-6 packed position layout
- Fix genesis-baker/src/bake/atlas_map.rs ghost format and bit-unpacking to use 11-11-6 layout
- Unify simulation coordinates in genesis-baker/src/bake/topology.rs, axon_growth.rs, and output_map.rs to 11-11-6
- Fix legacy dendrite target packing (24/8) in genesis-core/src/test_tick.rs and types.rs
- Make scripts/visualize_neuron.py interactive by default with PyQt6 or Tkinter backend, add --save flag for PNG export
- Add interactive 3D mode to scripts/visualize_ghosts.py with --show flag, using real shard.pos for neuron 3D positions
- Implement reading of shard.paths binary file to extract and render intermediate points of ghost axons across shards
- Parse manifest.toml and shard.toml for world_offset and dimensions to map Node A and Node B bounding boxes in 3D
- Add legend, metadata overlay, and statistics collection (neuron counts, ghost connection metrics) to visualize_ghosts.py
- Add ratatui and crossterm dependencies to genesis-node/Cargo.toml
- Implement tui module with DashboardState, ZoneMetrics, LogEntry, and responsive layout in genesis-node/src/tui/
- Replace SimpleReporter usage in main.rs, node/mod.rs, and boot.rs with Arc<Mutex<DashboardState>>
- Update shard_thread.rs to report zone metrics (spikes, phase) and log Night Phase latency in nanoseconds
- Add --log flag to Cli struct for plain text fallback mode, isolate TUI output by redirecting stdout/stderr
- Merge docs/specs/010_ide.md into 08_ide.md and delete the duplicate file
- Update reference in 011_cli_dashboard.md from 010_ide.md to 08_ide.md
- Fix broken links to design_specs.md in 9 spec files, replacing with README.md
- Convert // TODO comments to > [!NOTE] **[Planned]** markers in 7 locations across spec files

## [0.335.36] - 2026-03-09 19:55:46

**TUI mvp**

## [0.334.36] - 2026-03-09 15:07:46

**Integrate HiddenCortex and Hot-Reload for 3-Layer GNM CartPole**

### Added
- Integrate HiddenCortex zone (L2/3) to establish Sensory -> Hidden -> Motor topology
- Extend simulation.toml world definition to accommodate the three-layer architecture
- Update all zone configs (SensoryCortex, HiddenCortex, MotorCortex) for GNM integration
- Implement Hot-Reload of ZoneManifest with atomic settings block in NodeContext
- Add cyclic file checking in NodeRuntime for live manifest updates without restart
- Extend genesis-core/src/config/manifest.rs ZoneManifest to support settings block
- Modify genesis-node/src/boot.rs and genesis-node/src/node/mod.rs to integrate hot-reload logic
- Implement GPU Constant Memory hot-reload for physics and plasticity parameters
- Fix genesis-baker structural synchronization in src/main.rs
- Update example README with 3-layer guide and 100+ points record
- Achieve record 3100+ TPS in IntraNode mode after HiddenCortex integration
- Increase checkpoint frequency to 100,000 ticks for optimized I/O
- Refine L4 -> L5 targeting specificity in zone configurations
- Eliminate Action Selection Bias by removing `>=` favoring 0 and adding persistence on force equality
- Implement live_dashboard.py for real-time training graph visualization
- Finalize dashboard vertical layout 2.2
- Add weight analysis visualization with Green/Red color maps for all three cortices
- Update docs/specs/02_configuration.md and roadmap with Topology Distillation concept

## [0.319.36] - 2026-03-09 05:56:48

**GENESIS: Hierarchical Evolution & Training Breakthrough**

### Added
- Full 3-Layer Cortex: Deployed L4 (Sensory) -> L2/3 (Hidden) -> L5 (Motor) architecture.
- TPS Record: Hit 3100+ across 450k neurons. Hardware is absolutely singing!
- Training: Score surpassed 100 milestones thanks to the hidden layer and client-side bug fixes.
- Logic: Excised random tie-breaks; implemented action persistence (inertia).
- Analytics: Observed 99% inhibitory stabilization in the Motor layer. The network lives and thinks.
- Dashboard 2.0: Introduced TPS tracking, multi-SMA (25/100/300), and interactive scrolling.
- Specs: Established roadmap for graph distillation and hot-reloading for nocturnal phases.

## [0.312.36] - 2026-03-09 05:56:30

**GENESIS: Hierarchical Evolution & Training Breakthrough**

### Added
- Full 3-Layer Cortex: Deployed L4 (Sensory) -> L2/3 (Hidden) -> L5 (Motor) architecture.
- TPS Record: Hit 3100+ across 450k neurons. Hardware is absolutely singing!
- Training: Score surpassed 100 milestones thanks to the hidden layer and client-side bug fixes.
- Logic: Excised random tie-breaks; implemented action persistence (inertia).
- Analytics: Observed 99% inhibitory stabilization in the Motor layer. The network lives and thinks.
- Dashboard 2.0: Introduced TPS tracking, multi-SMA (25/100/300), and interactive scrolling.
- Specs: Established roadmap for graph distillation and hot-reloading for nocturnal phases.

## [0.305.36] - 2026-03-08 22:38:53

**IMPORTANT**

### Added
- GNM: deploy high-fidelity neuron library v2.1
- 1801 unique biphy-calibrated models (Allen/NeuroMorpho)
- fixed synaptic weights: Exc (0.7%), PV/Inh (3%) of threshold delta
- enforced invariants: signal_propagation_length >= refractory_period + 1
- metadata: improved meta-plasticity curves and GSOP Dead Zone fixes
- docs: added Credits.md with formal research attribution

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
- **True Hardware E2E Scalability (§2)** - verified 1M neurons simulating at real physics bounds
- E2E synthetic benchmark `e2e_test.rs` capturing real CUDA execution speeds bypassing CPU Baker overhead
- Real-time physics bounds metrics on GTX 1080 Ti equivalent: ~32k Ticks/s (1K), ~22k Ticks/s (10K), ~5k Ticks/s (100K)
- Proved memory safety and full pipeline closure from Virtual Input → GLIF → GSOP → Output Readout loops

## [0.13.5] - 2026-02-26

### Added
- **Readout Interface (Output §3)** - `record_readout_kernel` extracting flagged motor spikes
- Dense `output_history` buffer capturing batched readout spikes per tick inside VRAM
- `IoConfig` expanded in Core config with `OutputMap` and `readout_batch_ticks` for tiled outputs
- Atlas tiling generation for motor soma assignment via `.gxo` files in Baker
- DayPhase integration of readout recording into the simulation fast-path
- Removed obsolete `record_outputs.cu` and atomic spike routing logic
- **Sleep API and Spike Drop** - `is_sleeping` and `sleep_requested` added to runtime. Sleeping zones drop incoming spikes (Legalized Amnesia §2.3) and skip physics.
- **Secure Cross-Shard Geometry** - bounds check (`if ghost_id < total_axons`) in `apply_spike_batch_kernel`
- **Night Phase Checkpointing** - logic dumping `pre_sprout` states to disk before Baker processing

### Fixed
- SIGSEGV in mock-gpu runtime tests resolved
- Fix `local_axons_count` being shadowed after virtual axon append

## [0.12.3] - 2026-02-25

### Added
- **Input Interface (Virtual Axons §2.1)** - added `inject_inputs.cu` to map 1-bit external bitmasks to virtual axon firing
- Abstracted `grow_single_axon()` for reuse in Input map generation (`grow_input_maps`)
- Deterministic FNV-1a seeded routing for `pixel→soma` translation across virtual axons
- Generation of `.gxi` (Genesis eXternal Input) binary format with Header, Map Descriptors, and flat axon arrays
- Expanded VRAM state to load `.gxi` indirection tables (`map_pixel_to_axon`) and allocate bitmask buffers
- Batched bitmask upload capability (`upload_input_bitmask`) via DayPhase
- **Testing Architecture** - comprehensive unit testing for Cone Tracing algorithm (26 tests in genesis-baker) covering SpatialGrid sensing, trajectory steering, and multi-shard generation
- **Testing Architecture** - comprehensive testing of synaptogenesis (dendrite_connect) validating Rule of Uniqueness, Type Whitelist, Inhibitory signs, and Self-Exclusion with an ASCII visualizer

### Fixed
- Active Tail bounds alignment and axon sentinel refresh implemented
- Protective layers against Signal Superimposition and refraction bounds
- Fix broken indentation and redundant unsafe blocks in VramState drop

## [0.11.0] - 2026-02-24

### Added
- **Ghost Axons (§1.7)** - бесшовный рост аксонов через границы шардов
- `ShardBounds` structure with `full_world()` and `is_outside()` boundary detection
- `GhostPacket` inter-shard transfer format (entry point, direction, remaining steps)
- `inject_ghost_axons()` - ghost axon growth continuation in receiving shard
- Pipeline integration in `main.rs` with diagnostic logging
- Unit tests for boundary detection and ghost packet handling
- Updated spec `04_connectivity.md` §1.7 with full protocol description

## [0.10.1] - 2026-02-24

### Added
- **Power Score activation** - `compute_power_index` now called during Night Phase sprouting
- Whitelist filtering in `reconnect_empty_dendrites()` (was missing)
- `sprouting_weight_type` config parameter for soft type-matching scoring component

## [0.10.0] - 2026-02-24

### Added
- **Rule of Uniqueness (§1.4)** - `HashSet`-based deduplication prevents redundant axon connections
- **Dendrite Whitelist (§1.5)** - per-type compatibility filtering via `dendrite_whitelist` in blueprints
- **Configurable Initial Weight** - `initial_synapse_weight` moved to `blueprints.toml`
- Unit tests for whitelist and initial weight parsing
- Updated spec `04_connectivity.md` §1.4–1.5

## [0.9.0] - 2026-02-24

### Added
- **GPU LUT Expansion 4→16** - each of 16 neuron types gets a unique physical profile (GLIF/GSOP)
- **Voxel Uniqueness** - reject-sampling guarantees one voxel = at most one neuron
- `growth_vertical_bias`, `type_affinity`, `is_inhibitory` fields in blueprints
- Blueprints.toml updated with 4 base types (Vertical_Excitatory, Horizontal_Inhibitory, Stable_Excitatory, Relay_Excitatory)

## [0.8.0] - 2026-02-24

### Added
- **Binary Formatting (§2.1)** - formalized GSNS/GSAX header specs
- `InstanceConfig` refactored into dedicated `instance.rs`
- Default CLI paths updated to `config/zones/V1/*`
- E2E test script paths corrected

## [0.7.0] - 2026-02-23

### Added
- **Configuration Architecture (Spec 02 §1.1–1.3)** - `simulation.toml` parser in genesis-core
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
- **Night Phase IPC Baker Daemon** - `genesis-baker-daemon` executable running in background
- **Shared Memory Protocol (SHM)** - zero-copy transfer of weights and targets between CUDA runtime and CPU baker
- **Unix Sockets** - for JSON control messages (`night_start`, `night_done`) synchronization
- **Sort & Prune CUDA Kernel** - O(1) register Bitonic Sort (N=128) per neuron to auto-promote LTM/WM and prune weak connections
- Integration E2E IPC tests verifying full orchestrator pipeline handoff

## [0.5.0] - 2026-02-23


### Added
- **Smart Axon Growth: Cone Tracing** - iterative, biologically plausible axon sprouting
- `SpatialGrid` spatial hash map for O(1) neighbor lookup during growth
- V\_attract inverse-square law and piecewise tip geometry
- Variable-length `.axons` binary format (tip\_x, tip\_y, tip\_z, length per axon)

## [0.4.1] - 2026-02-23

### Added
- **Genesis IDE** - new `genesis-ide` crate with Bevy 3D viewer
- Orbital + Fly camera modes with mouse/keyboard control
- HUD overlay: FPS, neuron count, axon count, selected neuron info
- Neuron spheres colored by `type_mask`
- Baker exports `shard.positions`; IDE highlights spiking neurons via WebSocket glow (3 frames)

## [0.4.0] - 2026-02-23

### Added
- **Genesis Monitor** - WebSocket telemetry server broadcasting tick, phase, and real-time spike dense IDs
- Bevy 3D client renders neurons as glow-highlighted spheres synced to live VRAM state

## [0.3.0] - 2026-02-22

### Added
- **Ghost Axon Handover** - TCP Slow Path with VRAM reserve pool and Handover handshake
- Dynamic `SpikeRouter` route registration via slow path
- **Homeostatic Plasticity** - branchless penalty/decay in GLIF kernel
- Equilibrium validated across 100 CUDA ticks

## [0.2.0] - 2026-02-22

### Added
- `genesis-node` daemon: parses `shard.toml`, mounts VRAM, drives BSP ephemeral loop
- **Atlas Routing** - external Ghost Axons baked at compile time, zero GPU overhead at runtime

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
