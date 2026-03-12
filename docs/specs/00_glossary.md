# Genesis Architecture Glossary

**Version:** 0.0.2 ***"Stable MVP"***  
**Last Updated:** 2026-03-12  
**Scope:** Complete terminology reference across all 11 Genesis specifications

---

## Overview

This glossary defines ~60 core terms used throughout the Genesis neuromorphic simulation architecture. Terms are alphabetically organized with cross-references to primary spec sections.

---

## Terms

### Active Tail

The trailing edge of a dendritic tree segment that contains soma-proximal synapses still receiving spike signals. Contrasts with the "dormant head" in pruned branches.

**See also:** Pruning, Inertia Curves  
**Specs:** 04_connectivity.md (§4.3), 05_signal_physics.md (§5.2)

---

### Anatomical Blueprint (Anatomy)

TOML-formatted specification file defining:
- Brain zone topology (zones, dimensions, neuron densities)
- Layer arrangement and Z-order sorting
- Initial axon/dendrite connectivity patterns (via cones)

Generated from `.toml` files in `config/zones/*/anatomy.toml` by genesis-baker.

**See also:** Baking, Configuration  
**Specs:** 02_configuration.md (§2.1), 09_baking_pipeline.md (§9.1)

---

### Axon Sentinel

Boundary marker at the end of each neuron's axon output list on GPU. Enables O(1) detection of axon array bounds during spike propagation. Stored as `0xFFFFFFFF` soma ID.

**See also:** Dense Index, Pruning  
**Specs:** 05_signal_physics.md (§5.3), 07_gpu_runtime.md (§7.2)

---

### Baking (Compilation Pipeline)

Multi-stage process converting TOML configuration → GPU-ready binary formats:
1. **Parse:** Anatomy + Blueprint TOML
2. **Validate:** Check constraints (max dendrites, axon counts)
3. **Place:** Z-sort neurons, assign grid coordinates
4. **Connect:** Cone tracing, growth sprouting
5. **Serialize:** Output .gxi/.gxo/.ghosts binary files

Executed by genesis-baker, runs on CPU only.

**See also:** Anatomical Blueprint, Genesis-Baker, GXI/GXO  
**Specs:** 09_baking_pipeline.md (entire), 02_configuration.md (§2.3)

---

### Blueprint (Connectivity)

TOML specification of target connectivity patterns:
- Neuron type pairs (e.g., SensoryCortex → HiddenCortex)
- Connection probability or density
- Synaptic weight initialization distributions
- Growth/pruning rate schedules

Defines "wiring rules" that baking applies via cone tracing + sprouting.

**See also:** Anatomical Blueprint, Cone Tracing, Sprouting  
**Specs:** 04_connectivity.md (§4.1), 09_baking_pipeline.md (§9.2)

---

### Brain Shard (Shard)

Independently-executable GPU subdivision of a zone. Each shard:
- Contains ~1000-10000 neurons (memory-constrained)
- Has dedicated VRAM allocation (state, weights, inputs, outputs)
- Runs Day Phase kernels asynchronously
- Communicates via ghost axons (inter-shard) or network (inter-zone)

Multiple shards per zone enable large-scale training.

**See also:** Zone, Night Phase, Ghost Axon, Day Phase  
**Specs:** 06_distributed.md (§6.1), 07_gpu_runtime.md (§7.3)

---

### BSP Barrier (Bulk Synchronous Parallel)

Inter-zone synchronization primitive ensuring all zones complete Day Phase before advancing global tick. Implemented via:
- Barrier daemon monitoring zone completion timestamps
- Blocking gate preventing Night Phase until all zones ACK ready
- Used in MVP distributed mode (P1+)

**See also:** Day Phase, Night Phase, Distributed Mode  
**Specs:** 06_distributed.md (§6.2), 07_gpu_runtime.md (§7.5)

---

### Bare Metal Runtime (Genesis-Lite)

A streamlined, C/C++ implementation of the Genesis compute kernel designed for microcontrollers (e.g., ESP32-S3) and embedded systems. Operates without a full OS, targeting direct hardware interactions.

**See also:** ESP32, I/O Matrix  
**Specs:** 10_hardware_backends.md (§3.1)

---

### Channel Trait

Rust abstraction enabling pluggable I/O backends:
- `ExternalChannel`: UDP/WebSocket to external clients (CartPole, etc.)
- `GhostChannel`: Inter-shard spike batching
- `IntraGpuChannel`: D2H/H2D transfers and Constant Memory updates

Central to zone_runtime.rs orchestration.

**See also:** External I/O, Ghost Axon, IPC  
**Specs:** 07_gpu_runtime.md (§7.4), 08_io_matrix.md (§8.1)

---

### Columnar Layout (Structure of Arrays / SoA)

GPU memory organization storing neuron properties as separate linear arrays:
- `soma_positions[]`: X,Y,Z grid coordinates (PackedPosition)
- `v_soma[]`: Voltage state vectors (64 floats each)
- `gsop_matrix[]`: Growth/stasis/pruning state (VariantParameters)
- `dendrite_slots[][]`: 128-deep dendritic trees per neuron

Enables coalesced memory access and AVX-512 vectorization on CPU.

**See also:** PackedPosition, Warp Alignment, VramState  
**Specs:** 03_neuron_model.md (§3.2), 07_gpu_runtime.md (§7.2)

---

### Cone Tracing

Geometric algorithm for axon sprouting:
1. Define axon cone: origin soma, target zone, angle/distance limits
2. Enumerate neurons in target zone within cone bounds
3. Stochastically connect to subset (based on connection probability)
4. Initialize synapse weights from distribution

Used by baking pipeline to create realistic long-range connectivity.

**See also:** Sprouting, Blueprint, Baking  
**Specs:** 04_connectivity.md (§4.2), 09_baking_pipeline.md (§9.2)

---

### Day Phase

GPU-resident simulation cycle executing 6 CUDA kernels per tick:
1. **PropagateAxons**: Process dendritic tree inputs → soma voltage
2. **ApplySpikeBatch**: Add spike buffers to dendritic trees
3. **InjectInputs**: Add external I/O (CartPole, etc.)
4. **UpdateNeurons**: Integrate differential equations (Hodgkin-Huxley variant)
5. **ApplyGSOP**: Growth/stasis/pruning state transitions
6. **RecordOutputs**: Copy spike outputs to I/O matrices

Completes in ~10-100 ms on modern GPU. Followed by Night Phase (CPU-resident).

**See also:** Night Phase, CUDA Kernels, Tick  
**Specs:** 05_signal_physics.md (§5.4), 07_gpu_runtime.md (§7.3)

---

### Dense Index

Spike propagation data structure mapping neuron ID → dendritic tree children. Enables O(log N) lookup of which neurons receive input from a given axon.

Stored as bitmask array with Axon Sentinels marking boundaries.

**See also:** Axon Sentinel, Pruning  
**Specs:** 05_signal_physics.md (§5.3), 07_gpu_runtime.md (§7.2)

---

### External I/O

Communication channel between Genesis and external processes (CartPole client, visualization tools):
- **UDP 8081 (Input):** Receives sensory data (encoded as Input_Bitmask)
- **UDP 8082 (Output):** Sends motor commands (encoded as Output_History)
- **WebSocket (TelemetryServer):** Real-time telemetry (spikes, voltages)
- **TCP (GeometryServer):** Static geometry queries

Multiplexed via ExternalIoHeader (zone_hash, matrix_hash, payload_size).

**See also:** I/O Matrix, CartPole, ExternalIoServer  
**Specs:** 08_io_matrix.md (§8.2), 08_ide.md (§2.3)

---

### ExternalIoHeader

Binary packet format for UDP I/O:
```
Offset  Field         Type      Purpose
0-4     zone_hash     u32       Identifies receiving zone
4-8     matrix_hash   u32       Identifies I/O matrix version
8-12    payload_size  u32       Bytes in spike data
12+     payload       [u8]      Input_Bitmask or Output_History
```

Used by UDP 8081/8082 External I/O servers.

**See also:** External I/O, I/O Matrix  
**Specs:** 08_io_matrix.md (§8.2), 07_gpu_runtime.md (§7.4)

---

### Genesis-Baker (Baker)

CPU-only Rust application (genesis-baker crate) performing the Baking pipeline:
- Input: Anatomy + Blueprint TOML files
- Processing: Validation, placement, connectivity, serialization
- Output: `.gxi`, `.gxo`, `.ghosts` binary files in `baked/` directory

Runs on developer machine or CI/CD prior to runtime deployment.

**See also:** Baking, GXI/GXO, Genesis-Runtime  
**Specs:** 09_baking_pipeline.md (§9.1), project_structure.md (§genesis-baker/)

---

### Genesis-Core (Core)

Shared Rust library (genesis-core crate) providing:
- Type definitions (Signal, PackedPosition, VariantParameters)
- Constants (PackedPosition bit widths, MAX_DENDRITE_SLOTS=128)
- Mathematical primitives (GSOP curves, signal physics)
- IPC message framing (Channel trait, ghost protocol)

Depended on by baker, runtime, and IDE.

**See also:** PackedPosition, VariantParameters, Channel Trait  
**Specs:** 05_signal_physics.md (§5.1), project_structure.md (§genesis-core/)

---

### Genesis-IDE (IDE)

GPU-accelerated 3D visualization tool (genesis-ide crate):
- Built with Bevy engine
- Subscribes to TelemetryServer (WebSocket)
- Real-time rendering of neuron spikes, voltage heat maps, growth visualization
- Camera controls for zone inspection

Primarily for development/debugging, not production.

**See also:** TelemetryServer, Visualization  
**Specs:** 08_ide.md (entire), project_structure.md (§genesis-ide/)

---

### Genesis-Runtime (Runtime)

GPU-intensive Rust application (genesis-runtime crate) executing:
- Day Phase CUDA kernels (5 computational + 1 readout per tick)
- Night Phase state transitions (CPU-resident)
- I/O multiplexing (External I/O, Ghost Sync, Telemetry)
- Zone orchestration (per-shard task scheduling)

Primary production deployment target.

**See also:** Day Phase, Night Phase, Zone  
**Specs:** 07_gpu_runtime.md (entire), project_structure.md (§genesis-runtime/)

---

### Ghost Axon

Virtual representation of inter-shard axon connection. At runtime:
1. Outgoing spikes buffered locally (local shard)
2. Ghost Sync kernel transfers D2H
3. ExternalIoServer multiplexes to receiving shard
4. Receiving shard unpacks and injects into dendrites

Enables independent shard execution while maintaining biological connectivity.

**See also:** Brain Shard, Day Phase, Ghost Sync  
**Specs:** 06_distributed.md (§6.1), 05_signal_physics.md (§5.5)

---

### Ghost Sync Kernel

CUDA kernel (6th Day Phase kernel, executed after RecordOutputs) that:
1. Read spike output buffers
2. Construct GhostConnection packets (inter-shard links)
3. Transfer via H2D to ExternalIoServer
4. ExternalIoServer routes to receiving shards

Runs asynchronously with Night Phase execution.

**See also:** Ghost Axon, Day Phase, ExternalIoServer  
**Specs:** 05_signal_physics.md (§5.5), 07_gpu_runtime.md (§7.3)

---

### GLIF (Generalized Leaky Integrate-and-Fire)

Neuronal model variant incorporating:
- Subthreshold dynamics: `dv/dt = -(v - v_rest) / τ + I_in`
- Spike threshold: `v > v_thresh → emit spike`
- Refractoriness: Dead time after spike emission
- Adaptation: Current-dependent threshold increase

Genesis implements GLIF with 64-dimensional voltage state per neuron.

**See also:** Hodgkin-Huxley, Voltage State  
**Specs:** 03_neuron_model.md (§3.1), 05_signal_physics.md (§5.4)

---

### HIP / ROCm (AMD Backend)

The C++ heterogeneous-compute interface (HIP) and open-source platform (ROCm) for AMD GPUs. Genesis targets HIP as its primary alternative to CUDA for server-grade hardware diversification.

**See also:** Day Phase, GPU Runtime  
**Specs:** 10_hardware_backends.md (§2.1), 07_gpu_runtime.md (§1.6)

---

### GSOP (Growth, Stasis, Prune Operator)

State machine controlling synapse lifespan:
- **Growth** (G): Newly formed synapse, potentiating
- **Stasis** (S): Established synapse, stable strength
- **Pruning** (P): Marked for removal, inertial decay

Variant-encoded in VariantParameters (16 bits per synapse). Transitions driven by:
- Spike correlation (Hebbian rules)
- Weight magnitude (pruning small weights)
- Activity history (active tail priority)

**See also:** VariantParameters, Inertia Curves, Pruning  
**Specs:** 04_connectivity.md (§4.3), 07_gpu_runtime.md (§7.2)

---

### GXI (Genesis eXternal Input)

Binary file format for Input I/O matrices:
- **Header:** zone_id, matrix_hash, input_count, total_pixels
- **Data:** Bitmask array (1 bit per soma), indexed by input neuron ID
- **Size:** ~(total_pixels / 8) bytes

Generated by baking pipeline. Loaded by ExternalIoServer at runtime.

**See also:** GXO, I/O Matrix, External I/O  
**Specs:** 08_io_matrix.md (§8.1), 09_baking_pipeline.md (§9.3)

---

### GXO (Genesis eXternal Output)

Binary file format for Output I/O matrices:
- **Header:** zone_id, matrix_hash, output_count
- **Data:** SoA arrays (soma_ids[], spike_history[])
- **SoA Entry:** 12 bytes (soma_id: u32, spike_history: u64 bitmap over last 8 ticks)

Generated by baking pipeline. Used by RecordOutputs kernel to fill output buffers.

**See also:** GXI, I/O Matrix, RecordOutputs  
**Specs:** 08_io_matrix.md (§8.1), 09_baking_pipeline.md (§9.3)

---

### Hodgkin-Huxley

Classical neurophysiological model with separate gating variables for ion channels:
- Na (depolarizing), K (repolarizing), L (leak)
- Each with voltage-dependent opening/closing rates

Genesis uses simplified variant (GLIF-like) with fewer parameters but biological plausibility.

**See also:** GLIF, Voltage State, Neuron Model  
**Specs:** 03_neuron_model.md (§3.1), 05_signal_physics.md (§5.1)

---

### I/O Matrix

Interface abstraction mapping:
- **Input:** External sensory data → soma voltage injection
- **Output:** Spike records → external command signals

Stored in GXI (input routing), GXO (output recording), and ExternalIoHeader (packet envelope).

**See also:** GXI, GXO, External I/O  
**Specs:** 08_io_matrix.md (entire)

---

### Inertia Curves

Time-dependent decay profiles for synapses in GSOP Pruning state:
- Functionally: `strength(t) = strength_0 * exp(-t / τ_inertia)`
- Biologically: Represents post-synaptic physical dissolution
- Prevents abrupt connectivity loss

Active Tail synapses have longer inertias than dormant head synapses.

**See also:** GSOP, Active Tail, Pruning  
**Specs:** 04_connectivity.md (§4.3), 05_signal_physics.md (§5.2)

---

### LTM Slot / Working Memory (WM) Slot

Dendritic compartment classification:
- **LTM (Long-Term Memory):** Slow synapses, high integration window (~100 ms)
- **WM (Working Memory):** Fast synapses, short integration window (~10 ms)

Each neuron allocates 128 dendrite slots split between LTM/WM ratios (configurable).

**See also:** Dendrite, Segment, Neuron Model  
**Specs:** 03_neuron_model.md (§3.2), 04_connectivity.md (§4.2)

---

### Night Phase

CPU-resident state transition cycle following Day Phase (asynchronous):
- **Host Sleep:** Allow zone to sleep if input-quiet (configurable latency)
- **Plasticity Updates:** GSOP state transitions, weight updates
- **Pruning:** Remove marked synapses (actual deletion in LTM/WM slots)
- **Growth Opportunity:** Check sprouting conditions, queue new axons

Variable per-zone latency enables low-power idle states.

**See also:** Day Phase, GSOP, Pruning, Sprouting  
**Specs:** 06_distributed.md (§6.3), 07_gpu_runtime.md (§7.4)

---

### Neuron Model

Mathematical framework for soma dynamics. Genesis supports:
- **Primary:** GLIF (Generalized Leaky Integrate-and-Fire)
- **Optional:** Hodgkin-Huxley (reduced parameter set)

Parameters per neuron:
- Leak resistance, capacitance, resting potential, threshold
- Spike width, refractory period
- Adaptation time constants

Integrated via Euler method in UpdateNeurons kernel.

**See also:** GLIF, Hodgkin-Huxley, UpdateNeurons  
**Specs:** 03_neuron_model.md (entire)

---

### Neuromorphic Computing

Hardware architectures that emulate biological neural structures (e.g., Intel Loihi). Genesis biospecs (GSOP, GLIF) are designed for future mapping onto event-driven, asynchronous neuromorphic silicon.

**See also:** ASIC, GNM  
**Specs:** 10_hardware_backends.md (§4.1)

---

### PackedPosition

Compact encoding of 3D neuron coordinates:
```
Bits    Field      Range       Use
0-9     X          [0, 1023]   Grid column
10-19   Y          [0, 1023]   Grid row
20-29   Z          [0, 1023]   Layer depth
30-31   reserved   —           Future extension
```

Stored as u32. Enables coalesced GPU memory access (all positions in same cache line).

**See also:** Columnar Layout, Warp Alignment  
**Specs:** 03_neuron_model.md (§3.2), 07_gpu_runtime.md (§7.2)

---

### Pruning

Process removing low-strength synapses to reduce memory/compute:
1. Mark synapses in GSOP Pruning state
2. Apply inertia curve decay (exponential)
3. Delete from LTM/WM slots when strength → 0
4. Update dendritic trees

Triggered by activity patterns during Night Phase.

**See also:** GSOP, Inertia Curves, Active Tail, Night Phase  
**Specs:** 04_connectivity.md (§4.3), 06_distributed.md (§6.3)

---

### Readout Interface

CUDA kernel interface for extracting spike outputs:
```
__global__ void recordOutputs(
  const SoA_State* vram_state,
  const GxoFile* output_mapping,
  ExternalIoHeader* output_packet
)
```

Copies spike buffers to ExternalIoHeader payload for UDP 8082 transmission.

**See also:** RecordOutputs, GXO, External I/O  
**Specs:** 05_signal_physics.md (§5.5), 08_io_matrix.md (§8.1)

---

### RecordOutputs Kernel

6th CUDA kernel in Day Phase, writes spike records:
1. Read spike buffer (Output_History) from VRAM
2. Map to GXO output routing (soma_ids → matrix positions)
3. Fill ExternalIoHeader packet
4. Signal ExternalIoServer ready

Last computational step before Ghost Sync.

**See also:** Day Phase, GXO, Readout Interface  
**Specs:** 05_signal_physics.md (§5.5), 07_gpu_runtime.md (§7.3)

---

### Segment

Individual dendritic compartment (synaptic input point):
- Contains single synapse (presynaptic axon connection)
- Assigned to LTM or WM slot
- Has weight, GSOP state (variant), location

Segments are atomic units of connectivity; 128 per neuron max.

**See also:** Dendrite, LTM/WM Slot, Synapse  
**Specs:** 03_neuron_model.md (§3.2), 04_connectivity.md (§4.1)

---

### Shard

See **Brain Shard**.

---

### Signal Physics

Comprehensive model of spike propagation and integration:
- **Axon dynamics:** Threshold crossing, spike emission, transmission delay
- **Dendritic integration:** Segment-level synaptic currents → soma voltage
- **Soma dynamics:** GLIF equation, spike generation, refractoriness

Implemented across 5 Day Phase kernels (PropagateAxons, ApplySpikeBatch, InjectInputs, UpdateNeurons, ApplyGSOP).

**See also:** GLIF, Day Phase, Spike  
**Specs:** 05_signal_physics.md (entire)

---

### SoA

See **Columnar Layout**.

---

### Spike

Binary event (all-or-nothing) emitted by soma when voltage exceeds threshold:
- Timestamp: precise tick in global simulation timeline
- Source: soma ID (1-N where N = zone population)
- Destination: all presynaptic targets (multi-cast via Dense Index)

Propagated asynchronously during ApplySpikeBatch kernel.

**See also:** Axon, Signal Physics, Dense Index  
**Specs:** 05_signal_physics.md (§5.1), 07_gpu_runtime.md (§7.3)

---

### Sprouting

Biological process generating new axon connections:
1. Identify growth candidate neurons (high activity activity)
2. Define cone in target zone
3. Stochastically select targets (via cone tracing)
4. Initialize new synapses with small weights (Hebbian reset)

Triggered during Night Phase plasticity if conditions met.

**See also:** Cone Tracing, Night Phase, Growth (GSOP)  
**Specs:** 04_connectivity.md (§4.2), 09_baking_pipeline.md (§9.2)

---

### TelemetryFrameHeader

Binary header for telemetry WebSocket packets:
```
Offset  Field         Type      Purpose
0-4     magic         u32       0xDEADBEEF (validation)
4-12    tick          u64       Global simulation tick
12-16   spikes_count  u32       # spikes in this frame
16+     spike_data    [u32]     Spike soma IDs
```

Sent by TelemetryServer to genesis-ide every ~100 ms.

**See also:** TelemetryServer, Genesis-IDE  
**Specs:** 08_ide.md (§2.3), 07_gpu_runtime.md (§7.4)

---

### TelemetryServer

WebSocket service (genesis-runtime) publishing:
- Spike events (every Day Phase)
- Voltage snapshots (per soma, sampled)
- Weight distributions (plasticity state)

Consumed by genesis-IDE for real-time visualization.

**See also:** Genesis-IDE, External I/O, WebSocket  
**Specs:** 08_ide.md (§2.3), 07_gpu_runtime.md (§7.4)

---

### Tick

Atomic unit of simulation time. One tick ≈ 1 ms biological time.

- Day Phase: All 6 CUDA kernels execute within 1 tick (~10 ms wall clock on modern GPU)
- Night Phase: Plasticity updates between ticks
- Global: Synchronized via BSP Barrier across all zones

**See also:** Day Phase, Night Phase, BSP Barrier  
**Specs:** 05_signal_physics.md (§5.4), 07_gpu_runtime.md (§7.1)

---

### UpdateNeurons Kernel

4th CUDA kernel in Day Phase, integrates neuronal dynamics:
1. Read soma voltage state (v_soma)
2. Compute input current from dendritic trees (dendritic integration)
3. Solve GLIF ODE: `dv/dt = -(v - v_rest) / τ + I_in`
4. Detect threshold crossing → emit spike
5. Apply refractoriness masking

Core of biophysical simulation.

**See also:** GLIF, Day Phase, Signal Physics  
**Specs:** 05_signal_physics.md (§5.4), 07_gpu_runtime.md (§7.3)

---

### VariantParameters

Packed encoding of synapse state (16 bits per synapse):
```
Bits    Field       Type        Use
0-3     variant     enum(4)     GSOP state (G/S/P/dormant)
4-7     reserved    —           Growth substate
8-15    weight_idx  u8          Index into quantized weight table
```

Stored in VRAM columnar array. Enables atomic GSOP transitions.

**See also:** GSOP, Columnar Layout, PackedPosition  
**Specs:** 07_gpu_runtime.md (§7.2), 05_signal_physics.md (§5.3)

---

### Virtual Axon

Conceptual extension of axon output beyond physical shard boundaries:
- Physically: Stored in ghost signal buffer per shard
- Logically: Axon "projects" to receiving shards via ExternalIoServer
- At runtime: InjectInputs kernel unpacks on receiving side

Enables neuron-to-neuron connectivity without explicit memory copy.

**See also:** Ghost Axon, Brain Shard, InjectInputs  
**Specs:** 06_distributed.md (§6.1), 05_signal_physics.md (§5.5)

---

### v_seg (Voltage Segment State)

Per-segment voltage variable representing local dendritic compartment potential:
- Integrated as: `dv_seg/dt = (v_soma - v_seg) / τ_seg + I_synapse`
- Contributes to soma integration window
- Enables compartmental model richer than single-compartment LIF

Genesis stores 64 v_seg values per neuron in columnar array.

**See also:** Compartmental Model, Dendritic Integration  
**Specs:** 03_neuron_model.md (§3.2), 05_signal_physics.md (§5.1)

---

### VramState

Complete GPU memory structure for one shard:
```c
struct VramState {
  PackedPosition* soma_positions;      // X,Y,Z grid coords
  float* v_soma;                        // Soma voltages (64×N)
  VariantParameters* gsop_matrix;       // Synapse GSOP state
  u8* input_bitmask;                    // Current tick inputs
  u64* output_history;                  // Last 8 ticks spikes
  Dendrite* dendrite_slots[128];        // Segment trees
};
```

Allocated once per shard at runtime initialization.

**See also:** Brain Shard, Columnar Layout, Day Phase  
**Specs:** 07_gpu_runtime.md (§7.2), 09_baking_pipeline.md (§9.1)

---

### Warp Alignment

GPU memory access pattern optimization where:
- 32 neuron data (warp size on NVIDIA) stored consecutively
- All arrays (v_soma, gsop_matrix, etc.) follow same pattern
- Enables automatic memory coalescing in CUDA kernels

Critical for >90% peak memory bandwidth utilization.

**See also:** Columnar Layout, SoA, PackedPosition  
**Specs:** 07_gpu_runtime.md (§7.2), 05_signal_physics.md (§5.5)

---

### Zone

Independently-configurable brain region containing one or more shards:
- **Shards per zone:** 1-N (depends on neuron count, memory budget)
- **Connectivity:** Intra-zone via dedicated links; inter-zone via ExternalIoServer
- **Night Phase Latency:** Per-zone variable sleep window (enables power scaling)

E.g., SensoryCortex (1 zone) → HiddenCortex (1-2 zones) → MotorCortex (1 zone).

**See also:** Brain Shard, Night Phase, Distributed Mode  
**Specs:** 02_configuration.md (§2.2), 06_distributed.md (§6.1)

---

## Alphabetical Index

| A | B | C | D | E |
|---|---|---|---|---|
| Active Tail | Anatomical Blueprint | Channel Trait | Day Phase | External I/O |
| Axon Sentinel | Baking | Columnar Layout | Dense Index | ExternalIoHeader |
| — | Blueprint | Cone Tracing | — | — |
| — | Brain Shard | — | — | — |
| — | BSP Barrier | — | — | — |

| F | G | H | I | L |
|---|---|---|---|---|
| — | Genesis-Baker | Hodgkin-Huxley | I/O Matrix | LTM/WM Slot |
| — | Genesis-Core | — | — | — |
| — | Genesis-IDE | — | — | — |
| — | Genesis-Runtime | — | — | — |
| — | Ghost Axon | — | — | — |
| — | Ghost Sync Kernel | — | — | — |
| — | GLIF | — | — | — |
| — | GSOP | — | — | — |
| — | GXI | — | — | — |
| — | GXO | — | — | — |

| N | P | R | S | T |
|---|---|---|---|---|
| Neuron Model | PackedPosition | Readout Interface | Shard | TelemetryFrameHeader |
| Night Phase | Pruning | RecordOutputs Kernel | Signal Physics | TelemetryServer |
| — | — | — | SoA | Tick |
| — | — | — | Spike | — |
| — | — | — | Sprouting | — |
| — | — | — | Segment | — |

| U | V | W | Z |
|---|---|---|---|
| UpdateNeurons Kernel | VariantParameters | Warp Alignment | Zone |
| — | Virtual Axon | — | — |
| — | v_seg | — | — |
| — | VramState | — | — |

---

## Cross-Spec Reference Map

| Term | Primary Spec | Secondary Specs |
|------|--------------|-----------------|
| **Active Tail** | 04_connectivity | 05_signal_physics |
| **Anatomical Blueprint** | 02_configuration | 09_baking_pipeline |
| **Axon Sentinel** | 05_signal_physics | 07_gpu_runtime |
| **Baking** | 09_baking_pipeline | 02_configuration |
| **Brain Shard** | 06_distributed | 07_gpu_runtime |
| **BSP Barrier** | 06_distributed | 07_gpu_runtime |
| **Channel Trait** | 07_gpu_runtime | 08_io_matrix |
| **Columnar Layout** | 03_neuron_model | 07_gpu_runtime |
| **Cone Tracing** | 04_connectivity | 09_baking_pipeline |
| **Day Phase** | 05_signal_physics | 07_gpu_runtime |
| **Dense Index** | 05_signal_physics | 07_gpu_runtime |
| **External I/O** | 08_io_matrix | 08_ide, 07_gpu_runtime |
| **ExternalIoHeader** | 08_io_matrix | 07_gpu_runtime |
| **Genesis-Baker** | 09_baking_pipeline | 02_configuration |
| **Genesis-Core** | 05_signal_physics | project_structure |
| **Genesis-IDE** | 08_ide | 07_gpu_runtime |
| **Genesis-Runtime** | 07_gpu_runtime | 05_signal_physics |
| **Ghost Axon** | 06_distributed | 05_signal_physics |
| **Ghost Sync Kernel** | 05_signal_physics | 07_gpu_runtime |
| **GLIF** | 03_neuron_model | 05_signal_physics |
| **GSOP** | 04_connectivity | 07_gpu_runtime |
| **GXI** | 08_io_matrix | 09_baking_pipeline |
| **GXO** | 08_io_matrix | 09_baking_pipeline |
| **Hodgkin-Huxley** | 03_neuron_model | 05_signal_physics |
| **I/O Matrix** | 08_io_matrix | 08_ide |
| **Inertia Curves** | 04_connectivity | 05_signal_physics |
| **LTM/WM Slot** | 03_neuron_model | 04_connectivity |
| **Night Phase** | 06_distributed | 07_gpu_runtime |
| **Neuron Model** | 03_neuron_model | 05_signal_physics |
| **PackedPosition** | 03_neuron_model | 07_gpu_runtime |
| **Pruning** | 04_connectivity | 06_distributed |
| **Readout Interface** | 05_signal_physics | 08_io_matrix |
| **RecordOutputs Kernel** | 05_signal_physics | 07_gpu_runtime |
| **Segment** | 03_neuron_model | 04_connectivity |
| **Signal Physics** | 05_signal_physics | 07_gpu_runtime |
| **SoA** | 03_neuron_model | 07_gpu_runtime |
| **Spike** | 05_signal_physics | 07_gpu_runtime |
| **Sprouting** | 04_connectivity | 06_distributed |
| **TelemetryFrameHeader** | 08_ide | 07_gpu_runtime |
| **TelemetryServer** | 08_ide | 07_gpu_runtime |
| **Tick** | 05_signal_physics | 07_gpu_runtime |
| **UpdateNeurons Kernel** | 05_signal_physics | 07_gpu_runtime |
| **VariantParameters** | 07_gpu_runtime | 05_signal_physics |
| **Virtual Axon** | 06_distributed | 05_signal_physics |
| **v_seg** | 03_neuron_model | 05_signal_physics |
| **VramState** | 07_gpu_runtime | 09_baking_pipeline |
| **Warp Alignment** | 07_gpu_runtime | 05_signal_physics |
| **Zone** | 02_configuration | 06_distributed |

---

## Usage Guide

### For Spec Writers

When introducing a term:
1. Check if it exists in this glossary
2. If new, add entry (alphabetically) with:
   - **Definition** (1-3 sentences, technical but accessible)
   - **See also:** Related terms (3-5 synonyms/dependencies)
   - **Specs:** Primary + secondary references
3. Update Cross-Spec Reference Map

### For Readers

To understand an architecture component:
1. Find primary term in alphabetical index
2. Read definition + related terms
3. Follow **Specs:** links to detailed sections
4. Use Cross-Spec Reference Map to see multi-term interactions

### Maintenance Schedule

- **Monthly:** Verify all 10 specs for new terminology; update glossary
- **Per major update:** Regenerate Cross-Spec Reference Map
- **Yearly:** Full glossary audit (remove obsolete terms, merge duplicates)

---

**End of Glossary**
