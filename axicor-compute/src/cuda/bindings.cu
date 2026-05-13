#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// =====================================================================
// 1. VRAM Layout (mirrors axicor_core::layout::VramState)
// =====================================================================
extern "C" {

// [DOD] 32-byte aligned Burst Architecture
struct alignas(32) BurstHeads8 {
  uint32_t h0; uint32_t h1; uint32_t h2; uint32_t h3;
  uint32_t h4; uint32_t h5; uint32_t h6; uint32_t h7;
};

struct SoA_State {
  uint32_t padded_n;
  uint32_t total_axons;

  int32_t* __restrict__ voltage;
  uint8_t* __restrict__ flags;
  int32_t* __restrict__ threshold_offset;
  uint8_t* __restrict__ refractory_timer;
  uint32_t* __restrict__ soma_to_axon;

  uint32_t* __restrict__ dendrite_targets;
  int32_t* __restrict__ dendrite_weights;
  uint8_t* __restrict__ dendrite_timers;

  BurstHeads8* __restrict__ axon_heads;

  uint32_t* __restrict__ input_bitmask;
  uint8_t* __restrict__ output_history;
  uint32_t* __restrict__ telemetry_count;
  uint32_t* __restrict__ telemetry_spikes;
};

// 64 bytes per variant (1 L1 cache line). 16 variants = 1024 bytes in Constant Memory.
struct alignas(64) VariantParameters {
  // === Group 1: 32-bit fields (bytes 0..20) ===
  int32_t threshold;
  int32_t rest_potential;
  uint32_t leak_shift;
  int32_t homeostasis_penalty;
  uint32_t spontaneous_firing_period_ticks;

  // === Group 2: 16-bit fields (bytes 20..28) ===
  uint16_t initial_synapse_weight;
  uint16_t gsop_potentiation;
  uint16_t gsop_depression;
  uint16_t homeostasis_decay;

  // === Group 3: 8-bit fields (bytes 28..32) ===
  uint8_t refractory_period;
  uint8_t synapse_refractory_period;
  uint8_t signal_propagation_length;
  uint8_t is_inhibitory; // 1 = true (GABA), 0 = false (Glu)

  // === Group 4: Inertia Curve & AHP (bytes 32..48) ===
  uint8_t inertia_curve[8];                 // 32..40
  uint16_t ahp_amplitude;                   // 40..42
  uint8_t _pad[6];                          // 42..48

  // === Group 5: Adaptive Leak Hardware (bytes 48..58) ===
  int32_t adaptive_leak_min_shift;          // 48..52
  uint16_t adaptive_leak_gain;              // 52..54
  uint8_t adaptive_mode;                    // 54..55
  uint8_t _leak_pad[3];                     // 55..58

  // === Group 6: Dopamine Receptors & Padding (bytes 58..64) ===
  uint8_t d1_affinity;                       // 58..59
  uint8_t d2_affinity;                       // 59..60
  uint32_t heartbeat_m;                      // 60..64
};
}

// Variant LUT uploaded to GPU Constant Memory (448 bytes).
// Shared symbol. Also referenced in physics.cu.
extern __constant__ VariantParameters VARIANT_LUT[16];
__constant__ int16_t current_dopamine;

__global__ void cu_extract_telemetry_kernel(
    const uint8_t* __restrict__ soma_flags,
    uint32_t* __restrict__ out_ids,
    uint32_t* __restrict__ out_count,
    uint32_t padded_n
);

// Constants (mirrored from constants.rs)
#define MAX_DENDRITE_SLOTS 128
#define AXON_SENTINEL 0x80000000

__device__ __forceinline__ void push_burst_head(BurstHeads8* h, uint32_t v_seg) {
  h->h7 = h->h6;
  h->h6 = h->h5;
  h->h5 = h->h4;
  h->h4 = h->h3;
  h->h3 = h->h2;
  h->h2 = h->h1;
  h->h1 = h->h0;
  // [DOD FIX] Wrap-around u32. Propagate signal starting from segment 0.
  h->h0 = (uint32_t)(0 - v_seg); 
}

// =====================================================================
// Kernel 1: Input Injection (InjectInputs)
// Ref: 08_io_matrix.md §2.6
// =====================================================================
__global__ void
inject_inputs_kernel(const SoA_State state, const uint32_t* __restrict__ bitmask,
                     uint32_t virtual_offset, uint32_t current_tick,
                     uint8_t input_stride, uint32_t total_virtual_axons) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_virtual_axons)
    return;

  // Apply stride (temporal downsampling)
  if (input_stride > 0 && (current_tick % input_stride) != 0)
    return;

  // Calculate effective tick index into bitmask
  uint32_t effective_tick =
      (input_stride == 0) ? 0 : (current_tick / input_stride);
  uint32_t words_per_tick = (total_virtual_axons + 63) / 64 * 2;

  uint32_t word_idx = (effective_tick * words_per_tick) + (tid / 32);
  uint32_t bit_idx = tid % 32;

  // Broadcast read: 32 threads share the same L1 cache line
  uint32_t mask_word = bitmask[word_idx];

  // Active bit = Burst Shift
  if ((mask_word >> bit_idx) & 1) {
    BurstHeads8 b = state.axon_heads[virtual_offset + tid];
    push_burst_head(&b, 1);
    state.axon_heads[virtual_offset + tid] = b;
  }
}

// =====================================================================
// Kernel 2: Apply Ghost Spike Batch (Fast Path)
// Ref: 06_distributed.md §2.4 (Sender-Side Mapping)
// =====================================================================
__global__ void apply_spike_batch_kernel(SoA_State state,
                                         const uint32_t* __restrict__ schedule_buffer,
                                         const uint32_t* __restrict__ counts,
                                         uint32_t current_tick,
                                         uint32_t max_spikes_per_tick) {
  uint32_t num_spikes = counts[current_tick];
  if (num_spikes == 0)
    return; // Early Exit

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_spikes)
    return;

  // Calculate offset into schedule buffer
  uint32_t offset = current_tick * max_spikes_per_tick + tid;
  uint32_t target_axon = schedule_buffer[offset];

  // [DOD FIX] Bounds check against VRAM to prevent out-of-range writes
  if (target_axon < state.total_axons) {
    BurstHeads8 b = state.axon_heads[target_axon];
    push_burst_head(&b, 1);
    state.axon_heads[target_axon] = b;
  }
}

// =====================================================================
// Kernel 3: Propagate Axon Signal Wavefronts (PropagateAxons)
// Ref: 05_signal_physics.md §1.6
// =====================================================================
__global__ void propagate_axons_kernel(const SoA_State state, uint32_t v_seg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= state.total_axons)
    return;

  BurstHeads8 h = state.axon_heads[tid];
  if (h.h0 != AXON_SENTINEL) h.h0 += v_seg;
  if (h.h1 != AXON_SENTINEL) h.h1 += v_seg;
  if (h.h2 != AXON_SENTINEL) h.h2 += v_seg;
  if (h.h3 != AXON_SENTINEL) h.h3 += v_seg;
  if (h.h4 != AXON_SENTINEL) h.h4 += v_seg;
  if (h.h5 != AXON_SENTINEL) h.h5 += v_seg;
  if (h.h6 != AXON_SENTINEL) h.h6 += v_seg;
  if (h.h7 != AXON_SENTINEL) h.h7 += v_seg;
  state.axon_heads[tid] = h;
}

// =====================================================================
// Kernel 6: Record Readout Output (Direct Memory Access)
// Ref: 08_io_matrix.md §3.2
// =====================================================================
__global__ void record_readout_kernel(const SoA_State state,
                                      const uint32_t* __restrict__ mapped_soma_ids,
                                      uint32_t num_channels,
                                      uint32_t current_tick) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_channels)
    return;

  uint32_t target_soma = mapped_soma_ids[tid];
  uint8_t is_spiking = state.flags[target_soma] & 0x01;

  // Write to 2D array [sync_batch_ticks x num_channels]
  // Flat access, no atomics needed
  uint32_t out_idx = current_tick * num_channels + tid;
  state.output_history[out_idx] = is_spiking;
}

// =====================================================================
// C-ABI Entry Points for Rust (FFI Bindings)
// =====================================================================
extern "C" {

void *gpu_malloc(size_t size) {
  void *ptr = NULL;
  cudaMalloc(&ptr, size);
  return ptr;
}

void gpu_free(void *ptr) { cudaFree(ptr); }

void *gpu_host_alloc(size_t size) {
  void *ptr = NULL;
  cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
  return ptr;
}

void gpu_host_free(void *ptr) { cudaFreeHost(ptr); }

void gpu_memcpy_host_to_device_async(void *dst, const void *src, size_t size,
                                     cudaStream_t stream) {
  cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

void gpu_memcpy_device_to_host_async(void *dst, const void *src, size_t size,
                                     cudaStream_t stream) {
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

bool gpu_memcpy_host_to_device(void *dst, const void *src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("CUDA ERROR in gpu_memcpy_host_to_device (size %zu): %s\n", size,
           cudaGetErrorString(err));
  }
  return err == cudaSuccess;
}
void gpu_memcpy_device_to_host(void *dst, const void *src, size_t size) {
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "[gpu_memcpy_device_to_host] CUDA ERROR: %s (src=%p, size=%zu)\n",
            cudaGetErrorString(err), src, size);
  }
}

int32_t gpu_stream_create(cudaStream_t *out_stream) {
  return (int32_t)cudaStreamCreateWithFlags(out_stream, cudaStreamNonBlocking);
}

int32_t gpu_stream_destroy(cudaStream_t stream) {
  return (int32_t)cudaStreamDestroy(stream);
}

void gpu_stream_synchronize(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
}

void gpu_set_device(int32_t device_id) {
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    printf("FATAL CUDA ERROR in gpu_set_device: %s\n", cudaGetErrorString(err));
  }
}
void gpu_device_synchronize() { cudaDeviceSynchronize(); }

void gpu_synchronize() { cudaDeviceSynchronize(); }

void gpu_load_constants(const void *host_ptr) {
  // Upload 1024 bytes (16 x 64) to symbol VARIANT_LUT
  cudaMemcpyToSymbol(VARIANT_LUT, host_ptr, 1024, 0, cudaMemcpyHostToDevice);
}

void update_constant_memory_hot_reload(const VariantParameters *new_variants,
                                       cudaStream_t stream) {
  // Async variant reload without pipeline stall,
  // enabling Zero-Downtime Hot-Reload between BSP phases.
  cudaMemcpyToSymbolAsync(VARIANT_LUT, new_variants,
                          sizeof(VariantParameters) * 16, 0,
                          cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

// Note: SoA_State is passed by value (repr(C) match with Rust side).
void launch_inject_inputs(SoA_State vram, const uint32_t *bitmask,
                          uint32_t current_tick, uint32_t total_virtual_axons) {
  uint32_t threads = 256;
  uint32_t blocks = (total_virtual_axons + threads - 1) / threads;
  vram.input_bitmask = (uint32_t *)bitmask;
  inject_inputs_kernel<<<blocks, threads, 0, 0>>>(
      vram, vram.input_bitmask, 0, current_tick, 1, total_virtual_axons);
}

void launch_apply_spike_batch(SoA_State vram, const uint32_t *tick_schedule,
                              uint32_t tick_spikes_count) {
  if (tick_spikes_count == 0)
    return;
  uint32_t threads = 256;
  uint32_t blocks = (tick_spikes_count + threads - 1) / threads;
  // tick_schedule points to flat u32 array.
  // SpikeEvent from Rust is (u32 id, u32 offset), but on GPU side
  // apply_spike_batch_kernel takes uint32_t* schedule_buffer.
  // Note: SpikeEvent is a pair (u32 id, u32 offset), so we stride
  // accordingly. See doc ref: 10_rpc.md — ApplySpikeBatch tick_schedule.
  apply_spike_batch_kernel<<<blocks, threads, 0, 0>>>(
      vram, tick_schedule, &tick_spikes_count, 0, tick_spikes_count);
}

void launch_propagate_axons(SoA_State vram, uint32_t v_seg) {
  uint32_t threads = 256;
  uint32_t blocks = (vram.total_axons + threads - 1) / threads;
  propagate_axons_kernel<<<blocks, threads, 0, 0>>>(vram, v_seg);
}

void launch_record_readout(SoA_State vram, const uint32_t *mapped_soma_ids,
                           uint8_t *output_history, uint32_t current_tick,
                           uint32_t total_pixels) {
  uint32_t threads = 256;
  uint32_t blocks = (total_pixels + threads - 1) / threads;
  vram.output_history = output_history;
  record_readout_kernel<<<blocks, threads, 0, 0>>>(vram, mapped_soma_ids,
                                                   total_pixels, current_tick);
}

// ... rest of functions stay same or similar ...

// =====================================================================
// Kernel 7: Extract Outgoing Ghost Axon Spikes
// =====================================================================

#pragma pack(push, 1)
struct SpikeEvent {
  uint32_t ghost_id;
  uint32_t tick_offset;
};
#pragma pack(pop)

// Extract fired spikes into Pinned RAM for network transmission
__global__ void extract_outgoing_spikes_kernel(
    const BurstHeads8* __restrict__ axon_heads,
    const uint32_t* __restrict__ src_indices,   
    const uint32_t* __restrict__ dst_ghost_ids, 
    uint32_t count, uint32_t sync_batch_ticks, uint32_t v_seg,
    SpikeEvent* __restrict__ out_events, 
    uint32_t* __restrict__ out_count     
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;

  uint32_t local_axon = src_indices[tid];

  // [DOD FIX] Hardware Memory Protection
  // 0x80000000 = AXON_SENTINEL (empty pixel)
  // 0xFFFFFFFF = u32::MAX (soma without axon)
  if (local_axon >= 0x80000000u) return;

  BurstHeads8 h = axon_heads[local_axon];
  uint32_t ghost_id = dst_ghost_ids[tid];

  // Reinterpret BurstHeads8 as flat uint32_t array (Zero-cost cast)
  const uint32_t* heads = (const uint32_t*)&h;

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
      uint32_t head = heads[i];
      
      // Skip empty sentinel (AXON_SENTINEL = 0x80000000)
      // [DOD FIX] Only skip true Sentinel. Fresh spikes (0xFFFFFFFF) must pass the barrier.
      if (head == 0x80000000u) continue;

      uint32_t ticks_since_spike = head / v_seg;

      // Check if spike falls within current sync batch window
      if (ticks_since_spike < sync_batch_ticks) {
          uint32_t out_idx = atomicAdd(out_count, 1);
          out_events[out_idx].ghost_id = ghost_id;
          out_events[out_idx].tick_offset = sync_batch_ticks - 1 - ticks_since_spike;
      }
  }
}

void launch_extract_outgoing_spikes(const BurstHeads8 *axon_heads,
                                    const uint32_t *src_indices,
                                    const uint32_t *dst_ghost_ids,
                                    uint32_t count, uint32_t sync_batch_ticks,
                                    uint32_t v_seg,
                                    void *out_events, uint32_t *out_count,
                                    cudaStream_t stream) {
  uint32_t threads = 256;
  uint32_t blocks = (count + threads - 1) / threads;

  // Reset output counter before extraction
  cudaMemsetAsync(out_count, 0, sizeof(uint32_t), stream);

  extract_outgoing_spikes_kernel<<<blocks, threads, 0, stream>>>(
      axon_heads, src_indices, dst_ghost_ids, count, sync_batch_ticks, v_seg,
      (SpikeEvent *)out_events, out_count);
}
} // end extern "C"

#define WARP_SIZE 32

struct DendriteSlot {
  uint32_t target;
  int32_t weight; // [DOD FIX] i32 weights
  uint8_t timer;
};

// =====================================================================
// Kernel 8: Synapse Sort and Prune (Night Phase)
// =====================================================================
__global__ void sort_and_prune_kernel(SoA_State state, uint32_t padded_n, int16_t global_prune_threshold) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t flag = state.flags[tid];
  // [DOD FIX] Reset burst_count bits [3:1], preserve Type [7:4] and Spike bit [0]
  state.flags[tid] = flag & 0xF1;
  
  uint8_t variant_id = (flag >> 4) & 0x0F;
  // [DOD FIX] Shift human-readable threshold into i32 Mass Domain
  int32_t prune_threshold = (int32_t)global_prune_threshold << 16;

  // [DOD FIX] 32 threads per warp; 12-byte DendriteSlot x 128 fits within 48 KB Shared Memory.
  __shared__ DendriteSlot smem[32][MAX_DENDRITE_SLOTS];
  uint32_t lane_id = threadIdx.x;

  // 1. Load dendrites (Coalesced Load)
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    uint32_t col_idx = slot * padded_n + tid;
    smem[lane_id][slot].target = state.dendrite_targets[col_idx];
    smem[lane_id][slot].weight = state.dendrite_weights[col_idx];
    smem[lane_id][slot].timer = state.dendrite_timers[col_idx];
  }
  __syncwarp();

  // 2. Sort by weight (Insertion Sort in L1/Shared Memory)
  for (int i = 1; i < MAX_DENDRITE_SLOTS; ++i) {
    DendriteSlot key = smem[lane_id][i];
    if (key.target == 0)
      continue;

    int32_t key_abs = key.weight >= 0 ? key.weight : -key.weight;
    int j = i - 1;

    while (j >= 0) {
      DendriteSlot prev = smem[lane_id][j];
      int32_t prev_abs = prev.weight >= 0 ? prev.weight : -prev.weight;
      if (prev.target != 0 && prev_abs >= key_abs) {
        break;
      }
      smem[lane_id][j + 1] = prev;
      j = j - 1;
    }
    smem[lane_id][j + 1] = key;
  }

  // 3. PRUNING (remove weak synapses below threshold)
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    int32_t w = smem[lane_id][slot].weight;
    int32_t abs_w = w >= 0 ? w : -w;
    if (smem[lane_id][slot].target != 0 && abs_w < prune_threshold) {
      smem[lane_id][slot].target = 0;
      smem[lane_id][slot].weight = 0;
      smem[lane_id][slot].timer = 0;
    }
  }

  // 3.5 COMPACTION (Dense Rule enforcement)
  int compact_idx = 0;
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    if (smem[lane_id][slot].target != 0) {
      smem[lane_id][compact_idx] = smem[lane_id][slot];
      compact_idx++;
    }
  }
  for (int slot = compact_idx; slot < MAX_DENDRITE_SLOTS; ++slot) {
    smem[lane_id][slot].target = 0;
    smem[lane_id][slot].weight = 0;
    smem[lane_id][slot].timer = 0;
  }
  __syncwarp();

  // 4. Write back (Coalesced Store)
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    uint32_t col_idx = slot * padded_n + tid;
    state.dendrite_targets[col_idx] = smem[lane_id][slot].target;
    state.dendrite_weights[col_idx] = smem[lane_id][slot].weight;
    state.dendrite_timers[col_idx] = smem[lane_id][slot].timer;
  }
}

extern "C" {
// launch_sort_and_prune is defined below (requires ShardVramPtrs)
} // last existing extern "C"

// =====================================================================
//  New Memory Contract: ShardVramPtrs + cu_* functions
//
// WARNING: ShardVramPtrs field order must exactly match the Rust-side
// repr(C) struct and the .state binary layout. Mismatch = Silent Data Corruption.
//
// Flat memory layout (padded_n must be 32-aligned for Warp-Aligned access):
//   soma_voltage      [padded_n]        4 B   base ptr (= soma_voltage)
//   soma_flags        [padded_n]        1 B
//   threshold_offset  [padded_n]        4 B
//   timers            [padded_n]        1 B
//   soma_to_axon      [padded_n]        4 B
//   dendrite_targets  [padded_n x 128]  4 B
//   dendrite_weights  [padded_n x 128]  2 B
//   --- Separate cudaMalloc, separate cudaMemcpyAsync ---
//   axon_heads        [total_axons]     4 B   separate cudaMalloc
// =====================================================================

// Mirrors Rust #[repr(C)] struct ShardVramPtrs
struct ShardVramPtrs {
  int32_t* __restrict__ soma_voltage; // base ptr of state blob
  uint8_t* __restrict__ soma_flags;
  int32_t* __restrict__ threshold_offset;
  uint8_t* __restrict__ timers;
  uint32_t* __restrict__ soma_to_axon;
  uint32_t* __restrict__ dendrite_targets;
  int32_t* __restrict__ dendrite_weights;
  uint8_t* __restrict__ dendrite_timers;
  BurstHeads8* __restrict__ axon_heads; // Separate allocation
};

#define MAX_DENDRITES_SV 128

extern "C" {

// --- Allocation
// ----------------------------------------------------------------
// cudaMalloc: State (flat blob) + Axons (separate). Single monolithic allocation.
__global__ void init_sentinels_kernel(BurstHeads8* heads, uint32_t total) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        heads[i].h0 = 0x80000000; heads[i].h1 = 0x80000000;
        heads[i].h2 = 0x80000000; heads[i].h3 = 0x80000000;
        heads[i].h4 = 0x80000000; heads[i].h5 = 0x80000000;
        heads[i].h6 = 0x80000000; heads[i].h7 = 0x80000000;
    }
}

// Allocate VRAM for a single shard
int32_t cu_allocate_shard(uint32_t padded_n, uint32_t total_axons,
                          ShardVramPtrs *out_vram) {
  size_t sz_voltage = (size_t)padded_n * sizeof(int32_t);
  size_t sz_flags = (size_t)padded_n * sizeof(uint8_t);
  size_t sz_thresh = (size_t)padded_n * sizeof(int32_t);
  size_t sz_timers = (size_t)padded_n * sizeof(uint8_t);
  size_t sz_s2a = (size_t)padded_n * sizeof(uint32_t);
  size_t sz_targets = (size_t)padded_n * MAX_DENDRITES_SV * sizeof(uint32_t);
  size_t sz_weights = (size_t)padded_n * MAX_DENDRITES_SV * sizeof(int32_t);
  size_t sz_dtimers = (size_t)padded_n * MAX_DENDRITES_SV * sizeof(uint8_t);

  // [DOD FIX] Add 64 * 8 bytes of padding to guarantee 64B alignment for all 8 arrays
  size_t total_state = sz_voltage + sz_flags + sz_thresh + sz_timers + sz_s2a +
                       sz_targets + sz_weights + sz_dtimers + (64 * 8);

  // Flat Allocation: single cudaMalloc for all SoA arrays + alignment padding
  void *base = nullptr;
  cudaError_t err = cudaMalloc(&base, total_state);
  if (err != cudaSuccess) {
    fprintf(stderr, "[cu_allocate_shard] cudaMalloc state failed: %s\n",
            cudaGetErrorString(err));
    return (int32_t)err;
  }

  // Zero-fill entire state blob for deterministic initialization
  cudaMemset(base, 0, total_state);

  // Zero-Cost Partitioning:
  size_t off = 0;
  out_vram->soma_voltage = (int32_t *)((char *)base + off);
  off = (off + sz_voltage + 63) & ~63;
  out_vram->soma_flags = (uint8_t *)((char *)base + off);
  off = (off + sz_flags + 63) & ~63;
  out_vram->threshold_offset = (int32_t *)((char *)base + off);
  off = (off + sz_thresh + 63) & ~63;
  out_vram->timers = (uint8_t *)((char *)base + off);
  off = (off + sz_timers + 63) & ~63;
  out_vram->soma_to_axon = (uint32_t *)((char *)base + off);
  off = (off + sz_s2a + 63) & ~63;
  out_vram->dendrite_targets = (uint32_t *)((char *)base + off);
  off = (off + sz_targets + 63) & ~63;
  out_vram->dendrite_weights = (int32_t *)((char *)base + off);
  off = (off + sz_weights + 63) & ~63;
  out_vram->dendrite_timers = (uint8_t *)((char *)base + off);

  // Axon heads are allocated separately (total_axons != padded_n)
  err = cudaMalloc((void **)&out_vram->axon_heads,
                   (size_t)total_axons * sizeof(BurstHeads8));
  if (err != cudaSuccess) {
    fprintf(stderr, "[cu_allocate_shard] cudaMalloc axon_heads failed: %s\n",
            cudaGetErrorString(err));
    cudaFree(base);
    return (int32_t)err;
  }

  // [DOD FIX] Strict 32-bit Sentinel Initialization
  uint32_t threads = 256;
  uint32_t blocks = (total_axons + threads - 1) / threads;
  init_sentinels_kernel<<<blocks, threads>>>(out_vram->axon_heads, total_axons);

  return 0;
}

// --- DMA Upload: State
// ----------------------------------------------------------------
// .state file contains 7 arrays serialized contiguously, matching ShardVramPtrs layout.
// Since it's a Flat Allocation, base_ptr == soma_voltage. Single cudaMemcpyAsync
// uploads all 7 arrays at 100% PCIe bandwidth utilization.
int32_t cu_upload_state_blob(const ShardVramPtrs *vram, const void *state_blob,
                             size_t state_size) {
  cudaError_t err =
      cudaMemcpyAsync((void *)vram->soma_voltage, // base ptr of flat blob
                      state_blob, state_size, cudaMemcpyHostToDevice,
                      0 // default stream
      );
  if (err != cudaSuccess) {
    fprintf(stderr, "[cu_upload_state_blob] cudaMemcpyAsync failed: %s\n",
            cudaGetErrorString(err));
    return (int32_t)err;
  }
  // Block CPU until VRAM is ready. Init-phase latency is acceptable.
  cudaStreamSynchronize(0);
  return 0;
}

// --- DMA Upload: Axons
// ----------------------------------------------------------------
// .axons file is uploaded directly into BurstHeads8 axon_heads array.
int32_t cu_upload_axons_blob(const ShardVramPtrs *vram, const void *axons_blob,
                             size_t axons_size) {
  cudaError_t err = cudaMemcpyAsync((void *)vram->axon_heads, axons_blob,
                                    axons_size, cudaMemcpyHostToDevice, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "[cu_upload_axons_blob] cudaMemcpyAsync failed: %s\n",
            cudaGetErrorString(err));
    return (int32_t)err;
  }
  cudaStreamSynchronize(0);
  return 0;
}

// --- Free
// ----------------------------------------------------------------
// soma_voltage == base ptr of the state blob.
// Two cudaFree calls match the two cudaMalloc calls in cu_allocate_shard.
void cu_free_shard(ShardVramPtrs *vram) {
  if (vram->soma_voltage) {
    cudaFree((void *)vram->soma_voltage);
    vram->soma_voltage = nullptr;
    vram->soma_flags = nullptr;
    vram->threshold_offset = nullptr;
    vram->timers = nullptr;
    vram->soma_to_axon = nullptr;
    vram->dendrite_targets = nullptr;
    vram->dendrite_weights = nullptr;
    vram->dendrite_timers = nullptr;
  }
  if (vram->axon_heads) {
    cudaFree((void *)vram->axon_heads);
    vram->axon_heads = nullptr;
  }
}
}

// =====================================================================
// Night Phase: Sort & Prune (requires ShardVramPtrs, defined above)
// Block size = warp size (32 threads). Each thread processes its own Shared Memory lane.
// Launched on default stream to avoid concurrent kernel conflicts.
// =====================================================================
extern "C" void launch_sort_and_prune(const ShardVramPtrs *ptrs, uint32_t padded_n, int16_t prune_threshold) {
  // [DOD FIX] 32 threads per block to match warp size and 48KB Shared Memory budget
  dim3 threads(32, 1);
  dim3 blocks((padded_n + 32 - 1) / 32, 1);

  SoA_State state;
  state.dendrite_targets = ptrs->dendrite_targets;
  state.dendrite_weights = ptrs->dendrite_weights;
  state.dendrite_timers = ptrs->dendrite_timers;
  state.flags = ptrs->soma_flags; // Needed to reset burst_count bits

  sort_and_prune_kernel<<<blocks, threads>>>(state, padded_n, prune_threshold);
}

// ============================================================================
// I/O VRAM Allocations & DMA Transfers
// ============================================================================
extern "C" {

int32_t cu_allocate_io_buffers(
    uint32_t input_words,       // Total input bitmask size in u32 words
    uint32_t schedule_capacity, // sync_batch_ticks * MAX_SPIKES_PER_TICK
    uint32_t output_capacity,   // sync_batch_ticks * num_outputs
    uint32_t **d_input_bitmask, uint32_t **d_incoming_spikes,
    uint8_t **d_output_history) {
  if (input_words > 0) {
    if (cudaMalloc((void **)d_input_bitmask, input_words * sizeof(uint32_t)) !=
        cudaSuccess)
      return -1;
  } else {
    *d_input_bitmask = nullptr;
  }

  if (schedule_capacity > 0) {
    if (cudaMalloc((void **)d_incoming_spikes,
                   schedule_capacity * sizeof(uint32_t)) != cudaSuccess)
      return -1;
  } else {
    *d_incoming_spikes = nullptr;
  }

  if (output_capacity > 0) {
    if (cudaMalloc((void **)d_output_history,
                   output_capacity * sizeof(uint8_t)) != cudaSuccess)
      return -1;
  } else {
    *d_output_history = nullptr;
  }
  return 0;
}

int32_t cu_free_io_buffers(uint32_t *d_input_bitmask,
                           uint32_t *d_incoming_spikes,
                           uint8_t *d_output_history) {
  if (d_input_bitmask)
    cudaFree(d_input_bitmask);
  if (d_incoming_spikes)
    cudaFree(d_incoming_spikes);
  if (d_output_history)
    cudaFree(d_output_history);
  return 0;
}

int32_t cu_dma_h2d_io(uint32_t *d_input_bitmask,
                      const uint32_t *h_input_bitmask, uint32_t input_words,
                      uint32_t *d_incoming_spikes,
                      const uint32_t *h_incoming_spikes,
                      uint32_t schedule_capacity,
                      cudaStream_t stream) {
  // Async H2D transfers on the given Stream
  if (input_words > 0 && d_input_bitmask && h_input_bitmask) {
    cudaMemcpyAsync(d_input_bitmask, h_input_bitmask,
                    input_words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
  }
  if (schedule_capacity > 0 && d_incoming_spikes && h_incoming_spikes) {
    cudaMemcpyAsync(d_incoming_spikes, h_incoming_spikes,
                    schedule_capacity * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);
  }
  return 0;
}

int32_t cu_dma_d2h_io(uint8_t *h_output_history,
                      const uint8_t *d_output_history,
                      uint32_t output_capacity,
                      cudaStream_t stream) {
  if (output_capacity > 0 && d_output_history && h_output_history) {
    cudaMemcpyAsync(h_output_history, d_output_history,
                    output_capacity * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                    stream);
  }
  return 0;
}

void gpu_reset_telemetry_count(uint32_t* count_d, cudaStream_t stream) {
    cudaMemsetAsync(count_d, 0, sizeof(uint32_t), stream);
}

} // Final closing brace for extern "C"
