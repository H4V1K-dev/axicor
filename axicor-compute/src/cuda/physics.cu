#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

//     bindings.cu
struct alignas(32) BurstHeads8 {
  uint32_t h0; uint32_t h1; uint32_t h2; uint32_t h3;
  uint32_t h4; uint32_t h5; uint32_t h6; uint32_t h7;
};

struct ShardVramPtrs {
  int32_t* __restrict__ soma_voltage; // base ptr  state-
  uint8_t* __restrict__ soma_flags;
  int32_t* __restrict__ threshold_offset;
  uint8_t* __restrict__ timers;
  uint32_t* __restrict__ soma_to_axon;
  uint32_t* __restrict__ dendrite_targets;
  int32_t* __restrict__ dendrite_weights; // [DOD FIX]  int16_t*,    !
  uint8_t* __restrict__ dendrite_timers;
  BurstHeads8* __restrict__ axon_heads; //  
};

#define AXON_SENTINEL 0x80000000

__global__ void cu_reset_burst_counters_kernel(ShardVramPtrs vram, uint32_t padded_n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= padded_n) return;
    //  Type ID (0xF0)    (0x01),   [3:1]
    vram.soma_flags[tid] &= 0xF1;
}

__device__ __forceinline__ void push_burst_head(BurstHeads8* h, uint32_t v_seg) {
  h->h7 = h->h6;
  h->h6 = h->h5;
  h->h5 = h->h4;
  h->h4 = h->h3;
  h->h3 = h->h2;
  h->h2 = h->h1;
  h->h1 = h->h0;
  // [DOD FIX] Wrap-around u32.   Propagate    0.
  h->h0 = (uint32_t)(0 - v_seg); 
}

#define MAX_DENDRITES 128

//  64  (1 - L1). 16  = 1024   Constant Memory.
struct alignas(64) VariantParameters {
  // ===  1: 32-bit ( 0..20) ===
  int32_t threshold;
  int32_t rest_potential;
  uint32_t leak_shift;
  int32_t homeostasis_penalty;
  uint32_t spontaneous_firing_period_ticks;

  // ===  2: 16-bit ( 20..28) ===
  uint16_t initial_synapse_weight;
  uint16_t gsop_potentiation;
  uint16_t gsop_depression;
  uint16_t homeostasis_decay;

  // ===  3: 8-bit ( 28..32) ===
  uint8_t refractory_period;
  uint8_t synapse_refractory_period;
  uint8_t signal_propagation_length;
  uint8_t is_inhibitory; // 1 = true (GABA), 0 = false (Glu)

  // ===  4: Inertia Curve & AHP ( 32..48) ===
  uint8_t inertia_curve[8];                 // 32..40
  uint16_t ahp_amplitude;                   // 40..42
  uint8_t _pad[6];                          // 42..48

  // ===  5: Adaptive Leak Hardware ( 48..58) ===
  int32_t adaptive_leak_min_shift;          // 48..52
  uint16_t adaptive_leak_gain;              // 52..54
  uint8_t adaptive_mode;                    // 54..55
  uint8_t _leak_pad[3];                     // 55..58

  // ===  6: Dopamine Receptors & Padding ( 58..64) ===
  uint8_t d1_affinity;                       // 58..59
  uint8_t d2_affinity;                       // 59..60
  uint32_t heartbeat_m;                      // 60..64
};

//   . Rust      .
__constant__ VariantParameters VARIANT_LUT[16];

// ============================================================================
// 1. Inject Inputs Kernel (Virtual Axons)
//          
// ============================================================================
__global__ void cu_inject_inputs_kernel(BurstHeads8* __restrict__ axon_heads,
                                        const uint32_t* __restrict__ input_bitmask,
                                        uint32_t virtual_offset,
                                        uint32_t num_virtual_axons,
                                        uint32_t v_seg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_virtual_axons)
    return;

  //    2  ALU (  32   
  // shift)
  uint32_t word_idx = tid / 32;
  uint32_t bit_idx = tid % 32;
  bool is_active = (input_bitmask[word_idx] >> bit_idx) & 1;

  //  :     
  if (is_active) {
    BurstHeads8 h = axon_heads[virtual_offset + tid];
    push_burst_head(&h, v_seg);
    axon_heads[virtual_offset + tid] = h;
  }
}

// ============================================================================
// 2. Apply Spike Batch Kernel (Network / Ghost Axons)
// O(1)     Sender-Side Mapping
// ============================================================================
__global__ void cu_apply_spike_batch_kernel(BurstHeads8* __restrict__ axon_heads,
                                            const uint32_t* __restrict__ incoming_spikes,
                                            uint32_t num_incoming_spikes,
                                            uint32_t total_axons,
                                            uint32_t v_seg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_incoming_spikes)
    return;

  uint32_t ghost_id = incoming_spikes[tid];

  if (ghost_id >= total_axons)
    return;

  // [DOD FIX]    ghost_id,    tid!    .
  BurstHeads8 h = axon_heads[ghost_id];
  push_burst_head(&h, v_seg);
  axon_heads[ghost_id] = h;
}

// ============================================================================
// 3. Propagate Axons Kernel
// ============================================================================
__global__ void cu_propagate_axons_kernel(BurstHeads8* __restrict__ axon_heads,
                                          uint32_t total_axons,
                                          uint32_t v_seg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_axons)
    return;

  BurstHeads8 h = axon_heads[tid];
  if (h.h0 != AXON_SENTINEL) h.h0 += v_seg;
  if (h.h1 != AXON_SENTINEL) h.h1 += v_seg;
  if (h.h2 != AXON_SENTINEL) h.h2 += v_seg;
  if (h.h3 != AXON_SENTINEL) h.h3 += v_seg;
  if (h.h4 != AXON_SENTINEL) h.h4 += v_seg;
  if (h.h5 != AXON_SENTINEL) h.h5 += v_seg;
  if (h.h6 != AXON_SENTINEL) h.h6 += v_seg;
  if (h.h7 != AXON_SENTINEL) h.h7 += v_seg;
  axon_heads[tid] = h;
}

// ============================================================================
// 4. Update Neurons Kernel (GLIF + Dendritic Integration)
// ============================================================================
__global__ void cu_update_neurons_kernel(ShardVramPtrs vram,
                                         uint32_t padded_n, uint32_t total_axons, uint32_t current_tick, uint32_t v_seg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t flags = vram.soma_flags[tid];
  uint8_t timer = vram.timers[tid];

  flags &= ~0x01;

  uint8_t variant_id = (flags >> 4) & 0x0F;
  VariantParameters p = VARIANT_LUT[variant_id];

  // [DOD FIX] Branchless Homeostasis Decay (Zero Warp Divergence)
  int32_t thresh_offset = vram.threshold_offset[tid];
  int32_t decayed = thresh_offset - p.homeostasis_decay;
  thresh_offset = decayed & ~(decayed >> 31);

  if (timer > 0) {
    vram.timers[tid] = timer - 1;
    vram.soma_flags[tid] = flags;
    vram.threshold_offset[tid] = thresh_offset;
    return;
  }

  int32_t current_voltage = vram.soma_voltage[tid];
  int32_t i_in = 0;

  for (int i = 0; i < MAX_DENDRITES; i++) {
    uint32_t col_idx = i * padded_n + tid;
    uint32_t target_packed = vram.dendrite_targets[col_idx];
    if (target_packed == 0) break; // Hardware Early Exit

    // [DOD FIX] Synaptic Refractory Gate (Возвращаем вырезанную физику)
    uint8_t d_timer = vram.dendrite_timers[col_idx];
    if (d_timer > 0) {
        vram.dendrite_timers[col_idx] = d_timer - 1;
        continue; // ~90% тиков дендрит спит, пропускаем тяжелую математику
    }

    uint32_t raw_id = target_packed & 0x00FFFFFF;
    if (raw_id == 0) continue; // [DOD FIX] Игнорируем мусор, но не слепнем (continue вместо break)

    uint32_t target_id = raw_id - 1;
    if (target_id >= total_axons) continue; // Hardware VRAM Guard

    uint32_t seg_idx = target_packed >> 24;
    BurstHeads8 h = vram.axon_heads[target_id];
    uint32_t prop = p.signal_propagation_length;

    // Branchless 8-head bitwise OR
    bool hit = ((h.h0 - seg_idx) < prop) | ((h.h1 - seg_idx) < prop) |
               ((h.h2 - seg_idx) < prop) | ((h.h3 - seg_idx) < prop) |
               ((h.h4 - seg_idx) < prop) | ((h.h5 - seg_idx) < prop) |
               ((h.h6 - seg_idx) < prop) | ((h.h7 - seg_idx) < prop);

    if (hit) {
      int32_t charge = (int32_t)vram.dendrite_weights[col_idx] >> 16;
      i_in += charge;
      // [DOD FIX] Block synapse from multi-reading the same tail
      vram.dendrite_timers[col_idx] = p.synapse_refractory_period;
    }
  }

  // Adaptive GLIF Leak (Branchless)
  int32_t adaptive_sub = ((thresh_offset * p.adaptive_leak_gain) >> 8) * p.adaptive_mode;
  int32_t new_shift = (int32_t)p.leak_shift - adaptive_sub;
  new_shift = max(new_shift, p.adaptive_leak_min_shift);
  uint32_t current_shift = max(new_shift, 0);

  current_voltage += i_in; 

  // Shift-based Exponential Leak (Branchless)
  int32_t diff = current_voltage - p.rest_potential;
  current_voltage -= (diff >> current_shift);

  int32_t effective_threshold = p.threshold + thresh_offset;
  int32_t is_glif_spiking = (current_voltage >= effective_threshold) ? 1 : 0;

  // Spontaneous Firing (Heartbeat) - Branchless period check
  uint32_t phase = ((uint64_t)current_tick * p.heartbeat_m + (uint64_t)tid * 104729) & 0xFFFF;
  int32_t is_heartbeat = (p.heartbeat_m > 0 && phase < p.heartbeat_m) ? 1 : 0;

  //   (  Heartbeat)
  int32_t final_spike = is_glif_spiking | is_heartbeat;

  // AHP: Membrane reset with undershoot
  int32_t reset_v = p.rest_potential - (int32_t)p.ahp_amplitude;
  current_voltage = is_glif_spiking * reset_v + (1 - is_glif_spiking) * current_voltage;
  thresh_offset += is_glif_spiking * p.homeostasis_penalty;
  uint8_t new_timer = is_glif_spiking * p.refractory_period + (1 - is_glif_spiking) * vram.timers[tid];
  // 7. Axon heads shift on spike (Burst Shift)
  if (final_spike) {
    uint32_t my_axon = vram.soma_to_axon[tid];
    // [DOD FIX] Strict VRAM Guard before axon_heads access
    if (my_axon != 0xFFFFFFFF && my_axon < total_axons) {
      BurstHeads8 h = vram.axon_heads[my_axon];
      h.h7 = h.h6; h.h6 = h.h5; h.h5 = h.h4; h.h4 = h.h3;
      h.h3 = h.h2; h.h2 = h.h1; h.h1 = h.h0; h.h0 = (uint32_t)(0 - v_seg);
      vram.axon_heads[my_axon] = h; // 32-byte coalesced write
    }
  }

  // 8.   VRAM (Zero-Warp Divergence)
  vram.soma_voltage[tid] = current_voltage;

  // [DOD FIX] BDP Decay.   0,  final_spike == 0. ,  final_spike == 1.
  uint8_t burst_count = (flags >> 1) & 0x07;
  burst_count = final_spike * (burst_count + (burst_count < 7 ? 1 : 0));
  
  //  ,  Type ID ( 4 )
  vram.soma_flags[tid] = (flags & 0xF0) | (burst_count << 1) | (uint8_t)final_spike;
  vram.threshold_offset[tid] = thresh_offset;
  vram.timers[tid] = new_timer;
}

// ============================================================================
// 5. Apply GSOP Kernel (Spatial STDP Plasticity)
// ============================================================================
__global__ void cu_apply_gsop_kernel(ShardVramPtrs vram, uint32_t padded_n, uint32_t total_axons, int16_t dopamine) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t flags = vram.soma_flags[tid];
  if ((flags & 0x01) == 0) return;
  
  //      ( 1,     if)
  uint8_t burst_count = (flags >> 1) & 0x07;
  int32_t burst_mult = (burst_count > 0) ? burst_count : 1;

  uint8_t variant_id = (flags >> 4) & 0x0F;
  VariantParameters p = VARIANT_LUT[variant_id];

  for (int i = 0; i < MAX_DENDRITES; i++) {
    uint32_t col_idx = i * padded_n + tid;
    uint32_t target_packed = vram.dendrite_targets[col_idx];
    if (target_packed == 0) break;

    uint32_t raw_id = target_packed & 0x00FFFFFF;
    if (raw_id == 0) break; // [DOD FIX] Zero-Index Trap Protection!

    uint32_t target_id = raw_id - 1;
    if (target_id >= total_axons) continue; // [DOD FIX] Hardware VRAM Guard!
    uint32_t seg_idx = target_packed >> 24;
    BurstHeads8 b = vram.axon_heads[target_id];
    uint32_t len = p.signal_propagation_length;

    //    ()    
    uint32_t min_dist = 0xFFFFFFFF;
    uint32_t d;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        uint32_t head = ((uint32_t*)&b)[k];
        d = head - seg_idx;
        uint32_t valid_d = d | ((uint32_t)-(int32_t)(d >= len));
        min_dist = min(min_dist, valid_d);
    }

    bool is_active = (min_dist != 0xFFFFFFFF);

    int32_t w = vram.dendrite_weights[col_idx];
    int32_t sign = (w >= 0) ? 1 : -1;
    int32_t abs_w = (w >= 0) ? w : -w;

    // 1. Inertia Rank (1 , Branchless)
    uint32_t rank = abs_w >> 28;
    if (rank > 7)
      rank = 7;
    int32_t inertia = p.inertia_curve[rank];

    // Dopamine modulation (D1 boosts LTP, D2 suppresses LTD on reward)
    // Integer physics: (int16 * uint8) >> 7
    int32_t pot_mod = ((int32_t)dopamine * (int32_t)p.d1_affinity) >> 7;
    int32_t dep_mod = ((int32_t)dopamine * (int32_t)p.d2_affinity) >> 7;

    // D1  LTP  . D2  LTD   ( ).
    int32_t raw_pot = (int32_t)p.gsop_potentiation + pot_mod;
    int32_t raw_dep = (int32_t)p.gsop_depression - dep_mod;

    // Branchless clamp to 0 (Anti-Negative Guard)
    int32_t final_pot = raw_pot & ~(raw_pot >> 31);
    int32_t final_dep = raw_dep & ~(raw_dep >> 31);

    //        
    int32_t delta_pot = (final_pot * inertia * burst_mult) >> 7;
    int32_t delta_dep = (final_dep * inertia * burst_mult) >> 7;
    //  .  16      (>> 1)
    uint32_t cooling_shift = is_active ? (min_dist >> 4) : 0;

    // 3. Causal Delta    STDP
    int32_t delta = is_active ? (delta_pot >> cooling_shift) : -delta_dep;

    // 4. Slot Decay neutralized (fixed to 1.0x)
    int32_t decay = 128;
    delta = (delta * decay) >> 7; // [DOD FIX] Single Spatial Cooling

    // 5. Apply & Clamp
    int32_t new_abs = abs_w + delta;

    // [DOD FIX] Branchless clamp(0, val).  new_abs < 0,   0xFFFFFFFF,   0.
    //      ( Dale's Law Safety ).
    new_abs = new_abs & ~(new_abs >> 31);

    if (new_abs > 2140000000) {
      new_abs = 2140000000;
    }

    vram.dendrite_weights[col_idx] = (int32_t)(new_abs * sign);
  }
}

// ============================================================================
// 6. Record Readout Kernel (Output Matrix)
// ============================================================================
__global__ void cu_record_readout_kernel(const uint8_t* __restrict__ soma_flags,
                                         const uint32_t* __restrict__ mapped_soma_ids,
                                         uint8_t* __restrict__ output_history,
                                         uint32_t num_outputs,
                                         uint32_t padded_n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_outputs)
    return;

  uint32_t soma_id = mapped_soma_ids[tid];
  uint8_t is_spiking = 0;

  // [DOD]   Memory Out-of-Bounds.    .
  if (soma_id != 0xFFFFFFFF && soma_id < padded_n) { // [DOD FIX] Strict VRAM Guard
    is_spiking = soma_flags[soma_id] & 0x01;
  }

  output_history[tid] = is_spiking;
}

// ============================================================================
// 7. Warp-Aggregated Telemetry Extraction
// ============================================================================
__global__ void cu_extract_telemetry_kernel(
    const uint8_t* __restrict__ soma_flags,
    uint32_t* __restrict__ out_ids,
    uint32_t* __restrict__ out_count,
    uint32_t padded_n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x % 32;

    // 1.    ( 0)
    bool is_spiking = false;
    if (tid < padded_n) {
        is_spiking = (soma_flags[tid] & 0x01) != 0;
    }

    // 2. Ballot:       32-  
    uint32_t active_mask = __ballot_sync(0xFFFFFFFF, is_spiking);
    uint32_t warp_pop = __popc(active_mask);

    // 3. Leader (lane 0)   atomicAdd   
    uint32_t warp_offset = 0;
    if (lane == 0 && warp_pop > 0) {
        warp_offset = atomicAdd(out_count, warp_pop);
    }

    // 4. Leader   offset   
    warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);

    // 5.  ID     
    if (is_spiking) {
        //       (    )
        uint32_t local_rank = __popc(active_mask & ((1u << lane) - 1));
        out_ids[warp_offset + local_rank] = tid;
    }
}

// ============================================================================
// Intra-GPU Ghost Sync Kernel (Zero-Copy L2 Cache Routing)
// ============================================================================
__global__ void cu_ghost_sync_kernel(
    const BurstHeads8* __restrict__ src_heads,
    BurstHeads8* __restrict__ dst_heads,
    const uint32_t* __restrict__ src_indices,
    const uint32_t* __restrict__ dst_indices,
    uint32_t count,
    uint32_t dst_total_axons, // [DOD FIX] Hardware VRAM Guard
    uint32_t sync_batch_ticks,
    uint32_t v_seg
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint32_t src_axon = src_indices[tid];
    uint32_t dst_ghost = dst_indices[tid];

    // [DOD FIX] Hardware protection: 0x80000000 (Sentinel) and 0xFFFFFFFF (No axon)
    if (src_axon == 0x80000000 || src_axon == 0xFFFFFFFF) return;

    // [DOD FIX] Strict VRAM Guard. Never trust host code.
    if (dst_ghost >= dst_total_axons) return;

    BurstHeads8 s_h = src_heads[src_axon];
    BurstHeads8 d_h;
    
    // [DOD FIX] Temporal Shift (Branchless Vectorized Copy)
    uint32_t batch_shift = sync_batch_ticks * v_seg;

    d_h.h0 = (s_h.h0 == 0x80000000u) ? 0x80000000u : s_h.h0 - batch_shift;
    d_h.h1 = (s_h.h1 == 0x80000000u) ? 0x80000000u : s_h.h1 - batch_shift;
    d_h.h2 = (s_h.h2 == 0x80000000u) ? 0x80000000u : s_h.h2 - batch_shift;
    d_h.h3 = (s_h.h3 == 0x80000000u) ? 0x80000000u : s_h.h3 - batch_shift;
    d_h.h4 = (s_h.h4 == 0x80000000u) ? 0x80000000u : s_h.h4 - batch_shift;
    d_h.h5 = (s_h.h5 == 0x80000000u) ? 0x80000000u : s_h.h5 - batch_shift;
    d_h.h6 = (s_h.h6 == 0x80000000u) ? 0x80000000u : s_h.h6 - batch_shift;
    d_h.h7 = (s_h.h7 == 0x80000000u) ? 0x80000000u : s_h.h7 - batch_shift;

    dst_heads[dst_ghost] = d_h;
}

// ============================================================================
// DEBUG HARNESS (Epic 2) - Isolated Micro-Kernels for Electrophysiology
// ============================================================================
__global__ void cu_debug_inject_current_kernel(
    int32_t* __restrict__ soma_voltage,
    const uint32_t* __restrict__ target_tids,
    const int32_t* __restrict__ injection_uV,
    uint32_t count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint32_t target_soma = target_tids[tid];
    // [DOD] Atomic add protects against overlapping injection targets
    atomicAdd(&soma_voltage[target_soma], injection_uV[tid]);
}

__global__ void cu_debug_record_v_kernel(
    const int32_t* __restrict__ soma_voltage,
    const uint32_t* __restrict__ target_tids,
    int32_t* __restrict__ out_trace,
    uint32_t current_tick,
    uint32_t count,
    uint32_t max_ticks
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    if (current_tick >= max_ticks) return;

    uint32_t target_soma = target_tids[tid];
    // [DOD] Flat 2D projection: [target_index][tick_index]
    out_trace[tid * max_ticks + current_tick] = soma_voltage[target_soma];
}

extern "C" {
void launch_ghost_sync(
    const BurstHeads8* src_heads,
    BurstHeads8* dst_heads,
    const uint32_t* src_indices,
    const uint32_t* dst_indices,
    uint32_t count,
    uint32_t dst_total_axons, // [DOD FIX] C-ABI Sync
    uint32_t sync_batch_ticks,
    uint32_t v_seg,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    cu_ghost_sync_kernel<<<blocks, threads, 0, stream>>>(
        src_heads, dst_heads, src_indices, dst_indices, count, dst_total_axons, sync_batch_ticks, v_seg
    );
}

// ============================================================================
// Day Phase Orchestrator
// ============================================================================
int32_t cu_step_day_phase(const ShardVramPtrs *vram, uint32_t padded_n,
                          uint32_t total_axons, uint32_t v_seg, uint32_t current_tick,
                          // ---  (InjectInputs) ---
                          const uint32_t *input_bitmask,
                          uint32_t virtual_offset, uint32_t num_virtual_axons,
                          // ---  (ApplySpikeBatch) ---
                          const uint32_t *incoming_spikes,
                          uint32_t num_incoming_spikes,
                          // ---  (RecordReadout) ---
                          const uint32_t *mapped_soma_ids,
                          uint8_t *output_history, uint32_t num_outputs,
                          int16_t dopamine,
                          cudaStream_t stream) {
  int threads = 256;
  int blocks_n = (padded_n + threads - 1) / threads;

  // [DOD FIX]  burst_count ( [3:1])    
  cu_reset_burst_counters_kernel<<<blocks_n, threads, 0, stream>>>(*vram, padded_n);

  // 1. InjectInputs (       )
  if (num_virtual_axons > 0 && input_bitmask != nullptr) {
    int blocks_in = (num_virtual_axons + threads - 1) / threads;
    cu_inject_inputs_kernel<<<blocks_in, threads, 0, stream>>>(
        vram->axon_heads, input_bitmask, virtual_offset, num_virtual_axons, v_seg);
  }

  // 2. ApplySpikeBatch (    )
  if (num_incoming_spikes > 0 && incoming_spikes != nullptr) {
    int blocks_spikes = (num_incoming_spikes + threads - 1) / threads;
    cu_apply_spike_batch_kernel<<<blocks_spikes, threads, 0, stream>>>(
        vram->axon_heads, incoming_spikes, num_incoming_spikes, total_axons, v_seg);
  }

  // 3. PropagateAxons (   )
  int blocks_prop = (total_axons + threads - 1) / threads;
  cu_propagate_axons_kernel<<<blocks_prop, threads, 0, stream>>>(
      vram->axon_heads, total_axons, v_seg);

  // 4. UpdateNeurons (GLIF)
  int blocks_update = (padded_n + threads - 1) / threads;
  cu_update_neurons_kernel<<<blocks_update, threads, 0, stream>>>(
      *vram, padded_n, total_axons, current_tick, v_seg);

  // 5. ApplyGSOP ( 3D STDP)
  cu_apply_gsop_kernel<<<blocks_update, threads, 0, stream>>>(*vram, padded_n, total_axons, dopamine);

  // 6. RecordReadout
  if (num_outputs > 0 && mapped_soma_ids != nullptr &&
      output_history != nullptr) {
    int blocks_out = (num_outputs + threads - 1) / threads;
    cu_record_readout_kernel<<<blocks_out, threads, 0, stream>>>(
        vram->soma_flags, mapped_soma_ids, output_history, num_outputs, padded_n);
  }

  return 0;
}

extern "C" void cu_reset_burst_counters(const ShardVramPtrs *vram, uint32_t padded_n, cudaStream_t stream) {
  int threads = 256;
  int blocks = (padded_n + threads - 1) / threads;
  cu_reset_burst_counters_kernel<<<blocks, threads, 0, stream>>>(*vram, padded_n);
}

//        GPU
int32_t cu_upload_constant_memory(const VariantParameters *lut) {
  return cudaMemcpyToSymbol(VARIANT_LUT, lut, sizeof(VariantParameters) * 16);
}

void launch_debug_inject_current(
    int32_t* soma_voltage,
    const uint32_t* target_tids,
    const int32_t* injection_uV,
    uint32_t count,
    cudaStream_t stream
) {
    if (count == 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    cu_debug_inject_current_kernel<<<blocks, threads, 0, stream>>>(soma_voltage, target_tids, injection_uV, count);
}

void launch_debug_record_v(
    const int32_t* soma_voltage,
    const uint32_t* target_tids,
    int32_t* out_trace,
    uint32_t current_tick,
    uint32_t count,
    uint32_t max_ticks,
    cudaStream_t stream
) {
    if (count == 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    cu_debug_record_v_kernel<<<blocks, threads, 0, stream>>>(soma_voltage, target_tids, out_trace, current_tick, count, max_ticks);
}

void launch_extract_telemetry(
    const uint8_t* flags_d,
    uint32_t* out_ids_d,
    uint32_t* out_count_d,
    uint32_t padded_n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (padded_n + threads - 1) / threads;
    cu_extract_telemetry_kernel<<<blocks, threads, 0, stream>>>(
        flags_d, out_ids_d, out_count_d, padded_n
    );
}

} // extern "C"
