#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define AXON_SENTINEL 0x80000000u

// target_packed bit layout: [31..10] Axon_ID (22 bits) | [9..0] Segment_Index (10 bits)
#define TARGET_AXON_SHIFT 10
#define TARGET_SEG_MASK   0x3FFu

// Constant memory structures
struct alignas(32) VariantParameters {
  int32_t threshold;
  int32_t rest_potential;
  int32_t leak;
  int32_t homeostasis_penalty;
  int32_t homeostasis_decay;
  uint16_t gsop_potentiation;
  uint16_t gsop_depression;
  uint8_t refractory_period;
  uint8_t synapse_refractory;
  uint8_t slot_decay_ltm;
  uint8_t slot_decay_wm;
  uint8_t propagation_length;  // Active Tail length (per-variant, from blueprints)
  uint8_t _padding[3];
};

struct alignas(128) GenesisConstantMemory {
  VariantParameters variants[16];
  uint8_t inertia_lut[16];
  uint8_t _padding[112];
};

__constant__ GenesisConstantMemory const_mem;

extern "C" bool upload_constant_memory(const void *host_ptr) {
  cudaError_t err =
      cudaMemcpyToSymbol(const_mem, host_ptr, sizeof(GenesisConstantMemory));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA MemcpyToSymbol failed: %s\n",
            cudaGetErrorString(err));
    return false;
  }
  return true;
}

// ============================================================================
// 1. Propagate Axons Kernel (Spec 05 §1.6)
// ============================================================================
// Unconditional shift of ALL axons (Local + Ghost + Virtual).
// SENTINEL guard: skip inactive axons to avoid false positive overlap
// after ~59.6h of ticks. Sentinel Refresh handles long-running zones.
__global__ void propagate_axons_kernel(uint32_t total_axons,
                                       uint32_t *axon_heads, uint32_t v_seg) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_axons) {
    uint32_t head = axon_heads[idx];
    if (head != AXON_SENTINEL) {
      axon_heads[idx] = head + v_seg;
    }
  }
}

extern "C" void launch_propagate_axons(uint32_t total_axons,
                                       uint32_t *axon_heads, uint32_t v_seg,
                                       void *stream) {
  int blockSize = 256;
  int numBlocks = (total_axons + blockSize - 1) / blockSize;
  propagate_axons_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
      total_axons, axon_heads, v_seg);
}

// ============================================================================
// 2. Update Neurons Kernel — GLIF + Dendrite Integration (Spec 05 §1.5)
// ============================================================================
// Order: Homeostasis decay → Refractory gate → GLIF leak → Columnar dendrite
// loop (with per-slot refractory timer) → Threshold check → Fire/reset.
__global__ void update_neurons_kernel(uint32_t padded_n, int32_t *voltage,
                                      int32_t *threshold_offset,
                                      uint8_t *refractory_timer, uint8_t *flags,
                                      uint32_t *soma_to_axon,
                                      uint32_t *dendrite_targets,
                                      int16_t *dendrite_weights,
                                      uint8_t *dendrite_timers,
                                      uint32_t *axon_heads) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  // 1. Unpack type + load variant parameters (1 clock L1)
  uint8_t f = flags[tid];
  uint8_t type_mask = f >> 4;
  uint8_t variant = type_mask & 0xF;
  VariantParameters p = const_mem.variants[variant];

  // 2. Homeostasis decay — ALWAYS runs, even when soma is refractory (Spec §1.5)
  int32_t th_off = threshold_offset[tid];
  int32_t decayed = th_off - p.homeostasis_decay;
  th_off = decayed & ~(decayed >> 31); // Branchless max(0, val)

  // 3. Refractory early exit (~90% of threads)
  uint8_t ref_timer = refractory_timer[tid];
  if (ref_timer > 0) {
    refractory_timer[tid] = ref_timer - 1;
    threshold_offset[tid] = th_off;
    flags[tid] = f & ~0x1; // Clear spike flag
    return;
  }

  // 4. GLIF: Membrane leak (branchless clamp >= rest_potential)
  int32_t v = voltage[tid];
  int32_t leaked = v - p.leak;
  int32_t diff = leaked - p.rest_potential;
  v = p.rest_potential + (diff & ~(diff >> 31)); // max(rest, leaked)

  // 5. Columnar dendrite loop: 128 slots (Coalesced Access)
  for (int slot = 0; slot < 128; ++slot) {
    uint32_t col_idx = slot * padded_n + tid;

    // 5a. Dendrite Refractory Gate — saves 3 global reads when timer > 0
    uint8_t d_timer = dendrite_timers[col_idx];
    if (d_timer > 0) {
      dendrite_timers[col_idx] = d_timer - 1;
      continue;
    }

    // 5b. Empty slot → break (Columnar Defrag invariant from Night Phase:
    // first target==0 means all remaining slots are also empty)
    uint32_t target_packed = dendrite_targets[col_idx];
    if (target_packed == 0) break;

    // 5c. Active Tail Overlap Check (u32 overflow legalized)
    uint32_t axon_id = target_packed >> TARGET_AXON_SHIFT;
    uint32_t seg_idx = target_packed & TARGET_SEG_MASK;
    uint32_t head = axon_heads[axon_id];
    uint32_t dist = head - seg_idx;

    if (dist <= p.propagation_length) {
      // 5d. Voltage accumulation (i16→i32, sign baked in)
      v += (int32_t)dendrite_weights[col_idx];

      // 5e. Timer reset from Constant Memory — marks contact for GSOP
      dendrite_timers[col_idx] = p.synapse_refractory;
    }
  }

  // 6. Threshold Check & Fire
  int32_t eff_threshold = p.threshold + th_off;
  int32_t is_spiking = (v >= eff_threshold) ? 1 : 0;

  // Branchless state update
  v = is_spiking * p.rest_potential + (1 - is_spiking) * v;
  ref_timer = is_spiking * p.refractory_period;
  th_off += is_spiking * p.homeostasis_penalty;
  f = (f & 0xFE) | (uint8_t)is_spiking;

  // 7. Fire axon: predicated store (Spec: axon_heads[soma_to_axon[tid]] = 0)
  if (is_spiking) {
    uint32_t my_axon = soma_to_axon[tid];
    if (my_axon != 0xFFFFFFFF) {
      axon_heads[my_axon] = 0;
    }
  }

  // 8. Write-back to VRAM
  voltage[tid] = v;
  threshold_offset[tid] = th_off;
  refractory_timer[tid] = ref_timer;
  flags[tid] = f;
}

extern "C" void
launch_update_neurons(uint32_t padded_n, void *voltage, void *threshold_offset,
                      void *refractory_timer, void *flags, void *soma_to_axon,
                      void *dendrite_targets, void *dendrite_weights,
                      void *dendrite_timers,
                      void *axon_heads, void *stream) {
  int blockSize = 128; // Smaller block for high register count
  int numBlocks = (padded_n + blockSize - 1) / blockSize;
  update_neurons_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
      padded_n, (int32_t *)voltage, (int32_t *)threshold_offset,
      (uint8_t *)refractory_timer, (uint8_t *)flags, (uint32_t *)soma_to_axon,
      (uint32_t *)dendrite_targets, (int16_t *)dendrite_weights,
      (uint8_t *)dendrite_timers,
      (uint32_t *)axon_heads);
}

// ============================================================================
// 3. Apply GSOP Kernel — Plasticity (Spec 05 §1.3)
// ============================================================================
// Uses Timer-as-Contact-Flag: UpdateNeurons already recorded contact in
// dendrite_timers. timer == synapse_refractory → contact this tick.
// No redundant overlap check. Early exit for 99.9% of threads.
__global__ void apply_gsop_kernel(uint32_t padded_n, uint8_t *flags,
                                  uint32_t *dendrite_targets,
                                  int16_t *dendrite_weights,
                                  uint8_t *dendrite_timers) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  // 1. Early Exit: 99.9% of threads leave here (spike rate ~1-10 Hz)
  uint8_t f = flags[tid];
  if (!(f & 0x1)) return;

  // 2. Load variant parameters (L1 Cache, 1 clock)
  uint8_t type_mask = f >> 4;
  uint8_t variant = type_mask & 0xF;
  VariantParameters p = const_mem.variants[variant];

  // 3. Columnar Loop: 128 slots (Coalesced Access)
  for (int slot = 0; slot < 128; ++slot) {
    uint32_t col_idx = slot * padded_n + tid;

    uint32_t target_packed = dendrite_targets[col_idx];
    if (target_packed == 0) break; // Columnar defrag invariant

    // 4. Causal Check: Timer-as-Contact-Flag
    // UpdateNeurons wrote: timer = synapse_refractory on contact.
    // timer == synapse_refractory → causal (Potentiation)
    // timer < synapse_refractory → no contact or decaying (Depression)
    uint8_t d_timer = dendrite_timers[col_idx];
    uint32_t is_causal = (d_timer == p.synapse_refractory);

    // 5. Inertia Rank: abs(weight) >> 12 → 0..15 (Spec §2.4)
    int16_t w = dendrite_weights[col_idx];
    uint16_t abs_w = (w >= 0) ? (uint16_t)w : (uint16_t)(-w);
    uint8_t rank = abs_w >> 12;
    uint8_t inertia = const_mem.inertia_lut[rank];

    // 6. Branchless GSOP Math (Zero Float)
    uint16_t delta_pot = (p.gsop_potentiation * inertia) >> 7;
    uint16_t delta_dep = (p.gsop_depression * inertia) >> 7;
    int32_t delta = is_causal * delta_pot - (!is_causal) * delta_dep;

    // 7. Slot Decay: LTM/WM multipliers (Fixed-point: 128 = 1.0×)
    uint8_t decay = (slot < 80) ? p.slot_decay_ltm : p.slot_decay_wm;
    delta = (delta * decay) >> 7;

    // 8. Signed Clamp ±32767 (Branchless sign extraction)
    int32_t w_sign = ((int32_t)w >> 31) | 1; // +1 or -1
    int32_t new_abs = (int32_t)abs_w + delta;
    new_abs = (new_abs > 32767) ? 32767 : ((new_abs < 0) ? 0 : new_abs);
    dendrite_weights[col_idx] = (int16_t)(w_sign * new_abs);
  }
}

extern "C" void launch_apply_gsop(uint32_t padded_n, void *flags,
                                  void *dendrite_targets,
                                  void *dendrite_weights, void *dendrite_timers,
                                  void *stream) {
  int blockSize = 128;
  int numBlocks = (padded_n + blockSize - 1) / blockSize;
  apply_gsop_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
      padded_n, (uint8_t *)flags, (uint32_t *)dendrite_targets,
      (int16_t *)dendrite_weights, (uint8_t *)dendrite_timers);
}

