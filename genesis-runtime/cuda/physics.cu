#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define AXON_SENTINEL 0x80000000

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
  uint8_t _padding[4];
};

struct alignas(128) GenesisConstantMemory {
  VariantParameters variants[4];
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

// 1. Propagate Axons Kernel
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

// 2. Update Neurons Kernel (GLIF + Dendrite Integration)
__global__ void update_neurons_kernel(uint32_t padded_n, int32_t *voltage,
                                      int32_t *threshold_offset,
                                      uint8_t *refractory_timer, uint8_t *flags,
                                      uint32_t *soma_to_axon,
                                      uint32_t *dendrite_targets,
                                      int16_t *dendrite_weights,
                                      uint32_t *axon_heads) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t f = flags[tid];
  uint8_t type_mask = f >> 4;
  uint8_t variant = (type_mask >> 2) & 0x3;

  VariantParameters p = const_mem.variants[variant];

  // Read state
  int32_t v = voltage[tid];
  int32_t th_off = threshold_offset[tid];
  uint8_t ref_timer = refractory_timer[tid];

  // Clear spiking flag (bit 0)
  f &= ~1;

  if (ref_timer > 0) {
    ref_timer--;
  } else {
    // Leak
    v -= p.leak;

    // Dendrite summation (Columnar Layout)
    int32_t dendrite_sum = 0;
    for (int slot = 0; slot < 128; slot++) {
      uint32_t target_packed = dendrite_targets[slot * padded_n + tid];
      if (target_packed != 0) {
        uint32_t axon_id = target_packed >> 8;
        uint32_t segment = target_packed & 0xFF;

        // Active Tail Check
        uint32_t head = axon_heads[axon_id];
        if (head != AXON_SENTINEL && (head - segment) == 0) {
          dendrite_sum += dendrite_weights[slot * padded_n + tid];
        }
      }
    }

    v += dendrite_sum;

    // Spiking Check
    if (v >= (p.threshold + th_off)) {
      v = p.rest_potential;
      ref_timer = p.refractory_period;
      th_off += p.homeostasis_penalty;
      f |= 1; // Set spike flag

      // Fire local axon
      uint32_t my_axon = soma_to_axon[tid];
      if (my_axon != 0xFFFFFFFF) { // u32::MAX
        axon_heads[my_axon] = 0;
      }
    } else {
      // Decay threshold offset
      if (th_off > 0)
        th_off -= p.homeostasis_decay;
      if (th_off < 0)
        th_off = 0;
      if (v < p.rest_potential)
        v = p.rest_potential;
    }
  }

  // Write-back
  voltage[tid] = v;
  threshold_offset[tid] = th_off;
  refractory_timer[tid] = ref_timer;
  flags[tid] = f;
}

extern "C" void
launch_update_neurons(uint32_t padded_n, void *voltage, void *threshold_offset,
                      void *refractory_timer, void *flags, void *soma_to_axon,
                      void *dendrite_targets, void *dendrite_weights,
                      void *axon_heads, void *stream) {
  int blockSize = 128; // Smaller block for high register count
  int numBlocks = (padded_n + blockSize - 1) / blockSize;
  update_neurons_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
      padded_n, (int32_t *)voltage, (int32_t *)threshold_offset,
      (uint8_t *)refractory_timer, (uint8_t *)flags, (uint32_t *)soma_to_axon,
      (uint32_t *)dendrite_targets, (int16_t *)dendrite_weights,
      (uint32_t *)axon_heads);
}

// 3. Apply GSOP Kernel
__global__ void apply_gsop_kernel(uint32_t padded_n, uint8_t *flags,
                                  uint32_t *dendrite_targets,
                                  int16_t *dendrite_weights,
                                  uint8_t *dendrite_timers,
                                  uint32_t *axon_heads) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t f = flags[tid];
  uint8_t type_mask = f >> 4;
  uint8_t variant = (type_mask >> 2) & 0x3;
  VariantParameters p = const_mem.variants[variant];
  bool is_spiking = (f & 1) != 0;

  for (int slot = 0; slot < 128; ++slot) {
    uint32_t target_packed = dendrite_targets[slot * padded_n + tid];
    if (target_packed != 0) {
      uint32_t axon_id = target_packed >> 8;
      uint32_t segment = target_packed & 0xFF;
      uint32_t head = axon_heads[axon_id];

      uint8_t timer = dendrite_timers[slot * padded_n + tid];
      bool contact = (head != AXON_SENTINEL && (head - segment) == 0);

      if (contact) {
        // Pre-synaptic spike resets the timer
        timer = p.synapse_refractory;
      } else if (timer > 0) {
        timer--;
      }

      // Plasticity update (only if post-synaptic neuron spikes)
      if (is_spiking) {
        int16_t w = dendrite_weights[slot * padded_n + tid];
        int16_t abs_w = (w >= 0) ? w : -w;
        int sign = (w >= 0) ? 1 : -1;
        uint8_t rank = abs_w >> 11; // 0..15 range
        uint8_t inertia = const_mem.inertia_lut[rank];

        if (timer > 0) {
          // Potentiation: Recent pre-synaptic activity
          int32_t delta = (p.gsop_potentiation * inertia) >> 7;
          abs_w += delta;
          if (abs_w > 32767)
            abs_w = 32767;
        } else {
          // Depression: No recent pre-synaptic activity
          uint8_t decay_mult = (slot < 80) ? p.slot_decay_ltm : p.slot_decay_wm;
          int32_t delta = (p.gsop_depression * decay_mult * inertia) >> 14;
          abs_w -= delta;
          if (abs_w < 0)
            abs_w = 0;
        }
        dendrite_weights[slot * padded_n + tid] = sign * abs_w;
      }
      dendrite_timers[slot * padded_n + tid] = timer;
    }
  }
}

extern "C" void launch_apply_gsop(uint32_t padded_n, void *flags,
                                  void *dendrite_targets,
                                  void *dendrite_weights, void *dendrite_timers,
                                  void *axon_heads, void *stream) {
  int blockSize = 128;
  int numBlocks = (padded_n + blockSize - 1) / blockSize;
  apply_gsop_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
      padded_n, (uint8_t *)flags, (uint32_t *)dendrite_targets,
      (int16_t *)dendrite_weights, (uint8_t *)dendrite_timers,
      (uint32_t *)axon_heads);
}
