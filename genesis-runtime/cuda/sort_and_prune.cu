#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_DENDRITE_SLOTS 128

// 8 bytes per slot
struct alignas(8) DendriteSlot {
  uint32_t target_packed;
  int16_t weight;
  uint8_t timer;
  uint8_t _pad;
};

// 1. Sort & Prune Kernel
// Blocks must be strictly 32 threads (1 warp) to optimally use Shared Memory.
__global__ void sort_and_prune_kernel(uint32_t padded_n,
                                      uint32_t *dendrite_targets,
                                      int16_t *dendrite_weights,
                                      uint8_t *dendrite_timers,
                                      int16_t prune_threshold) {
  // 32 neurons per block. 128 slots per neuron. 8 bytes per slot.
  // Total shared memory: 32 * 128 * 8 = 32768 bytes (32 KB)
  __shared__ DendriteSlot smem[32][MAX_DENDRITE_SLOTS];

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x;

  bool active = tid < padded_n;

  // 1. Coalesced Read from Global Memory to Shared Memory
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    if (active) {
      uint32_t idx = slot * padded_n + tid;
      smem[lane][slot].target_packed = dendrite_targets[idx];
      smem[lane][slot].weight = dendrite_weights[idx];
      smem[lane][slot].timer = dendrite_timers[idx];
    }
  }

  __syncthreads(); // Wait for all reads in the block

  if (active) {
    // 2. Sequential Sort (Insertion Sort) per Thread over its 128 slots.
    // Sort by abs(weight) descending.
    for (int i = 1; i < MAX_DENDRITE_SLOTS; ++i) {
      DendriteSlot key = smem[lane][i];
      int16_t key_abs = (key.weight >= 0) ? key.weight : -key.weight;
      int j = i - 1;

      // Shift elements to the right to make room
      while (j >= 0) {
        int16_t w_j = smem[lane][j].weight;
        int16_t abs_j = (w_j >= 0) ? w_j : -w_j;

        // Target 0 (Sentinel empty slot) should always sink to the bottom
        bool key_is_empty = (key.target_packed == 0);
        bool j_is_empty = (smem[lane][j].target_packed == 0);

        bool should_swap = false;
        if (j_is_empty && !key_is_empty) {
          should_swap = true;
        } else if (!j_is_empty && !key_is_empty) {
          if (key_abs > abs_j) {
            should_swap = true;
          }
        }

        if (should_swap) {
          smem[lane][j + 1] = smem[lane][j];
          j = j - 1;
        } else {
          break;
        }
      }
      smem[lane][j + 1] = key;
    }

    // 3. Prune Connections Below Threshold
    for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
      int16_t w = smem[lane][slot].weight;
      int16_t abs_w = (w >= 0) ? w : -w;
      if (abs_w < prune_threshold) {
        smem[lane][slot].target_packed = 0; // Destroy connection structurally
      }
    }
  }

  __syncthreads(); // Synchronize before write back

  // 4. Coalesced Write back to Global Memory
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    if (active) {
      uint32_t idx = slot * padded_n + tid;
      dendrite_targets[idx] = smem[lane][slot].target_packed;
      dendrite_weights[idx] = smem[lane][slot].weight;
      dendrite_timers[idx] = smem[lane][slot].timer;
    }
  }
}

extern "C" void launch_sort_and_prune(uint32_t padded_n, void *dendrite_targets,
                                      void *dendrite_weights,
                                      void *dendrite_timers,
                                      int16_t prune_threshold, void *stream) {
  int blockSize =
      32; // Exactly 1 Warp to map directly to shared memory dimensions
  int numBlocks = (padded_n + blockSize - 1) / blockSize;
  sort_and_prune_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
      padded_n, (uint32_t *)dendrite_targets, (int16_t *)dendrite_weights,
      (uint8_t *)dendrite_timers, prune_threshold);
}
