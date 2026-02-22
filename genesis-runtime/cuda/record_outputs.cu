#include <cuda_runtime.h>
#include <stdint.h>

__global__ void record_outputs_kernel(uint32_t padded_n, uint8_t *flags,
                                      uint32_t *outbound_spikes_buffer,
                                      uint32_t *outbound_spikes_count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  // Read the flags array
  uint8_t f = flags[tid];
  bool is_spiking = (f & 1) != 0;

  if (is_spiking) {
    // Atomic increment of the global counter. Returns the old value to serve as
    // our index.
    uint32_t idx = atomicAdd(outbound_spikes_count, 1);

    // Note: In a production Z-Sort map, we might look up a secondary array here
    // to translate `tid` into a mapped output ID. For the base BSP
    // implementation, we output the raw Dense Neuron ID.
    outbound_spikes_buffer[idx] = tid;
  }
}

extern "C" void launch_record_outputs(uint32_t padded_n, void *flags,
                                      void *outbound_spikes_buffer,
                                      void *outbound_spikes_count,
                                      void *stream) {
  int blockSize = 128; // Standard optimization block size
  int numBlocks = (padded_n + blockSize - 1) / blockSize;

  record_outputs_kernel<<<numBlocks, blockSize, 0, (cudaStream_t)stream>>>(
      padded_n, (uint8_t *)flags, (uint32_t *)outbound_spikes_buffer,
      (uint32_t *)outbound_spikes_count);
}
