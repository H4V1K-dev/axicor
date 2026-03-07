#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// =====================================================================
// 1. VRAM Layout (Строгое совпадение с genesis_core::layout::VramState)
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
  int16_t* __restrict__ dendrite_weights;
  uint8_t* __restrict__ dendrite_timers;

  BurstHeads8* __restrict__ axon_heads;

  uint32_t* __restrict__ input_bitmask;
  uint8_t* __restrict__ output_history;
  uint32_t* __restrict__ telemetry_count;
  uint32_t* __restrict__ telemetry_spikes;
};

// Строго 128 байт. 16 типов = 2048 байт (влезает в constant memory)
struct VariantParameters {
  int32_t threshold;
  int32_t rest_potential;
  int32_t leak_rate;
  int32_t homeostasis_penalty;
  int32_t homeostasis_decay;
  int32_t gsop_potentiation;
  int32_t gsop_depression;
  uint8_t refractory_period;
  uint8_t synapse_refractory_period;
  uint8_t slot_decay_ltm;
  uint8_t slot_decay_wm;
  uint8_t signal_propagation_length;
  uint8_t ltm_slot_count;
  uint16_t heartbeat_m;      // DDS Phase Accumulator Multiplier
  int16_t inertia_curve[16]; // 32B — кривая инерции GSOP
  int16_t prune_threshold;   // Night Phase threshold
  uint8_t _pad2[58];         // Дополняем до 128 байт
};
}

// Глобальная константная память GPU (448 байт).
// Глобальная константная память. Определена в physics.cu.
extern __constant__ VariantParameters VARIANT_LUT[16];
__constant__ int16_t current_dopamine;

__global__ void cu_extract_telemetry_kernel(
    const uint8_t* __restrict__ soma_flags,
    uint32_t* __restrict__ out_ids,
    uint32_t* __restrict__ out_count,
    uint32_t padded_n
);

// Константы (совпадают с constants.rs)
#define MAX_DENDRITE_SLOTS 128
#define AXON_SENTINEL 0x80000000

__device__ __forceinline__ void push_burst_head(BurstHeads8* h) {
  h->h7 = h->h6;
  h->h6 = h->h5;
  h->h5 = h->h4;
  h->h4 = h->h3;
  h->h3 = h->h2;
  h->h2 = h->h1;
  h->h1 = h->h0;
  h->h0 = 0;
}

// =====================================================================
// Ядро 1: Инъекция внешних сигналов (InjectInputs)
// Спецификация: 08_io_matrix.md §2.6
// =====================================================================
__global__ void
inject_inputs_kernel(const SoA_State state, const uint32_t* __restrict__ bitmask,
                     uint32_t virtual_offset, uint32_t current_tick,
                     uint8_t input_stride, uint32_t total_virtual_axons) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_virtual_axons)
    return;

  // Поддержка stride (частоты инъекции)
  if (input_stride > 0 && (current_tick % input_stride) != 0)
    return;

  // Вычисляем индекс слова в плоском массиве bitmask
  uint32_t effective_tick =
      (input_stride == 0) ? 0 : (current_tick / input_stride);
  uint32_t words_per_tick = (total_virtual_axons + 31) / 32;

  uint32_t word_idx = (effective_tick * words_per_tick) + (tid / 32);
  uint32_t bit_idx = tid % 32;

  // Чтение маски (Broadcast read — 32 потока варпа читают одно и то же слово из
  // L1 кеша)
  uint32_t mask_word = bitmask[word_idx];

  // Рождение сигнала = Burst Shift
  if ((mask_word >> bit_idx) & 1) {
    BurstHeads8 b = state.axon_heads[virtual_offset + tid];
    push_burst_head(&b);
    state.axon_heads[virtual_offset + tid] = b;
  }
}

// =====================================================================
// Ядро 2: Инъекция Ghost-спайков (Fast Path)
// Спецификация: 06_distributed.md §2.4 (Sender-Side Mapping)
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

  // Вычисляем смещение для текущего тика
  uint32_t offset = current_tick * max_spikes_per_tick + tid;
  uint32_t target_axon = schedule_buffer[offset];

  // [DOD FIX] Жесткая защита VRAM от битых сетевых индексов
  if (target_axon < state.total_axons) {
    BurstHeads8 b = state.axon_heads[target_axon];
    push_burst_head(&b);
    state.axon_heads[target_axon] = b;
  }
}

// =====================================================================
// Ядро 3: Безусловный сдвиг голов всех аксонов (PropagateAxons)
// Спецификация: 05_signal_physics.md §1.6
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
// Ядро 6: Запись истории активности (Direct Memory Access)
// Спецификация: 08_io_matrix.md §3.2
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

  // Запись в 2D буфер [sync_batch_ticks × num_channels]
  // Строгий flat access, никаких atomics
  uint32_t out_idx = current_tick * num_channels + tid;
  state.output_history[out_idx] = is_spiking;
}

// =====================================================================
// C-ABI Экспорты для Rust (FFI Bindings)
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
  // Копируем 1024 байта (16 * 64) напрямую в symbol VARIANT_LUT
  cudaMemcpyToSymbol(VARIANT_LUT, host_ptr, 1024, 0, cudaMemcpyHostToDevice);
}

void update_constant_memory_hot_reload(const VariantParameters *new_variants,
                                       cudaStream_t stream) {
  // Асинхронно копируем новые параметры в константную память,
  // обеспечивая Zero-Downtime Hot-Reload на барьере BSP.
  cudaMemcpyToSymbolAsync(VARIANT_LUT, new_variants,
                          sizeof(VariantParameters) * 16, 0,
                          cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

extern "C" void update_global_dopamine(int16_t dopamine, void *stream) {
  cudaMemcpyToSymbolAsync(current_dopamine, &dopamine, sizeof(int16_t), 0,
                          cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

// Ланчеры. Мы передаем SoA_State по значению (repr(C) в Rust гарантирует
// совместимость).
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
  // Мы используем tick_schedule напрямую. Ядро ожидает массив u32.
  // SpikeEvent в Rust упакован так же, как Ghost_Payload в GPU (4B ID + 4B
  // offset). Но ядро apply_spike_batch_kernel ожидает просто uint32_t *
  // schedule_buffer. ВАКНИМАНИЕ: Если SpikeEvent - это (u32 id, u32 offset), то
  // нам нужно ядро, которое это понимает. Но спецификация Шага 10 говорит:
  // ApplySpikeBatch берет tick_schedule.
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
// Ядро 7: Синхронизация Ghost Axons
// =====================================================================
__global__ void ghost_sync_kernel(const BurstHeads8* __restrict__ src_axon_heads,
                                  BurstHeads8* __restrict__ dst_axon_heads,
                                  const uint32_t* __restrict__ src_indices,
                                  const uint32_t* __restrict__ dst_indices, uint32_t count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count)
    return;
  uint32_t src_idx = src_indices[tid];
  uint32_t dst_idx = dst_indices[tid];

  // [DOD FIX] Аппаратная защита от отравленных индексов.
  // Легальный axon_id << 2^31. Всё что >= 0x80000000:
  //   0x80000000 = AXON_SENTINEL (мёртвый аксон)
  //   0xFFFFFFFF = u32::MAX (сома без аксона, soma_to_axon default)
  // Такой src_idx → out-of-bounds в VRAM → Page Fault → Context Poison.
  if (src_idx < 0x80000000u) {
    dst_axon_heads[dst_idx] = src_axon_heads[src_idx];
  }
}

void launch_ghost_sync(const BurstHeads8 *src_heads, BurstHeads8 *dst_heads,
                       const uint32_t *src_indices, const uint32_t *dst_indices,
                       uint32_t count, cudaStream_t stream) {
  uint32_t threads = 256;
  uint32_t blocks = (count + threads - 1) / threads;
  ghost_sync_kernel<<<blocks, threads, 0, stream>>>(
      src_heads, dst_heads, src_indices, dst_indices, count);
}

#pragma pack(push, 1)
struct SpikeEvent {
  uint32_t ghost_id;
  uint32_t tick_offset;
};
#pragma pack(pop)

// Ядро компактизирует спайки из гигабайтного графа в плоский Pinned RAM буфер
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

  // Кастуем структуру к массиву для развернутого прохода (Zero-cost)
  const uint32_t* heads = (const uint32_t*)&h;

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
      uint32_t head = heads[i];
      
      // Игнорируем мертвые хвосты (AXON_SENTINEL = 0x80000000)
      if (head >= 0x70000000u) continue;

      uint32_t ticks_since_spike = head / v_seg;

      // Если спайк произошел ВНУТРИ текущего батча
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

  // Сброс счетчика перед запуском ядра
  cudaMemsetAsync(out_count, 0, sizeof(uint32_t), stream);

  extract_outgoing_spikes_kernel<<<blocks, threads, 0, stream>>>(
      axon_heads, src_indices, dst_ghost_ids, count, sync_batch_ticks, v_seg,
      (SpikeEvent *)out_events, out_count);
}
} // end extern "C"

#define WARP_SIZE 32

struct DendriteSlot {
  uint32_t target;
  int16_t weight;
  uint8_t timer;
};

// =====================================================================
// Ядро 8: Сортировка и прунинг синапсов (Night Phase)
// =====================================================================
__global__ void sort_and_prune_kernel(SoA_State state, uint32_t padded_n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  // Динамическое чтение порога для каждого нейрона
  uint8_t flag = state.flags[tid];
  uint8_t variant_id = (flag >> 4) & 0x0F;
  int16_t prune_threshold = VARIANT_LUT[variant_id].prune_threshold;

  __shared__ DendriteSlot smem[WARP_SIZE][MAX_DENDRITE_SLOTS];
  uint32_t lane_id = threadIdx.x;

  // 1. КООПЕРАТИВНОЕ ЧТЕНИЕ (Coalesced Load)
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    uint32_t col_idx = slot * padded_n + tid;
    smem[lane_id][slot].target = state.dendrite_targets[col_idx];
    smem[lane_id][slot].weight = state.dendrite_weights[col_idx];
    smem[lane_id][slot].timer = state.dendrite_timers[col_idx];
  }
  __syncwarp();

  // 2. ИНДИВИДУАЛЬНАЯ СОРТИРОВКА (Insertion Sort в L1/Shared Memory)
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

  // 3. PRUNING (Обрезка слабых связей)
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    int16_t w = smem[lane_id][slot].weight;
    int32_t abs_w = w >= 0 ? w : -w;
    if (smem[lane_id][slot].target != 0 && abs_w < prune_threshold) {
      smem[lane_id][slot].target = 0;
      smem[lane_id][slot].weight = 0;
      smem[lane_id][slot].timer = 0;
    }
  }
  __syncwarp();

  // 4. КООПЕРАТИВНАЯ ЗАПИСЬ (Coalesced Store)
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    uint32_t col_idx = slot * padded_n + tid;
    state.dendrite_targets[col_idx] = smem[lane_id][slot].target;
    state.dendrite_weights[col_idx] = smem[lane_id][slot].weight;
    state.dendrite_timers[col_idx] = smem[lane_id][slot].timer;
  }
}

extern "C" {
// launch_sort_and_prune с полной сигнатурой определен ниже (requires
// ShardVramPtrs)
} // last existing extern "C"

// =====================================================================
// § New Memory Contract: ShardVramPtrs + cu_* functions
//
// ЗАКОН: Порядок полей ShardVramPtrs строго совпадает с Rust-структурой
// и с порядком байт в .state блобе. Нарушение = Silent Data Corruption.
//
// Раскладка в памяти (padded_n кратно 32 → всё Warp-Aligned):
//   soma_voltage      [padded_n]       × 4 B  ← base ptr (= soma_voltage)
//   soma_flags        [padded_n]       × 1 B
//   threshold_offset  [padded_n]       × 4 B
//   timers            [padded_n]       × 1 B
//   soma_to_axon      [padded_n]       × 4 B
//   dendrite_targets  [padded_n × 128] × 4 B
//   dendrite_weights  [padded_n × 128] × 2 B
//   ── один cudaMalloc, один cudaMemcpyAsync ──
//   axon_heads        [total_axons]    × 4 B  ← отдельный cudaMalloc
// =====================================================================

// Зеркало Rust #[repr(C)] struct ShardVramPtrs
struct ShardVramPtrs {
  int32_t* __restrict__ soma_voltage; // base ptr всего state-блоба
  uint8_t* __restrict__ soma_flags;
  int32_t* __restrict__ threshold_offset;
  uint8_t* __restrict__ timers;
  uint32_t* __restrict__ soma_to_axon;
  uint32_t* __restrict__ dendrite_targets;
  int16_t* __restrict__ dendrite_weights;
  uint8_t* __restrict__ dendrite_timers;
  BurstHeads8* __restrict__ axon_heads; // отдельный буфер
};

#define MAX_DENDRITES_SV 128

extern "C" {

// ─── Аллокация
// ──────────────────────────────────────────────────────────────── Два
// cudaMalloc: State (непрерывный блок) + Axons (отдельный). Если хоть одна
// аллокация падает — откатываем и возвращаем ненулевой код.
int32_t cu_allocate_shard(uint32_t padded_n, uint32_t total_axons,
                          ShardVramPtrs *out_vram) {
  size_t sz_voltage = (size_t)padded_n * sizeof(int32_t);
  size_t sz_flags = (size_t)padded_n * sizeof(uint8_t);
  size_t sz_thresh = (size_t)padded_n * sizeof(int32_t);
  size_t sz_timers = (size_t)padded_n * sizeof(uint8_t);
  size_t sz_s2a = (size_t)padded_n * sizeof(uint32_t);
  size_t sz_targets = (size_t)padded_n * MAX_DENDRITES_SV * sizeof(uint32_t);
  size_t sz_weights = (size_t)padded_n * MAX_DENDRITES_SV * sizeof(int16_t);
  size_t sz_dtimers = (size_t)padded_n * MAX_DENDRITES_SV * sizeof(uint8_t);

  size_t total_state = sz_voltage + sz_flags + sz_thresh + sz_timers + sz_s2a +
                       sz_targets + sz_weights + sz_dtimers;

  // Единый Flat Allocation для всех полей сом + дендритов
  void *base = nullptr;
  cudaError_t err = cudaMalloc(&base, total_state);
  if (err != cudaSuccess) {
    fprintf(stderr, "[cu_allocate_shard] cudaMalloc state failed: %s\n",
            cudaGetErrorString(err));
    return (int32_t)err;
  }

  // Нулевая инициализация — гарантирует отсутствие мусора при первом тике
  cudaMemset(base, 0, total_state);

  // Zero-Cost Partitioning: раздаём указатели внутри одного буфера
  size_t off = 0;
  out_vram->soma_voltage = (int32_t *)((char *)base + off);
  off += sz_voltage;
  out_vram->soma_flags = (uint8_t *)((char *)base + off);
  off += sz_flags;
  out_vram->threshold_offset = (int32_t *)((char *)base + off);
  off += sz_thresh;
  out_vram->timers = (uint8_t *)((char *)base + off);
  off += sz_timers;
  out_vram->soma_to_axon = (uint32_t *)((char *)base + off);
  off += sz_s2a;
  out_vram->dendrite_targets = (uint32_t *)((char *)base + off);
  off += sz_targets;
  out_vram->dendrite_weights = (int16_t *)((char *)base + off);
  off += sz_weights;
  out_vram->dendrite_timers = (uint8_t *)((char *)base + off);

  // Аксоны — отдельная аллокация (total_axons ≠ padded_n)
  err = cudaMalloc((void **)&out_vram->axon_heads,
                   (size_t)total_axons * sizeof(BurstHeads8));
  if (err != cudaSuccess) {
    fprintf(stderr, "[cu_allocate_shard] cudaMalloc axon_heads failed: %s\n",
            cudaGetErrorString(err));
    cudaFree(base);
    return (int32_t)err;
  }

  // Инициализируем аксоны нулём (живые). Baker перезапишет нужные значения.
  cudaMemset(out_vram->axon_heads, 0, (size_t)total_axons * sizeof(BurstHeads8));

  return 0;
}

// ─── DMA Upload: State
// ──────────────────────────────────────────────────────── .state блоб содержит
// 7 массивов слитно, в том же порядке, что и ShardVramPtrs. Поскольку мы
// сделали Flat Allocation, base_ptr == soma_voltage. Один cudaMemcpyAsync
// заполняет ВСЕ 7 массивов на 100% пропускной способности PCIe.
int32_t cu_upload_state_blob(const ShardVramPtrs *vram, const void *state_blob,
                             size_t state_size) {
  cudaError_t err =
      cudaMemcpyAsync((void *)vram->soma_voltage, // base ptr блока
                      state_blob, state_size, cudaMemcpyHostToDevice,
                      0 // default stream
      );
  if (err != cudaSuccess) {
    fprintf(stderr, "[cu_upload_state_blob] cudaMemcpyAsync failed: %s\n",
            cudaGetErrorString(err));
    return (int32_t)err;
  }
  // Block CPU until VRAM is ready. Init-phase — latency не важна.
  cudaStreamSynchronize(0);
  return 0;
}

// ─── DMA Upload: Axons
// ──────────────────────────────────────────────────────── .axons блоб —
// плоский массив uint32_t axon_heads.
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

// ─── Free
// ─────────────────────────────────────────────────────────────────────
// soma_voltage == base ptr всего state-блока.
// Два cudaFree соответствуют двум cudaMalloc в cu_allocate_shard.
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
// Один блок = один нейрон (32 потока). Идеальная утилизация Shared Memory.
// Без stream — ночная фаза синхронна.
// =====================================================================
extern "C" void launch_sort_and_prune(const ShardVramPtrs *ptrs,
                                      uint32_t padded_n) {
  uint32_t threads = 32;
  uint32_t blocks = padded_n;

  SoA_State state = {};
  state.dendrite_targets = ptrs->dendrite_targets;
  state.dendrite_weights = ptrs->dendrite_weights;
  state.dendrite_timers = ptrs->dendrite_timers;
  state.flags = ptrs->soma_flags; // Передаем флаги для чтения варианта

  sort_and_prune_kernel<<<blocks, threads>>>(state, padded_n);
}

// ============================================================================
// I/O VRAM Allocations & DMA Transfers
// ============================================================================
extern "C" {

int32_t cu_allocate_io_buffers(
    uint32_t input_words,       // Размер битовой маски в u32
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

int32_t cu_dma_h2d_io(uint32_t *d_input_bitmask,
                      const uint32_t *h_input_bitmask, uint32_t input_words,
                      uint32_t *d_incoming_spikes,
                      const uint32_t *h_incoming_spikes,
                      uint32_t schedule_capacity) {
  // Асинхронная загрузка в Stream 0. CPU не блокируется!
  if (input_words > 0 && d_input_bitmask && h_input_bitmask) {
    cudaMemcpyAsync(d_input_bitmask, h_input_bitmask,
                    input_words * sizeof(uint32_t), cudaMemcpyHostToDevice, 0);
  }
  if (schedule_capacity > 0 && d_incoming_spikes && h_incoming_spikes) {
    cudaMemcpyAsync(d_incoming_spikes, h_incoming_spikes,
                    schedule_capacity * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, 0);
  }
  return 0;
}

int32_t cu_dma_d2h_io(uint8_t *h_output_history,
                      const uint8_t *d_output_history,
                      uint32_t output_capacity) {
  if (output_capacity > 0 && d_output_history && h_output_history) {
    cudaMemcpyAsync(h_output_history, d_output_history,
                    output_capacity * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                    0);
  }
  return 0;
}

void gpu_reset_telemetry_count(const ShardVramPtrs *ptrs, cudaStream_t stream) {
    // No-op for pinned memory as requested by specification/absence of telemetry_count.
}

void launch_extract_telemetry(const ShardVramPtrs *ptrs, uint32_t padded_n, uint32_t *out_ids, uint32_t *out_count_pinned, cudaStream_t stream) {
    if (out_ids == nullptr || out_count_pinned == nullptr || ptrs == nullptr) return;

    int threads = 256;
    int blocks = (padded_n + threads - 1) / threads;
    
    cu_extract_telemetry_kernel<<<blocks, threads, 0, stream>>>(
        ptrs->soma_flags,
        out_ids,
        out_count_pinned,
        padded_n
    );
}
} // Final closing brace for extern "C"
