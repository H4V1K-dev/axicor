#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// =====================================================================
// 1. VRAM Layout (Строгое совпадение с genesis_core::layout::VramState)
// =====================================================================
extern "C" {
struct SoA_State {
  int32_t *voltage;
  uint8_t *flags;
  int32_t *threshold_offset;
  uint8_t *refractory_timer;

  uint32_t *soma_to_axon; // Маппинг DenseID -> Local Axon ID

  uint32_t *dendrite_targets;
  int16_t *dendrite_weights;
  uint8_t *dendrite_timers;

  uint32_t *axon_heads;

  uint32_t *input_bitmask;
  uint8_t *output_history;

  uint32_t *telemetry_spikes;
  uint32_t *telemetry_count;
};

// Строгое выравнивание в 64 байта, как в Rust
struct VariantParameters {
  int32_t threshold;
  int32_t rest_potential;
  int32_t leak_rate;
  int32_t homeostasis_penalty;

  int16_t gsop_potentiation;
  int16_t gsop_depression;
  uint16_t homeostasis_decay;
  uint16_t signal_propagation_length;
  uint16_t conduction_velocity;
  uint16_t slot_decay_ltm;
  uint16_t slot_decay_wm;

  uint8_t refractory_period;
  uint8_t synapse_refractory_period;

  uint8_t inertia_curve[16];
  uint8_t _reserved[16];
};
}

// Глобальная константная память GPU (1024 байта).
// L1 кеш раздаст эти параметры всем варпам за 1 такт.
__constant__ VariantParameters VARIANT_LUT[16];

// Константы (совпадают с constants.rs)
#define MAX_DENDRITE_SLOTS 128
#define AXON_SENTINEL 0x80000000

// =====================================================================
// Ядро 1: Инъекция внешних сигналов (InjectInputs)
// Спецификация: 08_io_matrix.md §2.6
// =====================================================================
__global__ void inject_inputs_kernel(const SoA_State state,
                                     uint32_t virtual_offset,
                                     uint32_t current_tick,
                                     uint8_t input_stride,
                                     uint32_t total_virtual_axons) {
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
  uint32_t mask_word = state.input_bitmask[word_idx];

  // Рождение сигнала = сброс головы в 0
  if ((mask_word >> bit_idx) & 1) {
    state.axon_heads[virtual_offset + tid] = 0;
  }
}

// =====================================================================
// Ядро 2: Инъекция Ghost-спайков (Fast Path)
// Спецификация: 06_distributed.md §2.4 (Sender-Side Mapping)
// =====================================================================
__global__ void apply_spike_batch_kernel(SoA_State state,
                                         const uint32_t *schedule_buffer,
                                         const uint32_t *counts,
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

  // O(1) Инъекция без поиска (Sender-Side Mapping)
  state.axon_heads[target_axon] = 0;
}

// =====================================================================
// Ядро 3: Безусловный сдвиг голов всех аксонов (PropagateAxons)
// Спецификация: 05_signal_physics.md §1.6
// =====================================================================
__global__ void propagate_axons_kernel(const SoA_State state,
                                       uint32_t total_axons, uint32_t v_seg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_axons)
    return;

  // Безусловное чтение, сложение и запись.
  // Мёртвые аксоны с AXON_SENTINEL (0x80000000) просто увеличивают своё
  // значение. 100% утилизация ALU без ветвлений.
  state.axon_heads[tid] += v_seg;
}

__global__ void update_neurons_kernel(const SoA_State state,
                                      uint32_t padded_n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t flag = state.flags[tid];
  uint8_t type_idx = (flag >> 4) & 0xF;

  // Broadcast Read: загрузка параметров физики из L1 за 1 такт
  VariantParameters variant = VARIANT_LUT[type_idx];

  // 1. Refractory Gate (Hard Limit)
  uint8_t ref_timer = state.refractory_timer[tid];
  if (ref_timer > 0) {
    state.refractory_timer[tid] = ref_timer - 1;
    // Очищаем бит спайка, если он был
    state.flags[tid] = flag & 0xFE;
    return; // EARLY EXIT: Выбрасываем поток из исполнения
  }

  // 2. Leakage & Homeostasis Soft Limit (Branchless clamp)
  int32_t v = state.voltage[tid];
  v -= variant.leak_rate;
  if (v < variant.rest_potential)
    v = variant.rest_potential;

  int32_t t_offset = state.threshold_offset[tid];
  t_offset -= variant.homeostasis_decay;
  if (t_offset < 0)
    t_offset = 0;

  // 3. Dendrite Summation (Строгий Columnar Layout)
  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    uint32_t col_idx = slot * padded_n + tid; // 100% Coalesced Access
    uint32_t target = state.dendrite_targets[col_idx];

    if (target == 0)
      break; // Сортировка гарантирует: пустые слоты только в хвосте

    // Проверка синаптической рефрактерности
    uint8_t d_timer = state.dendrite_timers[col_idx];
    if (d_timer > 0) {
      state.dendrite_timers[col_idx] = d_timer - 1;
      continue;
    }

    // Распаковка target_packed: [31..28] Type | [27..8] AxonID | [7..0] SegIdx
    uint32_t axon_idx = (target >> 8) & 0xFFFFF;
    uint32_t seg_idx = target & 0xFF;

    uint32_t head = state.axon_heads[axon_idx];
    uint32_t dist = head - seg_idx;

    // Active Tail Check.
    // Если аксон мертв, head = 0x80000000. dist будет ~2 млрд, что >
    // propagation_length.
    if (dist <= variant.signal_propagation_length) {
      v += state.dendrite_weights[col_idx];
      state.dendrite_timers[col_idx] = variant.synapse_refractory_period;
    }
  }

  // 4. Threshold & Fire Check
  int32_t effective_thresh = variant.threshold + t_offset;
  bool is_spiking = v >= effective_thresh;

  if (is_spiking) {
    v = variant.rest_potential;
    state.refractory_timer[tid] = variant.refractory_period;
    t_offset += variant.homeostasis_penalty;

    // Рождение сигнала в локальном аксоне
    uint32_t my_axon = state.soma_to_axon[tid];
    state.axon_heads[my_axon] = 0;

    // Телеметрия: пакуем ID спайка в плоский массив для IDE
    uint32_t spike_idx = atomicAdd(state.telemetry_count, 1);
    // Защита от переполнения (допустим, лимит 500_000 спайков за батч)
    if (spike_idx < 500000) {
      state.telemetry_spikes[spike_idx] = tid;
    }
  }

  // 5. Write-back состояния
  state.voltage[tid] = v;
  state.threshold_offset[tid] = t_offset;
  state.flags[tid] = (flag & 0xFE) | is_spiking;
}

// =====================================================================
// Ядро 5: GSOP Пластичность (Timer-as-Contact-Flag)
// Спецификация: 04_connectivity.md §2.2
// =====================================================================
#define LTM_SLOT_COUNT 80 // Первые 80 слотов — стабильная память (LTM)

__global__ void apply_gsop_kernel(const SoA_State state, uint32_t padded_n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t flag = state.flags[tid];

  // БРУТАЛЬНЫЙ EARLY EXIT: Пластичность обсчитывается ТОЛЬКО в момент спайка
  // сомы.
  if ((flag & 0x01) == 0)
    return;

  uint8_t type_idx = (flag >> 4) & 0xF;
  VariantParameters variant = VARIANT_LUT[type_idx];

  for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
    uint32_t col_idx = slot * padded_n + tid; // 100% Coalesced Memory Access
    uint32_t target = state.dendrite_targets[col_idx];

    if (target == 0)
      break; // Сортировка гарантирует: пустые слоты только в хвосте

    int16_t w = state.dendrite_weights[col_idx];
    uint32_t abs_w = w >= 0 ? w : -w;

    // Индекс кривой инерции (16 рангов по 2048 единиц веса)
    uint32_t rank = abs_w >> 11;
    if (rank > 15)
      rank = 15;

    int32_t inertia = variant.inertia_curve[rank];
    uint8_t timer = state.dendrite_timers[col_idx];

    int32_t delta = 0;
    // Timer-as-Contact-Flag: Если таймер равен рефрактерности синапса — значит
    // этот дендрит касался активного хвоста сигнала именно в этом тике.
    if (timer == variant.synapse_refractory_period) {
      delta = (variant.gsop_potentiation * inertia) >> 7; // Усиление
    } else {
      delta = -((variant.gsop_depression * inertia) >> 7); // Ослабление (LTD)
    }

    int32_t new_w = w + delta;

    // Slot-based decay (Градиентная стабильность LTM vs WM)
    int32_t decay = (slot < LTM_SLOT_COUNT) ? variant.slot_decay_ltm
                                            : variant.slot_decay_wm;

    if (new_w > 0) {
      new_w -= decay;
      if (new_w < 0)
        new_w = 0;
    } else if (new_w < 0) {
      new_w += decay;
      if (new_w > 0)
        new_w = 0;
    }

    // Clamp to i16 (Защита от переполнения)
    if (new_w > 32767)
      new_w = 32767;
    if (new_w < -32768)
      new_w = -32768;

    state.dendrite_weights[col_idx] = (int16_t)new_w;
  }
}

// =====================================================================
// Ядро 6: Запись истории активности (Direct Memory Access)
// Спецификация: 08_io_matrix.md §3.2
// =====================================================================
__global__ void record_readout_kernel(const SoA_State state,
                                      const uint32_t *mapped_soma_ids,
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
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void gpu_stream_synchronize(cudaStream_t stream) {
  cudaStreamSynchronize(stream);
}

void gpu_device_synchronize() { cudaDeviceSynchronize(); }

void gpu_load_constants(const void *host_ptr) {
  // Копируем 1024 байта (16 * 64) напрямую в symbol VARIANT_LUT
  cudaMemcpyToSymbol(VARIANT_LUT, host_ptr, 1024, 0, cudaMemcpyHostToDevice);
}

// Ланчеры. Мы распаковываем указатель на VramState и передаем структуру по
// значению, чтобы она легла в регистры или Constant Memory при запуске ядра.
void launch_inject_inputs(const SoA_State *vram, uint32_t virtual_offset,
                          uint32_t current_tick_in_batch, uint8_t input_stride,
                          uint32_t total_virtual_axons, cudaStream_t stream) {
  uint32_t threads = 256;
  uint32_t blocks = (total_virtual_axons + threads - 1) / threads;
  inject_inputs_kernel<<<blocks, threads, 0, stream>>>(
      *vram, virtual_offset, current_tick_in_batch, input_stride,
      total_virtual_axons);
}

void launch_apply_spike_batch(const SoA_State *vram,
                              const uint32_t *schedule_buffer,
                              const uint32_t *counts, uint32_t current_tick,
                              uint32_t max_spikes_per_tick,
                              cudaStream_t stream) {
  // Мы запускаем ядро на максимальное теоретическое количество спайков,
  // но внутри ядра есть Early Exit, если actual counts[current_tick] меньше.
  // Чтобы запустить ровно нужное количество потоков, нам пришлось бы сначала
  // скопировать counts[current_tick] на GPU или CPU, что убьет latency.
  // Поэтому запускаем константное число потоков, и пусть GPU отсекает лишние.
  // Ожидаемое макс число спайков в тик обычно невелико (e.g. 1024), так что это
  // 1 блок.
  uint32_t threads = 256;
  uint32_t blocks = (max_spikes_per_tick + threads - 1) / threads;
  apply_spike_batch_kernel<<<blocks, threads, 0, stream>>>(
      *vram, schedule_buffer, counts, current_tick, max_spikes_per_tick);
}

void launch_propagate_axons(const SoA_State *vram, uint32_t total_axons,
                            uint32_t v_seg, cudaStream_t stream) {
  uint32_t threads = 256;
  uint32_t blocks = (total_axons + threads - 1) / threads;
  propagate_axons_kernel<<<blocks, threads, 0, stream>>>(*vram, total_axons,
                                                         v_seg);
}

void launch_update_neurons(const SoA_State *vram, uint32_t padded_n,
                           cudaStream_t stream) {
  uint32_t threads = 256;
  uint32_t blocks =
      padded_n / threads; // padded_n гарантированно кратно 32, поэтому деление
                          // без остатка для выравнивания
  if (blocks == 0 && padded_n > 0)
    blocks = 1;
  update_neurons_kernel<<<blocks, threads, 0, stream>>>(*vram, padded_n);
}

void launch_apply_gsop(const SoA_State *vram, uint32_t padded_n,
                       cudaStream_t stream) {
  uint32_t threads = 256;
  uint32_t blocks = padded_n / threads;
  if (blocks == 0 && padded_n > 0)
    blocks = 1;
  apply_gsop_kernel<<<blocks, threads, 0, stream>>>(*vram, padded_n);
}

void launch_record_readout(const SoA_State *vram,
                           const uint32_t *mapped_soma_ids, uint32_t num_somas,
                           uint32_t tick_offset, cudaStream_t stream) {
  uint32_t threads = 256;
  uint32_t blocks = (num_somas + threads - 1) / threads;
  record_readout_kernel<<<blocks, threads, 0, stream>>>(*vram, mapped_soma_ids,
                                                        num_somas, tick_offset);
}

// =====================================================================
// Ядро 7: Синхронизация Ghost Axons
// =====================================================================
__global__ void ghost_sync_kernel(const uint32_t *src_axon_heads,
                                  uint32_t *dst_axon_heads,
                                  const uint32_t *src_indices,
                                  const uint32_t *dst_indices, uint32_t count) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count)
    return;
  uint32_t src_idx = src_indices[tid];
  uint32_t dst_idx = dst_indices[tid];
  dst_axon_heads[dst_idx] = src_axon_heads[src_idx];
}

void launch_ghost_sync(const uint32_t *src_heads, uint32_t *dst_heads,
                       const uint32_t *src_indices, const uint32_t *dst_indices,
                       uint32_t count, cudaStream_t stream) {
  uint32_t threads = 256;
  uint32_t blocks = (count + threads - 1) / threads;
  ghost_sync_kernel<<<blocks, threads, 0, stream>>>(
      src_heads, dst_heads, src_indices, dst_indices, count);
}

void gpu_reset_telemetry_count(const SoA_State *vram, cudaStream_t stream) {
  cudaMemsetAsync(vram->telemetry_count, 0, sizeof(uint32_t), stream);
}

#pragma pack(push, 1)
struct SpikeEvent {
  uint32_t ghost_id;
  uint32_t tick_offset;
};
#pragma pack(pop)

// Ядро компактизирует спайки из гигабайтного графа в плоский Pinned RAM буфер
__global__ void extract_outgoing_spikes_kernel(
    const uint32_t *axon_heads,
    const uint32_t *src_indices,   // Локальные ID экспортируемых аксонов
    const uint32_t *dst_ghost_ids, // Их ID на удаленной машине
    uint32_t count, uint32_t sync_batch_ticks,
    SpikeEvent *out_events, // Указатель на Pinned RAM (Mapped)
    uint32_t *out_count     // Указатель на Pinned RAM (Mapped)
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count)
    return;

  uint32_t local_axon = src_indices[tid];
  uint32_t head = axon_heads[local_axon];

  // Математика сдвига: если head < размера батча, значит спайк родился в ЭТОМ
  // батче.
  if (head < sync_batch_ticks) {
    // Атомарно занимаем слот в выходном буфере (Zero-cost atomic в L2)
    uint32_t out_idx = atomicAdd(out_count, 1);

    out_events[out_idx].ghost_id = dst_ghost_ids[tid];
    // Восстанавливаем точный тик, на котором произошел спайк
    out_events[out_idx].tick_offset = sync_batch_ticks - 1 - head;
  }
}

void launch_extract_outgoing_spikes(const uint32_t *axon_heads,
                                    const uint32_t *src_indices,
                                    const uint32_t *dst_ghost_ids,
                                    uint32_t count, uint32_t sync_batch_ticks,
                                    void *out_events, uint32_t *out_count,
                                    cudaStream_t stream) {
  uint32_t threads = 256;
  uint32_t blocks = (count + threads - 1) / threads;

  // Сброс счетчика перед запуском ядра
  cudaMemsetAsync(out_count, 0, sizeof(uint32_t), stream);

  extract_outgoing_spikes_kernel<<<blocks, threads, 0, stream>>>(
      axon_heads, src_indices, dst_ghost_ids, count, sync_batch_ticks,
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
__global__ void sort_and_prune_kernel(SoA_State state, uint32_t padded_n,
                                      int16_t prune_threshold) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

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
void launch_sort_and_prune(uint32_t padded_n, uint32_t *dendrite_targets,
                           int16_t *dendrite_weights, uint8_t *dendrite_timers,
                           int16_t prune_threshold, cudaStream_t stream) {
  uint32_t threads = 32;
  uint32_t blocks = (padded_n + threads - 1) / threads;
  if (blocks == 0 && padded_n > 0)
    blocks = 1;

  SoA_State state;
  state.dendrite_targets = dendrite_targets;
  state.dendrite_weights = dendrite_weights;
  state.dendrite_timers = dendrite_timers;

  sort_and_prune_kernel<<<blocks, threads, 0, stream>>>(state, padded_n,
                                                        prune_threshold);
}
}
