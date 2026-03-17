#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// Дублируем контракт памяти из bindings.cu
struct alignas(32) BurstHeads8 {
  uint32_t h0; uint32_t h1; uint32_t h2; uint32_t h3;
  uint32_t h4; uint32_t h5; uint32_t h6; uint32_t h7;
};

struct ShardVramPtrs {
  int32_t* __restrict__ soma_voltage;
  uint8_t* __restrict__ soma_flags;
  int32_t* __restrict__ threshold_offset;
  uint8_t* __restrict__ timers;
  uint32_t* __restrict__ soma_to_axon;
  uint32_t* __restrict__ dendrite_targets;
  int16_t* __restrict__ dendrite_weights;
  BurstHeads8* __restrict__ axon_heads;
};

#define AXON_SENTINEL 0x80000000

__global__ void cu_reset_burst_counters_kernel(ShardVramPtrs vram, uint32_t padded_n) {
    uint32_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (tid >= padded_n) return;
    // Сохраняем Type ID (0xF0) и флаг спайка (0x01), обнуляем биты [3:1]
    vram.soma_flags[tid] &= 0xF1;
}

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

#define MAX_DENDRITES 128

// Строго 64 байта (1 кэш-линия L1). 16 типов = 1024 байта в Constant Memory.
struct alignas(64) VariantParameters {
  int32_t threshold;                  // 0..4
  int32_t rest_potential;             // 4..8
  int32_t leak_rate;                  // 8..12
  int32_t homeostasis_penalty;        // 12..16
  uint16_t homeostasis_decay;         // 16..18
  int16_t gsop_potentiation;          // 18..20
  int16_t gsop_depression;            // 20..22
  uint8_t refractory_period;          // 22..23
  uint8_t synapse_refractory_period;  // 23..24
  uint8_t slot_decay_ltm;             // 24..25
  uint8_t slot_decay_wm;              // 25..26
  uint8_t signal_propagation_length;  // 26..27
  uint8_t d1_affinity;                // 27..28 [D1 Рецептор]
  uint16_t heartbeat_m;               // 28..30
  uint8_t d2_affinity;                // 30..31 [D2 Рецептор]
  uint8_t ltm_slot_count;             // 31..32
  uint8_t inertia_curve[15];          // 32..47 (15 bytes, was i16)
  uint8_t _pad_inertia;               // 47..48 (i16 alignment)
  int16_t prune_threshold;            // 48..50
  uint8_t adaptive_leak_mode;         // 49..50
  uint8_t _pad_adaptive;              // 50..51
  int16_t dopamine_leak_gain;         // 51..53
  int16_t burst_leak_gain;            // 53..55
  int16_t leak_min;                   // 55..57
  int16_t leak_max;                   // 57..59
  uint8_t _reserved[4];               // 59..63 (total 64)
};
static_assert(sizeof(VariantParameters) == 64, "VariantParameters must stay 64 bytes");

// Глобальная константная память. Rust будет заливать сюда конфиг перед стартом.
__constant__ VariantParameters VARIANT_LUT[16];

// ============================================================================
// 1. Inject Inputs Kernel (Virtual Axons)
// Извлекает биты из плотной маски и сбрасывает головы виртуальным аксонам
// ============================================================================
__global__ void cu_inject_inputs_kernel(BurstHeads8* __restrict__ axon_heads,
                                        const uint32_t* __restrict__ input_bitmask,
                                        uint32_t virtual_offset,
                                        uint32_t num_virtual_axons) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_virtual_axons)
    return;

  // Извлечение бита за 2 такта ALU (деление на 32 компилятор оптимизирует в
  // shift)
  uint32_t word_idx = tid / 32;
  uint32_t bit_idx = tid % 32;
  bool is_active = (input_bitmask[word_idx] >> bit_idx) & 1;

  // Ветвление минимизировано: пишем только если есть пульс
  if (is_active) {
    BurstHeads8 h = axon_heads[virtual_offset + tid];
    push_burst_head(&h);
    axon_heads[virtual_offset + tid] = h;
  }
}

// ============================================================================
// 2. Apply Spike Batch Kernel (Network / Ghost Axons)
// O(1) инъекция сетевых спайков через Sender-Side Mapping
// ============================================================================
__global__ void cu_apply_spike_batch_kernel(BurstHeads8* __restrict__ axon_heads,
                                            const uint32_t* __restrict__ incoming_spikes,
                                            uint32_t num_incoming_spikes,
                                            uint32_t total_axons) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_incoming_spikes)
    return;

  // Sender-Side Mapping гарантирует, что incoming_spikes[tid] — это готовый
  // локальный индекс в массиве axon_heads. Никаких трансляций ID.
  uint32_t ghost_id = incoming_spikes[tid];

  // [DOD FIX] Жесткая защита VRAM от битых сетевых индексов
  if (ghost_id < total_axons) {
    BurstHeads8 h = axon_heads[ghost_id];
    push_burst_head(&h);
    axon_heads[ghost_id] = h;
  }
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
// Milestone 4: continuous or discrete adaptive leak, selected by adaptive_leak_mode
// ============================================================================
__global__ void cu_update_neurons_kernel(ShardVramPtrs vram,
                                         uint32_t padded_n, uint32_t current_tick,
                                         int16_t dopamine) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t flags = vram.soma_flags[tid];
  uint8_t timer = vram.timers[tid];

  flags &= ~0x01;

  if (timer > 0) {
    vram.timers[tid] = timer - 1;
    vram.soma_flags[tid] = flags;
    return;
  }

  uint8_t variant_id = (flags >> 4) & 0x0F;
  VariantParameters p = VARIANT_LUT[variant_id];

  int32_t current_voltage = vram.soma_voltage[tid];
  int32_t i_in = 0;

  for (int i = 0; i < MAX_DENDRITES; i++) {
    uint32_t col_idx = i * padded_n + tid;
    uint32_t target_packed = vram.dendrite_targets[col_idx];

    if (target_packed == 0)
      break;

    // [DOD FIX] Subtract 1 to undo +1 from pack_dendrite_target (Zero-Index
    // Trap)
    uint32_t target_id = (target_packed & 0x00FFFFFF) - 1;
    uint32_t seg_idx = target_packed >> 24;

    BurstHeads8 h = vram.axon_heads[target_id];
    uint32_t prop = p.signal_propagation_length;

    // Branchless 8-head bitwise OR
    bool hit = ((h.h0 - seg_idx) < prop) | ((h.h1 - seg_idx) < prop) |
               ((h.h2 - seg_idx) < prop) | ((h.h3 - seg_idx) < prop) |
               ((h.h4 - seg_idx) < prop) | ((h.h5 - seg_idx) < prop) |
               ((h.h6 - seg_idx) < prop) | ((h.h7 - seg_idx) < prop);

    if (hit) {
      i_in += (int32_t)vram.dendrite_weights[col_idx];
    }
  }

  // [DOD FIX] Branchless Homeostasis Decay (Zero Warp Divergence)
  int32_t thresh_offset = vram.threshold_offset[tid];
  int32_t decayed = thresh_offset - p.homeostasis_decay;
  // Если decayed < 0, Arithmetic shift (>> 31) даст 0xFFFFFFFF.
  // Инверсия (~) даст 0x00000000. В итоге decayed & 0 = 0.
  thresh_offset = decayed & ~(decayed >> 31);

  // 4. GLIF Leak (Двусторонняя утечка к rest_potential)
  current_voltage += i_in; // Применяем токи синапсов

  int32_t diff = current_voltage - p.rest_potential;

  // Branchless извлечение знака: 1 если > 0, -1 если < 0, 0 если 0
  int32_t sign = (diff > 0) - (diff < 0);
  int32_t abs_diff = diff * sign;

  uint8_t burst_count = (flags >> 1) & 0x07;

  // Milestone 4: adaptive leak path selection
  int32_t leak_used = p.leak_rate;
  if (p.adaptive_leak_mode != 0 && p.leak_min < p.leak_max) {
    int32_t lo = p.leak_min;
    int32_t hi = p.leak_max;
    int32_t dopamine_mod = (dopamine * p.dopamine_leak_gain) >> 7;
    int32_t burst_mod = (int32_t)burst_count * p.burst_leak_gain;

    if (p.adaptive_leak_mode == 1) {
      int32_t raw = p.leak_rate + dopamine_mod + burst_mod;
      int32_t clamped_lo = raw - ((raw - lo) & ((raw - lo) >> 31));
      int32_t clamped_hi = clamped_lo + ((hi - clamped_lo) & ((hi - clamped_lo) >> 31));
      leak_used = clamped_hi + ((1 - clamped_hi) & ((clamped_hi - 1) >> 31));
    } else {
      int32_t base_clamped_lo = p.leak_rate - ((p.leak_rate - lo) & ((p.leak_rate - lo) >> 31));
      int32_t base_clamped = base_clamped_lo + ((hi - base_clamped_lo) & ((hi - base_clamped_lo) >> 31));
      int32_t combined_mod = dopamine_mod + burst_mod;
      int32_t band = (hi - lo) >> 2;
      band = band + ((1 - band) & ((band - 1) >> 31));

      int32_t mode = 0; // stable
      if (burst_count >= 4 && burst_mod > 0) {
        mode = 3; // recovery
      } else if (combined_mod <= -band) {
        mode = 1; // responsive
      } else if (combined_mod >= band) {
        mode = 2; // excited
      }

      int32_t responsive = (base_clamped + lo) >> 1;
      int32_t excited = (base_clamped + hi + 1) >> 1;
      leak_used = base_clamped;
      if (mode == 1) {
        leak_used = responsive;
      } else if (mode == 2) {
        leak_used = excited;
      } else if (mode == 3) {
        leak_used = hi;
      }
      leak_used = leak_used + ((1 - leak_used) & ((leak_used - 1) >> 31));
    }
  }

  // Вычитаем утечку из модуля разницы
  int32_t leaked_abs = abs_diff - leak_used;

  // Branchless clamp: не даем модулю уйти ниже 0 (перелет через rest_potential)
  leaked_abs = leaked_abs & ~(leaked_abs >> 31);

  // Возвращаем знак и прибавляем к потенциалу покоя
  current_voltage = p.rest_potential + (sign * leaked_abs);

  int32_t effective_threshold = p.threshold + thresh_offset;
  int32_t is_glif_spiking = (current_voltage >= effective_threshold) ? 1 : 0;

  // DDS Phase Accumulator (§4 neuron_model.md)
  uint32_t phase = (current_tick * p.heartbeat_m + tid * 104729) & 0xFFFF;
  int32_t is_heartbeat = (p.heartbeat_m > 0 && phase < p.heartbeat_m) ? 1 : 0;

  // Итоговый спайк (ГЛИФ ИЛИ Heartbeat)
  int32_t final_spike = is_glif_spiking | is_heartbeat;

  // Мембрана и гомеостаз сбрасываются ТОЛЬКО от GLIF-спайка
  current_voltage = is_glif_spiking * p.rest_potential + (1 - is_glif_spiking) * current_voltage;
  thresh_offset += is_glif_spiking * p.homeostasis_penalty;
  uint8_t new_timer = is_glif_spiking * p.refractory_period + (1 - is_glif_spiking) * vram.timers[tid];

  // 7. Сдвиг голов аксона при спайке (Burst Shift)
  if (final_spike) {
    // [DOD FIX] Bit 0: Instant Spike (GSOP), Bit 1: Batch Accumulator (Sprouting)
    flags = (flags & 0xFC) | 0x03;

    uint32_t my_axon = vram.soma_to_axon[tid];
    if (my_axon != 0xFFFFFFFF) {
      BurstHeads8 h = vram.axon_heads[my_axon];
      push_burst_head(&h);
      vram.axon_heads[my_axon] = h;
    }
  } else {
    // [DOD FIX] Очищаем ТОЛЬКО мгновенный спайк. Аккумулятор (Bit 1) остается жив до Ночи.
    flags = flags & ~0x01;
  }

  // 8. Запись в VRAM
  vram.soma_voltage[tid] = current_voltage;
  uint8_t updated_burst_count = burst_count;
  updated_burst_count += final_spike * (updated_burst_count < 7);
  vram.soma_flags[tid] = (flags & 0xF0) | (updated_burst_count << 1) | (uint8_t)final_spike;
  vram.threshold_offset[tid] = thresh_offset;
  vram.timers[tid] = new_timer;
}

// ============================================================================
// 5. Apply GSOP Kernel (Spatial STDP Plasticity)
// ============================================================================
__global__ void cu_apply_gsop_kernel(ShardVramPtrs vram, uint32_t padded_n, int16_t dopamine) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= padded_n)
    return;

  uint8_t flags = vram.soma_flags[tid];
  if ((flags & 0x01) == 0)
    return;

  uint8_t variant_id = (flags >> 4) & 0x0F;
  VariantParameters p = VARIANT_LUT[variant_id];

  for (int i = 0; i < MAX_DENDRITES; i++) {
    uint32_t col_idx = i * padded_n + tid;
    uint32_t target_packed = vram.dendrite_targets[col_idx];

    if (target_packed == 0)
      break; // Пустые слоты в хвосте

    // [DOD FIX] Subtract 1 to undo +1 from pack_dendrite_target (Zero-Index
    // Trap)
    uint32_t target_id = (target_packed & 0x00FFFFFF) - 1;
    uint32_t seg_idx = target_packed >> 24;
    BurstHeads8 b = vram.axon_heads[target_id];
    uint32_t len = p.signal_propagation_length;

    // Ищем самую свежую (минимальную) дистанцию среди всех голов
    uint32_t min_dist = 0xFFFFFFFF;
    uint32_t d;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        uint32_t head = ((uint32_t*)&b)[k];
        d = head - seg_idx;
        min_dist = min(min_dist, (d < len) ? d : 0xFFFFFFFF);
    }

    bool is_active = (min_dist != 0xFFFFFFFF);

    int16_t w = vram.dendrite_weights[col_idx];
    int16_t sign = (w >= 0) ? 1 : -1;
    int32_t abs_w = (int32_t)w;
    if (abs_w < 0)
      abs_w = -abs_w;

    // 1. Inertia Rank (1 такт, Branchless)
    uint32_t rank = abs_w >> 11;
    if (rank > 14)
      rank = 14;
    int32_t inertia = p.inertia_curve[rank];

    // 2. Asymmetric Dopamine Modulation (D1/D2 Receptors)
    int32_t base_pot = p.gsop_potentiation;
    int32_t base_dep = p.gsop_depression;

    // Сдвиг >> 7 делит на 128 (1.0x множитель)
    int32_t pot_mod = (dopamine * p.d1_affinity) >> 7;
    int32_t dep_mod = (dopamine * p.d2_affinity) >> 7;

    // D1 усиливает LTP при награде. D2 давит LTD при награде (спасает связи).
    int32_t raw_pot = base_pot + pot_mod;
    int32_t raw_dep = base_dep - dep_mod;

    // Causal LTP может инвертироваться в штраф (нет clamp)
    // Anti-causal LTD не может инвертироваться в рост (оставляем clamp)
    int32_t final_dep = raw_dep & ~(raw_dep >> 31);

    int32_t delta_pot = (raw_pot * inertia) >> 7;
    int32_t delta_dep = (final_dep * inertia) >> 7;
    // Экспоненциальный сдвиг. Каждые 16 тиков сила обучения падает вдвое (>> 1)
    uint32_t cooling_shift = is_active ? (min_dist >> 4) : 0;

    // 3. Causal Delta с экспоненциальным остыванием STDP
    int32_t delta = is_active ? (delta_pot >> cooling_shift) : -delta_dep;

    // 4. Slot Decay
    int32_t decay = (i < p.ltm_slot_count) ? p.slot_decay_ltm : p.slot_decay_wm;
    delta = (delta * decay) >> (7 + cooling_shift);

    // 5. Apply & Clamp
    int32_t new_abs = abs_w + delta;

    // [DOD FIX] Branchless clamp(0, val). Если new_abs < 0, сдвиг даст 0xFFFFFFFF, инверсия даст 0.
    // Это предотвращает инверсию знака веса ( Dale's Law Safety ).
    new_abs = new_abs & ~(new_abs >> 31);

    if (new_abs > 32767) {
      new_abs = 32767;
    }

    vram.dendrite_weights[col_idx] = (int16_t)(new_abs * sign);
  }
}

// ============================================================================
// 6. Record Readout Kernel (Output Matrix)
// ============================================================================
__global__ void cu_record_readout_kernel(const uint8_t* __restrict__ soma_flags,
                                         const uint32_t* __restrict__ mapped_soma_ids,
                                         uint8_t* __restrict__ output_history,
                                         uint32_t num_outputs) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_outputs)
    return;

  uint32_t soma_id = mapped_soma_ids[tid];
  uint8_t is_spiking = 0;

  // [DOD] Защита от Memory Out-of-Bounds. Сентинел означает пустой пиксель.
  if (soma_id != 0xFFFFFFFF) {
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
    uint32_t lane = threadIdx.x % 64;

    // 1. Читаем флаг спайка (бит 0)
    bool is_spiking = false;
    if (tid < padded_n) {
        is_spiking = (soma_flags[tid] & 0x01) != 0;
    }

    // 2. Ballot: каждый поток выставляет свой бит в 32-битную маску варпа
    uint64_t active_mask = __ballot(is_spiking);
    uint32_t warp_pop = __popcll(active_mask);

    // 3. Leader (lane 0) делает единственный atomicAdd в глобальную память
    uint32_t warp_offset = 0;
    if (lane == 0 && warp_pop > 0) {
        warp_offset = atomicAdd(out_count, warp_pop);
    }

    // 4. Leader раздает полученный offset всем потокам варпа
    warp_offset = __shfl(warp_offset, 0);

    // 5. Запись ID в плоский массив без коллизий
    if (is_spiking) {
        // Вычисляем локальный индекс потока среди стреляющих (считаем единицы до текущего бита)
        uint32_t local_rank = __popcll(active_mask & ((1ull << lane) - 1));
        out_ids[warp_offset + local_rank] = tid;
    }
}

extern "C" {

// ============================================================================
// Day Phase Orchestrator
// ============================================================================
int32_t cu_step_day_phase(const ShardVramPtrs *vram, uint32_t padded_n,
                          uint32_t total_axons, uint32_t v_seg, uint32_t current_tick,
                          // --- ВХОДЫ (InjectInputs) ---
                          const uint32_t *input_bitmask,
                          uint32_t virtual_offset, uint32_t num_virtual_axons,
                          // --- СЕТЬ (ApplySpikeBatch) ---
                          const uint32_t *incoming_spikes,
                          uint32_t num_incoming_spikes,
                          // --- ВЫХОДЫ (RecordReadout) ---
                          const uint32_t *mapped_soma_ids,
                          uint8_t *output_history, uint32_t num_outputs,
                          int16_t dopamine,
                          hipStream_t stream) {
  int threads = 256;

  // Burst reset выполняется в shard.rs один раз в начале батча, не здесь.
  // Иначе burst_count сбрасывался бы каждый тик и не накапливался.

  // 1. InjectInputs (Только если есть виртуальные аксоны и передана маска)
  if (num_virtual_axons > 0 && input_bitmask != nullptr) {
    int blocks_in = (num_virtual_axons + threads - 1) / threads;
    cu_inject_inputs_kernel<<<blocks_in, threads, 0, stream>>>(
        vram->axon_heads, input_bitmask, virtual_offset, num_virtual_axons);
  }

  // 2. ApplySpikeBatch (Сетевые спайки от соседних зон)
  if (num_incoming_spikes > 0 && incoming_spikes != nullptr) {
    int blocks_spikes = (num_incoming_spikes + threads - 1) / threads;
    cu_apply_spike_batch_kernel<<<blocks_spikes, threads, 0, stream>>>(
        vram->axon_heads, incoming_spikes, num_incoming_spikes, total_axons);
  }

  // 3. PropagateAxons
  int blocks_axons = (total_axons + threads - 1) / threads;
  cu_propagate_axons_kernel<<<blocks_axons, threads, 0, stream>>>(vram->axon_heads,
                                                       total_axons, v_seg);

  // 4. UpdateNeurons (GLIF)
  int blocks_neurons = (padded_n + threads - 1) / threads;
  cu_update_neurons_kernel<<<blocks_neurons, threads, 0, stream>>>(*vram, padded_n, current_tick, dopamine);

  // 5. ApplyGSOP (Пластичность 3D STDP)
  cu_apply_gsop_kernel<<<blocks_neurons, threads, 0, stream>>>(*vram, padded_n, dopamine);

  // 6. RecordReadout
  if (num_outputs > 0 && mapped_soma_ids != nullptr &&
      output_history != nullptr) {
    int blocks_out = (num_outputs + threads - 1) / threads;
    cu_record_readout_kernel<<<blocks_out, threads, 0, stream>>>(
        vram->soma_flags, mapped_soma_ids, output_history, num_outputs);
  }

  return 0;
}

extern "C" void cu_reset_burst_counters(const ShardVramPtrs *vram, uint32_t padded_n, hipStream_t stream) {
  int threads = 256;
  int blocks = (padded_n + threads - 1) / threads;
  cu_reset_burst_counters_kernel<<<blocks, threads, 0, stream>>>(*vram, padded_n);
}

// Позволяет заливать параметры вариантов в константную память GPU
int32_t cu_upload_constant_memory(const VariantParameters *lut) {
    hipError_t err = hipMemcpyToSymbol(HIP_SYMBOL(VARIANT_LUT), lut, sizeof(VariantParameters) * 16, 0, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        void* ptr = nullptr;
        if (hipGetSymbolAddress(&ptr, HIP_SYMBOL(VARIANT_LUT)) == hipSuccess) {
            err = hipMemcpy(ptr, lut, sizeof(VariantParameters) * 16, hipMemcpyHostToDevice);
        } else if (hipGetSymbolAddress(&ptr, VARIANT_LUT) == hipSuccess) {
            err = hipMemcpy(ptr, lut, sizeof(VariantParameters) * 16, hipMemcpyHostToDevice);
        }
    }
    if (err != hipSuccess) {
        printf("HIP ERROR in cu_upload_constant_memory: %d\n", err);
    }
    return err;
}

void gpu_load_constants(const void *host_ptr) {
    if (hipMemcpyToSymbol(HIP_SYMBOL(VARIANT_LUT), host_ptr, 1024, 0, hipMemcpyHostToDevice) != hipSuccess) {
        void* ptr = nullptr;
        if (hipGetSymbolAddress(&ptr, HIP_SYMBOL(VARIANT_LUT)) == hipSuccess || hipGetSymbolAddress(&ptr, VARIANT_LUT) == hipSuccess) {
            (void)hipMemcpy(ptr, host_ptr, 1024, hipMemcpyHostToDevice);
        }
    }
}

void update_constant_memory_hot_reload(const VariantParameters *new_variants, hipStream_t stream) {
    if (hipMemcpyToSymbolAsync(HIP_SYMBOL(VARIANT_LUT), new_variants, sizeof(VariantParameters) * 16, 0, hipMemcpyHostToDevice, stream) != hipSuccess) {
        void* ptr = nullptr;
        if (hipGetSymbolAddress(&ptr, HIP_SYMBOL(VARIANT_LUT)) == hipSuccess || hipGetSymbolAddress(&ptr, VARIANT_LUT) == hipSuccess) {
            (void)hipMemcpyAsync(ptr, new_variants, sizeof(VariantParameters) * 16, hipMemcpyHostToDevice, stream);
        }
    }
}

void gpu_reset_telemetry_count(const ShardVramPtrs *ptrs, hipStream_t stream) {
    (void)ptrs; (void)stream;
}

void launch_extract_telemetry(const ShardVramPtrs *ptrs, uint32_t padded_n, uint32_t *out_ids, uint32_t *out_count_pinned, hipStream_t stream) {
    if (out_ids == nullptr || out_count_pinned == nullptr || ptrs == nullptr) return;
    int threads = 256;
    int blocks = (padded_n + threads - 1) / threads;
    cu_extract_telemetry_kernel<<<blocks, threads, 0, stream>>>(
        ptrs->soma_flags, out_ids, out_count_pinned, padded_n
    );
}

} // extern "C"
