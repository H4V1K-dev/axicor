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
  uint8_t* __restrict__ dendrite_timers;
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
// Строго 64 байта (1 кэш-линия L1)
struct alignas(64) VariantParameters {
  int32_t threshold;                        // 0..4
  int32_t rest_potential;                   // 4..8
  int32_t leak_rate;                        // 8..12
  int32_t homeostasis_penalty;              // 12..16
  uint32_t spontaneous_firing_period_ticks; // 16..20

  uint16_t initial_synapse_weight;          // 20..22
  uint16_t gsop_potentiation;               // 22..24
  uint16_t gsop_depression;                 // 24..26
  uint16_t homeostasis_decay;               // 26..28

  uint8_t refractory_period;                // 28..29
  uint8_t synapse_refractory_period;        // 29..30
  uint8_t signal_propagation_length;        // 30..31
  uint8_t is_inhibitory;                    // 31..32

  uint8_t inertia_curve[16];                // 32..48

  int32_t adaptive_leak_max;                // 48..52
  uint16_t adaptive_leak_gain;              // 52..54
  uint8_t adaptive_mode;                    // 54..55
  uint8_t _leak_pad[3];                     // 55..58

  uint8_t d1_affinity;                      // 58..59
  uint8_t d2_affinity;                      // 59..60
  uint8_t _pad[4];                          // 60..64
};

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
// ============================================================================
__global__ void cu_update_neurons_kernel(ShardVramPtrs vram,
                                         uint32_t padded_n, uint32_t current_tick) {
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

    // [DOD FIX] Subtract 1 to undo +1 from pack_dendrite_target (Zero-Index Trap)
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
  thresh_offset = decayed & ~(decayed >> 31);

  // 4. Adaptive GLIF Leak (Linear subtraction per user spec)
  int32_t current_leak = p.leak_rate;
  if (p.adaptive_mode == 1) {
    int32_t adaptive_add = (thresh_offset * p.adaptive_leak_gain) >> 8;
    current_leak += adaptive_add;

    int32_t over = current_leak - p.adaptive_leak_max;
    current_leak -= over & ~(over >> 31);
  }

  current_voltage += i_in; // Применяем токи синапсов
  int32_t diff = current_voltage - p.rest_potential;
  int32_t sign = (diff > 0) - (diff < 0);
  int32_t abs_diff = diff * sign;

  // Линейное вычитание адаптивной или статической утечки!
  int32_t leaked_abs = abs_diff - current_leak;
  leaked_abs = leaked_abs & ~(leaked_abs >> 31);

  // Возвращаем знак и прибавляем к потенциалу покоя
  current_voltage = p.rest_potential + (sign * leaked_abs);

  int32_t effective_threshold = p.threshold + thresh_offset;
  int32_t is_glif_spiking = (current_voltage >= effective_threshold) ? 1 : 0;

  // Spontaneous Firing (Heartbeat) - Branchless period check
  uint32_t period = p.spontaneous_firing_period_ticks;
  int32_t is_heartbeat = (period > 0 && ((current_tick + (tid * 104729)) % period) == 0) ? 1 : 0;

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
    flags = flags & ~0x01;
  }

  // 8. Запись в VRAM
  vram.soma_voltage[tid] = current_voltage;
  // [DOD FIX] Branchless BDP Counter update
  uint8_t burst_count = (flags >> 1) & 0x07;
  burst_count += final_spike * (burst_count < 7);
  vram.soma_flags[tid] = (flags & 0xF0) | (burst_count << 1) | (uint8_t)final_spike;
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
  if ((flags & 0x01) == 0) return;
  
  // [DOD FIX] Extraction of Burst-Dependent Plasticity multiple
  uint8_t burst_count = (flags >> 1) & 0x07;
  int32_t burst_mult = (burst_count > 0) ? burst_count : 1;

  uint8_t variant_id = (flags >> 4) & 0x0F;
  VariantParameters p = VARIANT_LUT[variant_id];

  for (int i = 0; i < MAX_DENDRITES; i++) {
    uint32_t col_idx = i * padded_n + tid;
    uint32_t target_packed = vram.dendrite_targets[col_idx];

    if (target_packed == 0)
      break; 

    uint32_t target_id = (target_packed & 0x00FFFFFF) - 1;
    uint32_t seg_idx = target_packed >> 24;
    BurstHeads8 b = vram.axon_heads[target_id];
    uint32_t len = p.signal_propagation_length;

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
    if (abs_w < 0) abs_w = -abs_w;

    int32_t rank = abs_w >> 11;
    if (rank > 15) rank = 15;
    int32_t inertia = p.inertia_curve[rank];

    // Dopamine modulation (D1 boosts LTP, D2 suppresses LTD on reward)
    // Integer physics: (int16 * uint8) >> 7
    int32_t pot_mod = ((int32_t)dopamine * (int32_t)p.d1_affinity) >> 7;
    int32_t dep_mod = ((int32_t)dopamine * (int32_t)p.d2_affinity) >> 7;

    int32_t raw_pot = (int32_t)p.gsop_potentiation + pot_mod;
    int32_t raw_dep = (int32_t)p.gsop_depression - dep_mod;

    int32_t final_dep = raw_dep & ~(raw_dep >> 31);

    // [BIT ACCURACY FIX] Multiplication BEFORE shift strictly exactly like CUDA
    int32_t delta_pot = (raw_pot * inertia * burst_mult) >> 7;
    int32_t delta_dep = (final_dep * inertia * burst_mult) >> 7;

    uint32_t cooling_shift = is_active ? (min_dist >> 4) : 0;

    int32_t delta = is_active ? (delta_pot >> cooling_shift) : -delta_dep;

    int32_t decay = 128;
    delta = (delta * decay) >> (7 + cooling_shift);

    int32_t new_abs = abs_w + delta;
    new_abs = new_abs & ~(new_abs >> 31);

    if (new_abs > 32767) new_abs = 32767;

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

    bool is_spiking = false;
    if (tid < padded_n) {
        is_spiking = (soma_flags[tid] & 0x01) != 0;
    }

    uint64_t active_mask = __ballot(is_spiking);
    uint32_t warp_pop = __popcll(active_mask);

    uint32_t warp_offset = 0;
    if (lane == 0 && warp_pop > 0) {
        warp_offset = atomicAdd(out_count, warp_pop);
    }

    warp_offset = __shfl(warp_offset, 0);

    if (is_spiking) {
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
                          const uint32_t *input_bitmask,
                          uint32_t virtual_offset, uint32_t num_virtual_axons,
                          const uint32_t *incoming_spikes,
                          uint32_t num_incoming_spikes,
                          const uint32_t *mapped_soma_ids,
                          uint8_t *output_history, uint32_t num_outputs,
                          int16_t dopamine,
                          hipStream_t stream) {
  int threads = 256;
  int blocks_n = (padded_n + threads - 1) / threads;

  cu_reset_burst_counters_kernel<<<blocks_n, threads, 0, stream>>>(*vram, padded_n);

  if (num_virtual_axons > 0 && input_bitmask != nullptr) {
    int blocks_in = (num_virtual_axons + threads - 1) / threads;
    cu_inject_inputs_kernel<<<blocks_in, threads, 0, stream>>>(
        vram->axon_heads, input_bitmask, virtual_offset, num_virtual_axons);
  }

  if (num_incoming_spikes > 0 && incoming_spikes != nullptr) {
    int blocks_spikes = (num_incoming_spikes + threads - 1) / threads;
    cu_apply_spike_batch_kernel<<<blocks_spikes, threads, 0, stream>>>(
        vram->axon_heads, incoming_spikes, num_incoming_spikes, total_axons);
  }

  int blocks_axons = (total_axons + threads - 1) / threads;
  cu_propagate_axons_kernel<<<blocks_axons, threads, 0, stream>>>(vram->axon_heads,
                                                       total_axons, v_seg);

  int blocks_neurons = (padded_n + threads - 1) / threads;
  cu_update_neurons_kernel<<<blocks_neurons, threads, 0, stream>>>(*vram, padded_n, current_tick);

  cu_apply_gsop_kernel<<<blocks_neurons, threads, 0, stream>>>(*vram, padded_n, dopamine);

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

int32_t cu_upload_constant_memory(const VariantParameters *lut) {
    hipError_t err = hipMemcpyToSymbol(HIP_SYMBOL(VARIANT_LUT), lut, sizeof(VariantParameters) * 16, 0, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        void* ptr = nullptr;
        if (hipGetSymbolAddress(&ptr, HIP_SYMBOL(VARIANT_LUT)) == hipSuccess || hipGetSymbolAddress(&ptr, VARIANT_LUT) == hipSuccess) {
            err = hipMemcpy(ptr, lut, sizeof(VariantParameters) * 16, hipMemcpyHostToDevice);
        }
    }
    return (int32_t)err;
}

void gpu_load_constants(const void *host_ptr) {
    (void)hipMemcpyToSymbol(HIP_SYMBOL(VARIANT_LUT), host_ptr, 1024, 0, hipMemcpyHostToDevice);
}

void update_constant_memory_hot_reload(const VariantParameters *new_variants, hipStream_t stream) {
    (void)hipMemcpyToSymbolAsync(HIP_SYMBOL(VARIANT_LUT), new_variants, sizeof(VariantParameters) * 16, 0, hipMemcpyHostToDevice, stream);
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
