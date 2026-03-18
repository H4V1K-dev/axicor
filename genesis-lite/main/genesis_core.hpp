#pragma once
#include <stdint.h>
#include <stddef.h>
#include <atomic>

#define MAX_DENDRITE_SLOTS 32
#define AXON_SENTINEL 0x80000000

struct alignas(32) BurstHeads8 {
    uint32_t h0; uint32_t h1; uint32_t h2; uint32_t h3;
    uint32_t h4; uint32_t h5; uint32_t h6; uint32_t h7;
};

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

  uint8_t _pad[6];                          // 58..64
};

// Read-Only топология (Мапится из Flash-памяти)
struct FlashTopology {
    uint32_t* dendrite_targets;
    uint32_t* soma_to_axon; 
};

// Hot State (Лежит в 520 KB SRAM)
struct SramState {
    uint32_t padded_n;
    uint32_t total_axons;

    int32_t* voltage;
    uint8_t* flags;
    int32_t* threshold_offset;
    uint8_t* refractory_timer;

    int16_t* dendrite_weights;
    uint8_t* dendrite_timers; // [DOD FIX] Таймеры синапсов (Columnar)
    BurstHeads8* axon_heads;
};

// [DOD FIX] Заголовок L7 протокола (Spike Batch V2)
struct alignas(16) SpikeBatchHeaderV2 {
    uint32_t src_zone_hash;
    uint32_t dst_zone_hash;
    uint32_t epoch;
    uint32_t is_last;
};

// [DOD] Сетевой пакет (Строго 8 байт)
struct alignas(8) SpikeEvent {
    uint32_t ghost_id;
    uint32_t tick_offset;
};

// [DOD FIX] Control Plane (Строго 8 байт, алиасинг SpikeEvent)
#define CTRL_MAGIC_DOPA 0x41504F44 // "DOPA" в Little-Endian

struct alignas(8) ControlPacket {
    uint32_t magic;     // Обязано быть CTRL_MAGIC_DOPA
    int16_t dopamine;   // Инъекция R-STDP (-32768..32767)
    uint16_t _pad;      // Выравнивание до 8 байт
};

// [DOD] Телеметрия для совместимости с live_dashboard.py (16 bytes: <Ifff)
struct alignas(16) DashboardFrame {
    uint32_t episode;
    float score;
    float tps;
    float is_done;
};

#define SPIKE_QUEUE_SIZE 256

// [DOD] Zero-Lock SPSC Queue (Core 0 -> Core 1)
struct alignas(32) LockFreeSpikeQueue {
    // Разносим head и tail по разным кэш-линиям процессора Xtensa (False Sharing protection)
    alignas(32) std::atomic<uint32_t> head{0};
    alignas(32) std::atomic<uint32_t> tail{0};
    SpikeEvent buffer[SPIKE_QUEUE_SIZE];

    void clear() {
        // [DOD FIX] Lock-free сброс: подтягиваем tail к head (Biological Amnesia)
        tail.store(head.load(std::memory_order_relaxed), std::memory_order_release);
    }

    bool push(const SpikeEvent& ev) {
        uint32_t curr_head = head.load(std::memory_order_relaxed);
        uint32_t next_head = (curr_head + 1) % SPIKE_QUEUE_SIZE;
        if (next_head == tail.load(std::memory_order_acquire)) return false; // Full
        buffer[curr_head] = ev;
        head.store(next_head, std::memory_order_release);
        return true;
    }

    bool pop(SpikeEvent& ev) {
        uint32_t curr_tail = tail.load(std::memory_order_relaxed);
        if (curr_tail == head.load(std::memory_order_acquire)) return false; // Empty
        ev = buffer[curr_tail];
        tail.store((curr_tail + 1) % SPIKE_QUEUE_SIZE, std::memory_order_release);
        return true;
    }
};

// [DOD FIX] Night Phase: Сортировка и прунинг синапсов (Core 1)
void sort_and_prune_kernel(SramState& sram, FlashTopology& flash, int16_t global_prune_threshold);
