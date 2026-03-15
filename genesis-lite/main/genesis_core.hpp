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

// Строго 64 байта (1 кэш-линия)
struct alignas(64) VariantParameters {
    int32_t threshold;
    int32_t rest_potential;
    int32_t leak_rate;
    int32_t homeostasis_penalty;
    uint16_t homeostasis_decay;
    int16_t gsop_potentiation;
    int16_t gsop_depression;
    uint8_t refractory_period;
    uint8_t synapse_refractory_period;
    uint8_t slot_decay_ltm;
    uint8_t slot_decay_wm;
    uint8_t signal_propagation_length;
    uint8_t d1_affinity;
    uint16_t heartbeat_m;
    uint8_t d2_affinity;
    uint8_t ltm_slot_count;
    int16_t inertia_curve[15]; // !! Changed to 15 elements to match specification
    int16_t prune_threshold;   // 2 bytes
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
    BurstHeads8* axon_heads;
};

// [DOD] Сетевой пакет (8 байт)
struct alignas(8) SpikeEvent {
    uint32_t ghost_id;
    uint32_t tick_offset;
};

// [DOD FIX] Control Plane для ESP-NOW
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
