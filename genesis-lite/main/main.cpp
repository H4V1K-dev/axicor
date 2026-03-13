#include <stdio.h>
#include <stdlib.h>
#include <math.h> // Для эмуляции датчика
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include <inttypes.h>
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "genesis_core.hpp"

VariantParameters VARIANT_LUT[2];
SramState sram;
FlashTopology flash;
int16_t global_dopamine = 0; 

// [DOD] Zero-Lock Motor Output (Core 1 -> Core 0)
struct alignas(32) MotorOut {
    std::atomic<uint32_t> left{0};
    std::atomic<uint32_t> right{0};
};

MotorOut motors;

// [DOD] Глобальный Ring Buffer между ядрами
LockFreeSpikeQueue rx_queue;

void init_brain(uint32_t num_neurons) {
    sram.padded_n = num_neurons;
    sram.total_axons = num_neurons;

    sram.voltage = (int32_t*)calloc(num_neurons, sizeof(int32_t));
    sram.flags = (uint8_t*)calloc(num_neurons, sizeof(uint8_t));
    sram.threshold_offset = (int32_t*)calloc(num_neurons, sizeof(int32_t));
    sram.refractory_timer = (uint8_t*)calloc(num_neurons, sizeof(uint8_t));
    sram.dendrite_weights = (int16_t*)calloc(num_neurons * MAX_DENDRITE_SLOTS, sizeof(int16_t));
    sram.axon_heads = (BurstHeads8*)calloc(num_neurons, sizeof(BurstHeads8));

    flash.dendrite_targets = (uint32_t*)calloc(num_neurons * MAX_DENDRITE_SLOTS, sizeof(uint32_t));
    flash.soma_to_axon = (uint32_t*)calloc(num_neurons, sizeof(uint32_t));

    for(uint32_t i = 0; i < num_neurons; i++) {
        sram.axon_heads[i].h0 = AXON_SENTINEL;
    }

    VARIANT_LUT[0].threshold = 400;
    VARIANT_LUT[0].rest_potential = 0;
    VARIANT_LUT[0].leak_rate = 10;
    VARIANT_LUT[0].refractory_period = 5;
    VARIANT_LUT[0].signal_propagation_length = 3;
    VARIANT_LUT[0].gsop_potentiation = 60;
    VARIANT_LUT[0].gsop_depression = 30;
    VARIANT_LUT[0].ltm_slot_count = 16;
    VARIANT_LUT[0].slot_decay_ltm = 128; 
    VARIANT_LUT[0].slot_decay_wm = 64;   
    VARIANT_LUT[0].d1_affinity = 128;
    VARIANT_LUT[0].d2_affinity = 128;
    for(int i=0; i<15; i++) VARIANT_LUT[0].inertia_curve[i] = 128 - (i * 8);

    printf("🧠 Genesis-Lite: %" PRIu32 " neurons. Memory Split: SRAM / Flash.\n", num_neurons);
}

inline uint32_t check_head_dist(uint32_t head, uint32_t seg_idx, uint32_t prop_len, uint32_t current_min) {
    uint32_t d = head - seg_idx;
    return (d <= prop_len && d < current_min) ? d : current_min;
}

// =========================================================
// CORE 1: Day Phase (HFT Compute)
// =========================================================
void day_phase_task(void *pvParameter) {
    uint32_t tick = 0;
    uint32_t v_seg = 1; 
    SpikeEvent ev;

    while(1) {
        int64_t start_time = esp_timer_get_time();

        // 0. Apply Spike Batch (Zero-Cost Injection)
        while (rx_queue.pop(ev)) {
            if (ev.ghost_id < sram.total_axons) {
                sram.axon_heads[ev.ghost_id].h0 = 0; 
            }
        }

        // 1. Propagate Axons
        for(uint32_t i = 0; i < sram.total_axons; i++) {
            if(sram.axon_heads[i].h0 != AXON_SENTINEL) sram.axon_heads[i].h0 += v_seg;
            if(sram.axon_heads[i].h1 != AXON_SENTINEL) sram.axon_heads[i].h1 += v_seg;
        }

        // 2. Update Neurons (GLIF + Dendrites)
        for(uint32_t tid = 0; tid < sram.padded_n; tid++) {
            uint8_t flags = sram.flags[tid];
            uint8_t variant_id = (flags >> 4) & 0x0F;
            VariantParameters p = VARIANT_LUT[variant_id]; 
            
            if (sram.refractory_timer[tid] > 0) {
                sram.refractory_timer[tid] -= 1;
                sram.flags[tid] &= ~0x01; 
                continue;
            }

            int32_t current_voltage = sram.voltage[tid];
            int32_t i_in = 0;

            for (int slot = 0; slot < MAX_DENDRITE_SLOTS; slot++) {
                uint32_t col_idx = slot * sram.padded_n + tid;
                uint32_t target_packed = flash.dendrite_targets[col_idx];
                if (target_packed == 0) break; 

                uint32_t seg_idx = target_packed >> 24;
                uint32_t axon_id = (target_packed & 0x00FFFFFF) - 1;

                BurstHeads8 h = sram.axon_heads[axon_id];
                uint32_t prop = p.signal_propagation_length;

                bool hit = ((h.h0 - seg_idx) <= prop) || ((h.h1 - seg_idx) <= prop); 
                if (hit) {
                    i_in += sram.dendrite_weights[col_idx];
                }
            }

            current_voltage += i_in;
            int32_t diff = current_voltage - p.rest_potential;
            int32_t sign = (diff > 0) - (diff < 0);
            int32_t leaked_abs = abs(diff) - p.leak_rate;
            leaked_abs = leaked_abs & ~(leaked_abs >> 31); 
            current_voltage = p.rest_potential + (sign * leaked_abs);

            int32_t effective_threshold = p.threshold + sram.threshold_offset[tid];
            if (current_voltage >= effective_threshold) {
                current_voltage = p.rest_potential;
                sram.refractory_timer[tid] = p.refractory_period;
                sram.flags[tid] = (flags & 0xFE) | 0x01; 
                
                sram.axon_heads[tid].h1 = sram.axon_heads[tid].h0;
                sram.axon_heads[tid].h0 = 0; 
            } else {
                sram.flags[tid] &= ~0x01;
            }

            sram.voltage[tid] = current_voltage;
        }

        // 3. Apply GSOP 
        for(uint32_t tid = 0; tid < sram.padded_n; tid++) {
            uint8_t flags = sram.flags[tid];
            if ((flags & 0x01) == 0) continue; 

            uint8_t variant_id = (flags >> 4) & 0x0F;
            VariantParameters p = VARIANT_LUT[variant_id]; 
            int32_t dopamine = global_dopamine;

            for (int slot = 0; slot < MAX_DENDRITE_SLOTS; slot++) {
                uint32_t col_idx = slot * sram.padded_n + tid;
                uint32_t target_packed = flash.dendrite_targets[col_idx];
                if (target_packed == 0) break;

                uint32_t seg_idx = target_packed >> 24;
                uint32_t axon_id = (target_packed & 0x00FFFFFF) - 1;
                BurstHeads8 h = sram.axon_heads[axon_id];
                uint32_t prop = p.signal_propagation_length;

                uint32_t min_dist = 0xFFFFFFFF;
                min_dist = check_head_dist(h.h0, seg_idx, prop, min_dist);
                min_dist = check_head_dist(h.h1, seg_idx, prop, min_dist);

                bool is_active = (min_dist != 0xFFFFFFFF);
                int16_t w = sram.dendrite_weights[col_idx];
                int32_t w_sign = (w >= 0) ? 1 : -1;
                int32_t abs_w = w >= 0 ? w : -w;

                uint32_t rank = abs_w >> 11;
                if (rank > 14) rank = 14;
                int32_t inertia = p.inertia_curve[rank];

                int32_t pot_mod = (dopamine * p.d1_affinity) >> 7;
                int32_t dep_mod = (dopamine * p.d2_affinity) >> 7;

                int32_t raw_pot = p.gsop_potentiation + pot_mod;
                int32_t raw_dep = p.gsop_depression - dep_mod;
                int32_t final_pot = raw_pot & ~(raw_pot >> 31);
                int32_t final_dep = raw_dep & ~(raw_dep >> 31);

                int32_t delta_pot = (final_pot * inertia) >> 7;
                int32_t delta_dep = (final_dep * inertia) >> 7;

                uint32_t cooling_shift = is_active ? (min_dist >> 4) : 0;
                int32_t delta = is_active ? (delta_pot >> cooling_shift) : -delta_dep;

                int32_t decay = (slot >= p.ltm_slot_count) ? p.slot_decay_wm : p.slot_decay_ltm;
                delta = (delta * decay) >> 7;

                int32_t new_abs = abs_w + delta;
                if (new_abs < 1) new_abs = 1;
                if (new_abs > 32767) new_abs = 32767;

                sram.dendrite_weights[col_idx] = (int16_t)(new_abs * w_sign);
            }
        }

        int64_t end_time = esp_timer_get_time();

        // 4. Record Readout (Motor Cortex)
        // Если нейрон 254 выстрелил - даем импульс на левый мотор
        if (sram.flags[254] & 0x01) {
            motors.left.fetch_add(1, std::memory_order_relaxed);
        }
        // Если нейрон 255 выстрелил - даем импульс на правый мотор
        if (sram.flags[255] & 0x01) {
            motors.right.fetch_add(1, std::memory_order_relaxed);
        }

        tick++;
        if (tick % 100 == 0) {
            printf("⚡ Tick %" PRIu32 " | Hot loop time: %" PRId64 " us\n", tick, (int64_t)(end_time - start_time));
        }

        if (tick % 10 == 0) {
            vTaskDelay(1 / portTICK_PERIOD_MS); 
        }
    }
}

// [DOD] Выполняется в контексте задачи Wi-Fi. Никаких аллокаций.
void on_esp_now_recv(const esp_now_recv_info_t *esp_now_info, const uint8_t *data, int data_len) {
    if (data_len == sizeof(SpikeEvent)) {
        SpikeEvent* ev = (SpikeEvent*)data;
        // Мгновенный атомарный вброс в кольцевой буфер
        if (!rx_queue.push(*ev)) {
            // Buffer full - легализованная амнезия (Biological Drop)
        }
    }
}

// =========================================================
// CORE 0: Pro Phase (Sensors I2C / PWM Motors / Network)
// =========================================================
void pro_core_task(void *pvParameter) {
    printf("📡 [Core 0] Initializing ESP-NOW Swarm Protocol...\n");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();

    ret = esp_wifi_init(&cfg);
    if (ret != ESP_OK) {
        printf("⚠️ [Core 0] Wi-Fi init failed (QEMU environment). Falling back to Offline Mode.\n");
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
        ESP_ERROR_CHECK(esp_wifi_start());
        ESP_ERROR_CHECK(esp_now_init());
        ESP_ERROR_CHECK(esp_now_register_recv_cb(on_esp_now_recv));
        printf("✅ [Core 0] ESP-NOW initialized.\n");
    }

    printf("🤖 [Core 0] Hardware I/O Loop Started.\n");

    while(1) {
        // 1. I2C Gyroscope Stub (Генерируем синусоиду наклона робота)
        float t = (float)esp_timer_get_time() / 1000000.0f;
        float angle = sinf(t * 2.0f); // Наклон от -1.0 до 1.0

        // 2. Population Coding (Сенсорный энкодер: float -> spikes)
        // Проецируем наклон на 10 рецепторных нейронов (аксоны 0..9)
        int center_id = (int)((angle + 1.0f) * 4.5f);
        if (center_id < 0) center_id = 0;
        if (center_id > 9) center_id = 9;

        SpikeEvent ev;
        ev.ghost_id = center_id; // Бьем спайком в конкретный сенсор
        ev.tick_offset = 0;
        rx_queue.push(ev);

        // 3. PWM Motor Out (Декодер: spikes -> Duty Cycle)
        // Вычитываем накопленные импульсы от Core 1 и обнуляем счетчик за 1 операцию
        uint32_t p_left = motors.left.exchange(0, std::memory_order_relaxed);
        uint32_t p_right = motors.right.exchange(0, std::memory_order_relaxed);

        if (p_left > 0 || p_right > 0) {
            printf("⚙️ [Motors] Angle: %5.2f | PWM Left: %3" PRIu32 " | PWM Right: %3" PRIu32 "\n", angle, p_left, p_right);
        }

        // Работаем на 20 Гц (типично для сервоприводов)
        vTaskDelay(50 / portTICK_PERIOD_MS); 
    }
}

extern "C" void app_main(void) {
    printf("🚀 Booting Genesis-Lite on FreeRTOS (Dual-Core)...\n");
    init_brain(256);

    // Поднимаем PRO Core (Сеть) на нулевом ядре
    xTaskCreatePinnedToCore(
        pro_core_task, "ProPhase", 4096, NULL, 5, NULL, 0 
    );

    // Поднимаем APP Core (Физика) на первом ядре
    xTaskCreatePinnedToCore(
        day_phase_task, "DayPhase", 8192, NULL, configMAX_PRIORITIES - 1, NULL, 1 
    );
}

