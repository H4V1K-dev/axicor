#include <stdio.h>
#include <stdlib.h>
#include <math.h> // Для эмуляции датчика
#include <algorithm> // Прунинг синапсов
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include <inttypes.h>
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "driver/i2c.h"
#include "driver/ledc.h"
#include "driver/spi_master.h"
#include "lwip/netdb.h"
#include "esp_partition.h"
#include "spi_flash_mmap.h"
#include "genesis_core.hpp"

VariantParameters VARIANT_LUT[2];
SramState sram;
FlashTopology flash;
std::atomic<int16_t> global_dopamine{0}; 
std::atomic<uint32_t> global_tick{0};

spi_device_handle_t tui_spi;
uint8_t* tui_dma_buffer = nullptr; // [DOD FIX] Объявляем глобально!

#define SAFE_CALLOC(n, size) ({ \
    void* ptr = calloc((n), (size)); \
    if (!ptr) { printf("FATAL OOM: size %d\n", (int)((n)*(size))); abort(); } \
    ptr; \
})

// [DOD] Zero-Lock Motor Output (Core 1 -> Core 0)
struct alignas(32) MotorOut {
    std::atomic<uint32_t> left{0};
    std::atomic<uint32_t> right{0};
};

MotorOut motors;

// [DOD] Глобальный Ring Buffer между ядрами
LockFreeSpikeQueue rx_queue;

void init_brain() { // [DOD FIX] Без аргументов!
    // 1. Мапим Flash первым делом
    const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_SPIFFS, "brain_topo");
    if (!part) { printf("FATAL: Partition 'brain_topo' not found! Did you flash it?\n"); abort(); }

    spi_flash_mmap_handle_t mmap_handle;
    const void* map_ptr;
    // [DOD FIX] Мапим только 1МБ вместо 8МБ, чтобы не исчерпать vaddr range на ESP32
    uint32_t map_size = (part->size > 1024*1024) ? 1024*1024 : part->size;
    esp_err_t err = esp_partition_mmap(part, 0, map_size, ESP_PARTITION_MMAP_DATA, &map_ptr, &mmap_handle);
    if (err != ESP_OK) { printf("FATAL: spi_flash_mmap failed with code %d\n", err); abort(); }

    // 2. Читаем C-ABI Заголовок
    uint32_t* header = (uint32_t*)map_ptr;
    if (header[0] != 0x4F504F54) { // Magic "TOPO"
        printf("FATAL: Invalid Magic 0x%08X in Flash. Did you run distill_esp32.py?\n", (unsigned int)header[0]); abort();
    }
    uint32_t num_neurons = header[1];
    printf("🧠 Flash Header OK! Auto-detected Neurons: %lu\n", (unsigned long)num_neurons);

    sram.padded_n = num_neurons;
    sram.total_axons = num_neurons;

    // 3. Выделяем SRAM под реальный размер графа
    sram.voltage = (int32_t*)SAFE_CALLOC(num_neurons, sizeof(int32_t));
    sram.flags = (uint8_t*)SAFE_CALLOC(num_neurons, sizeof(uint8_t));
    sram.threshold_offset = (int32_t*)SAFE_CALLOC(num_neurons, sizeof(int32_t));
    sram.refractory_timer = (uint8_t*)SAFE_CALLOC(num_neurons, sizeof(uint8_t));
    sram.dendrite_weights = (int16_t*)SAFE_CALLOC(num_neurons * MAX_DENDRITE_SLOTS, sizeof(int16_t));
    sram.dendrite_timers = (uint8_t*)SAFE_CALLOC(num_neurons * MAX_DENDRITE_SLOTS, sizeof(uint8_t));
    
    // Выравнивание строго по 32 байтам для векторных инструкций Xtensa
    sram.axon_heads = (BurstHeads8*)heap_caps_aligned_calloc(
        32, num_neurons, sizeof(BurstHeads8), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL
    );
    if (!sram.axon_heads) { printf("FATAL OOM: axon_heads\n"); abort(); }

    // 4. Сдвигаем указатели Flash на 64 байта (пропускаем заголовок)
    flash.dendrite_targets = (uint32_t*)((uint8_t*)map_ptr + 64);
    uint32_t targets_size_bytes = num_neurons * MAX_DENDRITE_SLOTS * sizeof(uint32_t);
    flash.soma_to_axon = (uint32_t*)((uint8_t*)map_ptr + 64 + targets_size_bytes);
    
    printf("✅ MMAP Topology loaded from Flash: %d bytes mapped\n", (int)part->size);

    // 5. TUI DMA буфер
    size_t dma_buf_size = 240 * 10 * 2;
    tui_dma_buffer = (uint8_t*)heap_caps_malloc(dma_buf_size, MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
    if (!tui_dma_buffer) { printf("FATAL OOM: TUI DMA\n"); abort(); }

    for(uint32_t i = 0; i < num_neurons; i++) {
        sram.axon_heads[i].h0 = AXON_SENTINEL;
        sram.axon_heads[i].h1 = AXON_SENTINEL;
        sram.axon_heads[i].h2 = AXON_SENTINEL;
        sram.axon_heads[i].h3 = AXON_SENTINEL;
        sram.axon_heads[i].h4 = AXON_SENTINEL;
        sram.axon_heads[i].h5 = AXON_SENTINEL;
        sram.axon_heads[i].h6 = AXON_SENTINEL;
        sram.axon_heads[i].h7 = AXON_SENTINEL;
    }

    VARIANT_LUT[0].threshold = 400;
    VARIANT_LUT[0].rest_potential = 0;
    VARIANT_LUT[0].leak_rate = 10;
    VARIANT_LUT[0].refractory_period = 5;
    VARIANT_LUT[0].signal_propagation_length = 3;
    VARIANT_LUT[0].gsop_potentiation = 60;
    VARIANT_LUT[0].gsop_depression = 30;
    for(int i=0; i<15; i++) VARIANT_LUT[0].inertia_curve[i] = 128 - (i * 8);

    printf("🧠 Genesis-Lite: %" PRIu32 " neurons. Memory Split: SRAM / Flash.\n", num_neurons);
}


// [DOD] Minimal 5x7 Font (ASCII 32-90)
// Инициализация через массив, совместимый с C++ (без десигнаторов)
uint8_t font5x7[91][5] = {0};

void init_font() {
    auto set_char = [](char c, uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3, uint8_t b4) {
        font5x7[(int)c][0] = b0; font5x7[(int)c][1] = b1; font5x7[(int)c][2] = b2;
        font5x7[(int)c][3] = b3; font5x7[(int)c][4] = b4;
    };
    set_char(' ', 0x00, 0x00, 0x00, 0x00, 0x00);
    set_char(':', 0x00, 0x00, 0x24, 0x00, 0x00);
    set_char('-', 0x08, 0x08, 0x08, 0x08, 0x08);
    set_char('0', 0x3E, 0x51, 0x49, 0x45, 0x3E);
    set_char('1', 0x00, 0x42, 0x7F, 0x40, 0x00);
    set_char('2', 0x42, 0x61, 0x51, 0x49, 0x46);
    set_char('3', 0x21, 0x41, 0x45, 0x4B, 0x31);
    set_char('4', 0x18, 0x14, 0x12, 0x7F, 0x10);
    set_char('5', 0x27, 0x45, 0x45, 0x45, 0x39);
    set_char('6', 0x3C, 0x4A, 0x49, 0x49, 0x30);
    set_char('7', 0x01, 0x71, 0x09, 0x05, 0x03);
    set_char('8', 0x36, 0x49, 0x49, 0x49, 0x36);
    set_char('9', 0x06, 0x49, 0x49, 0x29, 0x1E);
    set_char('A', 0x7E, 0x11, 0x11, 0x11, 0x7E);
    set_char('D', 0x7F, 0x41, 0x41, 0x22, 0x1C);
    set_char('L', 0x7F, 0x40, 0x40, 0x40, 0x40);
    set_char('P', 0x7F, 0x09, 0x09, 0x09, 0x06);
    set_char('R', 0x7F, 0x09, 0x19, 0x29, 0x46);
    set_char('S', 0x46, 0x49, 0x49, 0x49, 0x31);
    set_char('T', 0x01, 0x01, 0x7F, 0x01, 0x01);
}

void draw_char(uint8_t* buf, int cursor_x, char c) {
    if (cursor_x < 0 || cursor_x > 234) return;
    if (c < 32 || c > 90) c = ' ';
    for (int cx = 0; cx < 5; cx++) {
        uint8_t line = font5x7[(int)c][cx];
        for (int cy = 0; cy < 7; cy++) {
            int px_idx = (cy * 240 + cursor_x + cx) * 2;
            if (line & 0x01) {
                buf[px_idx] = 0x07;     // Green High
                buf[px_idx + 1] = 0xE0; // Green Low
            } else {
                buf[px_idx] = 0x00;
                buf[px_idx + 1] = 0x00;
            }
            line >>= 1;
        }
    }
}

void draw_number(uint8_t* buf, int& x, int32_t val) {
    if (val < 0) {
        draw_char(buf, x, '-'); x += 6;
        val = -val;
    }
    char digits[10];
    int len = 0;
    if (val == 0) digits[len++] = '0';
    while (val > 0 && len < 10) {
        digits[len++] = '0' + (val % 10);
        val /= 10;
    }
    for (int i = len - 1; i >= 0; i--) {
        draw_char(buf, x, digits[i]);
        x += 6;
    }
}

void tui_render_stats_int(uint8_t* buffer, uint32_t tps, int32_t dopamine, uint32_t m_l, uint32_t m_r) {
    memset(buffer, 0, 240 * 10 * 2);
    int x = 4;
    draw_char(buffer, x, 'T'); x += 6; draw_char(buffer, x, 'P'); x += 6; draw_char(buffer, x, 'S'); x += 6; draw_char(buffer, x, ':'); x += 6;
    draw_number(buffer, x, tps); x += 12;
    draw_char(buffer, x, 'D'); x += 6; draw_char(buffer, x, ':'); x += 6;
    draw_number(buffer, x, dopamine); x += 12;
    draw_char(buffer, x, 'L'); x += 6; draw_char(buffer, x, ':'); x += 6;
    draw_number(buffer, x, m_l); x += 12;
    draw_char(buffer, x, 'R'); x += 6; draw_char(buffer, x, ':'); x += 6;
    draw_number(buffer, x, m_r);
}

// [DOD FIX] Night Phase: Сортировка и прунинг синапсов (Core 1)
void sort_and_prune_kernel(SramState& sram, FlashTopology& flash, int16_t global_prune_threshold) {
    int16_t threshold = global_prune_threshold;
    
    // [DOD FIX] Оптимизация под Columnar Layout (SRAM Locality)
    for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
        for (uint32_t tid = 0; tid < sram.padded_n; tid++) {
            uint32_t col_idx = slot * sram.padded_n + tid;
            uint32_t target = flash.dendrite_targets[col_idx];
            if (target == 0) continue;

            int16_t w = sram.dendrite_weights[col_idx];
            // [DOD FIX] Обнуляем только SRAM. Flash-память остается неизменной.
            // Связь структурно жива, но электрически мертва (Amnesia).
            if (std::abs((int)w) < threshold) {
                sram.dendrite_weights[col_idx] = 0;
            }
        }
    }
}

void day_phase_task(void *pvParameter) {
    uint32_t v_seg = 1; 
    SpikeEvent ev;

    while(1) {
        int64_t start_time = esp_timer_get_time();

        uint32_t tick = global_tick.load(std::memory_order_relaxed);

        // [DOD FIX] Сброс burst_count (биты [3:1]) в начале каждого батча (hardware contract sync)
        for(uint32_t tid = 0; tid < sram.padded_n; tid++) {
            sram.flags[tid] &= 0xF1;
        }

        // [DOD FIX] Night Phase (каждые 100 тиков) — прунинг и сортировка
        if (tick % 100 == 0) {
            sort_and_prune_kernel(sram, flash, 15); // Global Prune Threshold = 15
        }

        // 0. Apply Spike Batch (Zero-Cost Injection)
        while (rx_queue.pop(ev)) {
			if (ev.ghost_id < sram.total_axons) {
				BurstHeads8& ah = sram.axon_heads[ev.ghost_id];
				ah.h7 = ah.h6;
				ah.h6 = ah.h5;
				ah.h5 = ah.h4;
				ah.h4 = ah.h3;
				ah.h3 = ah.h2;
				ah.h2 = ah.h1;
				ah.h1 = ah.h0;
				ah.h0 = 0;
			}
		}

        // 1. Propagate Axons
        for(uint32_t i = 0; i < sram.total_axons; i++) {
            BurstHeads8& ah = sram.axon_heads[i];
			uint32_t mask;

			mask = -(ah.h0 != AXON_SENTINEL); ah.h0 += v_seg & mask;
			mask = -(ah.h1 != AXON_SENTINEL); ah.h1 += v_seg & mask;
			mask = -(ah.h2 != AXON_SENTINEL); ah.h2 += v_seg & mask;
			mask = -(ah.h3 != AXON_SENTINEL); ah.h3 += v_seg & mask;
			mask = -(ah.h4 != AXON_SENTINEL); ah.h4 += v_seg & mask;
			mask = -(ah.h5 != AXON_SENTINEL); ah.h5 += v_seg & mask;
			mask = -(ah.h6 != AXON_SENTINEL); ah.h6 += v_seg & mask;
			mask = -(ah.h7 != AXON_SENTINEL); ah.h7 += v_seg & mask;
        }

        // 2. Update Neurons (GLIF + Dendrites)
        for(uint32_t tid = 0; tid < sram.padded_n; tid++) {
            uint8_t flags = sram.flags[tid];
            uint8_t variant_id = (flags >> 4) & 0x0F;
            
            // [DOD FIX] Zero-Copy Read из L1 Cache (Никаких глубоких копий!)
            const VariantParameters& p = VARIANT_LUT[variant_id];

            if (sram.refractory_timer[tid] > 0) {
                sram.refractory_timer[tid] -= 1;
                sram.flags[tid] &= ~0x01;
                continue;
            }

            int32_t current_voltage = sram.voltage[tid];
            int32_t i_in = 0;

            // [DOD FIX] Выгружаем константу в регистр ДО входа в горячий цикл
            uint32_t prop = p.signal_propagation_length;

            for (int slot = 0; slot < MAX_DENDRITE_SLOTS; ++slot) {
                uint32_t col_idx = slot * sram.padded_n + tid;
                uint32_t target_packed = flash.dendrite_targets[col_idx];

                if (target_packed == 0) break; // Аппаратный Early Exit

                uint32_t seg_idx = target_packed >> 24;
                uint32_t axon_id = (target_packed & 0x00FFFFFF) - 1;
                // [DOD FIX] True Zero-Copy Reference. Нет копирования 32 байт!
                const BurstHeads8& h = sram.axon_heads[axon_id];

                // [DOD FIX] Branchless 8-head Hit Detection (SIMD-like OR)
                uint32_t is_active = 
                    ((h.h0 - seg_idx) <= prop) |
                    ((h.h1 - seg_idx) <= prop) |
                    ((h.h2 - seg_idx) <= prop) |
                    ((h.h3 - seg_idx) <= prop) |
                    ((h.h4 - seg_idx) <= prop) |
                    ((h.h5 - seg_idx) <= prop) |
                    ((h.h6 - seg_idx) <= prop) |
                    ((h.h7 - seg_idx) <= prop);

                int16_t w = sram.dendrite_weights[col_idx];
                
                // [DOD FIX] Branchless 1-cycle ALU masking
                i_in += (w & -(int32_t)is_active);
            }

            current_voltage += i_in;
            int32_t diff = current_voltage - p.rest_potential;
            int32_t sign = (diff > 0) - (diff < 0);
            int32_t abs_mask = diff >> 31;
			int32_t leaked_abs = ((diff ^ abs_mask) - abs_mask) - p.leak_rate;
            leaked_abs = leaked_abs & ~(leaked_abs >> 31); 
            current_voltage = p.rest_potential + (sign * leaked_abs);

            int32_t effective_threshold = p.threshold + sram.threshold_offset[tid];
            if (current_voltage >= effective_threshold) {
                current_voltage = p.rest_potential;
                sram.refractory_timer[tid] = p.refractory_period;
				
                uint8_t burst_count = (flags >> 1) & 0x07;
                burst_count += (burst_count < 7);
                sram.flags[tid] = (flags & 0xF0) | (burst_count << 1) | 0x01;
                
                BurstHeads8& ah = sram.axon_heads[tid];
                ah.h7 = ah.h6;
                ah.h6 = ah.h5;
                ah.h5 = ah.h4;
                ah.h4 = ah.h3;
                ah.h3 = ah.h2;
                ah.h2 = ah.h1;
                ah.h1 = ah.h0;
                ah.h0 = 0;

            } else {
                // [DOD FIX] Снимаем только мгновенный спайк, аккумулятор живет до Ночи
                sram.flags[tid] &= ~0x01;
            }

            sram.voltage[tid] = current_voltage;
        }

        // 3. Apply GSOP 
        for(uint32_t tid = 0; tid < sram.padded_n; tid++) {
            uint8_t flags = sram.flags[tid];
            if ((flags & 0x01) == 0) continue; 

            uint8_t variant_id = (flags >> 4) & 0x0F;
            // [DOD FIX] Zero-Copy
            const VariantParameters& p = VARIANT_LUT[variant_id];
            int32_t dopamine = global_dopamine.load(std::memory_order_relaxed);

            for (int slot = 0; slot < MAX_DENDRITE_SLOTS; slot++) {
                uint32_t col_idx = slot * sram.padded_n + tid;

                // [DOD FIX] Проверка синаптической рефрактерности (O(1) Branchless-friendly)
                if (sram.dendrite_timers[col_idx] > 0) {
                    sram.dendrite_timers[col_idx] -= 1;
                    continue; // Синапс спит, пропускаем тяжелую математику
                }

                uint32_t target_packed = flash.dendrite_targets[col_idx];
                if (target_packed == 0) break;

                // [DOD FIX] Поднимаем чтение веса. Предиктор переходов легко проглотит этот branch.
                int16_t w = sram.dendrite_weights[col_idx];
                if (w == 0) continue; 

                uint32_t seg_idx = target_packed >> 24;
                uint32_t axon_id = (target_packed & 0x00FFFFFF) - 1;
                
                // [DOD FIX] True Zero-Copy для GSOP
                const BurstHeads8& h = sram.axon_heads[axon_id];
                uint32_t prop = p.signal_propagation_length;

                // [DOD FIX] Branchless 8-head Hit Detection
                // Побитовое ИЛИ (|) запрещает компилятору создавать ветвления (jmp/br).
                // Все 8 сравнений выполняются линейно.
                uint32_t is_active = 
                    ((h.h0 - seg_idx) <= prop) |
                    ((h.h1 - seg_idx) <= prop) |
                    ((h.h2 - seg_idx) <= prop) |
                    ((h.h3 - seg_idx) <= prop) |
                    ((h.h4 - seg_idx) <= prop) |
                    ((h.h5 - seg_idx) <= prop) |
                    ((h.h6 - seg_idx) <= prop) |
                    ((h.h7 - seg_idx) <= prop);

                if (is_active) {
                    sram.dendrite_timers[col_idx] = p.synapse_refractory_period;
                }

                int32_t w_sign = (w >= 0) ? 1 : -1;
                int32_t abs_w = w >= 0 ? w : -w;

                uint32_t rank = abs_w >> 11;
                if (rank > 14) rank = 14;
                int32_t inertia = p.inertia_curve[rank];

                int32_t raw_pot = p.gsop_potentiation;
                int32_t raw_dep = p.gsop_depression;
                int32_t final_pot = raw_pot & ~(raw_pot >> 31);
                int32_t final_dep = raw_dep & ~(raw_dep >> 31);

                uint8_t burst_count = (flags >> 1) & 0x07;
                int32_t burst_mult = (burst_count > 0) ? burst_count : 1;

                // [DOD FIX] Умножение до сдвига строго как в CUDA. Нет cooling_shift.
                int32_t delta_pot = (final_pot * inertia * burst_mult) >> 7;
                int32_t delta_dep = (final_dep * inertia * burst_mult) >> 7;

                int32_t delta = is_active ? delta_pot : -delta_dep;

                // Fixed Slot Decay = 1.0x (128)
                delta = (delta * 128) >> 7;

                int32_t new_abs = abs_w + delta;
                
                // Branchless clamp to 0..32767
                new_abs = new_abs & ~(new_abs >> 31);
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

        global_tick.fetch_add(1, std::memory_order_relaxed);
        if (tick % 100 == 0) {
            printf("⚡ Tick %" PRIu32 " | Hot loop time: %" PRId64 " us\n", tick, (int64_t)(end_time - start_time));
        }
    }
}

// [DOD] Hardware Pins & Config
#define I2C_MASTER_SCL_IO 22
#define I2C_MASTER_SDA_IO 21
#define I2C_MASTER_NUM I2C_NUM_0
#define MPU6050_ADDR 0x68

#define MOTOR_LEFT_PIN 18
#define MOTOR_RIGHT_PIN 19

#define TUI_SPI_HOST    SPI2_HOST
#define TUI_PIN_MOSI    23
#define TUI_PIN_SCLK    18
#define TUI_PIN_CS      5

void init_hardware() {
    printf("⚙️ [Hardware] Initializing I2C & PWM...\n");
    // 1. I2C Master Init
    i2c_config_t conf = {};
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = (gpio_num_t)I2C_MASTER_SDA_IO;
    conf.scl_io_num = (gpio_num_t)I2C_MASTER_SCL_IO;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = 400000;
    i2c_param_config(I2C_MASTER_NUM, &conf);
    i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);

    // Выводим MPU6050 из спящего режима
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MPU6050_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, 0x6B, true); // PWR_MGMT_1
    i2c_master_write_byte(cmd, 0x00, true); // Wake up
    i2c_master_stop(cmd);
    i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);

    // 2. LEDC (PWM) Init for Motors/Servos (50 Hz, 13-bit)
    ledc_timer_config_t ledc_timer = {};
    ledc_timer.speed_mode       = LEDC_LOW_SPEED_MODE;
    ledc_timer.duty_resolution  = LEDC_TIMER_13_BIT;
    ledc_timer.timer_num        = LEDC_TIMER_0;
    ledc_timer.freq_hz          = 50;
    ledc_timer.clk_cfg          = LEDC_AUTO_CLK;
    ledc_timer_config(&ledc_timer);

    ledc_channel_config_t ledc_ch[2] = {
        { .gpio_num = MOTOR_LEFT_PIN,  .speed_mode = LEDC_LOW_SPEED_MODE, .channel = LEDC_CHANNEL_0, .intr_type = LEDC_INTR_DISABLE, .timer_sel = LEDC_TIMER_0, .duty = 0, .hpoint = 0, .flags = { .output_invert = 0 } },
        { .gpio_num = MOTOR_RIGHT_PIN, .speed_mode = LEDC_LOW_SPEED_MODE, .channel = LEDC_CHANNEL_1, .intr_type = LEDC_INTR_DISABLE, .timer_sel = LEDC_TIMER_0, .duty = 0, .hpoint = 0, .flags = { .output_invert = 0 } }
    };
    for (int i = 0; i < 2; i++) {
        ledc_channel_config(&ledc_ch[i]);
    }

    // [DOD FIX] Zero-Init гарантирует отсутствие проблем с порядком полей в C++
    spi_bus_config_t buscfg = {};
    buscfg.mosi_io_num = TUI_PIN_MOSI;
    buscfg.miso_io_num = -1;
    buscfg.sclk_io_num = TUI_PIN_SCLK;
    buscfg.quadwp_io_num = -1;
    buscfg.quadhd_io_num = -1;
    buscfg.max_transfer_sz = 240 * 10 * 2;

    spi_device_interface_config_t devcfg = {};
    devcfg.clock_speed_hz = 20 * 1000 * 1000;
    devcfg.mode = 0;
    devcfg.spics_io_num = TUI_PIN_CS;
    devcfg.queue_size = 7;
    ESP_ERROR_CHECK(spi_bus_initialize(TUI_SPI_HOST, &buscfg, SPI_DMA_CH_AUTO));
    ESP_ERROR_CHECK(spi_bus_add_device(TUI_SPI_HOST, &devcfg, &tui_spi));
}

// [DOD FIX] Integer-only I2C read (accel_x)
int16_t i2c_read_accel_x() {
    uint8_t data[2] = {0};
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MPU6050_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, 0x3B, true); // ACCEL_XOUT_H
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MPU6050_ADDR << 1) | I2C_MASTER_READ, true);
    i2c_master_read(cmd, data, 2, I2C_MASTER_LAST_NACK);
    i2c_master_stop(cmd);
    i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(10));
    i2c_cmd_link_delete(cmd);
    return (int16_t)((data[0] << 8) | data[1]);
}

#define MY_ZONE_HASH 0xDEADBEEF // Заменить на реальный хэш или задать через макрос
#define SYNC_BATCH_TICKS 20     // Размер батча HFT

// =========================================================
// CORE 0: Pro Phase (Sensors I2C / PWM Motors / Network)
// =========================================================
void pro_core_task(void *pvParameter) {
    printf("📡 [Core 0] Initializing LwIP UDP HFT Stack...\n");
    // ... [Omitted nvs/wifi/socket init for brevity, but it's there in the file]

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
        printf("✅ [Core 0] Wi-Fi initialized.\n");
    }

    // 0. Настройка UDP сокета
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(5000);
    bind(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

    init_hardware();
    printf("🤖 [Core 0] Hardware I/O Loop Started.\n");

    static uint32_t last_ui_tick = 0;
    static int64_t last_ui_time = esp_timer_get_time();
    static spi_transaction_t trans;
    static bool trans_in_flight = false;

    while (1) {
        // 1. NON-BLOCKING СЕТЬ (Fast-Path UDP)
        uint8_t rx_buffer[1472]; // Max MTU Ethernet
        struct sockaddr_in source_addr;
        socklen_t socklen = sizeof(source_addr);
        int len = recvfrom(sock, rx_buffer, sizeof(rx_buffer), MSG_DONTWAIT, (struct sockaddr *)&source_addr, &socklen);
        
        if (len >= (int)sizeof(SpikeBatchHeaderV2)) {
            SpikeBatchHeaderV2* header = (SpikeBatchHeaderV2*)rx_buffer;
            
            if (header->dst_zone_hash == MY_ZONE_HASH) {
                uint32_t current_tick_sync = global_tick.load(std::memory_order_relaxed);
                uint32_t current_epoch = current_tick_sync / SYNC_BATCH_TICKS;

                if (header->epoch > current_epoch) {
                    // Self-Healing: Сеть ушла вперед. Сбрасываем старый мусор.
                    rx_queue.clear();
                    global_tick.store(header->epoch * SYNC_BATCH_TICKS, std::memory_order_release);
                    current_epoch = header->epoch;
                }

                if (header->epoch == current_epoch) {
                    int events_len = len - sizeof(SpikeBatchHeaderV2);
                    SpikeEvent* incoming = (SpikeEvent*)(rx_buffer + sizeof(SpikeBatchHeaderV2));
                    int count = events_len / sizeof(SpikeEvent);

                    for (int i = 0; i < count; i++) {
                        if (incoming[i].ghost_id == CTRL_MAGIC_DOPA) {
                            ControlPacket* ctrl = (ControlPacket*)&incoming[i];
                            global_dopamine.store(ctrl->dopamine, std::memory_order_relaxed);
                        } else {
                            rx_queue.push(incoming[i]);
                        }
                    }
                }
            }
        }

        // 2. INTEGER I2C SENSOR & POPULATION ENCODER
        // Опрашиваем гироскоп (accel_x: -16384 .. +16384)
        int16_t accel_x = i2c_read_accel_x(); 
        
        // Переводим в диапазон 0..32768 (Integer Physics)
        int32_t normalized_accel = (int32_t)accel_x + 16384; 
        if (normalized_accel < 0) normalized_accel = 0;
        if (normalized_accel > 32768) normalized_accel = 32768;
        
        // [DOD FIX] Мапим на 10 рецепторных нейронов (0..9) БЕЗ f32
        // (normalized_accel * 10) / 32768  ==  (normalized_accel * 10) >> 15
        uint32_t center_id = (normalized_accel * 10) >> 15;
        if (center_id > 9) center_id = 9;

        SpikeEvent ev = {center_id, 0};
        rx_queue.push(ev);

        // 3. PWM Motor Out (Декодер: rate -> Duty Cycle)
        uint32_t p_left = motors.left.exchange(0, std::memory_order_relaxed);
        uint32_t p_right = motors.right.exchange(0, std::memory_order_relaxed);
        uint32_t duty_left = 400 + (p_left * 8);
        if (duty_left > 800) duty_left = 800;
        uint32_t duty_right = 400 + (p_right * 8);
        if (duty_right > 800) duty_right = 800;
        ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, duty_left);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1, duty_right);
        ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1);

        // 4. LOCK-FREE TUI & DMA (Раз в 500мс)
        int64_t now = esp_timer_get_time();
        if (now - last_ui_time > 500000) {
            uint32_t dt_us = (uint32_t)(now - last_ui_time);
            uint32_t current_tick_ui = global_tick.load(std::memory_order_relaxed);
            
            // [DOD FIX] 64-bit Integer Math. НИКАКИХ FLOAT!
            uint32_t tps_ui = (uint32_t)(((uint64_t)(current_tick_ui - last_ui_tick) * 1000000ULL) / dt_us);
            
            last_ui_tick = current_tick_ui;
            last_ui_time = now;

            // Проверяем статус прошлого кадра
            if (trans_in_flight) {
                spi_transaction_t* ret_trans;
                if (spi_device_get_trans_result(tui_spi, &ret_trans, 0) == ESP_OK) {
                    trans_in_flight = false;
                }
            }

            if (!trans_in_flight) {
                // [DOD FIX] Lock-Free Render stats into DMA buffer
                tui_render_stats_int(
                    tui_dma_buffer, 
                    tps_ui, 
                    global_dopamine.load(std::memory_order_relaxed),
                    motors.left.load(std::memory_order_relaxed),
                    motors.right.load(std::memory_order_relaxed)
                );

                memset(&trans, 0, sizeof(spi_transaction_t));
                trans.length = 240 * 10 * 2 * 8; // В битах
                trans.tx_buffer = tui_dma_buffer;
                
                // Запуск аппаратного DMA без блокировки ядра
                if (spi_device_queue_trans(tui_spi, &trans, 0) == ESP_OK) {
                    trans_in_flight = true;
                }
            }
        }

        // Если сетевых пакетов не было, отдаем 1 тик (1 мс) ОС, чтобы не вешать Watchdog
        // В HFT-режиме это предотвращает голодание других задач FreeRTOS
        // [DOD FIX] Гарантированный возврат кванта планировщику (IDLE0 WDT)
        // При CONFIG_FREERTOS_HZ=1000 это ровно 1 мс.
        if (len <= 0) {
            vTaskDelay(1); // Хардкод 1 тика вместо макроса
        }
        }
        }


extern "C" void app_main(void) {
    // [DOD FIX] Даем UART консоли 2 секунды на подключение, чтобы ловить паники!
    vTaskDelay(pdMS_TO_TICKS(2000));
    printf("🚀 Booting Genesis-Lite on FreeRTOS (Dual-Core)...\n");
    
    init_brain(); // Вызов без хардкода!
    init_font();  // [DOD FIX] Инициализация шрифта

    // Поднимаем PRO Core (Сеть) на нулевом ядре
    xTaskCreatePinnedToCore(
        pro_core_task, "ProPhase", 4096, NULL, 5, NULL, 0 
    );

    // Поднимаем APP Core (Физика) на первом ядре
    xTaskCreatePinnedToCore(
        day_phase_task, "DayPhase", 8192, NULL, configMAX_PRIORITIES - 1, NULL, 1 
    );
}


