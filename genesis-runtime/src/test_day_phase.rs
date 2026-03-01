use genesis_core::layout::{VramState, align_to_warp};
use genesis_core::constants::{MAX_DENDRITE_SLOTS, AXON_SENTINEL};
use genesis_core::config::blueprints::{GenesisConstantMemory, VariantParameters};
use crate::ffi;
use std::ptr;

#[test]
fn test_day_phase_physics_pipeline() {
    unsafe {
        let stream = ptr::null_mut(); // Default synchronous stream для тестов
        
        // 1. Настройка констант (LUT)
        let mut const_mem: GenesisConstantMemory = unsafe { std::mem::zeroed() };
        const_mem.variants[0] = VariantParameters {
            threshold: 10,
            rest_potential: 0,
            leak_rate: 0,
            homeostasis_penalty: 2,
            gsop_potentiation: 100, // Сильное обучение
            gsop_depression: 10,
            homeostasis_decay: 0,
            signal_propagation_length: 5,
            conduction_velocity: 1,
            slot_decay_ltm: 5,
            slot_decay_wm: 10,
            refractory_period: 2,
            synapse_refractory_period: 2,
            inertia_curve: [128; 16], // Коэффициент 1.0 (128/128)
            _reserved: [0; 16],
        };
        ffi::gpu_load_constants(&const_mem as *const _ as *const std::ffi::c_void);

        // 2. Аллокация (1 нейрон, 2 аксона, батч 3 тика)
        let padded_n = align_to_warp(1);
        let total_axons = 32;

        // 3. Формирование начального состояния (Host)
        let mut h_voltage = vec![0i32; padded_n];
        let mut h_flags = vec![0u8; padded_n]; // Тип 0 (биты 4-7 = 0)
        let mut h_threshold_offset = vec![0i32; padded_n];
        let mut h_refractory = vec![0u8; padded_n];
        
        let mut h_soma_to_axon = vec![0u32; padded_n];
        h_soma_to_axon[0] = 0; // Сома 0 владеет Локальным Аксоном 0

        let mut h_targets = vec![0u32; MAX_DENDRITE_SLOTS * padded_n];
        let mut h_weights = vec![0i16; MAX_DENDRITE_SLOTS * padded_n];
        let mut h_timers = vec![0u8; MAX_DENDRITE_SLOTS * padded_n];
        
        // Подключаем дендрит 0 сомы 0 к Виртуальному Аксону 1
        // target_packed: [31..28] Type=0 | [27..8] AxonID=1 | [7..0] SegIdx=0
        h_targets[0] = (0 << 28) | (1 << 8) | 0;
        h_weights[0] = 15; // > threshold, гарантирует спайк

        let mut h_axon_heads = vec![AXON_SENTINEL; total_axons];
        
        // Настройка входов: 3 тика, 1 word (32 бита). 
        // В 0-й тик подаем сигнал (бит 0 = 1). Остальные тики = 0.
        let mut h_input_bitmask = vec![0u32; 3];
        h_input_bitmask[0] = 1; 

        // Настройка выходов: 1 канал смотрит на сому 0
        let h_mapped_soma_ids = vec![0u32; 1];
        let d_mapped_soma_ids = ffi::gpu_malloc(4);
        ffi::gpu_memcpy_host_to_device_async(d_mapped_soma_ids, h_mapped_soma_ids.as_ptr() as *const _, 4, stream);

        // 4. Загрузка в VRAM
        let d_voltage = ffi::gpu_malloc(padded_n * 4);
        let d_flags = ffi::gpu_malloc(padded_n);
        let d_threshold_offset = ffi::gpu_malloc(padded_n * 4);
        let d_refractory_timer = ffi::gpu_malloc(padded_n);
        let d_soma_to_axon = ffi::gpu_malloc(padded_n * 4);
        let d_dendrite_targets = ffi::gpu_malloc(MAX_DENDRITE_SLOTS * padded_n * 4);
        let d_dendrite_weights = ffi::gpu_malloc(MAX_DENDRITE_SLOTS * padded_n * 2);
        let d_dendrite_timers = ffi::gpu_malloc(MAX_DENDRITE_SLOTS * padded_n);
        let d_axon_heads = ffi::gpu_malloc(total_axons * 4);
        let d_input_bitmask = ffi::gpu_malloc(3 * 4);
        let d_output_history = ffi::gpu_malloc(3 * 1);

        ffi::gpu_memcpy_host_to_device_async(d_voltage, h_voltage.as_ptr() as *const _, padded_n * 4, stream);
        ffi::gpu_memcpy_host_to_device_async(d_flags, h_flags.as_ptr() as *const _, padded_n, stream);
        ffi::gpu_memcpy_host_to_device_async(d_threshold_offset, h_threshold_offset.as_ptr() as *const _, padded_n * 4, stream);
        ffi::gpu_memcpy_host_to_device_async(d_refractory_timer, h_refractory.as_ptr() as *const _, padded_n, stream);
        ffi::gpu_memcpy_host_to_device_async(d_soma_to_axon, h_soma_to_axon.as_ptr() as *const _, padded_n * 4, stream);
        ffi::gpu_memcpy_host_to_device_async(d_dendrite_targets, h_targets.as_ptr() as *const _, MAX_DENDRITE_SLOTS * padded_n * 4, stream);
        ffi::gpu_memcpy_host_to_device_async(d_dendrite_weights, h_weights.as_ptr() as *const _, MAX_DENDRITE_SLOTS * padded_n * 2, stream);
        ffi::gpu_memcpy_host_to_device_async(d_dendrite_timers, h_timers.as_ptr() as *const _, MAX_DENDRITE_SLOTS * padded_n, stream);
        ffi::gpu_memcpy_host_to_device_async(d_axon_heads, h_axon_heads.as_ptr() as *const _, total_axons * 4, stream);
        ffi::gpu_memcpy_host_to_device_async(d_input_bitmask, h_input_bitmask.as_ptr() as *const _, 3 * 4, stream);
        
        let vram = VramState {
            voltage: d_voltage as *mut i32,
            flags: d_flags as *mut u8,
            threshold_offset: d_threshold_offset as *mut i32,
            refractory_timer: d_refractory_timer as *mut u8,
            soma_to_axon: d_soma_to_axon as *mut u32,
            dendrite_targets: d_dendrite_targets as *mut u32,
            dendrite_weights: d_dendrite_weights as *mut i16,
            dendrite_timers: d_dendrite_timers as *mut u8,
            axon_heads: d_axon_heads as *mut u32,
            input_bitmask: d_input_bitmask as *mut u32,
            output_history: std::ptr::null_mut(),
            telemetry_spikes: std::ptr::null_mut(),
            telemetry_count: std::ptr::null_mut(),
        };

        ffi::gpu_stream_synchronize(stream);

        // 5. Прогон Батча (Эмуляция execute_day_batch)
        let vram_ptr = &vram as *const VramState;
        let virtual_offset = 1; // Виртуальные аксоны начинаются с индекса 1
        let total_virtual_axons = 1;
        let input_stride = 1;
        let v_seg = 1;
        let num_output_channels = 1;

        for tick in 0..3 {
            ffi::launch_inject_inputs(vram_ptr, virtual_offset, tick, input_stride, total_virtual_axons, stream);
            // Ядро 2 пропускаем (сетевые спайки)
            ffi::launch_propagate_axons(vram_ptr, total_axons as u32, v_seg, stream);
            ffi::launch_update_neurons(vram_ptr, padded_n as u32, stream);
            ffi::launch_apply_gsop(vram_ptr, padded_n as u32, stream);
            ffi::launch_record_readout(vram_ptr, d_mapped_soma_ids as *const u32, num_output_channels, tick, stream);
        }
        ffi::gpu_stream_synchronize(stream);

        // 6. Скачивание результатов (Verify)
        let mut out_history = vec![0u8; 3];
        ffi::gpu_memcpy_device_to_host_async(out_history.as_mut_ptr() as *mut _, vram.output_history as *const _, 3, stream);
        
        let mut out_axons = vec![0u32; total_axons];
        ffi::gpu_memcpy_device_to_host_async(out_axons.as_mut_ptr() as *mut _, vram.axon_heads as *const _, total_axons * 4, stream);

        let mut out_weights = vec![0i16; MAX_DENDRITE_SLOTS * padded_n];
        ffi::gpu_memcpy_device_to_host_async(out_weights.as_mut_ptr() as *mut _, vram.dendrite_weights as *const _, MAX_DENDRITE_SLOTS * padded_n * 2, stream);

        let mut out_thresh_off = vec![0i32; padded_n];
        ffi::gpu_memcpy_device_to_host_async(out_thresh_off.as_mut_ptr() as *mut _, vram.threshold_offset as *const _, padded_n * 4, stream);

        let mut out_voltage = vec![0i32; padded_n];
        ffi::gpu_memcpy_device_to_host_async(out_voltage.as_mut_ptr() as *mut _, vram.voltage as *const _, padded_n * 4, stream);

        let mut out_flags = vec![0u8; padded_n];
        ffi::gpu_memcpy_device_to_host_async(out_flags.as_mut_ptr() as *mut _, vram.flags as *const _, padded_n, stream);

        ffi::gpu_stream_synchronize(stream);
        
        ffi::gpu_free(d_mapped_soma_ids);
        ffi::gpu_free(d_voltage);
        ffi::gpu_free(d_flags);
        ffi::gpu_free(d_threshold_offset);
        ffi::gpu_free(d_refractory_timer);
        ffi::gpu_free(d_soma_to_axon);
        ffi::gpu_free(d_dendrite_targets);
        ffi::gpu_free(d_dendrite_weights);
        ffi::gpu_free(d_dendrite_timers);
        ffi::gpu_free(d_axon_heads);
        ffi::gpu_free(d_input_bitmask);
        ffi::gpu_free(d_output_history);

        // ==========================================
        // 7. ASSERTIONS (Проверка механической симпатии)
        // ==========================================
        
        // Тик 0: Вход = 1 -> Вирт.аксон = 1 -> Нейрон бьет (15 >= 10) -> Лок.аксон = 0 -> Спайк записан.
        // Тик 1: Нейрон в рефрактерности. Аксоны сдвигаются.
        // Тик 2: Нейрон вышел из рефрактерности, но входа нет.
        
        assert_eq!(out_history[0], 1, "Neuron MUST spike at Tick 0");
        assert_eq!(out_history[1], 0, "Neuron MUST be in refractory at Tick 1");
        assert_eq!(out_history[2], 0, "Neuron MUST remain silent at Tick 2");

        // Аксон 0 (Локальный) родился в тик 0. Прошел 2 сдвига (в тике 1 и 2). Значение = 2.
        assert_eq!(out_axons[0], 2, "Local axon head should be 2");
        
        // Аксон 1 (Виртуальный) сброшен InjectInputs в тик 0. Прошел 3 сдвига (в тике 0, 1, 2).
        // В тике 0 InjectInputs ставит 0, затем Propagate дает 1. 
        // Тик 1: +1 -> 2. Тик 2: +1 -> 3.
        assert_eq!(out_axons[1], 3, "Virtual axon head should be 3");

        // Вес 0 прошел GSOP Potentiation в тик 0! 
        // delta = (100 * 128) >> 7 = 100.
        // new_weight = 15 + 100 = 115.
        // decay (LTM slot < 80) = 5. new_weight = 115 - 5 = 110.
        // Тик 1 и 2: Спайка нет, GSOP Early Exit, вес не меняется!
        assert_eq!(out_weights[0], 110, "STDP GSOP MUST potentiate weight correctly");

        // Threshold Penalty
        // В тик 0 спайк дал penalty = 2.
        // Тик 1: decay=0 -> остаток 2. Тик 2: остаток 2.
        assert_eq!(out_thresh_off[0], 2, "Homeostasis penalty MUST be applied");
        
        println!("GPU Pipeline Test Passed with Absolute Mechanical Sympathy!");
    }
}
