use genesis_core::layout::VramState;
use std::ffi::c_void;

/// Опак-тип для CUDA Stream. В Rust мы не знаем его структуру, просто таскаем указатель.
pub type CudaStream = *mut c_void;

// =============================================================================
// § ShardVramPtrs — Абсолютный источник правды для VRAM-раскладки шарда.
//
// ЗАКОН: Порядок полей ЗАПРЕЩЕНО менять. Он определяет побайтовую раскладку
// бинарного блоба .state, который baker пишет на диск, а compute заливает через
// один cudaMemcpyAsync. Нарушение → Silent Data Corruption.
//
// Размеры массивов (N = padded_n, кратно 32):
//   soma_voltage       [N]     i32   | 4N bytes
//   soma_flags         [N]     u8    | 1N bytes
//   threshold_offset   [N]     i32   | 4N bytes
//   timers             [N]     u8    | 1N bytes
//   soma_to_axon       [N]     u32   | 4N bytes
//   dendrite_targets   [128*N] u32   | 512N bytes
//   dendrite_weights   [128*N] i16   | 256N bytes
//   dendrite_timers    [128*N] u8    | 128N bytes
//   axon_heads         [A]     u32   | 4A bytes  (A = total_axons)
// =============================================================================
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShardVramPtrs {
    // --- Soma State (Длина = padded_n) ---
    pub soma_voltage:      *mut i32,   // GLIF потенциал
    pub soma_flags:        *mut u8,    // [7:4] Type | [0] is_spiking
    pub threshold_offset:  *mut i32,   // Гомеостаз (накапливаемый штраф)
    pub timers:            *mut u8,    // Рефрактерный счётчик
    pub soma_to_axon:      *mut u32,   // Индекс первого аксона (в axon_heads)

    // --- Columnar Dendrites (Длина = padded_n * MAX_DENDRITES) ---
    pub dendrite_targets:  *mut u32,   // Packed: DenseID + SegmentOffset
    pub dendrite_weights:  *mut i16,   // Синаптический вес [-32768..+32767]
    pub dendrite_timers:   *mut u8,    // Синаптическая рефрактерность

    // --- Axon Heads (Длина = total_axons: Local + Ghost + Virtual) ---
    pub axon_heads:        *mut genesis_core::layout::BurstHeads8,
}

unsafe impl Send for ShardVramPtrs {}
unsafe impl Sync for ShardVramPtrs {}

/// Параметры физики для одного типа (варианта) нейронов.
/// [ЗАКОН]: Размер ДОЛЖЕН быть строго 128 байт для выравнивания в Constant Memory.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct VariantParameters {
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_rate: i32,
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: i32,
    pub gsop_potentiation: i32,
    pub gsop_depression: i32,
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,
    pub slot_decay_ltm: u8,
    pub slot_decay_wm: u8,
    pub signal_propagation_length: u8,
    pub ltm_slot_count: u8,
    pub heartbeat_m: u16,      // [DOD FIX] DDS Multiplier
    pub inertia_curve: [i16; 16],
    pub prune_threshold: i16,
    pub _pad2a: [u8; 32],
    pub _pad2b: [u8; 26],
}

#[cfg_attr(not(feature = "mock-gpu"), link(name = "genesis_cuda", kind = "static"))]
extern "C" {
    // =====================================================================
    // § Новый Zero-Cost контракт (cu_* функции)
    // =====================================================================

    /// Аллоцирует VRAM для одного шарда и записывает указатели в out_vram.
    /// Возвращает 0 при успехе, ненулевой cudaError_t при ошибке.
    pub fn cu_allocate_shard(
        padded_n:    u32,
        total_axons: u32,
        out_vram:    *mut ShardVramPtrs,
    ) -> i32;

    /// Zero-Cost DMA Upload: один cudaMemcpyAsync для всего .state блоба.
    /// state_blob — плоский массив байт в порядке полей ShardVramPtrs.
    /// Возвращает 0 при успехе.
    pub fn cu_upload_state_blob(
        vram:        *const ShardVramPtrs,
        state_blob:  *const c_void,
        state_size:  usize,
    ) -> i32;

    /// Загружает .axons блоб (плоский [total_axons]u32) в axon_heads.
    /// Возвращает 0 при успехе.
    pub fn cu_upload_axons_blob(
        vram:        *const ShardVramPtrs,
        axons_blob:  *const c_void,
        axons_size:  usize,
    ) -> i32;

    /// Освобождает все VRAM-буферы, ассоциированные с шардом.
    pub fn cu_free_shard(vram: *mut ShardVramPtrs);

    /// Day Phase Orchestrator: запускает 6 ядер асинхронно.
    pub fn cu_step_day_phase(
        vram: *const ShardVramPtrs,
        padded_n: u32,
        total_axons: u32,
        v_seg: u32,
        current_tick: u32, // <--- PLUMB
        input_bitmask: *const u32,
        virtual_offset: u32,
        num_virtual_axons: u32,
        incoming_spikes: *const u32,
        num_incoming_spikes: u32,
        mapped_soma_ids: *const u32,
        output_history: *mut u8,
        num_outputs: u32,
    ) -> i32;

    /// Глобальная константная память GPU (448 байт).
    pub fn cu_upload_constant_memory(lut: *const VariantParameters) -> i32;

    // =====================================================================
    // § Управление памятью и потоками (legacy helpers, используются memory.rs)
    // =====================================================================
    pub fn gpu_malloc(size: usize) -> *mut c_void;
    pub fn gpu_free(dev_ptr: *mut c_void);

    pub fn gpu_host_alloc(size: usize) -> *mut c_void;
    pub fn gpu_host_free(ptr: *mut c_void);

    pub fn gpu_memcpy_host_to_device_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: CudaStream,
    );

    pub fn gpu_memcpy_device_to_host_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: CudaStream,
    );

    pub fn gpu_memcpy_host_to_device(
        dst_dev: *mut c_void,
        src_host: *const c_void,
        size: usize,
    ) -> bool;

    pub fn gpu_memcpy_device_to_host(
        dst_host: *mut c_void,
        src_dev: *const c_void,
        size: usize,
    ) -> bool;

    pub fn gpu_stream_synchronize(stream: CudaStream);
    pub fn gpu_set_device(device_id: i32);
    pub fn gpu_device_synchronize();
    
    /// Barrier: blocks CPU until all previous commands in the default stream are done.
    pub fn gpu_synchronize();
    
    // Загрузка Blueprint-параметров в Constant Memory GPU
    pub fn gpu_load_constants(host_ptr: *const c_void);
    pub fn update_constant_memory_hot_reload(new_variants: *const genesis_core::config::manifest::GpuVariantParameters, stream: CudaStream);
    pub fn update_global_dopamine(dopamine: i16, stream: CudaStream);

    pub fn launch_sort_and_prune(
        ptrs: *const ShardVramPtrs,
        padded_n: u32,
    );
    
    pub fn launch_extract_outgoing_spikes(
        axon_heads: *const genesis_core::layout::BurstHeads8,
        src_indices: *const u32,
        dst_ghost_ids: *const u32,
        count: u32,
        sync_batch_ticks: u32,
        v_seg: u32,
        out_events: *mut c_void,
        out_count: *mut u32,
        stream: CudaStream,
    );
    
    pub fn launch_ghost_sync(
        src_heads: *const genesis_core::layout::BurstHeads8,
        dst_heads: *mut genesis_core::layout::BurstHeads8,
        src_indices: *const u32,
        dst_indices: *const u32,
        count: u32,
        stream: CudaStream,
    );

    // =====================================================================
    // 2. Day Phase Pipeline (6 ядер строго по спецификации Шага 10)
    // =====================================================================

    /// Ядро 1: Инъекция внешних сигналов.
    /// [VramState, bitmask, current_tick, total_virtual_axons]
    pub fn launch_inject_inputs(
        vram: VramState,
        bitmask: *const u32,
        current_tick: u32,
        total_virtual_axons: u32,
    );

    /// Ядро 2: Инъекция сетевых спайков.
    /// [VramState, tick_schedule, tick_spikes_count]
    pub fn launch_apply_spike_batch(
        vram: VramState,
        tick_schedule: *const genesis_core::ipc::SpikeEvent,
        tick_spikes_count: u32,
    );

    /// Ядро 3: Безусловный сдвиг голов всех аксонов.
    pub fn launch_propagate_axons(
        vram: VramState,
        v_seg: u32,
    );

    /// Ядро 4: GLIF Физика, суммация дендритов.
    pub fn launch_update_neurons(
        vram: VramState,
        constants_ptr: *const c_void,
        current_tick: u32,
    );

    /// Ядро 5: Пластичность GSOP.
    pub fn launch_apply_gsop(
        vram: VramState,
    );

    /// Ядро 6: Вывод активности сом (RecordReadout).
    pub fn launch_record_readout(
        vram: VramState,
        mapped_soma_ids: *const u32,
        output_history: *mut u8,
        current_tick: u32,
        total_pixels: u32,
    );

    pub fn gpu_reset_telemetry_count(
        ptrs: *const ShardVramPtrs,
        stream: CudaStream,
    );

    pub fn launch_extract_telemetry(
        ptrs: *const ShardVramPtrs,
        padded_n: u32,
        out_ids: *mut u32,
        out_count_pinned: *mut u32,
        stream: CudaStream,
    );

    pub fn cu_allocate_io_buffers(
        input_words: u32,
        schedule_capacity: u32,
        output_capacity: u32,
        d_input_bitmask: *mut *mut u32,
        d_incoming_spikes: *mut *mut u32,
        d_output_history: *mut *mut u8,
    ) -> i32;

    pub fn cu_dma_h2d_io(
        d_input_bitmask: *mut u32,
        h_input_bitmask: *const u32,
        input_words: u32,
        d_incoming_spikes: *mut u32,
        h_incoming_spikes: *const u32,
        schedule_capacity: u32,
    ) -> i32;

    pub fn cu_dma_d2h_io(
        h_output_history: *mut u8,
        d_output_history: *const u8,
        output_capacity: u32,
    ) -> i32;
}
