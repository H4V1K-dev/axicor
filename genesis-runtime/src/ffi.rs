use genesis_core::layout::VramState;
use std::ffi::c_void;

/// Опак-тип для CUDA Stream. В Rust мы не знаем его структуру, просто таскаем указатель.
pub type CudaStream = *mut c_void;

#[link(name = "genesis_cuda", kind = "static")]
extern "C" {
    // =====================================================================
    // 1. Управление памятью и потоками (Zero-Copy DMA)
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
    pub fn gpu_device_synchronize();
    
    // Загрузка Blueprint-параметров в Constant Memory GPU
    pub fn gpu_load_constants(host_ptr: *const c_void);

    pub fn launch_sort_and_prune(
        padded_n: u32,
        targets: *mut c_void,
        weights: *mut c_void,
        timers: *mut c_void,
        prune_threshold: i16,
        stream: CudaStream,
    );
    
    pub fn launch_extract_outgoing_spikes(
        axon_heads: *const u32,
        src_indices: *const u32,
        dst_ghost_ids: *const u32,
        count: u32,
        sync_batch_ticks: u32,
        out_events: *mut c_void,
        out_count: *mut u32,
        stream: CudaStream,
    );
    
    pub fn launch_ghost_sync(
        src_heads: *const u32,
        dst_heads: *mut u32,
        src_indices: *const u32,
        dst_indices: *const u32,
        count: u32,
        stream: CudaStream,
    );

    // =====================================================================
    // 2. Day Phase Pipeline (6 ядер строго по спецификации)
    // =====================================================================

    /// Ядро 1: Инъекция внешних сигналов в Virtual Axons (Single-Tick Pulse)
    pub fn launch_inject_inputs(
        vram: *const VramState,
        virtual_offset: u32,
        current_tick_in_batch: u32,
        input_stride: u8,
        total_virtual_axons: u32,
        stream: CudaStream,
    );

    /// Ядро 2: Инъекция сетевых спайков из расписания (Strict BSP)
    pub fn launch_apply_spike_batch(
        vram: *const VramState,
        schedule_buffer: *const u32,
        counts: *const u32,
        current_tick: u32,
        max_spikes_per_tick: u32,
        stream: CudaStream,
    );

    /// Ядро 3: Безусловный сдвиг голов всех аксонов (Local + Ghost + Virtual)
    pub fn launch_propagate_axons(
        vram: *const VramState,
        total_axons: u32,
        v_seg: u32,
        stream: CudaStream,
    );

    /// Ядро 4: GLIF Физика, суммация дендритов, срыв спайков
    pub fn launch_update_neurons(
        vram: *const VramState,
        padded_n: u32,
        stream: CudaStream,
    );

    /// Ядро 5: Пластичность GSOP (Timer-as-Contact-Flag)
    pub fn launch_apply_gsop(
        vram: *const VramState,
        padded_n: u32,
        stream: CudaStream,
    );

    /// Ядро 6: Вывод активности сом в Output_History (Z-Sort Population Code)
    pub fn launch_record_readout(
        vram: *const VramState,
        mapped_soma_ids: *const u32, // Указатель на массив ID сом, привязанных к пикселям
        num_output_channels: u32,
        current_tick_in_batch: u32,
        stream: CudaStream,
    );

    pub fn gpu_reset_telemetry_count(
        vram: *const genesis_core::layout::VramState,
        stream: CudaStream,
    );
}
