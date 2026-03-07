use crate::memory::VramState;
use std::ptr;

pub struct IoDeviceBuffers {
    pub d_input_bitmask: *mut u32,
    pub d_incoming_spikes: *mut u32,
    pub d_output_history: *mut u8,
    pub max_spikes_per_tick: u32,
    pub input_words_per_tick: u32,
    pub num_outputs: u32,
}

unsafe impl Send for IoDeviceBuffers {}
unsafe impl Sync for IoDeviceBuffers {}

/// Главный двигатель шарда. Владеет VRAM и вызывает CUDA-ядра.
/// Этот крейт не знает ничего про файлы или сеть, только про указатели и VRAM.
pub struct ShardEngine {
    pub vram: VramState,
    // В будущем здесь будут cudaStream_t и cudaEvent_t для асинхронности
}

unsafe impl Send for ShardEngine {}
unsafe impl Sync for ShardEngine {}

impl ShardEngine {
    pub fn new(vram: VramState) -> Self {
        Self { vram }
    }

    /// Выполняет весь батч (sync_batch_ticks) автономно на GPU.
    pub fn step_day_phase_batch(
        &mut self,
        sync_batch_ticks: u32,
        io_buffers: &IoDeviceBuffers,
        h_input_bitmask: Option<&[u32]>,
        h_incoming_spikes: Option<&[u32]>, // Плоский schedule.ghost_ids
        h_spike_counts: &[u32],            // schedule.counts
        virtual_offset: u32,
        num_virtual_axons: u32,
        mapped_soma_ids_device: *const u32, // Загружается при старте шарда
        v_seg: u32,
        tick_base: u32, // <--- ADD
    ) {
        // 1. Bulk DMA H2D (Входы и Сетевые Спайки)
        let total_input_words = io_buffers.input_words_per_tick * sync_batch_ticks;
        let total_schedule_capacity = io_buffers.max_spikes_per_tick * sync_batch_ticks;

        unsafe {
            crate::ffi::cu_dma_h2d_io(
                io_buffers.d_input_bitmask,
                h_input_bitmask.map_or(ptr::null(), |s| s.as_ptr()),
                if h_input_bitmask.is_some() { total_input_words } else { 0 },
                io_buffers.d_incoming_spikes,
                h_incoming_spikes.map_or(ptr::null(), |s| s.as_ptr()),
                if h_incoming_spikes.is_some() { total_schedule_capacity } else { 0 },
            );
        }

        // 2. Hot Loop по тикам

        for tick in 0..sync_batch_ticks {
            let global_tick = tick_base + tick; // Compute true time

            // Вычисляем O(1) смещения для указателей на текущий тик
            let tick_input_ptr = if io_buffers.d_input_bitmask.is_null() {
                ptr::null()
            } else {
                unsafe { io_buffers.d_input_bitmask.add((tick * io_buffers.input_words_per_tick) as usize) }
            };

            let tick_spikes_ptr = if io_buffers.d_incoming_spikes.is_null() {
                ptr::null()
            } else {
                unsafe { io_buffers.d_incoming_spikes.add((tick * io_buffers.max_spikes_per_tick) as usize) }
            };

            let tick_output_ptr = if io_buffers.d_output_history.is_null() {
                ptr::null_mut()
            } else {
                unsafe { io_buffers.d_output_history.add((tick * io_buffers.num_outputs) as usize) }
            };

            let current_spikes_count = h_spike_counts[tick as usize];

            // 3. Вызов 6-ядерного конвейера для ОДНОГО тика
            let err = unsafe {
                crate::ffi::cu_step_day_phase(
                    &self.vram.ptrs,
                    self.vram.padded_n,
                    self.vram.total_axons,
                    v_seg,
                    global_tick, // <--- PLUMB
                    tick_input_ptr,
                    virtual_offset,
                    num_virtual_axons,
                    tick_spikes_ptr,
                    current_spikes_count,
                    mapped_soma_ids_device,
                    tick_output_ptr,
                    io_buffers.num_outputs,
                )
            };
            assert_eq!(err, 0, "FATAL: Day Phase Pipeline failed at tick {}", tick);
        }

        // 4. Ждем завершения всего батча (Синхронизация барьера BSP)
        #[cfg(not(feature = "mock-gpu"))]
        unsafe { crate::ffi::gpu_stream_synchronize(std::ptr::null_mut()) };
    }
}
