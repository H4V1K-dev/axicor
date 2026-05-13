use axicor_core::layout::{VariantParameters, VramState};
use std::ffi::c_void;

/// Opaque type for CUDA Stream. In Rust we don't know its structure, we just pass the pointer.
pub type CudaStream = *mut c_void;

// =============================================================================
//  ShardVramPtrs  The absolute source of truth for the shard's VRAM layout.
//
// LAW: The order of fields is FORBIDDEN to change. It defines the byte-for-byte layout
// of the binary .state blob that baker writes to disk and compute uploads via
// a single cudaMemcpyAsync. Violation  Silent Data Corruption.
//
// Array sizes (N = padded_n, multiple of 32):
//   soma_voltage       [N]     i32   | 4N bytes
//   soma_flags         [N]     u8    | 1N bytes
//   threshold_offset   [N]     i32   | 4N bytes
//   timers             [N]     u8    | 1N bytes
//   soma_to_axon       [N]     u32   | 4N bytes
//   dendrite_targets   [128*N] u32   | 512N bytes
//   dendrite_weights   [128*N] i32   | 512N bytes
//   dendrite_timers    [128*N] u8    | 128N bytes
//   axon_heads         [A]     u32   | 4A bytes  (A = total_axons)
// =============================================================================
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShardVramPtrs {
    pub soma_voltage: *mut i32,
    pub soma_flags: *mut u8,
    pub threshold_offset: *mut i32,
    pub timers: *mut u8,
    pub soma_to_axon: *mut u32,
    pub dendrite_targets: *mut u32,
    pub dendrite_weights: *mut i32, // СТРОГО i32 (Mass Domain)
    pub dendrite_timers: *mut u8,
    pub axon_heads: *mut axicor_core::layout::BurstHeads8,
}

unsafe impl Send for ShardVramPtrs {}
unsafe impl Sync for ShardVramPtrs {}

#[cfg_attr(
    all(not(feature = "mock-gpu"), not(feature = "amd")),
    link(name = "axicor_cuda", kind = "static")
)]
#[cfg_attr(
    all(not(feature = "mock-gpu"), feature = "amd"),
    link(name = "axicor_amd", kind = "static")
)]
extern "C" {
    // =====================================================================
    //  New Zero-Cost contract (cu_* functions)
    // =====================================================================

    /// Allocates VRAM for a single shard and writes pointers to out_vram.
    /// Returns 0 on success, non-zero cudaError_t on error.
    pub fn cu_allocate_shard(padded_n: u32, total_axons: u32, out_vram: *mut ShardVramPtrs) -> i32;

    pub fn cu_reset_burst_counters(ptrs: *const ShardVramPtrs, padded_n: u32, stream: CudaStream);

    /// Zero-Cost DMA Upload: one cudaMemcpyAsync for the entire .state blob.
    /// state_blob  flat byte array in the order of ShardVramPtrs fields.
    /// Returns 0 on success.
    pub fn cu_upload_state_blob(
        vram: *const ShardVramPtrs,
        state_blob: *const c_void,
        state_size: usize,
    ) -> i32;

    /// Loads the .axons blob (flat [total_axons]u32) into axon_heads.
    /// Returns 0 on success.
    pub fn cu_upload_axons_blob(
        vram: *const ShardVramPtrs,
        axons_blob: *const c_void,
        axons_size: usize,
    ) -> i32;

    /// Frees all VRAM buffers associated with the shard.
    pub fn cu_free_shard(vram: *mut ShardVramPtrs);

    /// Day Phase Orchestrator: launches 6 kernels asynchronously.
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
        dopamine: i16,
        stream: CudaStream,
    ) -> i32;

    /// Global constant GPU memory (448 bytes).
    pub fn cu_upload_constant_memory(lut: *const VariantParameters) -> i32;

    // =====================================================================
    //  Memory and stream management (legacy helpers, used by memory.rs)
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

    pub fn gpu_stream_create(out_stream: *mut CudaStream) -> i32;
    pub fn gpu_stream_destroy(stream: CudaStream) -> i32;

    pub fn gpu_stream_synchronize(stream: CudaStream);
    pub fn gpu_set_device(device_id: i32);
    pub fn gpu_device_synchronize();

    /// Barrier: blocks CPU until all previous commands in the default stream are done.
    pub fn gpu_synchronize();

    // Loading Blueprint parameters into GPU Constant Memory
    pub fn gpu_load_constants(host_ptr: *const c_void);
    pub fn update_constant_memory_hot_reload(
        new_variants: *const VariantParameters,
        stream: CudaStream,
    );

    pub fn launch_sort_and_prune(ptrs: *const ShardVramPtrs, padded_n: u32, prune_threshold: i16);

    pub fn launch_extract_outgoing_spikes(
        axon_heads: *const axicor_core::layout::BurstHeads8,
        src_indices: *const u32,
        dst_ghost_ids: *const u32,
        count: u32,
        sync_batch_ticks: u32,
        v_seg: u32,
        out_events: *mut c_void,
        out_count: *mut u32,
        stream: CudaStream,
    );

    // [DOD FIX] Epic 2: Debug Harness Micro-Kernels
    pub fn launch_debug_inject_current(
        soma_voltage: *mut i32,
        target_tids: *const u32,
        injection_uv: *const i32,
        count: u32,
        stream: CudaStream,
    );

    pub fn launch_debug_record_v(
        soma_voltage: *const i32,
        target_tids: *const u32,
        out_trace: *mut i32,
        current_tick: u32,
        count: u32,
        max_ticks: u32,
        stream: CudaStream,
    );

    pub fn launch_ghost_sync(
        src_heads: *const axicor_core::layout::BurstHeads8,
        dst_heads: *mut axicor_core::layout::BurstHeads8,
        src_indices: *const u32,
        dst_indices: *const u32,
        count: u32,
        dst_total_axons: u32,
        sync_batch_ticks: u32, // [DOD FIX]
        v_seg: u32,            // [DOD FIX]
        stream: CudaStream,
    );

    // =====================================================================
    // 2. Day Phase Pipeline (6 kernels strictly according to Step 10 specification)
    // =====================================================================

    /// Kernel 1: Injection of external signals.
    /// [VramState, bitmask, current_tick, total_virtual_axons]
    pub fn launch_inject_inputs(
        vram: VramState,
        bitmask: *const u32,
        current_tick: u32,
        total_virtual_axons: u32,
    );

    /// Kernel 2: Injection of network spikes.
    /// [VramState, tick_schedule, tick_spikes_count]
    pub fn launch_apply_spike_batch(
        vram: VramState,
        tick_schedule: *const axicor_core::ipc::SpikeEvent,
        tick_spikes_count: u32,
    );

    /// Kernel 3: Unconditional shift of all axon heads.
    pub fn launch_propagate_axons(vram: VramState, v_seg: u32);

    /// Kernel 4: GLIF Physics, dendrite summation.
    pub fn launch_update_neurons(vram: VramState, constants_ptr: *const c_void, current_tick: u32);

    /// Kernel 5: GSOP Plasticity.
    pub fn launch_apply_gsop(vram: VramState);

    /// Kernel 6: Soma activity readout (RecordReadout).
    pub fn launch_record_readout(
        vram: VramState,
        mapped_soma_ids: *const u32,
        output_history: *mut u8,
        current_tick: u32,
        total_pixels: u32,
    );

    // Zero-Cost atomic reset before batch
    pub fn gpu_reset_telemetry_count(count_d: *mut u32, stream: CudaStream);

    // Hardware spike aggregation
    pub fn launch_extract_telemetry(
        flags_d: *const u8,
        out_ids_d: *mut u32,
        out_count_d: *mut u32,
        padded_n: u32,
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

    pub fn cu_free_io_buffers(
        d_input_bitmask: *mut u32,
        d_incoming_spikes: *mut u32,
        d_output_history: *mut u8,
    );

    pub fn cu_dma_h2d_io(
        d_input_bitmask: *mut u32,
        h_input_bitmask: *const u32,
        input_words: u32,
        d_incoming_spikes: *mut u32,
        h_incoming_spikes: *const u32,
        schedule_capacity: u32,
        stream: CudaStream,
    ) -> i32;

    pub fn cu_dma_d2h_io(
        h_output_history: *mut u8,
        d_output_history: *const u8,
        output_capacity: u32,
        stream: CudaStream,
    ) -> i32;
}
