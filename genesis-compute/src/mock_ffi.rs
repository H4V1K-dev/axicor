use genesis_core::layout::VramState;
use crate::ffi::ShardVramPtrs;
use genesis_core::ipc::SpikeEvent;
use std::sync::Mutex;
use std::ffi::c_void;
use std::ptr;

// ─────────────────────────────────────────────────────────────────────────────
// TDD Call Logger
// ─────────────────────────────────────────────────────────────────────────────

static CALL_LOG: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());

pub fn clear_call_log() {
    CALL_LOG.lock().unwrap().clear();
}

pub fn get_call_log() -> Vec<(String, usize)> {
    CALL_LOG.lock().unwrap().clone()
}

fn log_call(name: &str, ptr_addr: usize) {
    CALL_LOG.lock().unwrap().push((name.to_string(), ptr_addr));
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory Management
// ─────────────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn gpu_malloc(size: usize) -> *mut c_void {
    unsafe { libc::malloc(size) as *mut c_void }
}

#[no_mangle]
pub extern "C" fn gpu_free(dev_ptr: *mut c_void) {
    unsafe { libc::free(dev_ptr) }
}

#[no_mangle]
pub extern "C" fn gpu_host_alloc(size: usize) -> *mut c_void {
    unsafe { libc::malloc(size) as *mut c_void }
}

#[no_mangle]
pub extern "C" fn gpu_host_free(dev_ptr: *mut c_void) {
    unsafe { libc::free(dev_ptr) }
}

#[no_mangle]
pub extern "C" fn gpu_memcpy_host_to_device(
    dst_dev: *mut c_void,
    src_host: *const c_void,
    size: usize,
) -> bool {
    unsafe { ptr::copy_nonoverlapping(src_host as *const u8, dst_dev as *mut u8, size); }
    true
}

#[no_mangle]
pub extern "C" fn gpu_memcpy_device_to_host(
    dst_host: *mut c_void,
    src_dev: *const c_void,
    size: usize,
) -> bool {
    unsafe { ptr::copy_nonoverlapping(src_dev as *const u8, dst_host as *mut u8, size); }
    true
}

#[no_mangle]
pub extern "C" fn gpu_memcpy_host_to_device_async(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    _stream: *mut c_void,
) {
    unsafe { ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size); }
}

#[no_mangle]
pub extern "C" fn gpu_memcpy_device_to_host_async(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    _stream: *mut c_void,
) {
    unsafe { ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size); }
}

#[no_mangle]
pub extern "C" fn gpu_memcpy_peer_async(
    dst: *mut c_void,
    _dst_dev: i32,
    src: *const c_void,
    _src_dev: i32,
    size: usize,
    _stream: *mut c_void,
) -> bool {
    unsafe { ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size); }
    true
}

#[no_mangle] pub extern "C" fn gpu_stream_create(out_stream: *mut *mut c_void) -> i32 {
    unsafe { *out_stream = std::ptr::null_mut(); }
    0
}
#[no_mangle] pub extern "C" fn gpu_stream_destroy(_stream: *mut c_void) -> i32 { 0 }

#[no_mangle] pub extern "C" fn gpu_stream_synchronize(_stream: *mut c_void) {}
#[no_mangle] pub extern "C" fn gpu_synchronize() {}

#[no_mangle]
pub extern "C" fn gpu_set_device(_device_id: i32) {}

#[no_mangle]
pub extern "C" fn gpu_device_synchronize() {}

#[no_mangle]
pub extern "C" fn gpu_load_constants(_host_ptr: *const c_void) {}

#[no_mangle]
pub extern "C" fn upload_constant_memory(_host_ptr: *const c_void) -> bool { true }

#[no_mangle]
pub extern "C" fn update_constant_memory_hot_reload(
    _new_variants: *const genesis_core::layout::VariantParameters,
    _stream: *mut c_void,
) {}

// ─────────────────────────────────────────────────────────────────────────────
// Day Phase Kernel Launches (6 kernels — Шаг 10)
// ─────────────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn launch_inject_inputs(
    _vram: VramState,
    bitmask: *const u32,
    _current_tick: u32,
    _total_virtual_axons: u32,
    _stream: *mut c_void,
) -> i32 {
    log_call("InjectInputs", bitmask as usize);
    0
}

#[no_mangle]
pub extern "C" fn launch_apply_spike_batch(
    _vram: VramState,
    tick_schedule: *const SpikeEvent,
    _tick_spikes_count: u32,
    _stream: *mut c_void,
) -> i32 {
    log_call("ApplySpikeBatch", tick_schedule as usize);
    0
}

#[no_mangle]
pub extern "C" fn launch_propagate_axons(
    _vram: VramState,
    _v_seg: u32,
    _stream: *mut c_void,
) -> i32 {
    log_call("PropagateAxons", 0);
    0
}

#[no_mangle]
pub extern "C" fn launch_update_neurons(
    _vram: VramState,
    _constants_ptr: *const c_void,
    _current_tick: u32,
    _stream: *mut c_void,
) -> i32 {
    log_call("UpdateNeurons", 0);
    0
}

#[no_mangle]
pub extern "C" fn launch_apply_gsop(
    _vram: VramState,
    _stream: *mut c_void,
) -> i32 {
    log_call("ApplyGSOP", 0);
    0
}

#[no_mangle]
pub extern "C" fn launch_record_readout(
    _vram: VramState,
    _mapped_soma_ids: *const u32,
    _output_history: *mut u8,
    _num_outputs: u32,
    _dopamine: i16,
    _stream: *mut c_void,
) -> i32 {
    log_call("RecordReadout", 0);
    0
}

// ─────────────────────────────────────────────────────────────────────────────
// Auxiliary Kernel Launches — No-Ops
// ─────────────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn launch_sort_and_prune(
    _ptrs: *const ShardVramPtrs,
    _padded_n: u32,
    _prune_threshold: i16,
) {
    log_call("SortAndPrune", 0);
}

#[no_mangle]
pub extern "C" fn launch_extract_outgoing_spikes(
    _axon_heads: *const genesis_core::layout::BurstHeads8,
    _src_indices: *const u32,
    _dst_ghost_ids: *const u32,
    _count: u32,
    _sync_batch_ticks: u32,
    _v_seg: u32,
    _out_events: *mut c_void,
    _out_count: *mut u32,
    _stream: *mut c_void,
) -> i32 { 0 }

#[no_mangle]
pub extern "C" fn launch_ghost_sync(
    _src_heads: *const genesis_core::layout::BurstHeads8,
    _dst_heads: *mut genesis_core::layout::BurstHeads8,
    _d_incoming_spikes: *mut u32,
    _h_incoming_spikes: *const u32,
    _schedule_capacity: u32,
    _stream: *mut c_void,
) -> i32 { 0 }

#[no_mangle]
pub extern "C" fn gpu_reset_telemetry_count(
    _ptrs: *const ShardVramPtrs,
    _stream: *mut c_void,
) {
    log_call("ResetTelemetryCount", 0);
}

#[no_mangle]
pub extern "C" fn launch_extract_telemetry(
    _ptrs: *const ShardVramPtrs,
    _padded_n: u32,
    _out_ids: *mut u32,
    out_count_pinned: *mut u32,
    _stream: *mut c_void,
) {
    log_call("ExtractTelemetry", 0);
    if !out_count_pinned.is_null() {
        unsafe { std::ptr::write_volatile(out_count_pinned, 0); }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Zero-Cost cu_* ABI wrappers used by the current runtime
// ─────────────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cu_allocate_shard(
    padded_n: u32,
    total_axons: u32,
    out_vram: *mut ShardVramPtrs,
) -> i32 {
    if out_vram.is_null() {
        return 1;
    }

    let n = padded_n as usize;
    let dendrites = n * 128;
    let a = total_axons as usize;

    unsafe {
        (*out_vram).soma_voltage = libc::calloc(n, std::mem::size_of::<i32>()) as *mut i32;
        (*out_vram).soma_flags = libc::calloc(n, std::mem::size_of::<u8>()) as *mut u8;
        (*out_vram).threshold_offset = libc::calloc(n, std::mem::size_of::<i32>()) as *mut i32;
        (*out_vram).timers = libc::calloc(n, std::mem::size_of::<u8>()) as *mut u8;
        (*out_vram).soma_to_axon = libc::calloc(n, std::mem::size_of::<u32>()) as *mut u32;
        (*out_vram).dendrite_targets = libc::calloc(dendrites, std::mem::size_of::<u32>()) as *mut u32;
        (*out_vram).dendrite_weights = libc::calloc(dendrites, std::mem::size_of::<i16>()) as *mut i16;
        (*out_vram).dendrite_timers = libc::calloc(dendrites, std::mem::size_of::<u8>()) as *mut u8;
        (*out_vram).axon_heads = libc::calloc(a, std::mem::size_of::<genesis_core::layout::BurstHeads8>())
            as *mut genesis_core::layout::BurstHeads8;
    }

    log_call("cu_allocate_shard", padded_n as usize);
    0
}

#[no_mangle]
pub extern "C" fn cu_upload_state_blob(
    vram: *const ShardVramPtrs,
    state_blob: *const c_void,
    state_size: usize,
) -> i32 {
    if vram.is_null() || state_blob.is_null() {
        return 1;
    }

    let bytes = unsafe { std::slice::from_raw_parts(state_blob as *const u8, state_size) };
    let offsets = crate::memory::compute_state_offsets({
        // The runtime passes the exact blob size for the current padded_n; recover N from the
        // already allocated soma pointers by trusting the serialized layout calculator.
        // Mock ABI only needs deterministic memcpy, not dynamic introspection.
        let mut guess = 32usize;
        while crate::memory::calculate_state_blob_size(guess).1 != state_size && guess < (1 << 20) {
            guess += 32;
        }
        guess
    });

    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr().add(offsets.soma_voltage), (*vram).soma_voltage as *mut u8, offsets.soma_flags - offsets.soma_voltage);
        ptr::copy_nonoverlapping(bytes.as_ptr().add(offsets.soma_flags), (*vram).soma_flags, offsets.threshold_offset - offsets.soma_flags);
        ptr::copy_nonoverlapping(bytes.as_ptr().add(offsets.threshold_offset), (*vram).threshold_offset as *mut u8, offsets.timers - offsets.threshold_offset);
        ptr::copy_nonoverlapping(bytes.as_ptr().add(offsets.timers), (*vram).timers, offsets.soma_to_axon - offsets.timers);
        ptr::copy_nonoverlapping(bytes.as_ptr().add(offsets.soma_to_axon), (*vram).soma_to_axon as *mut u8, offsets.dendrite_targets - offsets.soma_to_axon);
        ptr::copy_nonoverlapping(bytes.as_ptr().add(offsets.dendrite_targets), (*vram).dendrite_targets as *mut u8, offsets.dendrite_weights - offsets.dendrite_targets);
        ptr::copy_nonoverlapping(bytes.as_ptr().add(offsets.dendrite_weights), (*vram).dendrite_weights as *mut u8, offsets.dendrite_timers - offsets.dendrite_weights);
        ptr::copy_nonoverlapping(bytes.as_ptr().add(offsets.dendrite_timers), (*vram).dendrite_timers, offsets.total_bytes - offsets.dendrite_timers);
    }

    0
}

#[no_mangle]
pub extern "C" fn cu_upload_axons_blob(
    vram: *const ShardVramPtrs,
    axons_blob: *const c_void,
    axons_size: usize,
) -> i32 {
    if vram.is_null() || axons_blob.is_null() {
        return 1;
    }
    unsafe {
        ptr::copy_nonoverlapping(
            axons_blob as *const u8,
            (*vram).axon_heads as *mut u8,
            axons_size,
        );
    }
    0
}

#[no_mangle]
pub extern "C" fn cu_free_shard(vram: *mut ShardVramPtrs) {
    if vram.is_null() {
        return;
    }
    unsafe {
        libc::free((*vram).soma_voltage as *mut c_void);
        libc::free((*vram).soma_flags as *mut c_void);
        libc::free((*vram).threshold_offset as *mut c_void);
        libc::free((*vram).timers as *mut c_void);
        libc::free((*vram).soma_to_axon as *mut c_void);
        libc::free((*vram).dendrite_targets as *mut c_void);
        libc::free((*vram).dendrite_weights as *mut c_void);
        libc::free((*vram).dendrite_timers as *mut c_void);
        libc::free((*vram).axon_heads as *mut c_void);
    }
}

#[no_mangle]
pub extern "C" fn cu_reset_burst_counters(
    _ptrs: *const ShardVramPtrs,
    _padded_n: u32,
    _stream: *mut c_void,
) {
}

#[no_mangle]
pub extern "C" fn cu_step_day_phase(
    _vram: *const ShardVramPtrs,
    _padded_n: u32,
    _total_axons: u32,
    _v_seg: u32,
    _current_tick: u32,
    _input_bitmask: *const u32,
    _virtual_offset: u32,
    _num_virtual_axons: u32,
    _incoming_spikes: *const u32,
    _num_incoming_spikes: u32,
    _mapped_soma_ids: *const u32,
    output_history: *mut u8,
    num_outputs: u32,
    _dopamine: i16,
    _stream: *mut c_void,
) -> i32 {
    if !output_history.is_null() && num_outputs > 0 {
        unsafe { std::ptr::write_bytes(output_history, 0, num_outputs as usize); }
    }
    0
}

#[no_mangle]
pub extern "C" fn cu_upload_constant_memory(_lut: *const genesis_core::layout::VariantParameters) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn cu_allocate_io_buffers(
    input_words: u32,
    schedule_capacity: u32,
    output_capacity: u32,
    d_input_bitmask: *mut *mut u32,
    d_incoming_spikes: *mut *mut u32,
    d_output_history: *mut *mut u8,
) -> i32 {
    unsafe {
        if !d_input_bitmask.is_null() {
            *d_input_bitmask = libc::calloc(input_words as usize, std::mem::size_of::<u32>()) as *mut u32;
        }
        if !d_incoming_spikes.is_null() {
            *d_incoming_spikes = libc::calloc(schedule_capacity as usize, std::mem::size_of::<u32>()) as *mut u32;
        }
        if !d_output_history.is_null() {
            *d_output_history = libc::calloc(output_capacity as usize, std::mem::size_of::<u8>()) as *mut u8;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn cu_dma_h2d_io(
    d_input_bitmask: *mut u32,
    h_input_bitmask: *const u32,
    input_words: u32,
    d_incoming_spikes: *mut u32,
    h_incoming_spikes: *const u32,
    schedule_capacity: u32,
    _stream: *mut c_void,
) -> i32 {
    unsafe {
        if !d_input_bitmask.is_null() && !h_input_bitmask.is_null() && input_words > 0 {
            ptr::copy_nonoverlapping(h_input_bitmask, d_input_bitmask, input_words as usize);
        }
        if !d_incoming_spikes.is_null() && !h_incoming_spikes.is_null() && schedule_capacity > 0 {
            ptr::copy_nonoverlapping(h_incoming_spikes, d_incoming_spikes, schedule_capacity as usize);
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn cu_dma_d2h_io(
    h_output_history: *mut u8,
    d_output_history: *const u8,
    output_capacity: u32,
    _stream: *mut c_void,
) -> i32 {
    unsafe {
        if !h_output_history.is_null() && !d_output_history.is_null() && output_capacity > 0 {
            ptr::copy_nonoverlapping(d_output_history, h_output_history, output_capacity as usize);
        }
    }
    0
}
