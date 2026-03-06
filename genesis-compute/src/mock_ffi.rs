
use genesis_core::layout::VramState;
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

#[no_mangle] pub extern "C" fn gpu_set_device(_device_id: i32) {}
#[no_mangle] pub extern "C" fn gpu_device_synchronize() {}
#[no_mangle] pub extern "C" fn gpu_stream_synchronize(_stream: *mut c_void) {}
#[no_mangle] pub extern "C" fn gpu_synchronize() {}

#[no_mangle]
pub extern "C" fn gpu_load_constants(_host_ptr: *const c_void) {}

#[no_mangle]
pub extern "C" fn upload_constant_memory(_host_ptr: *const c_void) -> bool { true }

#[no_mangle]
pub extern "C" fn update_constant_memory_hot_reload(
    _new_variants: *const c_void,
    _stream: *mut c_void,
) {}

#[no_mangle]
pub extern "C" fn update_global_dopamine(_dopamine: i16, _stream: *mut c_void) {}

// ─────────────────────────────────────────────────────────────────────────────
// Day Phase Kernel Launches (6 kernels — Шаг 10)
// ─────────────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn launch_inject_inputs(
    _vram: VramState,
    bitmask: *const u32,
    _current_tick: u32,
    _total_virtual_axons: u32,
) {
    log_call("InjectInputs", bitmask as usize);
}

#[no_mangle]
pub extern "C" fn launch_apply_spike_batch(
    _vram: VramState,
    tick_schedule: *const SpikeEvent,
    _tick_spikes_count: u32,
) {
    log_call("ApplySpikeBatch", tick_schedule as usize);
}

#[no_mangle]
pub extern "C" fn launch_propagate_axons(
    _vram: VramState,
    _v_seg: u32,
) {
    log_call("PropagateAxons", 0);
}

#[no_mangle]
pub extern "C" fn launch_update_neurons(
    _vram: VramState,
    _constants_ptr: *const c_void,
    _current_tick: u32,
) {
    log_call("UpdateNeurons", 0);
}

#[no_mangle]
pub extern "C" fn launch_apply_gsop(
    _vram: VramState,
) {
    log_call("ApplyGSOP", 0);
}

#[no_mangle]
pub extern "C" fn launch_record_readout(
    _vram: VramState,
    _mapped_soma_ids: *const u32,
    _output_history: *mut u8,
    _current_tick: u32,
    _total_pixels: u32,
) {
    log_call("RecordReadout", 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Auxiliary Kernel Launches — No-Ops
// ─────────────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn launch_sort_and_prune(
    _vram: VramState,
    prune_threshold: i16,
) {
    log_call("SortAndPrune", prune_threshold as usize);
}

#[no_mangle]
pub extern "C" fn launch_extract_outgoing_spikes(
    _axon_heads: *const genesis_core::layout::BurstHeads8,
    _src_indices: *const u32,
    _dst_ghost_ids: *const u32,
    _count: u32,
    _sync_batch_ticks: u32,
    _out_events: *mut c_void,
    _out_count: *mut u32,
    _stream: *mut c_void,
) {}

#[no_mangle]
pub extern "C" fn launch_ghost_sync(
    _src_heads: *const genesis_core::layout::BurstHeads8,
    _dst_heads: *mut genesis_core::layout::BurstHeads8,
    _src_indices: *const u32,
    _dst_indices: *const u32,
    _count: u32,
    _stream: *mut c_void,
) {}



#[no_mangle]
pub extern "C" fn launch_extract_telemetry(
    _vram: *const crate::ffi::ShardVramPtrs,
    _padded_n: u32,
    _out_ids: *mut u32,
    out_count_pinned: *mut u32,
    _stream: *mut c_void,
) {
    log_call("ExtractTelemetry", 0);
    if !out_count_pinned.is_null() {
        unsafe { ptr::write_volatile(out_count_pinned, 0); }
    }
}
