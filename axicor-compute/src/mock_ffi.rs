use crate::ffi::ShardVramPtrs;
use crate::{bindings, cpu};
use axicor_core::ipc::SpikeEvent;
use axicor_core::layout::{VariantParameters, VramState};
use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;

// -----------------------------------------------------------------------------
// TDD Call Logger
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Memory Management (DOD Size-Prefixed Allocator)
// -----------------------------------------------------------------------------

use std::alloc::{alloc_zeroed, dealloc, Layout};

unsafe fn mock_alloc_64(size: usize) -> *mut c_void {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align_unchecked(size + 64, 64);
    let ptr = alloc_zeroed(layout);
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    *(ptr as *mut usize) = size;
    ptr.add(64) as *mut c_void
}

unsafe fn mock_free_64(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let real_ptr = (ptr as *mut u8).sub(64);
    let size = *(real_ptr as *const usize);
    let layout = Layout::from_size_align_unchecked(size + 64, 64);
    dealloc(real_ptr, layout);
}

#[no_mangle]
pub extern "C" fn gpu_malloc(size: usize) -> *mut c_void {
    unsafe { mock_alloc_64(size) }
}

/// # Safety
/// `dev_ptr` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gpu_free(dev_ptr: *mut c_void) {
    unsafe { mock_free_64(dev_ptr) }
}

#[no_mangle]
pub extern "C" fn gpu_host_alloc(size: usize) -> *mut c_void {
    unsafe { mock_alloc_64(size) }
}

/// # Safety
/// `ptr` must be valid.
#[no_mangle]
pub unsafe extern "C" fn gpu_host_free(dev_ptr: *mut c_void) {
    unsafe { mock_free_64(dev_ptr) }
}

#[no_mangle]
pub extern "C" fn gpu_memcpy_host_to_device(
    dst_dev: *mut c_void,
    src_host: *const c_void,
    size: usize,
) -> bool {
    unsafe {
        ptr::copy_nonoverlapping(src_host as *const u8, dst_dev as *mut u8, size);
    }
    true
}

#[no_mangle]
pub extern "C" fn gpu_memcpy_device_to_host(
    dst_host: *mut c_void,
    src_dev: *const c_void,
    size: usize,
) -> bool {
    unsafe {
        ptr::copy_nonoverlapping(src_dev as *const u8, dst_host as *mut u8, size);
    }
    true
}

#[no_mangle]
pub extern "C" fn gpu_memcpy_host_to_device_async(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    _stream: *mut c_void,
) {
    unsafe {
        ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size);
    }
}

#[no_mangle]
pub extern "C" fn gpu_memcpy_device_to_host_async(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
    _stream: *mut c_void,
) {
    unsafe {
        ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size);
    }
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
    unsafe {
        ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size);
    }
    true
}

/// # Safety
/// `out_stream` must be valid and mutable.
#[no_mangle]
pub unsafe extern "C" fn gpu_stream_create(out_stream: *mut *mut c_void) -> i32 {
    unsafe {
        *out_stream = std::ptr::null_mut();
    }
    0
}
#[no_mangle]
pub extern "C" fn gpu_stream_destroy(_stream: *mut c_void) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn gpu_stream_synchronize(_stream: *mut c_void) {}
#[no_mangle]
pub extern "C" fn gpu_synchronize() {}

#[no_mangle]
pub extern "C" fn gpu_set_device(_device_id: i32) {}

#[no_mangle]
pub extern "C" fn gpu_device_synchronize() {}

#[no_mangle]
pub extern "C" fn gpu_load_constants(_host_ptr: *const c_void) {}

#[no_mangle]
pub extern "C" fn upload_constant_memory(_host_ptr: *const c_void) -> bool {
    true
}

#[no_mangle]
pub extern "C" fn update_constant_memory_hot_reload(
    _new_variants: *const axicor_core::layout::VariantParameters,
    _stream: *mut c_void,
) {
}

/// # Safety
/// Caller must ensure pointers are valid.
#[no_mangle]
pub unsafe extern "C" fn cu_allocate_shard(
    padded_n: u32,
    total_axons: u32,
    out_vram: *mut ShardVramPtrs,
) -> i32 {
    bindings::cpu_allocate_shard(padded_n, total_axons, out_vram)
}

/// # Safety
/// - All VRAM pointers inside `ShardVramPtrs` MUST be 64-byte aligned (L1/L2 Cache Invariant).
/// - `padded_n` MUST be an exact multiple of the hardware warp size (32 for CUDA, 64 for HIP).
/// - Host-side I/O arrays MUST be allocated as pinned memory (`gpu_host_alloc`) to prevent PCIe page faults.
#[no_mangle]
pub unsafe extern "C" fn cu_reset_burst_counters(
    _ptrs: *const ShardVramPtrs,
    _padded_n: u32,
    _stream: *mut c_void,
) {
}

/// # Safety
/// - All VRAM pointers inside `ShardVramPtrs` MUST be 64-byte aligned (L1/L2 Cache Invariant).
/// - `padded_n` MUST be an exact multiple of the hardware warp size (32 for CUDA, 64 for HIP).
/// - Host-side I/O arrays MUST be allocated as pinned memory (`gpu_host_alloc`) to prevent PCIe page faults.
#[no_mangle]
pub unsafe extern "C" fn cu_upload_state_blob(
    vram: *const ShardVramPtrs,
    state_blob: *const c_void,
    state_size: usize,
) -> i32 {
    bindings::cpu_upload_state_blob(vram, state_blob, state_size)
}

/// # Safety
/// Caller must ensure pointers are valid and aligned to 64 bytes.
#[no_mangle]
pub unsafe extern "C" fn cu_upload_axons_blob(
    vram: *const ShardVramPtrs,
    axons_blob: *const c_void,
    axons_size: usize,
) -> i32 {
    bindings::cpu_upload_axons_blob(vram, axons_blob, axons_size)
}

/// # Safety
/// - All VRAM pointers inside `ShardVramPtrs` MUST be 64-byte aligned (L1/L2 Cache Invariant).
/// - `padded_n` MUST be an exact multiple of the hardware warp size (32 for CUDA, 64 for HIP).
/// - Host-side I/O arrays MUST be allocated as pinned memory (`gpu_host_alloc`) to prevent PCIe page faults.
#[no_mangle]
pub unsafe extern "C" fn cu_free_shard(vram: *mut ShardVramPtrs) {
    bindings::cpu_free_shard(vram);
}

/// # Safety
/// - All VRAM pointers inside `ShardVramPtrs` MUST be 64-byte aligned (L1/L2 Cache Invariant).
/// - `padded_n` MUST be an exact multiple of the hardware warp size (32 for CUDA, 64 for HIP).
/// - Host-side I/O arrays MUST be allocated as pinned memory (`gpu_host_alloc`) to prevent PCIe page faults.
#[no_mangle]
pub unsafe extern "C" fn cu_step_day_phase(
    vram: *const ShardVramPtrs,
    padded_n: u32,
    total_axons: u32,
    v_seg: u32,
    _current_tick: u32,
    input_bitmask: *const u32,
    virtual_offset: u32,
    num_virtual_axons: u32,
    incoming_spikes: *const u32,
    num_incoming_spikes: u32,
    mapped_soma_ids: *const u32,
    output_history: *mut u8,
    num_outputs: u32,
    dopamine: i16,
    _stream: *mut c_void,
) -> i32 {
    let ptrs = &*vram;

    let axon_heads = std::slice::from_raw_parts_mut(ptrs.axon_heads, total_axons as usize);
    if !input_bitmask.is_null() && num_virtual_axons != 0 {
        let input_words = (num_virtual_axons as usize).div_ceil(32);
        let mask = std::slice::from_raw_parts(input_bitmask, input_words);
        cpu::physics::cpu_inject_inputs(axon_heads, mask, virtual_offset, num_virtual_axons, v_seg);
    }

    if !incoming_spikes.is_null() && num_incoming_spikes != 0 {
        let spikes = std::slice::from_raw_parts(incoming_spikes, num_incoming_spikes as usize);
        cpu::physics::cpu_apply_spike_batch(axon_heads, spikes, v_seg);
    }

    cpu::physics::cpu_propagate_axons(axon_heads, v_seg);
    cpu::physics::cpu_update_neurons(ptrs, padded_n, total_axons, 0, v_seg);
    cpu::physics::cpu_apply_gsop(ptrs, padded_n, total_axons, dopamine);

    if !output_history.is_null() && !mapped_soma_ids.is_null() && num_outputs != 0 {
        let soma_flags = std::slice::from_raw_parts(ptrs.soma_flags, padded_n as usize);
        let mapped_ids = std::slice::from_raw_parts(mapped_soma_ids, num_outputs as usize);
        let history = std::slice::from_raw_parts_mut(output_history, num_outputs as usize);
        cpu::physics::cpu_record_outputs(soma_flags, mapped_ids, history, 0, num_outputs);
    }

    0
}

/// # Safety
/// Caller must ensure `lut` points to 16 VariantParameters.
#[no_mangle]
pub unsafe extern "C" fn cu_upload_constant_memory(lut: *const VariantParameters) -> i32 {
    bindings::cpu_upload_constant_memory(lut);
    0
}

// -----------------------------------------------------------------------------
// Day Phase Kernel Launches (6 kernels  Step 10)
// -----------------------------------------------------------------------------

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
pub extern "C" fn launch_apply_gsop(_vram: VramState, _stream: *mut c_void) -> i32 {
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

// -----------------------------------------------------------------------------
// Auxiliary Kernel Launches  No-Ops
// -----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn launch_sort_and_prune(
    ptrs: *const ShardVramPtrs,
    padded_n: u32,
    prune_threshold: i16,
) {
    unsafe {
        crate::cpu::physics::cpu_sort_and_prune(&*ptrs, padded_n, prune_threshold);
    }
}

#[no_mangle]
pub extern "C" fn launch_extract_outgoing_spikes(
    _axon_heads: *const axicor_core::layout::BurstHeads8,
    _src_indices: *const u32,
    _dst_ghost_ids: *const u32,
    _count: u32,
    _sync_batch_ticks: u32,
    _v_seg: u32,
    _out_events: *mut c_void,
    _out_count: *mut u32,
    _stream: *mut c_void,
) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn launch_ghost_sync(
    src_heads: *const axicor_core::layout::BurstHeads8,
    dst_heads: *mut axicor_core::layout::BurstHeads8,
    src_indices: *const u32,
    dst_indices: *const u32,
    count: u32,
    dst_total_axons: u32,
    sync_batch_ticks: u32,
    v_seg: u32,
    _stream: *mut std::ffi::c_void,
) -> i32 {
    let batch_shift = sync_batch_ticks * v_seg;
    for tid in 0..count as usize {
        let src_axon = unsafe { *src_indices.add(tid) };
        let dst_ghost = unsafe { *dst_indices.add(tid) };
        
        if src_axon == 0x80000000 || src_axon == 0xFFFFFFFF || dst_ghost >= dst_total_axons { 
            continue; 
        }

        let src = unsafe { *src_heads.add(src_axon as usize) };
        println!("DEBUG: tid={}, src_axon={}, src.h0={}", tid, src_axon, src.h0);

        let mut dst = axicor_core::layout::BurstHeads8::empty(0x80000000);
        dst.h0 = if src.h0 == 0x80000000 { 0x80000000 } else { src.h0.wrapping_sub(batch_shift) };
        dst.h1 = if src.h1 == 0x80000000 { 0x80000000 } else { src.h1.wrapping_sub(batch_shift) };
        dst.h2 = if src.h2 == 0x80000000 { 0x80000000 } else { src.h2.wrapping_sub(batch_shift) };
        dst.h3 = if src.h3 == 0x80000000 { 0x80000000 } else { src.h3.wrapping_sub(batch_shift) };
        dst.h4 = if src.h4 == 0x80000000 { 0x80000000 } else { src.h4.wrapping_sub(batch_shift) };
        dst.h5 = if src.h5 == 0x80000000 { 0x80000000 } else { src.h5.wrapping_sub(batch_shift) };
        dst.h6 = if src.h6 == 0x80000000 { 0x80000000 } else { src.h6.wrapping_sub(batch_shift) };
        dst.h7 = if src.h7 == 0x80000000 { 0x80000000 } else { src.h7.wrapping_sub(batch_shift) };
        unsafe { *dst_heads.add(dst_ghost as usize) = dst; }
    }
    0
}

#[no_mangle]
pub extern "C" fn gpu_reset_telemetry_count(_count_d: *mut u32, _stream: *mut std::ffi::c_void) {}

/// # Safety
/// Pointers must be valid and correctly sized according to `padded_n`.
#[no_mangle]
pub unsafe extern "C" fn launch_extract_telemetry(
    flags_d: *const u8,
    out_ids_d: *mut u32,
    out_count_d: *mut u32,
    padded_n: u32,
    _stream: *mut std::ffi::c_void,
) {
    unsafe {
        let flags = std::slice::from_raw_parts(flags_d, padded_n as usize);
        let out_ids = std::slice::from_raw_parts_mut(out_ids_d, padded_n as usize);
        *out_count_d = cpu::physics::cpu_extract_telemetry(flags, out_ids);
    }
}

unsafe fn aligned_alloc_zeroed<T>(count: usize) -> *mut T {
    if count == 0 {
        return ptr::null_mut();
    }
    let size = count * std::mem::size_of::<T>();
    mock_alloc_64(size) as *mut T
}

/// # Safety
/// Caller must ensure all double-pointers are valid.
#[no_mangle]
pub unsafe extern "C" fn cu_allocate_io_buffers(
    input_words: u32,
    schedule_capacity: u32,
    output_capacity: u32,
    d_input_bitmask: *mut *mut u32,
    d_incoming_spikes: *mut *mut u32,
    d_output_history: *mut *mut u8,
) -> i32 {
    *d_input_bitmask = aligned_alloc_zeroed::<u32>(input_words as usize);
    *d_incoming_spikes = aligned_alloc_zeroed::<u32>(schedule_capacity as usize);
    *d_output_history = aligned_alloc_zeroed::<u8>(output_capacity as usize);
    0
}

/// # Safety
/// - All VRAM pointers inside `ShardVramPtrs` MUST be 64-byte aligned (L1/L2 Cache Invariant).
/// - `padded_n` MUST be an exact multiple of the hardware warp size (32 for CUDA, 64 for HIP).
/// - Host-side I/O arrays MUST be allocated as pinned memory (`gpu_host_alloc`) to prevent PCIe page faults.
#[no_mangle]
pub unsafe extern "C" fn cu_free_io_buffers(
    d_input_bitmask: *mut u32,
    d_incoming_spikes: *mut u32,
    d_output_history: *mut u8,
) {
    if !d_input_bitmask.is_null() {
        mock_free_64(d_input_bitmask as *mut c_void);
    }
    if !d_incoming_spikes.is_null() {
        mock_free_64(d_incoming_spikes as *mut c_void);
    }
    if !d_output_history.is_null() {
        mock_free_64(d_output_history as *mut c_void);
    }
}

/// # Safety
/// - All VRAM pointers inside `ShardVramPtrs` MUST be 64-byte aligned (L1/L2 Cache Invariant).
/// - `padded_n` MUST be an exact multiple of the hardware warp size (32 for CUDA, 64 for HIP).
/// - Host-side I/O arrays MUST be allocated as pinned memory (`gpu_host_alloc`) to prevent PCIe page faults.
#[no_mangle]
pub unsafe extern "C" fn cu_dma_h2d_io(
    d_input_bitmask: *mut u32,
    h_input_bitmask: *const u32,
    input_words: u32,
    d_incoming_spikes: *mut u32,
    h_incoming_spikes: *const u32,
    schedule_capacity: u32,
    _stream: *mut c_void,
) -> i32 {
    if !d_input_bitmask.is_null() && !h_input_bitmask.is_null() && input_words != 0 {
        ptr::copy_nonoverlapping(h_input_bitmask, d_input_bitmask, input_words as usize);
    }
    if !d_incoming_spikes.is_null() && !h_incoming_spikes.is_null() && schedule_capacity != 0 {
        ptr::copy_nonoverlapping(
            h_incoming_spikes,
            d_incoming_spikes,
            schedule_capacity as usize,
        );
    }
    0
}

/// # Safety
/// - All host-side buffers MUST be Pinned RAM.
#[no_mangle]
pub unsafe extern "C" fn cu_dma_d2h_io(
    h_output_history: *mut u8,
    d_output_history: *const u8,
    output_capacity: u32,
    _stream: *mut c_void,
) -> i32 {
    if !h_output_history.is_null() && !d_output_history.is_null() && output_capacity != 0 {
        ptr::copy_nonoverlapping(d_output_history, h_output_history, output_capacity as usize);
    }
    0
}

#[no_mangle]
pub extern "C" fn launch_debug_inject_current(
    _soma_voltage: *mut i32,
    _target_tids: *const u32,
    _injection_uv: *const i32,
    _count: u32,
    _stream: *mut c_void,
) {
    // Handled by CPU module directly in non-GPU builds
}

#[no_mangle]
pub extern "C" fn launch_debug_record_v(
    _soma_voltage: *const i32,
    _target_tids: *const u32,
    _out_trace: *mut i32,
    _current_tick: u32,
    _count: u32,
    _max_ticks: u32,
    _stream: *mut c_void,
) {
    // Handled by CPU module directly
}
