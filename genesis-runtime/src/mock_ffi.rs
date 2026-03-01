#![cfg(feature = "mock-gpu")]

use std::ffi::c_void;
use std::ptr;

/// Compiles ONLY when `--features mock-gpu` is active (usually during tests).
/// Provides host-memory substitutes for CUDA FFI functions.

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
    // Both are host pointers in mock mode
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
    // Both are host pointers in mock mode
    unsafe {
        ptr::copy_nonoverlapping(src_dev as *const u8, dst_host as *mut u8, size);
    }
    true
}

#[no_mangle]
pub extern "C" fn gpu_device_synchronize() {}

#[no_mangle]
pub extern "C" fn gpu_stream_synchronize(_stream: *mut c_void) {}

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

#[no_mangle]
pub extern "C" fn upload_constant_memory(_host_ptr: *const c_void) -> bool {
    true
}

// --------------------------------------------------------------------------
// Kernel Launches (No-Op for network/ghost axon tests)
// --------------------------------------------------------------------------

#[no_mangle] pub extern "C" fn launch_propagate_axons(_1: u32, _2: *mut c_void, _3: u32, _4: *mut c_void) {}
#[no_mangle] pub extern "C" fn launch_update_neurons(_1: u32, _2: *mut c_void, _3: *mut c_void, _4: *mut c_void, _5: *mut c_void, _6: *const c_void, _7: *const c_void, _8: *mut c_void, _9: *mut c_void, _10: *mut c_void, _11: *mut c_void) {}
#[no_mangle] pub extern "C" fn launch_apply_gsop(_1: u32, _2: *const c_void, _3: *const c_void, _4: *mut c_void, _5: *mut c_void, _6: *mut c_void) {}
#[no_mangle] pub extern "C" fn launch_apply_spike_batch_impl(_1: u32, _2: *const c_void, _3: *mut c_void, _4: u32, _5: *mut c_void) {}
#[no_mangle] pub extern "C" fn launch_record_readout(_1: *const c_void, _2: *const c_void, _3: *mut c_void, _4: u32, _5: u32, _6: *mut c_void) {}
#[no_mangle] pub extern "C" fn launch_sort_and_prune(_1: u32, _2: u32, _3: i16, _4: *mut c_void, _5: *mut c_void, _6: *mut c_void, _7: *mut c_void, _8: *mut c_void) {}
#[no_mangle] pub extern "C" fn launch_inject_inputs(_1: *mut c_void, _2: *const c_void, _3: *const c_void, _4: u32, _5: u32, _6: *mut c_void) {}
#[no_mangle] pub extern "C" fn gpu_reset_telemetry_count(_1: *const c_void, _2: *mut c_void) {}
