use crate::ffi::ShardVramPtrs;
use axicor_core::constants::AXON_SENTINEL;
use axicor_core::layout::{BurstHeads8, VariantParameters};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ffi::c_void;

// =============================================================================
//  Size-Prefixed Allocator (DOD workaround for Layout loss)
// =============================================================================
unsafe fn alloc_aligned_with_prefix(size: usize, align: usize) -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(size + align, align);
    let ptr = alloc_zeroed(layout);
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    // Write size at the very beginning (metadata)
    *(ptr as *mut usize) = size;

    // Return shifted pointer, perfectly aligned to L1/L2 cache line
    ptr.add(align)
}

unsafe fn free_aligned_with_prefix(ptr: *mut u8, align: usize) {
    if ptr.is_null() {
        return;
    }
    let real_ptr = ptr.sub(align);
    let size = *(real_ptr as *const usize);
    let layout = Layout::from_size_align_unchecked(size + align, align);
    dealloc(real_ptr, layout);
}

// =============================================================================
// 1.1 Constant Memory Emulation (LUT)
// =============================================================================

#[repr(C, align(64))]
pub struct CpuConstantMemory {
    pub variants: [VariantParameters; 16],
}

pub static mut VARIANT_LUT: CpuConstantMemory = unsafe { std::mem::zeroed() };

/// # Safety
/// Caller must ensure `lut` points to 16 valid VariantParameters aligned to 64 bytes.
pub unsafe fn cpu_upload_constant_memory(lut: *const VariantParameters) {
    let src = std::slice::from_raw_parts(lut, 16);
    std::ptr::copy_nonoverlapping(
        src.as_ptr(),
        std::ptr::addr_of_mut!(VARIANT_LUT.variants) as *mut VariantParameters,
        16,
    );
}

// =============================================================================
// 1.2 VRAM Allocation via std::alloc
// =============================================================================

/// # Safety
/// Caller must ensure pointer is valid and correctly aligned.
pub unsafe fn cpu_allocate_shard(
    padded_n: u32,
    total_axons: u32,
    out_vram: *mut ShardVramPtrs,
) -> i32 {
    let n = padded_n as usize;

    let sz_voltage = n * 4;
    let sz_flags = n * 1;
    let sz_thresh = n * 4;
    let sz_timers = n * 1;
    let sz_s2a = n * 4;
    let sz_targets = n * 128 * 4;
    let sz_weights = n * 128 * 4; // Strictly 4 bytes (i32)
    let sz_dtimers = n * 128 * 1;

    // [DOD FIX] + (64 * 8) compensates for cache lines for all 8 SoA arrays
    let total_state_size = sz_voltage + sz_flags + sz_thresh + sz_timers 
        + sz_s2a + sz_targets + sz_weights + sz_dtimers + (64 * 8);

    // Base pointer is aligned to 64 bytes
    let base_ptr = alloc_aligned_with_prefix(total_state_size, 64);
    if base_ptr.is_null() {
        return -1;
    }

    // [DOD FIX] Strict C-ABI 64-Byte Padding (L2 Cache Line)
    let mut off = 0;
    
    (*out_vram).soma_voltage = base_ptr.add(off) as *mut i32;
    off = (off + sz_voltage + 63) & !63;

    (*out_vram).soma_flags = base_ptr.add(off);
    off = (off + sz_flags + 63) & !63;

    (*out_vram).threshold_offset = base_ptr.add(off) as *mut i32;
    off = (off + sz_thresh + 63) & !63;

    (*out_vram).timers = base_ptr.add(off);
    off = (off + sz_timers + 63) & !63;

    (*out_vram).soma_to_axon = base_ptr.add(off) as *mut u32;
    off = (off + sz_s2a + 63) & !63;

    (*out_vram).dendrite_targets = base_ptr.add(off) as *mut u32;
    off = (off + sz_targets + 63) & !63;

    (*out_vram).dendrite_weights = base_ptr.add(off) as *mut i32;
    off = (off + sz_weights + 63) & !63;

    (*out_vram).dendrite_timers = base_ptr.add(off);

    // Allocate memory for axon heads
    let total_axons_size = (total_axons as usize) * std::mem::size_of::<BurstHeads8>();
    let axons_ptr = alloc_aligned_with_prefix(total_axons_size, 32);
    if axons_ptr.is_null() {
        free_aligned_with_prefix(base_ptr, 64);
        return -1;
    }
    
    (*out_vram).axon_heads = axons_ptr as *mut BurstHeads8;
    
    // Initialize axons with sentinels
    let axon_slice = std::slice::from_raw_parts_mut((*out_vram).axon_heads, total_axons as usize);
    for h in axon_slice {
        *h = BurstHeads8::empty(AXON_SENTINEL);
    }

    0
}

// =============================================================================
// 1.4 Zero-Copy DMA and Free
// =============================================================================

/// # Safety
/// Caller must ensure pointers are valid and aligned to 64 bytes.
pub unsafe fn cpu_upload_state_blob(
    vram: *const ShardVramPtrs,
    state_blob: *const c_void,
    state_size: usize,
) -> i32 {
    std::ptr::copy_nonoverlapping(state_blob, (*vram).soma_voltage as *mut c_void, state_size);
    0
}

/// # Safety
/// Caller must ensure pointers are valid and aligned to 64 bytes.
pub unsafe fn cpu_upload_axons_blob(
    vram: *const ShardVramPtrs,
    axons_blob: *const c_void,
    axons_size: usize,
) -> i32 {
    std::ptr::copy_nonoverlapping(axons_blob, (*vram).axon_heads as *mut c_void, axons_size);
    0
}

/// # Safety
/// Caller must ensure `vram` is a valid pointer.
pub unsafe fn cpu_free_shard(vram: *mut ShardVramPtrs) {
    if vram.is_null() {
        return;
    }

    let ptrs = &mut *vram;

    if !ptrs.soma_voltage.is_null() {
        free_aligned_with_prefix(ptrs.soma_voltage as *mut u8, 64);
    }

    if !ptrs.axon_heads.is_null() {
        free_aligned_with_prefix(ptrs.axon_heads as *mut u8, 32);
    }

    std::ptr::write_bytes(vram, 0, 1);
}

// =============================================================================
// 1.5 Testing Logic
// =============================================================================

#[cfg(test)]
pub(crate) static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cu_shard_allocation_logic() {
        let mut ptrs: ShardVramPtrs = unsafe { std::mem::zeroed() };
        let padded_n = 64;
        let total_axons = 100;

        let res = unsafe { cpu_allocate_shard(padded_n, total_axons, &mut ptrs) };
        assert_eq!(res, 0);

        let base_addr = ptrs.soma_voltage as usize;
        assert_eq!(base_addr % 64, 0);

        let targets_addr = ptrs.dendrite_targets as usize;
        assert_eq!(targets_addr - base_addr, 14 * 64);

        unsafe {
            let first_head = *ptrs.axon_heads;
            assert_eq!(first_head.h0, AXON_SENTINEL);
        }

        unsafe { cpu_free_shard(&mut ptrs) };
        assert!(ptrs.soma_voltage.is_null());
        assert!(ptrs.axon_heads.is_null());
        assert!(ptrs.dendrite_targets.is_null());
    }

    #[test]
    fn test_constant_memory_layout() {
        assert_eq!(std::mem::size_of::<CpuConstantMemory>(), 1024);
        assert_eq!(std::mem::align_of::<CpuConstantMemory>(), 64);
    }
}
