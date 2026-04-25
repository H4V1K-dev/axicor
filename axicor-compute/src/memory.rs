use crate::ffi;
use crate::ffi::ShardVramPtrs;
use axicor_core::layout::align_to_warp;
use std::ffi::c_void;

// =============================================================================
//  SoA Layout Calculator
//
//  Calculates the memory layout for the .state blob to ensure binary 
//  compatibility between Baker (host) and Compute (GPU) backends.
//  Enforces strict 64-byte alignment for DMA-friendly PCIe transactions.
// =============================================================================

/// Maximum number of dendrites per neuron (Hardware Hard Constraint: 128).
pub const MAX_DENDRITES: usize = 128;

/// Calculates (padded_n, total_state_bytes) required for the state allocation.
///
/// **.state Binary Layout** (Strictly matches ShardVramPtrs field order):
/// ```text
/// soma_voltage      [padded_n]             4 B (i32)
/// soma_flags        [padded_n]             1 B (u8)
/// threshold_offset  [padded_n]             4 B (i32)
/// timers            [padded_n]             1 B (u8)
/// soma_to_axon      [padded_n]             4 B (u32)
/// dendrite_targets  [padded_n * 128]       4 B (u32)
/// dendrite_weights  [padded_n * 128]       4 B (i32, Mass Domain)
/// dendrite_timers   [padded_n * 128]       1 B (u8)
/// ```
/// Note: `axon_heads` are stored in a separate .axons blob.
#[inline]
pub fn calculate_state_blob_size(neuron_count: usize) -> (usize, usize) {
    let padded_n = align_to_warp(neuron_count);

    let soma_voltage_sz = padded_n * std::mem::size_of::<i32>();
    let soma_flags_sz = padded_n * std::mem::size_of::<u8>();
    let threshold_sz = padded_n * std::mem::size_of::<i32>();
    let timers_sz = padded_n * std::mem::size_of::<u8>();
    let soma_to_axon_sz = padded_n * std::mem::size_of::<u32>();
    let dendrite_targets_sz = padded_n * MAX_DENDRITES * std::mem::size_of::<u32>();
    let dendrite_weights_sz = padded_n * MAX_DENDRITES * std::mem::size_of::<i32>();
    let dendrite_timers_sz = padded_n * MAX_DENDRITES * std::mem::size_of::<u8>();

    let mut total = 0;
    total = (total + soma_voltage_sz + 63) & !63;
    total = (total + soma_flags_sz + 63) & !63;
    total = (total + threshold_sz + 63) & !63;
    total = (total + timers_sz + 63) & !63;
    total = (total + soma_to_axon_sz + 63) & !63;
    total = (total + dendrite_targets_sz + 63) & !63;
    total = (total + dendrite_weights_sz + 63) & !63;
    total = (total + dendrite_timers_sz + 63) & !63;

    (padded_n, total)
}

/// Offsets within the .state blob for DMA pointer arithmetic.
/// Used by both Baker and Compute to synchronize memory mapping.
#[derive(Debug, Clone, Copy)]
pub struct StateOffsets {
    pub soma_voltage: usize,
    pub soma_flags: usize,
    pub threshold_offset: usize,
    pub timers: usize,
    pub soma_to_axon: usize,
    pub dendrite_targets: usize,
    pub dendrite_weights: usize,
    pub dendrite_timers: usize,
    pub total_bytes: usize,
}

pub fn compute_state_offsets(padded_n: usize) -> StateOffsets {
    let mut off = 0;
    let soma_voltage = off;
    off = (off + padded_n * 4 + 63) & !63;
    let soma_flags = off;
    off = (off + padded_n + 63) & !63;
    let threshold_offset = off;
    off = (off + padded_n * 4 + 63) & !63;
    let timers = off;
    off = (off + padded_n + 63) & !63;
    let soma_to_axon = off;
    off = (off + padded_n * 4 + 63) & !63;
    let dendrite_targets = off;
    off = (off + padded_n * MAX_DENDRITES * 4 + 63) & !63;
    let dendrite_weights = off;
    off = (off + padded_n * MAX_DENDRITES * 4 + 63) & !63;
    let dendrite_timers = off;
    off = (off + padded_n * MAX_DENDRITES + 63) & !63;
    StateOffsets {
        soma_voltage,
        soma_flags,
        threshold_offset,
        timers,
        soma_to_axon,
        dendrite_targets,
        dendrite_weights,
        dendrite_timers,
        total_bytes: off,
    }
}

// =============================================================================
//  VramState - Shard VRAM Lifecycle Manager
//
//  Handles GPU-side memory allocation and DMA synchronization.
//  Lifecycle:
//    allocate() -> cu_allocate_shard (Hardware Batch Allocation)
//    upload()   -> cu_upload_state_blob (Zero-Copy DMA transaction)
//    Drop       -> cu_free_shard (VRAM reclamation)
// =============================================================================

pub struct VramState {
    pub ptrs: ShardVramPtrs,
    pub padded_n: u32,
    pub total_axons: u32,
    pub total_ghosts: u32,
    pub use_gpu: bool,
}

unsafe impl Send for VramState {}
unsafe impl Sync for VramState {}

impl VramState {
    /// Allocates device or host-mocked memory for the entire shard state.
    pub fn allocate(padded_n: u32, total_axons: u32, total_ghosts: u32, use_gpu: bool) -> Self {
        let mut ptrs = ShardVramPtrs {
            soma_voltage: std::ptr::null_mut(),
            soma_flags: std::ptr::null_mut(),
            threshold_offset: std::ptr::null_mut(),
            timers: std::ptr::null_mut(),
            soma_to_axon: std::ptr::null_mut(),
            dendrite_targets: std::ptr::null_mut(),
            dendrite_weights: std::ptr::null_mut(),
            dendrite_timers: std::ptr::null_mut(),
            axon_heads: std::ptr::null_mut(),
        };

        if use_gpu {
            let err = unsafe { ffi::cu_allocate_shard(padded_n, total_axons, &mut ptrs) };
            assert_eq!(
                err, 0,
                "FATAL: cu_allocate_shard (GPU) failed (cudaError={})",
                err
            );
        } else {
            let err =
                unsafe { crate::bindings::cpu_allocate_shard(padded_n, total_axons, &mut ptrs) };
            assert_eq!(err, 0, "FATAL: cpu_allocate_shard failed (res={})", err);
        }

        Self {
            ptrs,
            padded_n,
            total_axons,
            total_ghosts,
            use_gpu,
        }
    }

    // [DOD FIX] Strict VRAM Layout: Local -> Virtual -> Ghosts
    pub fn virtual_offset(&self) -> u32 {
        self.padded_n
    }

    /// Performs Zero-Copy DMA: uploads the .state blob directly to VRAM.
    pub fn upload_state(&self, flat_blob: &[u8]) {
        let (_, expected) = calculate_state_blob_size(self.padded_n as usize);
        assert_eq!(
            flat_blob.len(),
            expected,
            "FATAL: .state blob size mismatch: got {} expected {}",
            flat_blob.len(),
            expected
        );

        if self.use_gpu {
            let err = unsafe {
                ffi::cu_upload_state_blob(
                    &self.ptrs,
                    flat_blob.as_ptr() as *const c_void,
                    flat_blob.len(),
                )
            };
            assert_eq!(
                err, 0,
                "FATAL: cu_upload_state_blob DMA failed (cudaError={})",
                err
            );
        } else {
            let err = unsafe {
                crate::bindings::cpu_upload_state_blob(
                    &self.ptrs,
                    flat_blob.as_ptr() as *const c_void,
                    flat_blob.len(),
                )
            };
            assert_eq!(err, 0, "FATAL: cpu_upload_state_blob failed");
        }
    }

    /// Only axons are stored separately: loads `axon_heads` directly.
    pub fn upload_axon_heads(&self, axon_heads_blob: &[u8]) {
        let expected =
            (self.total_axons as usize) * std::mem::size_of::<axicor_core::layout::BurstHeads8>();
        let actual = axon_heads_blob.len();

        if actual > expected {
            panic!(
                "FATAL: axon_heads blob too large: got {} expected max {}",
                actual, expected
            );
        }

        if actual == 0 {
            return;
        }

        if self.use_gpu {
            let err = unsafe {
                ffi::cu_upload_axons_blob(
                    &self.ptrs,
                    axon_heads_blob.as_ptr() as *const c_void,
                    actual,
                )
            };
            assert_eq!(
                err, 0,
                "FATAL: cu_upload_axons_blob failed (cudaError={})",
                err
            );
        } else {
            let err = unsafe {
                crate::bindings::cpu_upload_axons_blob(
                    &self.ptrs,
                    axon_heads_blob.as_ptr() as *const c_void,
                    actual,
                )
            };
            assert_eq!(err, 0, "FATAL: cpu_upload_axons_blob failed");
        }
    }
}

impl Drop for VramState {
    fn drop(&mut self) {
        unsafe {
            if self.use_gpu {
                ffi::cu_free_shard(&mut self.ptrs);
            } else {
                crate::bindings::cpu_free_shard(&mut self.ptrs);
            }
        }
    }
}

// =============================================================================
//  PinnedBuffer - DMA-Optimized Memory (Page-Locked RAM)
//
//  Essential for high-speed cudaMemcpyAsync operations.
//  Prevents CPU-side page faults during PCIe transactions.
// =============================================================================

pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> PinnedBuffer<T> {
    pub fn new(len: usize) -> anyhow::Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }
        let bytes = len * std::mem::size_of::<T>();
        let ptr = unsafe { ffi::gpu_host_alloc(bytes) } as *mut T;
        if ptr.is_null() {
            anyhow::bail!("PinnedBuffer: gpu_host_alloc failed for {} bytes", bytes);
        }
        Ok(Self { ptr, len })
    }

    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::gpu_host_free(self.ptr as *mut c_void) };
        }
    }
}

unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}
