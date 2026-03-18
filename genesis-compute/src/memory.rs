use genesis_core::layout::align_to_warp;
use std::ffi::c_void;
use crate::ffi;
use crate::ffi::ShardVramPtrs;

// =============================================================================
// § SoA Layout Calculator
//
// Единственный источник правды для размеров массивов .state-блоба.
// baker использует эти цифры для сериализации.
// compute использует их для валидации перед DMA.
// =============================================================================

/// Максимальное число дендритных слотов на нейрон (Hard Constraint §8).
pub const MAX_DENDRITES: usize = 128;

/// Возвращает (padded_n, total_state_bytes) для шарда с данным числом нейронов.
///
/// **Порядок байт в .state блобе** (должен совпадать с ShardVramPtrs):
/// ```text
/// soma_voltage      [padded_n]            × 4 B
/// soma_flags        [padded_n]            × 1 B
/// threshold_offset  [padded_n]            × 4 B
/// timers            [padded_n]            × 1 B
/// soma_to_axon      [padded_n]            × 4 B
/// dendrite_targets  [padded_n × 128]      × 4 B
/// dendrite_weights  [padded_n × 128]      × 2 B
/// dendrite_timers   [padded_n × 128]      × 1 B
/// ```
/// Массив `axon_heads` хранится в отдельном `.axons`-файле.
#[inline]
pub fn calculate_state_blob_size(neuron_count: usize) -> (usize, usize) {
    let padded_n = align_to_warp(neuron_count);

    let soma_voltage_sz     = padded_n * std::mem::size_of::<i32>();
    let soma_flags_sz       = padded_n * std::mem::size_of::<u8>();
    let threshold_sz        = padded_n * std::mem::size_of::<i32>();
    let timers_sz           = padded_n * std::mem::size_of::<u8>();
    let soma_to_axon_sz     = padded_n * std::mem::size_of::<u32>();
    let dendrite_targets_sz = padded_n * MAX_DENDRITES * std::mem::size_of::<u32>();
    let dendrite_weights_sz = padded_n * MAX_DENDRITES * std::mem::size_of::<i16>();
    let dendrite_timers_sz  = padded_n * MAX_DENDRITES * std::mem::size_of::<u8>();

    let total = soma_voltage_sz
        + soma_flags_sz
        + threshold_sz
        + timers_sz
        + soma_to_axon_sz
        + dendrite_targets_sz
        + dendrite_weights_sz
        + dendrite_timers_sz;

    (padded_n, total)
}

/// Точные байтовые смещения каждого поля внутри .state блоба.
/// **Используется baker'ом для сериализации и compute для DMA pointer arithmetic.**
#[derive(Debug, Clone, Copy)]
pub struct StateOffsets {
    pub soma_voltage:     usize,
    pub soma_flags:       usize,
    pub threshold_offset: usize,
    pub timers:           usize,
    pub soma_to_axon:     usize,
    pub dendrite_targets: usize,
    pub dendrite_weights: usize,
    pub dendrite_timers:  usize,
    pub total_bytes:      usize,
}

pub fn compute_state_offsets(padded_n: usize) -> StateOffsets {
    let mut off = 0;
    let soma_voltage     = off; off += padded_n * 4;
    let soma_flags       = off; off += padded_n * 1;
    let threshold_offset = off; off += padded_n * 4;
    let timers           = off; off += padded_n * 1;
    let soma_to_axon     = off; off += padded_n * 4;
    let dendrite_targets = off; off += padded_n * MAX_DENDRITES * 4;
    let dendrite_weights = off; off += padded_n * MAX_DENDRITES * 2;
    let dendrite_timers  = off; off += padded_n * MAX_DENDRITES * 1;
    StateOffsets {
        soma_voltage, soma_flags, threshold_offset, timers,
        soma_to_axon, dendrite_targets, dendrite_weights, dendrite_timers,
        total_bytes: off,
    }
}

// =============================================================================
// § VramState — Владелец VRAM для одного шарда
//
// Единственный источник правды для GPU-ресурсов.
// Правила:
//  • allocate() → один cu_allocate_shard (внутри CUDA один BatchAlloc)
//  • upload()   → один cu_upload_state_blob (Zero-Copy DMA)
//  • Drop       → cu_free_shard
// =============================================================================

pub struct VramState {
    pub ptrs:        ShardVramPtrs,
    pub padded_n:    u32,
    pub total_axons: u32,
    pub total_ghosts: u32,
}

unsafe impl Send for VramState {}
unsafe impl Sync for VramState {}

impl VramState {
    /// Аллоцирует VRAM для шарда. Паникует при ошибке CUDA — это Fail-Fast политика.
    pub fn allocate(padded_n: u32, total_axons: u32, total_ghosts: u32) -> Self {
        let mut ptrs = ShardVramPtrs {
            soma_voltage:     std::ptr::null_mut(),
            soma_flags:       std::ptr::null_mut(),
            threshold_offset: std::ptr::null_mut(),
            timers:           std::ptr::null_mut(),
            soma_to_axon:     std::ptr::null_mut(),
            dendrite_targets: std::ptr::null_mut(),
            dendrite_weights: std::ptr::null_mut(),
            dendrite_timers:  std::ptr::null_mut(),
            axon_heads:       std::ptr::null_mut(),
        };
        let err = unsafe { ffi::cu_allocate_shard(padded_n, total_axons, &mut ptrs) };
        assert_eq!(err, 0, "FATAL: cu_allocate_shard failed (cudaError={})", err);
        Self { ptrs, padded_n, total_axons, total_ghosts }
    }

    // [DOD FIX] Virtual axons start after local AND ghosts
    pub fn virtual_offset(&self) -> u32 {
        self.padded_n + self.total_ghosts 
    }

    /// Zero-Copy DMA: заливает плоский .state блоб в VRAM.
    /// Размер блоба верифицируется до начала DMA — никакого partial write.
    pub fn upload_state(&self, flat_blob: &[u8]) {
        let (_, expected) = calculate_state_blob_size(self.padded_n as usize);
        assert_eq!(
            flat_blob.len(), expected,
            "FATAL: .state blob size mismatch: got {} expected {}. Baker/Runtime version skew!",
            flat_blob.len(), expected
        );
        let err = unsafe {
            ffi::cu_upload_state_blob(
                &self.ptrs,
                flat_blob.as_ptr() as *const c_void,
                flat_blob.len(),
            )
        };
        assert_eq!(err, 0, "FATAL: cu_upload_state_blob DMA failed (cudaError={})", err);
    }

    /// Только аксоны хранятся отдельно: загружает `axon_heads` напрямую.
    pub fn upload_axon_heads(&self, axon_heads_blob: &[u8]) {
        let expected = (self.total_axons as usize) * std::mem::size_of::<genesis_core::layout::BurstHeads8>();
        let actual = axon_heads_blob.len();
        
        // [DOD FIX] Allow smaller blobs for pre-allocated Dynamic Capacity Routing.
        // We only upload what we have on disk; the rest remains initialized (Sentinel).
        if actual > expected {
            panic!(
                "FATAL: axon_heads blob too large: got {} expected max {} (Check your ghost_capacity in manifest.toml vs baked artifacts)",
                actual, expected
            );
        }
        
        if actual == 0 { return; }

        let err = unsafe {
            ffi::cu_upload_axons_blob(
                &self.ptrs,
                axon_heads_blob.as_ptr() as *const c_void,
                actual,
            )
        };
        assert_eq!(err, 0, "FATAL: cu_upload_axons_blob failed (cudaError={})", err);
    }
}

impl Drop for VramState {
    fn drop(&mut self) {
        unsafe {
            ffi::cu_free_shard(&mut self.ptrs);
        }
    }
}

// =============================================================================
// § PinnedBuffer — DMA-ready Memory (Pinned RAM)
//
// Используется для входных масок и истории выходов, чтобы cudaMemcpyAsync
// работал на максимальной скорости PCIe.
// =============================================================================

pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> PinnedBuffer<T> {
    pub fn new(len: usize) -> anyhow::Result<Self> {
        if len == 0 {
            return Ok(Self { ptr: std::ptr::null_mut(), len: 0 });
        }
        let bytes = len * std::mem::size_of::<T>();
        let ptr = unsafe { ffi::gpu_host_alloc(bytes) } as *mut T;
        if ptr.is_null() {
            anyhow::bail!("PinnedBuffer: gpu_host_alloc failed for {} bytes", bytes);
        }
        Ok(Self { ptr, len })
    }

    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 { return &[]; }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 { return &mut []; }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&self) -> *mut T { self.ptr }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
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
