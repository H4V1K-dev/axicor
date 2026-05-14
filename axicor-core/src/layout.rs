use crate::constants::MAX_DENDRITE_SLOTS;
use crate::types::Voltage;
use bytemuck::{Pod, Zeroable};

pub const MAX_DENDRITES: usize = MAX_DENDRITE_SLOTS;

/// Neuron type parameter structure.
/// 64 bytes = 1 GPU L2 cache line. 16 types  64B = 1024B = entire __constant__ buffer.
/// Exactly one cache line per type  100% Coalesced Access, zero False Sharing.
#[rustfmt::skip]
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct VariantParameters {
    // === Block 1: 32-bit (Offsets 0..20) ===
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_shift: u32,
    pub homeostasis_penalty: i32,
    pub spontaneous_firing_period_ticks: u32,

    // === Block 2: 16-bit (Offsets 20..28) ===
    pub initial_synapse_weight: u16,
    pub gsop_potentiation: u16,
    pub gsop_depression: u16,
    pub homeostasis_decay: u16,

    // === Block 3: 8-bit (Offsets 28..32) ===
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,
    pub signal_propagation_length: u8,
    pub is_inhibitory: u8, // 1 = true (GABA), 0 = false (Glu)

    // === Block 4: Arrays (Offsets 32..48) ===
    pub inertia_curve: [u8; 8],
    pub ahp_amplitude: u16,
    pub _pad: [u8; 6],

    // === Block 5: Adaptive Leak Hardware (Offsets 48..58) ===
    pub adaptive_leak_min_shift: i32, // 48..52
    pub adaptive_leak_gain: u16,      // 52..54
    pub adaptive_mode: u8,            // 54..55
    pub _leak_pad: [u8; 3],           // 55..58

    // === Block 6: Pad (Offsets 58..64) ===
    pub d1_affinity: u8,
    pub d2_affinity: u8,
    pub heartbeat_m: u32,
}

const _: () = assert!(std::mem::size_of::<VariantParameters>() == 64);
const _: () = assert!(
    std::mem::align_of::<VariantParameters>() == 64,
    "Alignment violation!"
);

// [DOD] 32-byte alignment guarantees 8 heads are loaded in 1 L1 cache transaction.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BurstHeads8 {
    pub h0: u32,
    pub h1: u32,
    pub h2: u32,
    pub h3: u32,
    pub h4: u32,
    pub h5: u32,
    pub h6: u32,
    pub h7: u32,
}

impl BurstHeads8 {
    pub const fn empty(sentinel: u32) -> Self {
        Self {
            h0: sentinel,
            h1: sentinel,
            h2: sentinel,
            h3: sentinel,
            h4: sentinel,
            h5: sentinel,
            h6: sentinel,
            h7: sentinel,
        }
    }
}

const _: () = assert!(std::mem::size_of::<BurstHeads8>() == 32);
const _: () = assert!(std::mem::align_of::<BurstHeads8>() == 32);

/// Warp Alignment: size must be a multiple of 64 (AMD Wavefront + L2 Cache Line)
pub const fn align_to_warp(count: usize) -> usize {
    (count + 63) & !63
}

/// State file header (.state)
/// Exactly 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateFileHeader {
    pub magic: [u8; 4], // "GSNS" (Axicor State)
    pub version: u32,
    pub padded_n: u32,
    pub total_axons: u32,
}

impl StateFileHeader {
    pub fn new(padded_n: u32, total_axons: u32) -> Self {
        Self {
            magic: *b"GSNS",
            version: 1,
            padded_n,
            total_axons,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return None;
        }
        unsafe { Some(&*(bytes.as_ptr() as *const Self)) }
    }
}

const _: () = assert!(std::mem::size_of::<StateFileHeader>() == 16);

/// Axon file header (.axons)
/// Exactly 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxonsFileHeader {
    pub magic: [u8; 4], // "GSAX" (Axicor Axons)
    pub version: u32,
    pub total_axons: u32,
    pub _padding: u32,
}

const _: () = assert!(std::mem::size_of::<AxonsFileHeader>() == 16);

impl AxonsFileHeader {
    pub fn new(total_axons: u32) -> Self {
        Self {
            magic: *b"GSAX",
            version: 1,
            total_axons,
            _padding: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return None;
        }
        unsafe { Some(&*(bytes.as_ptr() as *const Self)) }
    }
}

pub const PATHS_MAGIC: u32 = 0x50415448; // "PATH"
pub const MAX_SEGMENTS_PER_AXON: usize = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PathsFileHeader {
    pub magic: u32,
    pub version: u32,
    pub total_axons: u32,
    pub max_segments: u32,
}

const _: () = assert!(
    std::mem::size_of::<PathsFileHeader>() == 16,
    "PathsFileHeader must be 16 bytes"
);

/// Calculates exact size of `shard.paths` file.
/// Guarantees the segments matrix starts at an address multiple of 64 bytes.
pub const fn calculate_paths_file_size(total_axons: usize) -> usize {
    let header_sz = std::mem::size_of::<PathsFileHeader>(); // 16
    let lengths_sz = total_axons;

    // Align to 64 bytes
    let padding = (64 - ((header_sz + lengths_sz) % 64)) % 64;

    let matrix_sz = total_axons * MAX_SEGMENTS_PER_AXON * 4; // 4 bytes per PackedPosition

    header_sz + lengths_sz + padding + matrix_sz
}

/// Offset to start of the matrix
pub const fn calculate_paths_matrix_offset(total_axons: usize) -> usize {
    let header_sz = 16;
    let lengths_sz = total_axons;
    let padding = (64 - ((header_sz + lengths_sz) % 64)) % 64;
    header_sz + lengths_sz + padding
}

/// Host-side SoA state of a shard.
/// Used for baking and disk I/O.
// MONOLITH: LOW — ShardStateSoA and VramState are central data structures that may grow into blobs.
// REFACTOR: Ensure future extensions use discrete "State Planes" to maintain SoA cache efficiency.
#[repr(C)]
pub struct ShardStateSoA {
    pub padded_n: usize, // Must be multiple of 32 (Warp Alignment)

    // --- Soma Hot State ---
    pub voltage: Vec<Voltage>,
    pub flags: Vec<u8>,
    pub threshold_offset: Vec<i32>,
    pub refractory_timer: Vec<u8>,

    // --- Columnar Dendrites (Size = MAX_DENDRITES * padded_n) ---
    pub dendrite_targets: Vec<u32>, // Dense ID + Segment Offset
    pub dendrite_weights: Vec<i32>,
    pub dendrite_timers: Vec<u8>,

    // --- Axon Heads (Size = total_axons) ---
    pub axon_heads: Vec<BurstHeads8>,
}

impl ShardStateSoA {
    /// Initializes a new shard.
    /// - `padded_n`: number of neurons (multiple of 32).
    /// - `total_axons`: total axons (local + ghost + virtual).
    pub fn new(padded_n: usize, total_axons: usize) -> Self {
        assert!(
            padded_n.is_multiple_of(32),
            "padded_n must be warp-aligned (multiple of 32)"
        );

        Self {
            padded_n,
            voltage: vec![0; padded_n],
            flags: vec![0; padded_n],
            threshold_offset: vec![0; padded_n],
            refractory_timer: vec![0; padded_n],

            dendrite_targets: vec![0; MAX_DENDRITES * padded_n],
            dendrite_weights: vec![0; MAX_DENDRITES * padded_n],
            dendrite_timers: vec![0; MAX_DENDRITES * padded_n],

            axon_heads: vec![BurstHeads8::empty(0); total_axons],
        }
    }

    /// Calculates flat index for Coalesced Access on GPU
    #[inline(always)]
    pub fn columnar_idx(padded_n: usize, neuron_idx: usize, slot: usize) -> usize {
        assert!(neuron_idx < padded_n && slot < MAX_DENDRITES,
    "columnar_idx: neuron_idx={neuron_idx} >= padded_n={padded_n} or slot={slot} >= {MAX_DENDRITES}");
        slot * padded_n + neuron_idx
    }
}

// ---------------------------------------------------------------------------
// 1.3 Dendrite Target Packing (Preventing the Zero-Index Trap)
// ---------------------------------------------------------------------------

use crate::constants::TARGET_SEG_SHIFT;

/// Packs Axon_ID and segment offset.
/// Applies +1 to Axon_ID so target == 0 always means "empty slot".
#[inline(always)]
pub fn pack_dendrite_target(axon_id: u32, segment_offset: u32) -> u32 {
    // u32::MAX is the marker for an empty pixel/slot at compile time
    if axon_id == u32::MAX { 
        return 0; // Hardware Early Exit trigger for GPU
    }
    // Offset + 1. Protects the 0th axon from becoming 0x00000000
    let safe_id = (axon_id + 1) & 0x00FFFFFF;
    let offset = (segment_offset & 0xFF) << 24;
    offset | safe_id
}

/// Extracts Axon_ID (accounting for -1 reverse shift).
#[inline(always)]
pub fn unpack_axon_id(target_packed: u32) -> u32 {
    let raw = target_packed & 0x00FFFFFF;
    if raw == 0 {
        return u32::MAX;
    }
    raw - 1
}

/// Extracts segment offset [0..255].
#[inline(always)]
pub const fn unpack_segment_offset(target_packed: u32) -> u32 {
    target_packed >> TARGET_SEG_SHIFT
}

/// CUDA-compatible structure for SoA FFI.
/// Contains raw pointers to Host/Device memory.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VramState {
    pub padded_n: u32,
    pub total_axons: u32,

    // Soma Hot State
    pub voltage: *mut i32,
    pub flags: *mut u8,
    pub threshold_offset: *mut i32,
    pub refractory_timer: *mut u8,
    pub soma_to_axon: *mut u32,

    // Dendrites (Size: 128 * padded_n)
    pub dendrite_targets: *mut u32,
    pub dendrite_weights: *mut i32,
    pub dendrite_timers: *mut u8,

    // Axons (Size: total_axons)
    pub axon_heads: *mut BurstHeads8,

    // I/O & Telemetry
    pub input_bitmask: *mut u32,
    pub output_history: *mut u8,
    pub telemetry_count: *mut u32,
    pub telemetry_spikes: *mut u32,
}

impl VramState {
    /// # Safety
    /// The caller must ensure that `ShardStateSoA` arrays are padded to a multiple of 32
    /// and correctly aligned. Data lifetime must exceed the returned layout struct.
    pub unsafe fn from_soa(soa: &mut ShardStateSoA) -> Self {
        Self {
            padded_n: soa.padded_n as u32,
            total_axons: soa.axon_heads.len() as u32,

            voltage: soa.voltage.as_mut_ptr(),
            flags: soa.flags.as_mut_ptr(),
            threshold_offset: soa.threshold_offset.as_mut_ptr(),
            refractory_timer: soa.refractory_timer.as_mut_ptr(),
            soma_to_axon: std::ptr::null_mut(),

            dendrite_targets: soa.dendrite_targets.as_mut_ptr(),
            dendrite_weights: soa.dendrite_weights.as_mut_ptr(),
            dendrite_timers: soa.dendrite_timers.as_mut_ptr(),

            axon_heads: soa.axon_heads.as_mut_ptr(),

            input_bitmask: std::ptr::null_mut(),
            output_history: std::ptr::null_mut(),
            telemetry_count: std::ptr::null_mut(),
            telemetry_spikes: std::ptr::null_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_soa_allocation() {
        let n = 1024;
        let axons = 5000;
        let soa = ShardStateSoA::new(n, axons);

        assert_eq!(soa.padded_n, n);
        assert_eq!(soa.voltage.len(), n);
        assert_eq!(soa.dendrite_weights.len(), n * 128);
        assert_eq!(soa.axon_heads.len(), axons);
    }

    #[test]
    fn test_vram_state_pointer_mapping() {
        let mut soa = ShardStateSoA::new(32, 100);
        soa.voltage[0] = 42;
        soa.axon_heads[99] = BurstHeads8::empty(123);

        unsafe {
            let vram = VramState::from_soa(&mut soa);
            assert_eq!(vram.padded_n, 32);
            assert_eq!(vram.total_axons, 100);
            assert_eq!(*vram.voltage, 42);
            assert_eq!((*vram.axon_heads.add(99)).h0, 123);
        }
    }

    #[test]
    fn test_header_sizes() {
        assert_eq!(std::mem::size_of::<StateFileHeader>(), 16);
        assert_eq!(std::mem::size_of::<AxonsFileHeader>(), 16);
    }

    #[test]
    fn test_columnar_idx_logic() {
        let n = 1024;
        // slot 0, neuron 0 -> 0
        assert_eq!(ShardStateSoA::columnar_idx(n, 0, 0), 0);
        // slot 1, neuron 0 -> 1024
        assert_eq!(ShardStateSoA::columnar_idx(n, 0, 1), 1024);
        // slot 0, neuron 1 -> 1
        assert_eq!(ShardStateSoA::columnar_idx(n, 1, 0), 1);
    }

    #[test]
    fn test_dendrite_target_packing() {
        // Zero-Index Trap check: axon 0, segment 0 -> must NOT be 0
        let t0 = pack_dendrite_target(0, 0);
        assert_ne!(t0, 0, "Zero-Index Trap: axon=0, seg=0 packed to 0!");
        assert_eq!(unpack_axon_id(t0), 0);
        assert_eq!(unpack_segment_offset(t0), 0);

        // Max range check
        let t_max = pack_dendrite_target(0x00FF_FFFE, 255);
        assert_eq!(unpack_axon_id(t_max), 0x00FF_FFFE);
        assert_eq!(unpack_segment_offset(t_max), 255);

        // Check mask isolation
        let t_mix = pack_dendrite_target(0x123456, 0xAB);
        assert_eq!(unpack_axon_id(t_mix), 0x123456);
        assert_eq!(unpack_segment_offset(t_mix), 0xAB);
    }
}
