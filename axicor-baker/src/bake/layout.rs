use axicor_compute::memory::{calculate_state_blob_size, MAX_DENDRITES};
use axicor_core::constants::{AXON_SENTINEL, MAX_DENDRITE_SLOTS};
use axicor_core::layout::align_to_warp;
use bytemuck::cast_slice;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Strict contract for local shard data after Phase A (before inter-zone connections).
pub struct CompiledShard {
    pub _zone_name: String,
    pub local_axons_count: usize,
    pub ghost_offset_base: u32, // [DOD FIX] Exact start index for ghosts
    pub ghost_capacity_limit: u32, // [DOD FIX] Hard limit from config
    /// Dense ID -> Axon ID mapping
    pub soma_to_axon_map: Vec<u32>,
    /// Packed 32-bit coordinates (X|Y|Z|Type)
    pub packed_positions: Vec<u32>,
    /// Physical zone dimensions in voxels (W, D, H)
    pub _bounds_voxels: (u32, u32, u32),
    /// Physical dimensions in microns (W, D) for the UV-Atlas
    pub bounds_um: (f32, f32),
}

/// Intermediate SoA structure on CPU before disk dump.
/// Guarantees proper padding for CUDA warps.
pub struct ShardSoA {
    pub padded_n: usize,
    pub _total_axons: usize,

    // Dynamic soma state
    pub voltage: Vec<i32>,
    pub flags: Vec<u8>,
    pub threshold_offset: Vec<i32>,
    pub refractory_timer: Vec<u8>,

    // Transposed dendrite matrix (Columnar Layout)
    pub dendrite_targets: Vec<u32>,
    pub dendrite_weights: Vec<i32>,
    pub dendrite_timers: Vec<u8>, // Refractory timers for synapses

    // Axons
    pub axon_heads: Vec<axicor_core::layout::BurstHeads8>,
    pub axon_tips_uvw: Vec<u32>, // PackedTip -> .geom
    pub axon_dirs_xyz: Vec<u32>, // PackedDir -> .geom

    // NEW FIELDS
    pub axon_lengths: Vec<u8>, // size: total_axons
    pub axon_paths: Vec<u32>,  // size: total_axons * 256

    // Mapping: soma_idx  axon_idx
    pub soma_to_axon: Vec<u32>,

    /// Packed soma positions (u32: 11-bit X, 11-bit Y, 6-bit Z, 4-bit Type)
    pub soma_positions: Vec<u32>,
}

impl ShardSoA {
    /// Allocates arrays of required size, filling them with zeros or sentinels.
    /// Automatically applies align_to_warp for N and Axons.
    pub fn new(raw_neuron_count: usize, raw_axon_count: usize) -> Self {
        let padded_n = align_to_warp(raw_neuron_count);
        let total_axons = align_to_warp(raw_axon_count);

        Self {
            padded_n,
            _total_axons: total_axons,
            voltage: vec![0; padded_n],
            flags: vec![0; padded_n],
            threshold_offset: vec![0; padded_n],
            refractory_timer: vec![0; padded_n],

            dendrite_targets: vec![0; MAX_DENDRITES * padded_n],
            dendrite_weights: vec![0; MAX_DENDRITES * padded_n],
            dendrite_timers: vec![0; MAX_DENDRITES * padded_n],

            // Hard invariant: empty axons MUST be 0x80000000
            axon_heads: vec![axicor_core::layout::BurstHeads8::empty(AXON_SENTINEL); total_axons],
            axon_tips_uvw: vec![0; total_axons],
            axon_dirs_xyz: vec![0; total_axons],

            axon_lengths: vec![0; total_axons],
            axon_paths: vec![0; total_axons * axicor_core::layout::MAX_SEGMENTS_PER_AXON],

            soma_to_axon: vec![u32::MAX; padded_n],
            soma_positions: vec![0; padded_n],
        }
    }

    /// Calculates flat index for Coalesced Access on GPU.
    #[inline(always)]
    pub fn _columnar_idx(padded_n: usize, neuron_idx: usize, slot: usize) -> usize {
        debug_assert!(neuron_idx < padded_n && slot < MAX_DENDRITE_SLOTS);
        slot * padded_n + neuron_idx
    }

    /// Dumps SoA structures to binary files. Zero-cost for loading at runtime.
    pub fn dump_to_disk(&self, out_dir: &Path) {
        let state_path = out_dir.join("shard.state");
        write_state_blob(
            &state_path,
            self.padded_n,
            &self.voltage,
            &self.flags,
            &self.threshold_offset,
            &self.refractory_timer,
            &self.soma_to_axon,
            &self.dendrite_targets,
            &self.dendrite_weights,
            &self.dendrite_timers,
        )
        .expect("Failed to write state blob");

        let axons_path = out_dir.join("shard.axons");
        write_axons_blob(&axons_path, &self.axon_heads).expect("Failed to write axons blob");

        // [DOD FIX] Geometry export for Night Phase
        let paths_path = out_dir.join("shard.paths");
        write_paths_blob(
            &paths_path,
            self.axon_heads.len(),
            &self.axon_lengths,
            &self.axon_paths,
        )
        .expect("Failed to write paths blob");

        let pos_path = out_dir.join("shard.pos");
        std::fs::write(&pos_path, bytemuck::cast_slice(&self.soma_positions))
            .expect("Failed to write pos blob");
    }
}

/// Zero-Cost assembly of .state binary blob.
/// Arrays must have length `padded_n`.
#[allow(clippy::too_many_arguments)]
pub fn write_state_blob(
    path: &Path,
    padded_n: usize,
    voltages: &[i32],
    flags: &[u8],
    thresholds: &[i32],
    timers: &[u8],
    soma_to_axon: &[u32],
    dendrite_targets: &[u32], // Length: padded_n * 128
    dendrite_weights: &[i32], // Length: padded_n * 128
    dendrite_timers: &[u8],   // Length: padded_n * 128
) -> std::io::Result<()> {
    let (_, total_size) = calculate_state_blob_size(padded_n);

    // Pre-allocate exact size to prevent memory reallocations
    let mut blob = Vec::with_capacity(total_size);

    // [DOD FIX] C-ABI Warp Alignment Padding for Monolithic DMA
    macro_rules! push_aligned {
        ($data:expr) => {
            blob.extend_from_slice(cast_slice($data));
            let pad = (64 - (blob.len() % 64)) % 64;
            blob.extend(std::iter::repeat(0).take(pad));
        };
    }

    push_aligned!(&voltages[..padded_n]);
    push_aligned!(&flags[..padded_n]);
    push_aligned!(&thresholds[..padded_n]);
    push_aligned!(&timers[..padded_n]);
    push_aligned!(&soma_to_axon[..padded_n]);
    push_aligned!(&dendrite_targets[..padded_n * MAX_DENDRITES]);
    push_aligned!(&dendrite_weights[..padded_n * MAX_DENDRITES]);
    push_aligned!(&dendrite_timers[..padded_n * MAX_DENDRITES]);

    assert_eq!(
        blob.len(),
        total_size,
        "FATAL: State blob size mismatch before disk flush. Check axicor-compute/src/memory.rs alignment!"
    );

    let mut file = File::create(path)?;
    file.write_all(&blob)?;
    Ok(())
}

/// Export of .axons blob.
pub fn write_axons_blob(
    path: &Path,
    axon_heads: &[axicor_core::layout::BurstHeads8], // Length: total_axons
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(cast_slice(axon_heads))?;
    Ok(())
}

pub fn write_paths_blob(
    path: &std::path::Path,
    total_axons: usize,
    lengths: &[u8],
    paths_matrix: &[u32],
) -> std::io::Result<()> {
    let total_size = axicor_core::layout::calculate_paths_file_size(total_axons);
    let matrix_offset = axicor_core::layout::calculate_paths_matrix_offset(total_axons);

    let mut blob = vec![0u8; total_size];

    // Header
    let header = axicor_core::layout::PathsFileHeader {
        magic: axicor_core::layout::PATHS_MAGIC,
        version: 1,
        total_axons: total_axons as u32,
        max_segments: axicor_core::layout::MAX_SEGMENTS_PER_AXON as u32,
    };

    // Copying without allocations
    unsafe {
        std::ptr::copy_nonoverlapping(&header as *const _ as *const u8, blob.as_mut_ptr(), 16);

        std::ptr::copy_nonoverlapping(lengths.as_ptr(), blob.as_mut_ptr().add(16), total_axons);

        std::ptr::copy_nonoverlapping(
            paths_matrix.as_ptr() as *const u8,
            blob.as_mut_ptr().add(matrix_offset),
            total_axons * axicor_core::layout::MAX_SEGMENTS_PER_AXON * 4,
        );
    }

    std::fs::write(path, &blob)
}
