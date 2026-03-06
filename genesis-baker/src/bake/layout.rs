use genesis_core::constants::{MAX_DENDRITE_SLOTS, AXON_SENTINEL};
use genesis_core::layout::align_to_warp;
use std::io::Write;
use std::path::Path;
use std::fs::File;
use bytemuck::cast_slice;
use genesis_compute::memory::{calculate_state_blob_size, MAX_DENDRITES};

/// Строгий контракт данных локального шарда после Фазы A (до межзональных связей).
pub struct CompiledShard {
    pub zone_name: String,
    pub local_axons_count: usize,
    /// Маппинг Dense ID -> Axon ID
    pub soma_to_axon_map: Vec<u32>,
    /// Упакованные 32-битные координаты (X|Y|Z|Type)
    pub packed_positions: Vec<u32>,
    /// Физические размеры зоны в вокселях (W, D, H)
    pub bounds_voxels: (u32, u32, u32),
    /// Физические размеры в микронах (W, D) для UV-Атласа
    pub bounds_um: (f32, f32),
}

/// Промежуточная SoA-структура на CPU перед дампом на диск.
/// Гарантирует правильный padding для CUDA варпов.
pub struct ShardSoA {
    pub padded_n: usize,
    pub total_axons: usize,

    // Динамическое состояние сом
    pub voltage: Vec<i32>,
    pub flags: Vec<u8>,
    pub threshold_offset: Vec<i32>,
    pub refractory_timer: Vec<u8>,

    // Транспонированная матрица дендритов (Columnar Layout)
    pub dendrite_targets: Vec<u32>,
    pub dendrite_weights: Vec<i16>,
    pub dendrite_timers: Vec<u8>, // Refractory timers for synapses

    // Аксоны
    pub axon_heads: Vec<genesis_core::layout::BurstHeads8>,
    pub axon_tips_uvw: Vec<u32>, // PackedTip -> .geom
    pub axon_dirs_xyz: Vec<u32>, // PackedDir -> .geom

    // Маппинг: soma_idx → axon_idx
    pub soma_to_axon: Vec<u32>,

    /// Упакованные позиции сом (u32: 11-бит X, 11-бит Y, 6-бит Z, 4-бит Type)
    pub soma_positions: Vec<u32>,
}

impl ShardSoA {
    /// Аллоцирует массивы нужного размера, заполняя их нулями или сентинелами.
    /// Автоматически применяет align_to_warp для N и Axons.
    pub fn new(raw_neuron_count: usize, raw_axon_count: usize) -> Self {
        let padded_n = align_to_warp(raw_neuron_count);
        let total_axons = align_to_warp(raw_axon_count);

        Self {
            padded_n,
            total_axons,
            voltage: vec![0; padded_n],
            flags: vec![0; padded_n],
            threshold_offset: vec![0; padded_n],
            refractory_timer: vec![0; padded_n],

            dendrite_targets: vec![0; MAX_DENDRITES * padded_n],
            dendrite_weights: vec![0; MAX_DENDRITES * padded_n],
            dendrite_timers: vec![0; MAX_DENDRITES * padded_n],

            // Хард-инвариант: пустые аксоны ОБЯЗАНЫ быть 0x80000000
            axon_heads: vec![genesis_core::layout::BurstHeads8::empty(AXON_SENTINEL); total_axons],
            axon_tips_uvw: vec![0; total_axons],
            axon_dirs_xyz: vec![0; total_axons],

            soma_to_axon: vec![u32::MAX; padded_n],
            soma_positions: vec![0; padded_n],
        }
    }

    /// Вычисляет плоский индекс для Coalesced Access на GPU.
    #[inline(always)]
    pub fn columnar_idx(padded_n: usize, neuron_idx: usize, slot: usize) -> usize {
        debug_assert!(neuron_idx < padded_n && slot < MAX_DENDRITE_SLOTS);
        slot * padded_n + neuron_idx
    }

    /// Дамп SoA-структур в бинарные файлы. Zero-cost для загрузки в рантайме.
    pub fn dump_to_disk(&self, out_dir: &Path) {
        // [Warp Alignment Check]
        assert!(self.padded_n % 32 == 0, "CRITICAL: padded_n must be multiple of 32");

        let state_path = out_dir.join("shard.state");
        let axons_path = out_dir.join("shard.axons");
        let geom_path = out_dir.join("shard.geom");
        let pos_path = out_dir.join("shard.pos");

        // 1. .state (Somas + Dendrites)
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
        ).expect("Failed to write .state blob");

        // 2. .axons (Heads only)
        write_axons_blob(&axons_path, &self.axon_heads).expect("Failed to write .axons blob");

        // 3. .geom (Tips + Dirs for visualization and growth logic)
        // Note: Not loaded into VRAM by ShardEngine, but needed by baker and telemetry
        let mut geom_file = File::create(geom_path).expect("Failed to create .geom file");
        geom_file.write_all(cast_slice(&self.axon_tips_uvw)).unwrap();
        geom_file.write_all(cast_slice(&self.axon_dirs_xyz)).unwrap();

        // 4. .pos (Packed soma positions u32: Type|Z|Y|X)
        let mut pos_file = File::create(pos_path).expect("Failed to create .pos file");
        pos_file.write_all(cast_slice(&self.soma_positions)).unwrap();
    }
}

/// Zero-Cost сборка бинарного блоба .state.
/// Массивы обязаны быть длины `padded_n`.
pub fn write_state_blob(
    path: &Path,
    padded_n: usize,
    voltages: &[i32],
    flags: &[u8],
    thresholds: &[i32],
    timers: &[u8],
    soma_to_axon: &[u32],
    dendrite_targets: &[u32], // Длина: padded_n * 128
    dendrite_weights: &[i16], // Длина: padded_n * 128
    dendrite_timers: &[u8],  // Длина: padded_n * 128
) -> std::io::Result<()> {
    let (_, total_size) = calculate_state_blob_size(padded_n);
    
    // Преаллокация точного размера для предотвращения реаллокаций памяти
    let mut blob = Vec::with_capacity(total_size);

    // [Contract] Строгая последовательность укладки байт согласно ShardVramPtrs
    blob.extend_from_slice(cast_slice(&voltages[..padded_n]));
    blob.extend_from_slice(cast_slice(&flags[..padded_n]));
    blob.extend_from_slice(cast_slice(&thresholds[..padded_n]));
    blob.extend_from_slice(cast_slice(&timers[..padded_n]));
    blob.extend_from_slice(cast_slice(&soma_to_axon[..padded_n]));
    blob.extend_from_slice(cast_slice(&dendrite_targets[..padded_n * MAX_DENDRITES]));
    blob.extend_from_slice(cast_slice(&dendrite_weights[..padded_n * MAX_DENDRITES]));
    blob.extend_from_slice(cast_slice(&dendrite_timers[..padded_n * MAX_DENDRITES]));

    assert_eq!(blob.len(), total_size, "FATAL: State blob size mismatch before disk flush");

    let mut file = File::create(path)?;
    file.write_all(&blob)?;
    Ok(())
}

/// Выгрузка .axons блоба.
pub fn write_axons_blob(
    path: &Path,
    axon_heads: &[genesis_core::layout::BurstHeads8], // Длина: total_axons
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(cast_slice(axon_heads))?;
    Ok(())
}

