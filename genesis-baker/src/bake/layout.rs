use genesis_core::constants::{MAX_DENDRITE_SLOTS, AXON_SENTINEL};
use genesis_core::layout::align_to_warp;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::fs::File;

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
    pub dendrite_timers: Vec<u8>,

    // Аксоны
    pub axon_heads: Vec<u32>,
    pub axon_tips_uvw: Vec<u32>, // PackedTip
    pub axon_dirs_xyz: Vec<u32>, // PackedDir

    // Маппинг: soma_idx → axon_idx
    pub soma_to_axon: Vec<u32>,
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

            dendrite_targets: vec![0; MAX_DENDRITE_SLOTS * padded_n],
            dendrite_weights: vec![0; MAX_DENDRITE_SLOTS * padded_n],
            dendrite_timers: vec![0; MAX_DENDRITE_SLOTS * padded_n],

            // Хард-инвариант: пустые аксоны ОБЯЗАНЫ быть 0x80000000
            axon_heads: vec![AXON_SENTINEL; total_axons],
            axon_tips_uvw: vec![0; total_axons],
            axon_dirs_xyz: vec![0; total_axons],

            soma_to_axon: vec![u32::MAX; padded_n],
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
        // ⚠️ ИНВАРИАНТ ВЫРАВНИВАНИЯ (01_foundations.md §2.2)
        debug_assert!(self.padded_n % 32 == 0, "CRITICAL: padded_n ({}) must be warp-aligned (multiple of 32)", self.padded_n);
        
        let state_path = out_dir.join("shard.state");
        let axons_path = out_dir.join("shard.axons");

        let mut state_file = File::create(state_path).expect("Failed to create .state file");
        let state_header = genesis_core::layout::StateFileHeader::new(
            self.padded_n as u32, 
            self.total_axons as u32
        );
        
        self.write_state_blob(&mut state_file, &state_header).expect("Failed to write state blob");

        // 2. Дамп аксонов (.axons)
        let mut axons_file = BufWriter::new(File::create(axons_path).expect("Failed to create .axons file"));
        let header = genesis_core::layout::AxonsFileHeader::new(self.total_axons as u32);
        axons_file.write_all(header.as_bytes()).unwrap();

        write_raw_slice(&mut axons_file, &self.axon_tips_uvw);
        write_raw_slice(&mut axons_file, &self.axon_dirs_xyz);
    }

    /// Zero-Copy Serializer (06_baker_io.md §2.1)
    pub fn write_state_blob(&self, file: &mut File, header: &genesis_core::layout::StateFileHeader) -> std::io::Result<()> {
        // [Contract §1.2.1] The file must be a byte-perfect image of VRAM.
        // We write all arrays sequentially.
        
        let pn = self.padded_n;
        let pa = self.total_axons;
        let dc = MAX_DENDRITE_SLOTS * pn;

        // 1. Заголовок
        file.write_all(header.as_bytes())?;

        // 2. Проливка сырой памяти (POD)
        unsafe {
            write_pod_slice(file, &self.voltage[..pn])?;
            write_pod_slice(file, &self.flags[..pn])?;
            write_pod_slice(file, &self.threshold_offset[..pn])?;
            write_pod_slice(file, &self.refractory_timer[..pn])?;
            write_pod_slice(file, &self.soma_to_axon[..pn])?;

            // Транспонированные дендриты
            write_pod_slice(file, &self.dendrite_targets[..dc])?;
            write_pod_slice(file, &self.dendrite_weights[..dc])?;
            write_pod_slice(file, &self.dendrite_timers[..dc])?;
            
            // Аксоны (Heads) - Fixed: must be pa (padded axons)
            write_pod_slice(file, &self.axon_heads[..pa])?;
        }
        
        Ok(())
    }
}

/// Helper для записи сырых слайсов в файл (Zero-Copy)
#[inline(always)]
unsafe fn write_pod_slice<T>(file: &mut File, data: &[T]) -> std::io::Result<()> {
    let bytes = std::slice::from_raw_parts(
        data.as_ptr() as *const u8,
        data.len() * std::mem::size_of::<T>()
    );
    file.write_all(bytes)
}

/// Старый хелпер для BufWriter (остается для .axons)
fn write_raw_slice<T>(writer: &mut BufWriter<File>, data: &[T]) {
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    };
    writer.write_all(byte_slice).expect("Failed to write raw layout bytes");
}
