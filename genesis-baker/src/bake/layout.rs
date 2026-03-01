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

            soma_to_axon: vec![u32::MAX; padded_n],
        }
    }

    /// Дамп SoA-структур в бинарные файлы. Zero-cost для загрузки в рантайме.
    pub fn dump_to_disk(&self, out_dir: &Path) {
        let state_path = out_dir.join("shard.state");
        let axons_path = out_dir.join("shard.axons");

        // 1. Дамп состояния сом и дендритов (.state)
        let mut state_file = BufWriter::new(File::create(state_path).expect("Failed to create .state file"));
        
        let state_header = genesis_core::layout::StateFileHeader::new(
            self.padded_n as u32, 
            self.total_axons as u32
        );
        state_file.write_all(state_header.as_bytes()).unwrap();

        // Пишем сырые байты без сериализации.
        // Порядок ОБЯЗАН совпадать с математикой смещений в memory.rs!
        write_raw_slice(&mut state_file, &self.voltage);
        write_raw_slice(&mut state_file, &self.flags);
        write_raw_slice(&mut state_file, &self.threshold_offset);
        write_raw_slice(&mut state_file, &self.refractory_timer);
        write_raw_slice(&mut state_file, &self.soma_to_axon); // Missing previously!

        write_raw_slice(&mut state_file, &self.dendrite_targets);
        write_raw_slice(&mut state_file, &self.dendrite_weights);
        write_raw_slice(&mut state_file, &self.dendrite_timers);
        
        // memory.rs also expects axon_heads at the very end of state_bytes!
        write_raw_slice(&mut state_file, &self.axon_heads);

        // 2. Дамп аксонов (.axons)
        // Even if serialize_axons isn't used here, we MUST write the header so `VramState::load_shard` doesn't panic.
        let mut axons_file = BufWriter::new(File::create(axons_path).expect("Failed to create .axons file"));
        let header = genesis_core::layout::AxonsFileHeader::new(self.total_axons as u32);
        axons_file.write_all(header.as_bytes()).unwrap();
    }
}

/// Helper для записи сырых слайсов в файл (убийца serde)
fn write_raw_slice<T>(writer: &mut BufWriter<File>, data: &[T]) {
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    };
    writer.write_all(byte_slice).expect("Failed to write raw layout bytes");
}
