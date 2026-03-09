use std::io::{Read, Write};
use std::path::PathBuf;
use clap::Parser;

use genesis_core::ipc::{
    shm_file_path, ShmHeader, ShmState,
    default_socket_path,
};
use genesis_core::config::manifest::ZoneManifest;
use genesis_core::config::blueprints::BlueprintsConfig;
use genesis_core::constants::MAX_DENDRITE_SLOTS;


struct NightPhaseContext {
    _baked_dir: PathBuf,
    _layer_ranges: Vec<genesis_baker::bake::axon_growth::LayerZRange>,
    _neuron_types: Vec<genesis_core::config::blueprints::NeuronType>,
    _sim_config: genesis_baker::parser::simulation::SimulationConfig,
    _shard_bounds: genesis_baker::bake::axon_growth::ShardBounds,
    _master_seed: u64,
    
    // [Шаг 1] Ghost Allocator initialization
    /// Индекс первого свободного слота для Ghost-аксонов
    /// Начинается сразу после локальных аксонов: next_ghost_slot = manifest.memory.padded_n
    _next_ghost_slot_base: u32,
    /// Максимальное количество аксонов (включая ghost capacity)
    _total_axons_max: u32,
    
    _total_ghosts: u32,
    _max_x: u32,
    _max_y: u32,
    
    _axon_heads: Vec<genesis_core::layout::BurstHeads8>,
    _soma_to_axon: Vec<u32>,

    _geom_mmap: memmap2::MmapMut,

    // НОВЫЕ ММАПЫ
    _paths_mmap: memmap2::MmapMut,
    _pos_mmap: memmap2::Mmap,
}

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    zone_hash: u32,
    #[arg(long)]
    baked_dir: PathBuf,
    #[arg(long)]
    brain: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();
    let zone_hash = cli.zone_hash;

    // Путь к brain.toml для поиска blueprints.toml
    let brain_toml: PathBuf = cli.brain.unwrap_or_else(|| PathBuf::from("config/brain.toml"));

    // 1. Читаем манифест шарда, чтобы узнать точные размеры VRAM
    let manifest_path = cli.baked_dir.join("manifest.toml");
    let manifest_str = std::fs::read_to_string(&manifest_path).expect("Failed to read manifest.toml");
    let manifest: ZoneManifest = toml::from_str(&manifest_str).expect("Failed to parse manifest");

    let padded_n = manifest.memory.padded_n as u32;
    let total_axons = (manifest.memory.virtual_axons + manifest.memory.ghost_capacity + manifest.memory.padded_n) as u32;

    // 2. Вычисляем размер SHM
    // ShmHeader (64 байта) + Weights (128 * N * 2 байта) + Targets (128 * N * 4 байта) + Handovers
    let weights_size = padded_n * 128 * 2;
    let targets_size = padded_n * 128 * 4;
    // [DOD FIX] Резервируем память под плоский массив Handovers (160 KB)
    let handovers_size = (genesis_core::ipc::MAX_HANDOVERS_PER_NIGHT * 16) as u32; // 16 bytes per AxonHandoverEvent

    let shm_len = 64 + weights_size + targets_size + handovers_size;

    // 3. Создаем file-backed shared memory (cross-platform: Linux + Windows)
    let shm_path = shm_file_path(cli.zone_hash);
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&shm_path)
        .expect("Daemon failed to create SHM file");
    file.set_len(shm_len as u64).expect("Failed to set SHM size");

    let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file).expect("Daemon failed to mmap SHM") };

    // 4. Инициализируем заголовок контракта
    let header = ShmHeader::new(cli.zone_hash, padded_n, total_axons);
    unsafe { std::ptr::write(mmap.as_mut_ptr() as *mut ShmHeader, header) };

    println!("[Baker Daemon {:08X}] SHM Allocated: {} MB at {:?}. Listening for IPC...", cli.zone_hash, shm_len / 1024 / 1024, shm_path);

    // Загружаем blueprints.toml для Dale's Law
    let blueprints = load_blueprints(&brain_toml, zone_hash);

    println!("🧠 Genesis Baker Daemon starting (zone_hash={:08X})", zone_hash);
    println!("   Loaded {} neuron types", blueprints.as_ref().map(|b| b.neuron_types.len()).unwrap_or(0));

    // [DOD FIX] Кешируем конфиги для inject_ghost_axons — один раз при старте
    let mut night_ctx = build_night_context(&cli.baked_dir, &brain_toml, zone_hash);

    let socket_addr = default_socket_path(zone_hash);

    // Keep file open so mmap stays valid
    let _shm_file = file;

    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&socket_addr);
        let listener = std::os::unix::net::UnixListener::bind(&socket_addr)
            .expect(&format!("FATAL: Cannot bind Unix socket {}", socket_addr));
        println!("🔌 Listening on {}", socket_addr);
        println!("   Waiting for Night Phase requests from genesis-node...");
        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    if let Err(e) = run_night_phase(s, zone_hash, blueprints.as_ref(), night_ctx.as_mut(), mmap.as_mut_ptr() as *mut u8) {
                        eprintln!("❌ Night Phase error: {}", e);
                    }
                }
                Err(e) => eprintln!("Connection error: {}", e),
            }
        }
    }

    #[cfg(windows)]
    {
        let listener = std::net::TcpListener::bind(&socket_addr)
            .expect(&format!("FATAL: Cannot bind TCP {}", socket_addr));
        println!("🔌 Listening on {}", socket_addr);
        println!("   Waiting for Night Phase requests from genesis-node...");
        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    if let Err(e) = run_night_phase(s, zone_hash, blueprints.as_ref(), night_ctx.as_ref(), mmap.as_mut_ptr() as *mut u8) {
                        eprintln!("❌ Night Phase error: {}", e);
                    }
                }
                Err(e) => eprintln!("Connection error: {}", e),
            }
        }
    }
}

fn load_blueprints(brain_toml: &PathBuf, zone_hash: u32) -> Option<BlueprintsConfig> {
    // [DOD FIX] Parse brain.toml properly and find the zone matching our zone_hash
    if let Ok(src) = std::fs::read_to_string(brain_toml) {
        if let Ok(brain_config) = toml::from_str::<genesis_core::config::brain::BrainConfig>(&src) {
            for zone in &brain_config.zones {
                let zh = genesis_core::hash::fnv1a_32(zone.name.as_bytes());
                if zh == zone_hash {
                    let bp_path = std::path::Path::new(&zone.blueprints);
                    if bp_path.exists() {
                        match BlueprintsConfig::load(bp_path) {
                            Ok(bp) => {
                                println!("   Blueprints loaded from {:?} (zone: {})", bp_path, zone.name);
                                return Some(bp);
                            }
                            Err(e) => eprintln!("⚠️  Failed to load blueprints from {:?}: {}", bp_path, e),
                        }
                    } else {
                        eprintln!("⚠️  Blueprints path {:?} does not exist for zone {}", bp_path, zone.name);
                    }
                    break;
                }
            }
        }
    }

    eprintln!("⚠️  blueprints.toml not found for zone 0x{:08X} — Dale's Law will use default weights", zone_hash);
    None
}


/// Загружает конфиги для inject_ghost_axons один раз при старте Демона.
/// Option<NightPhaseContext> — None если конфиги не найдены (graceful degradation).
fn build_night_context(baked_dir: &PathBuf, brain_toml: &PathBuf, zone_hash: u32) -> Option<NightPhaseContext> {
    use genesis_baker::bake::axon_growth::{compute_layer_ranges, ShardBounds};
    use genesis_baker::parser::simulation::SimulationConfig;

    // Читаем shard.toml (InstanceConfig) из BrainDNA поддиректории
    let dna_dir = baked_dir.join("BrainDNA");
    let shard_cfg = genesis_core::config::InstanceConfig::load(&dna_dir.join("shard.toml")).ok()?;

    // Читаем simulation.toml
    let brain_dir = brain_toml.parent().unwrap_or(std::path::Path::new("."));
    let sim_path = brain_dir.join("simulation.toml");
    let sim_config = SimulationConfig::load(&sim_path)
        .map_err(|e| eprintln!("[Daemon] Cannot load simulation.toml: {}", e)).ok()?;

    // Читаем blueprints для NeuronType list
    let bp = load_blueprints(brain_toml, zone_hash)?;
    let neuron_types = bp.neuron_types.clone();

    // Читаем anatomy из BrainDNA
    let anatomy_path = dna_dir.join("anatomy.toml");
    let anatomy = genesis_baker::parser::anatomy::Anatomy::load(&anatomy_path)
        .map_err(|e| eprintln!("[Daemon] Cannot load anatomy.toml: {}", e)).ok()?;

    let layer_ranges = compute_layer_ranges(&anatomy, &sim_config);

    // ShardBounds из shard.toml (world_offset + dimensions)
    let shard_bounds = ShardBounds::from_config(&shard_cfg);

    // master_seed — детерминированный (Инвариант #7)
    let master_seed = genesis_core::seed::MasterSeed::from_str("GENESIS").raw();

    // [Шаг 1] Читаем манифест для определения базы next_ghost_slot
    let manifest_path = baked_dir.join("manifest.toml");
    let manifest_str = std::fs::read_to_string(&manifest_path)
        .map_err(|e| eprintln!("[Daemon] Cannot read manifest.toml: {}", e)).ok()?;
    let manifest: genesis_core::config::manifest::ZoneManifest = toml::from_str(&manifest_str)
        .map_err(|e| eprintln!("[Daemon] Cannot parse manifest.toml: {}", e)).ok()?;

    let padded_n = manifest.memory.padded_n as u32;
    // [DOD FIX] Вся математика аксонов обязана быть выровнена по варпам (кратно 32)
    let raw_axons = manifest.memory.padded_n + manifest.memory.virtual_axons + manifest.memory.ghost_capacity;
    let total_axons_max = ((raw_axons + 31) & !31) as u32;

    // [DOD FIX] Zero-Copy Memory Mapping instead of std::fs::read.
    // Guarantees OS page alignment (4096 bytes) -> bytemuck is strictly safe.
    let state_path = baked_dir.join("shard.state");
    let state_file = std::fs::File::open(&state_path).ok()?;
    let state_mmap = unsafe { memmap2::Mmap::map(&state_file).ok()? };

    let axons_path = baked_dir.join("shard.axons");
    let axons_file = std::fs::File::open(&axons_path).ok()?;
    let axons_mmap = unsafe { memmap2::Mmap::map(&axons_file).ok()? };

    // Strict validation of the C-ABI Boundary invariant
    assert_eq!(
        state_mmap.as_ptr() as usize % 64,
        0,
        "FATAL C-ABI BOUNDARY: Memory-mapped .state file is not 64-byte aligned! OS page mapping failed."
    );

    let (axon_heads, soma_to_axon) = {
        // Zero-cost casting without allocations
        let mut axon_heads: Vec<genesis_core::layout::BurstHeads8> = {
            if !axons_mmap.is_empty() {
                // [DOD FIX] No legacy 16-byte header. Pure raw slice.
                bytemuck::cast_slice(&axons_mmap[..]).to_vec()
            } else {
                vec![genesis_core::layout::BurstHeads8::empty(genesis_core::constants::AXON_SENTINEL); total_axons_max as usize]
            }
        };
        axon_heads.truncate(total_axons_max as usize);

        let data = &state_mmap[..];
        let state_size = data.len();

        let s2a = if state_size > 0 {
            let bytes_per_neuron = 4 + 1 + 4 + 1 + 4 + genesis_core::constants::MAX_DENDRITE_SLOTS * (4 + 2 + 1);
            let padded_n = (state_size / bytes_per_neuron) as u32;

            let voltage_offset = 0;
            let flags_offset = voltage_offset + 4 * (padded_n as usize);
            let thresholds_offset = flags_offset + (padded_n as usize);
            let timers_offset = thresholds_offset + 4 * (padded_n as usize);
            let soma_to_axon_offset = timers_offset + (padded_n as usize);
            let soma_to_axon_end = soma_to_axon_offset + 4 * (padded_n as usize);

            if data.len() >= soma_to_axon_end {
                let s2a_slice = &data[soma_to_axon_offset..soma_to_axon_end];
                bytemuck::cast_slice(s2a_slice).to_vec()
            } else {
                vec![u32::MAX; padded_n as usize]
            }
        } else {
            vec![u32::MAX; 0]
        };

        (axon_heads, s2a)
    };

    // [DOD FIX] total_ghosts берется из манифеста
    let total_ghosts = manifest.memory.ghost_capacity as u32;
    let max_x = shard_cfg.dimensions.w;
    let max_y = shard_cfg.dimensions.d;

    let geom_path = baked_dir.join("shard.geom");
    let geom_file = std::fs::OpenOptions::new().read(true).write(true).open(&geom_path).ok()?;
    let geom_mmap = unsafe { memmap2::MmapMut::map_mut(&geom_file).ok()? };

    println!("[Daemon] Loaded {} axon geometries (next_ghost_slot_base={})", total_axons_max, padded_n);

    // Загрузка Paths (R/W) и Pos (R/O)
    let paths_path = baked_dir.join("shard.paths");
    let paths_file = std::fs::OpenOptions::new().read(true).write(true).open(&paths_path).ok()?;
    let paths_mmap = unsafe { memmap2::MmapMut::map_mut(&paths_file).ok()? };

    let pos_path = baked_dir.join("shard.pos");
    let pos_file = std::fs::File::open(&pos_path).ok()?;
    let pos_mmap = unsafe { memmap2::Mmap::map(&pos_file).ok()? };


    Some(NightPhaseContext {
        _baked_dir: baked_dir.clone(),
        _layer_ranges: layer_ranges,
        _neuron_types: neuron_types,
        _sim_config: sim_config,
        _shard_bounds: shard_bounds,
        _master_seed: master_seed,
        _next_ghost_slot_base: padded_n,
        _total_axons_max: total_axons_max,
        _total_ghosts: total_ghosts,
        _max_x: max_x,
        _max_y: max_y,
        _axon_heads: axon_heads,
        _soma_to_axon: soma_to_axon,
        _geom_mmap: geom_mmap,
        _paths_mmap: paths_mmap,
        _pos_mmap: pos_mmap,
    })
}

fn run_night_phase<S: Read + Write>(
    mut stream: S,
    _zone_hash: u32,
    blueprints: Option<&BlueprintsConfig>,
    mut ctx: Option<&mut NightPhaseContext>, // [DOD] Mut reference
    shm_ptr: *mut u8,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read binary BakeRequest (16 bytes)
    let mut req_buf = [0u8; 16];
    stream.read_exact(&mut req_buf)?;
    
    let req: &genesis_core::ipc::BakeRequest = unsafe { &*(req_buf.as_ptr() as *const _) };
    if req.magic != genesis_core::ipc::BAKE_MAGIC {
        return Err(format!("Invalid BAKE magic: {:08X}", req.magic).into());
    }
    
    println!("🌙 Night Phase trigger received (tick={}, prune={})", req.current_tick, req.prune_threshold);

    // 2. Validate SHM Header
    let hdr_ptr = shm_ptr as *mut ShmHeader;
    let hdr = unsafe { &mut *hdr_ptr };
    hdr.validate().map_err(|e| format!("SHM validation failed: {}", e))?;

    let padded_n = hdr.padded_n as usize;
    let w_off = hdr.weights_offset as usize;
    let t_off = hdr.targets_offset as usize;
    let h_off = hdr.handovers_offset as usize;
    let f_off = hdr.flags_offset as usize; // [DOD FIX] Flags extraction
    let h_count = hdr.handovers_count as usize;
    let slot_n = padded_n * MAX_DENDRITE_SLOTS;

    let h_count_max = genesis_core::ipc::MAX_HANDOVERS_PER_NIGHT as usize;
    
    // 3. Obtain slices directly from SHM (Zero-Copy)
    let (weights, targets, flags, handovers) = unsafe {
        let w_ptr = shm_ptr.add(w_off) as *mut i16;
        let t_ptr = shm_ptr.add(t_off) as *mut u32;
        let f_ptr = shm_ptr.add(f_off) as *const u8;
        let h_ptr = shm_ptr.add(h_off) as *mut genesis_core::ipc::AxonHandoverEvent;
        (
            std::slice::from_raw_parts_mut(w_ptr, slot_n),
            std::slice::from_raw_parts_mut(t_ptr, slot_n),
            std::slice::from_raw_parts(f_ptr, padded_n),
            std::slice::from_raw_parts_mut(h_ptr, h_count_max),
        )
    };

    // 4. CPU Sprouting & Living Axons (Zero-Copy)
    let (new_synapses, generated_handovers) = if let Some(ctx) = ctx.as_deref_mut() {
        // Каст заголовка и вычисление смещений
        let paths_hdr = unsafe { &*(ctx._paths_mmap.as_ptr() as *const genesis_core::layout::PathsFileHeader) };
        let paths_total_axons = paths_hdr.total_axons as usize;

        let lengths_slice = unsafe {
            std::slice::from_raw_parts_mut(
                ctx._paths_mmap.as_mut_ptr().add(16),
                paths_total_axons
            )
        };

        let matrix_offset = genesis_core::layout::calculate_paths_matrix_offset(paths_total_axons);
        let paths_slice = unsafe {
            std::slice::from_raw_parts_mut(
                ctx._paths_mmap.as_mut_ptr().add(matrix_offset) as *mut u32,
                paths_total_axons * 256
            )
        };

        let soma_positions = unsafe {
            std::slice::from_raw_parts(
                ctx._pos_mmap.as_ptr() as *const u32,
                padded_n
            )
        };

        // Извлекаем tips и dirs из _geom_mmap
        let geom_total_axons = ctx._total_axons_max as usize;
        let tips_slice = unsafe {
            std::slice::from_raw_parts_mut(ctx._geom_mmap.as_mut_ptr() as *mut u32, geom_total_axons)
        };
        let dirs_slice = unsafe {
            std::slice::from_raw_parts_mut(ctx._geom_mmap.as_mut_ptr().add(geom_total_axons * 4) as *mut u32, geom_total_axons)
        };

        genesis_baker::bake::sprouting::run_sprouting_pass(
            targets,
            weights,
            flags,
            handovers,            // NEW: передаем очередь
            h_count,              // НОВЫЙ ПАРАМЕТР
            tips_slice,           // Из MmapMut
            dirs_slice,           // Из MmapMut
            &ctx._soma_to_axon,
            padded_n,
            ctx._total_ghosts as usize, // NEW
            ctx._max_x,           // NEW
            ctx._max_y,           // NEW
            blueprints,
            hdr.epoch,
            lengths_slice,   // NEW
            paths_slice,     // NEW
            soma_positions,  // NEW
        )
    } else {
        (0, 0)
    };

    // Обновляем счетчик сгенерированных хэндоверов
    unsafe {
        (*hdr_ptr).handovers_count = generated_handovers as u32;
    }

    if let Some(ctx) = ctx.as_deref_mut() {
        // [DOD FIX] Асинхронный сброс грязных страниц на SSD (Crash Tolerance).
        // Не блокирует поток, ОС сама скинет данные на диск в фоне.
        let _ = ctx._paths_mmap.flush_async();
        let _ = ctx._geom_mmap.flush_async();
    }

    println!("   ↳ Sprouted {} new synapses, handovers: {}", new_synapses, generated_handovers);

    // 5. GSOP Plasticity / Ghost Integration (TODO/Placeholders)
    // Here we would use `handovers` for GSOP or Growth logic.
    // For now, mirroring the task's focus on IPC.

    // 6. Signal Done via Shared Memory state
    unsafe {
        shm_ptr.add(5).write_volatile(ShmState::NightDone as u8);
    }

    // 7. Binary Acknowledgement (4 bytes)
    let ack = genesis_core::ipc::BAKE_READY_MAGIC.to_le_bytes();
    stream.write_all(&ack)?;
    stream.flush()?;

    println!("🌅 Night Phase complete ({} new synapses)", new_synapses);
    Ok(())
}

