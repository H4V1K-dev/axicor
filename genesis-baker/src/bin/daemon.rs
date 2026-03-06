use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use clap::Parser;

use genesis_core::ipc::{
    shm_name, ShmHeader, ShmState,
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
    
    // [Шаг 4] Геометрия аксонов, загруженная один раз при старте
    /// axon_tips_uvw: Vec<u32> — упакованные Z|Y|X координаты кончиков (по одному на аксон)
    _axon_tips_uvw: Vec<u32>,
    /// axon_dirs_xyz: Vec<u32> — упакованные направления (по одному на аксон)
    _axon_dirs_xyz: Vec<u32>,
    /// axon_heads: Vec<genesis_core::layout::BurstHeads8> — состояние аксонных голов (для инициализации новых ghost аксонов)
    _axon_heads: Vec<genesis_core::layout::BurstHeads8>,
    
    // [Шаг 4] soma_to_axon маппинг для интеграции новых ghost axons 
    /// soma_to_axon: Vec<u32> — маппинг soma_idx → axon_idx
    _soma_to_axon: Vec<u32>,

    // [Phase 41.2] Types Cache (1 byte per axon) and Whitelist bitmasks
    _neuron_types_cache: Vec<u8>,
    _whitelist_masks: [u16; 16],
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

    // 3. Создаем POSIX Shared Memory (O_CREAT | O_TRUNC выжигает старые данные)
    let c_name = std::ffi::CString::new(shm_name(cli.zone_hash)).unwrap();
    let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC, 0o666) };
    if fd < 0 { panic!("Daemon failed to create SHM"); }
    unsafe { libc::ftruncate(fd, shm_len as libc::off_t) };

    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), shm_len as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0) };
    
    // 4. Инициализируем заголовок контракта
    let header = ShmHeader::new(cli.zone_hash, padded_n, total_axons);
    unsafe { std::ptr::write(ptr as *mut ShmHeader, header) };

    unsafe { libc::close(fd) };

    println!("[Baker Daemon {:08X}] SHM Allocated: {} MB. Listening for IPC...", cli.zone_hash, shm_len / 1024 / 1024);

    // Загружаем blueprints.toml для Dale's Law
    let blueprints = load_blueprints(&brain_toml);

    println!("🧠 Genesis Baker Daemon starting (zone_hash={:08X})", zone_hash);
    println!("   Loaded {} neuron types", blueprints.as_ref().map(|b| b.neuron_types.len()).unwrap_or(0));

    // [DOD FIX] Кешируем конфиги для inject_ghost_axons — один раз при старте
    let night_ctx = build_night_context(&cli.baked_dir, &brain_toml);

    let socket_path = default_socket_path(zone_hash);

    // Удаляем старый сокет если остался от прошлого запуска
    let _ = std::fs::remove_file(&socket_path);

    let listener = UnixListener::bind(&socket_path)
        .expect(&format!("FATAL: Cannot bind Unix socket {}", socket_path));

    println!("🔌 Listening on {}", socket_path);
    println!("   Waiting for Night Phase requests from genesis-node...");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let bp_ref = blueprints.as_ref();
                if let Err(e) = handle_night_phase(stream, zone_hash, bp_ref, night_ctx.as_ref(), ptr as *mut u8) {
                    eprintln!("❌ Night Phase error: {}", e);
                }
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }
}

fn load_blueprints(brain_toml: &PathBuf) -> Option<BlueprintsConfig> {
    // [DOD FIX] Читаем brain.toml и берём поле `blueprints` из первой зоны.
    // Это универсальный путь — работает для любого Brain (CartPole, RobotBrain, etc.)
    if let Ok(src) = std::fs::read_to_string(brain_toml) {
        // Ищем первую строку вида `blueprints = "..."` в файле
        for line in src.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("blueprints") {
                if let Some(after_eq) = trimmed.splitn(2, '=').nth(1) {
                    let path_str = after_eq.trim().trim_matches('"');
                    let bp_path = std::path::Path::new(path_str);
                    if bp_path.exists() {
                        match BlueprintsConfig::load(bp_path) {
                            Ok(bp) => {
                                println!("   Blueprints loaded from {:?}", bp_path);
                                return Some(bp);
                            }
                            Err(e) => eprintln!("⚠️  Failed to load blueprints from {:?}: {}", bp_path, e),
                        }
                    }
                }
            }
        }
    }

    eprintln!("⚠️  blueprints.toml not found — Dale's Law will use default weights");
    None
}


/// Загружает конфиги для inject_ghost_axons один раз при старте Демона.
/// Option<NightPhaseContext> — None если конфиги не найдены (graceful degradation).
fn build_night_context(baked_dir: &PathBuf, brain_toml: &PathBuf) -> Option<NightPhaseContext> {
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
    let bp = load_blueprints(brain_toml)?;
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
    let total_axons_max = (manifest.memory.padded_n + manifest.memory.virtual_axons + manifest.memory.ghost_capacity) as u32;

    // [Шаг 4] Загружаем геометрию аксонов из дисковых дампов
    // Файл .axons: просто header + axon_heads (u32 × total_axons)
    let axons_path = baked_dir.join("shard.axons");
    let axon_heads = if axons_path.exists() {
        let data = std::fs::read(&axons_path)
            .map_err(|e| eprintln!("[Daemon] Cannot read shard.axons: {}", e)).ok()?;
        // Пропускаем 16B заголовок (AxonsFileHeader), затем BurstHeads8×total_axons
        if data.len() > 16 {
            let slice = &data[16..];
            let count = slice.len() / 32;
            bytemuck::cast_slice(slice)
                .iter()
                .take(count.min(total_axons_max as usize))
                .copied()
                .collect()
        } else {
            vec![genesis_core::layout::BurstHeads8::empty(genesis_core::constants::AXON_SENTINEL); total_axons_max as usize]
        }
    } else {
        vec![genesis_core::layout::BurstHeads8::empty(genesis_core::constants::AXON_SENTINEL); total_axons_max as usize]
    };

    // Файл .geom: axon_tips_uvw (u32 × total_axons) + axon_dirs_xyz (u32 × total_axons)
    let geom_path = baked_dir.join("shard.geom");
    let (axon_tips_uvw, axon_dirs_xyz) = if geom_path.exists() {
        let data = std::fs::read(&geom_path)
            .map_err(|e| eprintln!("[Daemon] Cannot read shard.geom: {}", e)).ok()?;
        // Каждый аксон — 2 × u32, всего 8 * total_axons байт
        let count = total_axons_max as usize;
        let expected_size = 8 * count;
        if data.len() >= expected_size {
            let tips = bytemuck::cast_slice::<u8, u32>(&data[0..4*count])
                .iter().copied().collect();
            let dirs = bytemuck::cast_slice::<u8, u32>(&data[4*count..8*count])
                .iter().copied().collect();
            (tips, dirs)
        } else {
            (vec![0; count], vec![0; count])
        }
    } else {
        (vec![0; total_axons_max as usize], vec![0; total_axons_max as usize])
    };

    println!("[Daemon] Loaded {} axon geometries (next_ghost_slot_base={})", total_axons_max, padded_n);

    // [Шаг 4] Загружаем soma_to_axon маппинг для интеграции новых ghost axons
    // Файл .state: [padded_n * u32 voltages] + [padded_n * u8 flags] + [padded_n * u32 thresholds] + [padded_n * u8 timers] + [padded_n * u32 soma_to_axon] + ...
    let soma_to_axon = {
        let state_path = baked_dir.join("shard.state");
        if state_path.exists() {
            let data = std::fs::read(&state_path)
                .map_err(|e| eprintln!("[Daemon] Cannot read shard.state: {}", e)).ok()?;
            
            // Вычисляем offset soma_to_axon в .state бломе
            // Структура: [u32 voltages: 4*N] + [u8 flags: N] + [u32 thresholds: 4*N] + [u8 timers: N] + [u32 soma_to_axon: 4*N]
            let voltage_offset = 0;
            let flags_offset = voltage_offset + 4 * (padded_n as usize);
            let thresholds_offset = flags_offset + (padded_n as usize);
            let timers_offset = thresholds_offset + 4 * (padded_n as usize);
            let soma_to_axon_offset = timers_offset + (padded_n as usize);
            let soma_to_axon_end = soma_to_axon_offset + 4 * (padded_n as usize);
            
            if data.len() >= soma_to_axon_end {
                bytemuck::cast_slice::<u8, u32>(&data[soma_to_axon_offset..soma_to_axon_end])
                    .iter()
                    .copied()
                    .collect()
            } else {
                vec![u32::MAX; padded_n as usize]
            }
        } else {
            vec![u32::MAX; padded_n as usize]
        }
    };

    // [Phase 41.2] Извлечение типов (сдвиг >> 4) из shard.state
    let neuron_types_cache = {
        let state_path = baked_dir.join("shard.state");
        if state_path.exists() {
            let data = std::fs::read(&state_path).unwrap_or_default();
            // Структура: [u32 voltages: 4*N] + [u8 flags: N]
            let flags_offset = 4 * (padded_n as usize);
            let flags_end = flags_offset + (padded_n as usize);
            if data.len() >= flags_end {
                data[flags_offset..flags_end].iter().map(|f| (f >> 4) & 0x0F).collect()
            } else { vec![0; padded_n as usize] }
        } else { vec![0; padded_n as usize] }
    };

    let mut whitelist_masks = [0u16; 16];
    for (i, nt) in neuron_types.iter().enumerate().take(16) {
        let mut mask = 0u16;
        if nt.dendrite_whitelist.is_empty() {
            mask = 0xFFFF; // All types allowed if whitelist is empty
        } else {
            for allowed_name in &nt.dendrite_whitelist {
                for (j, other_nt) in neuron_types.iter().enumerate().take(16) {
                    if &other_nt.name == allowed_name {
                        mask |= 1 << j;
                    }
                }
            }
        }
        whitelist_masks[i] = mask;
    }

    Some(NightPhaseContext {
        _baked_dir: baked_dir.clone(),
        _layer_ranges: layer_ranges,
        _neuron_types: neuron_types,
        _sim_config: sim_config,
        _shard_bounds: shard_bounds,
        _master_seed: master_seed,
        _next_ghost_slot_base: padded_n,
        _total_axons_max: total_axons_max,
        _axon_tips_uvw: axon_tips_uvw,
        _axon_dirs_xyz: axon_dirs_xyz,
        _axon_heads: axon_heads,
        _soma_to_axon: soma_to_axon,
        _neuron_types_cache: neuron_types_cache,
        _whitelist_masks: whitelist_masks,
    })
}

fn handle_night_phase(
    mut stream: UnixStream,
    _zone_hash: u32,
    blueprints: Option<&BlueprintsConfig>,
    ctx: Option<&NightPhaseContext>,
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
    let h_count = hdr.handovers_count as usize;
    let slot_n = padded_n * MAX_DENDRITE_SLOTS;

    // 3. Obtain slices directly from SHM (Zero-Copy)
    let (weights, targets, _handovers) = unsafe {
        let w_ptr = shm_ptr.add(w_off) as *mut i16;
        let t_ptr = shm_ptr.add(t_off) as *mut u32;
        let h_ptr = shm_ptr.add(h_off) as *const genesis_core::ipc::AxonHandoverEvent;
        (
            std::slice::from_raw_parts_mut(w_ptr, slot_n),
            std::slice::from_raw_parts_mut(t_ptr, slot_n),
            std::slice::from_raw_parts(h_ptr, h_count),
        )
    };

    // 4. CPU Sprouting (Zero-Copy)
    let empty_cache: &[u8] = &[];
    let default_masks = [0xFFFF; 16];
    let (axon_types_cache, whitelist_masks) = if let Some(c) = ctx {
        (c._neuron_types_cache.as_slice(), &c._whitelist_masks)
    } else {
        (empty_cache, &default_masks)
    };

    let new_synapses = genesis_baker::bake::sprouting::run_sprouting_pass(
        targets,
        weights,
        padded_n,
        blueprints,
        hdr.epoch,
        axon_types_cache,
        whitelist_masks,
    );

    println!("   ↳ Sprouted {} new synapses", new_synapses);

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

