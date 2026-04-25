use clap::Parser;
use std::io::{Read, Write};
use std::path::PathBuf;

use axicor_core::config::blueprints::BlueprintsConfig;
use axicor_core::config::manifest::ZoneManifest;
use axicor_core::ipc::{default_socket_path, shm_file_path, ShmHeader};

struct NightPhaseContext {
    _baked_dir: PathBuf,
    _layer_ranges: Vec<axicor_baker::bake::axon_growth::LayerZRange>,
    _neuron_types: Vec<axicor_core::config::blueprints::NeuronType>,
    _sim_config: axicor_baker::parser::simulation::SimulationConfig,
    _shard_bounds: axicor_baker::bake::axon_growth::ShardBounds,
    _master_seed: u64,

    _next_ghost_slot_base: u32,
    _total_axons_max: u32,
    _total_ghosts: u32,
    _virtual_axons: u32, // [DOD FIX]
    _max_x: u32,
    _max_y: u32,
    _max_z: u32,

    _axon_heads: Vec<axicor_core::layout::BurstHeads8>,
    _soma_to_axon: Vec<u32>,

    _geom_mmap: memmap2::MmapMut,
    _paths_mmap: memmap2::MmapMut,
    _pos_mmap: memmap2::Mmap,
}

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    zone_hash: u32,
    #[arg(long)]
    baked_dir: PathBuf,
    // [DOD FIX] Accepting manifest from /dev/shm
    #[arg(long)]
    manifest: PathBuf,
}

fn main() {
    let (non_blocking_writer, _guard) = tracing_appender::non_blocking(std::io::stdout());
    tracing_subscriber::fmt()
        .with_writer(non_blocking_writer)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();
    let zone_hash = cli.zone_hash;

    // 1. Read shard manifest from /dev/shm
    let manifest_path = &cli.manifest;
    let manifest_str =
        std::fs::read_to_string(manifest_path).expect("Failed to read manifest.toml");
    let manifest: ZoneManifest = toml::from_str(&manifest_str).expect("Failed to parse manifest");

    let padded_n = manifest.memory.padded_n as u32;
    let total_axons = (manifest.memory.virtual_axons
        + manifest.memory.ghost_capacity
        + manifest.memory.padded_n) as u32;

    // 2. Calculate SHM size
    let shm_len = axicor_core::ipc::shm_size(padded_n as usize);

    // 3. Create file-backed shared memory
    let shm_path = shm_file_path(cli.zone_hash);
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&shm_path)
        .expect("Daemon failed to create SHM file");
    file.set_len(shm_len as u64)
        .expect("Failed to set SHM size");

    let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file).expect("Daemon failed to mmap SHM") };

    // 4. Initialize contract header
    let header = ShmHeader::new(cli.zone_hash, padded_n, total_axons);
    unsafe { std::ptr::write(mmap.as_mut_ptr() as *mut ShmHeader, header) };

    tracing::info!(
        "[Axicor Baker Daemon {:08X}] SHM Allocated: {} MB at {:?}. Listening for IPC...",
        cli.zone_hash,
        shm_len / 1024 / 1024,
        shm_path
    );

    // Load blueprints.toml from SRAM folder
    let blueprints = load_blueprints(&cli.baked_dir);

    tracing::info!(
        " Axicor Baker Daemon starting (zone_hash={:08X})",
        zone_hash
    );
    tracing::info!(
        "   Loaded {} neuron types",
        blueprints
            .as_ref()
            .map(|b| b.neuron_types.len())
            .unwrap_or(0)
    );

    // Cache configs
    let mut night_ctx = build_night_context(&cli.baked_dir, &cli.manifest, zone_hash);

    let socket_addr = default_socket_path(zone_hash);
    let _shm_file = file;

    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&socket_addr);
        let listener = std::os::unix::net::UnixListener::bind(&socket_addr)
            .unwrap_or_else(|_| panic!("FATAL: Cannot bind Unix socket {}", socket_addr));
        tracing::info!(" Listening on {}", socket_addr);
        tracing::info!("   Waiting for Night Phase requests from axicor-node...");
        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    if let Err(e) = run_night_phase(
                        s,
                        zone_hash,
                        blueprints.as_ref(),
                        night_ctx.as_mut(),
                        mmap.as_mut_ptr(),
                    ) {
                        tracing::error!("[ERROR] Night Phase error: {}", e);
                    }
                }
                Err(e) => tracing::error!("Connection error: {}", e),
            }
        }
    }

    #[cfg(windows)]
    {
        let listener = std::net::TcpListener::bind(&socket_addr)
            .unwrap_or_else(|_| panic!("FATAL: Cannot bind TCP {}", socket_addr));
        tracing::info!(" Listening on {}", socket_addr);
        tracing::info!("   Waiting for Night Phase requests from axicor-node...");
        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    if let Err(e) = run_night_phase(
                        s,
                        zone_hash,
                        blueprints.as_ref(),
                        night_ctx.as_mut(),
                        mmap.as_mut_ptr(),
                    ) {
                        tracing::error!("[ERROR] Night Phase error: {}", e);
                    }
                }
                Err(e) => tracing::error!("Connection error: {}", e),
            }
        }
    }
}

fn load_blueprints(baked_dir: &std::path::Path) -> Option<BlueprintsConfig> {
    let bp_path = baked_dir.join("BrainDNA").join("blueprints.toml");
    if bp_path.exists() {
        match BlueprintsConfig::load(&bp_path) {
            Ok(bp) => {
                tracing::info!("   Blueprints loaded from {:?}", bp_path);
                return Some(bp);
            }
            Err(e) => tracing::warn!(
                "[WARN]  Failed to load blueprints from {:?}: {}",
                bp_path,
                e
            ),
        }
    } else {
        tracing::warn!("[WARN]  blueprints.toml not found at {:?}", bp_path);
    }
    None
}

#[allow(clippy::ptr_arg)]
fn build_night_context(
    baked_dir: &std::path::PathBuf,
    manifest_path: &std::path::PathBuf,
    _zone_hash: u32,
) -> Option<NightPhaseContext> {
    use axicor_baker::bake::axon_growth::{compute_layer_ranges, ShardBounds};
    use axicor_baker::parser::simulation::SimulationConfig;

    let dna_dir = baked_dir.join("BrainDNA");
    let shard_cfg = axicor_core::config::InstanceConfig::load(&dna_dir.join("shard.toml"))
        .map_err(|e| tracing::error!("[Daemon] Cannot load shard.toml: {}", e))
        .ok()?;
    let sim_config = SimulationConfig::load(&dna_dir.join("simulation.toml"))
        .map_err(|e| tracing::error!("[Daemon] Cannot load simulation.toml: {}", e))
        .ok()?;
    let bp = load_blueprints(baked_dir)?;
    let neuron_types = bp.neuron_types.clone();
    let anatomy = axicor_baker::parser::anatomy::Anatomy::load(&dna_dir.join("anatomy.toml"))
        .map_err(|e| tracing::error!("[Daemon] Cannot load anatomy.toml: {}", e))
        .ok()?;

    let layer_ranges = compute_layer_ranges(&anatomy, &sim_config);
    let shard_bounds = ShardBounds::from_config(&shard_cfg);
    let master_seed = axicor_core::seed::MasterSeed::from_str("AXICOR").raw();

    let manifest_str = std::fs::read_to_string(manifest_path)
        .map_err(|e| tracing::error!("[Daemon] Cannot read manifest.toml: {}", e))
        .ok()?;
    let manifest: axicor_core::config::manifest::ZoneManifest = toml::from_str(&manifest_str)
        .map_err(|e| tracing::error!("[Daemon] Cannot parse manifest.toml: {}", e))
        .ok()?;

    let padded_n = manifest.memory.padded_n as u32;
    let raw_axons =
        manifest.memory.padded_n + manifest.memory.virtual_axons + manifest.memory.ghost_capacity;
    let total_axons_max = ((raw_axons + 31) & !31) as u32;

    let chk_state = baked_dir.join("checkpoint.state");
    let chk_axons = baked_dir.join("checkpoint.axons");
    let base_state = baked_dir.join("shard.state");
    let base_axons = baked_dir.join("shard.axons");

    let (state_path, axons_path) = if chk_state.exists() && chk_axons.exists() {
        (chk_state, chk_axons)
    } else {
        (base_state, base_axons)
    };

    let state_file = std::fs::File::open(&state_path).ok()?;
    let state_mmap = unsafe { memmap2::Mmap::map(&state_file).ok()? };

    let axons_file = std::fs::File::open(&axons_path).ok()?;
    let axons_mmap = unsafe { memmap2::Mmap::map(&axons_file).ok()? };

    assert_eq!(
        state_mmap.as_ptr() as usize % 64,
        0,
        "FATAL C-ABI BOUNDARY: Memory-mapped .state file is not 64-byte aligned! OS page mapping failed."
    );
    assert_eq!(
        axons_mmap.as_ptr() as usize % 32,
        0,
        "FATAL C-ABI BOUNDARY: Memory-mapped .axons file is not 32-byte aligned!"
    );

    let (axon_heads, soma_to_axon) = {
        let mut axon_heads: Vec<axicor_core::layout::BurstHeads8> = {
            if !axons_mmap.is_empty() {
                bytemuck::cast_slice(&axons_mmap[..]).to_vec()
            } else {
                vec![
                    axicor_core::layout::BurstHeads8::empty(axicor_core::constants::AXON_SENTINEL);
                    total_axons_max as usize
                ]
            }
        };
        axon_heads.truncate(total_axons_max as usize);

        let data = &state_mmap[..];
        let state_size = data.len();

        let s2a = if state_size > 0 {
            let bytes_per_neuron =
                4 + 1 + 4 + 1 + 4 + axicor_core::constants::MAX_DENDRITE_SLOTS * (4 + 4 + 1);

            if state_size % bytes_per_neuron != 0 {
                panic!(
                    "FATAL C-ABI MISMATCH: .state file size ({}) is not a multiple of {} bytes. Run build_brain.py to re-bake.",
                    state_size, bytes_per_neuron
                );
            }

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

    let total_ghosts = manifest.memory.ghost_capacity as u32;
    let max_x = shard_cfg.dimensions.w;
    let max_y = shard_cfg.dimensions.d;

    let geom_path = baked_dir.join("shard.geom");
    let geom_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&geom_path)
        .ok()?;
    let geom_mmap = unsafe { memmap2::MmapMut::map_mut(&geom_file).ok()? };

    tracing::info!(
        "[Axicor Daemon] Loaded {} axon geometries (next_ghost_slot_base={})",
        total_axons_max,
        padded_n
    );

    let paths_path = baked_dir.join("shard.paths");
    let paths_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&paths_path)
        .ok()?;
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
        _virtual_axons: manifest.memory.virtual_axons as u32, // [DOD FIX]
        _max_x: max_x,
        _max_y: max_y,
        _max_z: shard_cfg.dimensions.h,
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
    mut ctx: Option<&mut NightPhaseContext>,
    shm_ptr: *mut u8,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read binary BakeRequest (16 bytes)
    let mut req_buf = [0u8; 16];
    stream.read_exact(&mut req_buf)?;

    let req: &axicor_core::ipc::BakeRequest = unsafe { &*(req_buf.as_ptr() as *const _) };
    if req.magic != axicor_core::ipc::BAKE_MAGIC {
        return Err(format!("Invalid BAKE magic: {:08X}", req.magic).into());
    }

    tracing::info!(
        " Night Phase trigger received (tick={}, prune={}, max_sprouts={})",
        req.current_tick,
        req.prune_threshold,
        req.max_sprouts
    );

    // [DOD FIX] Read ghost owner map (Origin Tracking)
    let total_ghosts = ctx.as_ref().map(|c| c._total_ghosts as usize).unwrap_or(0);
    let mut ghost_origins = vec![0u32; total_ghosts];
    if total_ghosts > 0 {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(ghost_origins.as_mut_ptr() as *mut u8, total_ghosts * 4)
        };
        stream.read_exact(bytes)?;
    }

    // 2. Validate SHM Header
    let hdr_ptr = shm_ptr as *mut ShmHeader;
    let hdr = unsafe { &mut *hdr_ptr };
    hdr.validate()
        .map_err(|e| format!("SHM validation failed: {}", e))?;

    let padded_n = hdr.padded_n as usize;
    let w_off = hdr.weights_offset as usize;
    let t_off = hdr.targets_offset as usize;
    let h_off = hdr.handovers_offset as usize;
    let f_off = hdr.flags_offset as usize; // [DOD FIX] Flags extraction
    let h_count = hdr.handovers_count as usize;
    let slot_n = padded_n * axicor_core::constants::MAX_DENDRITE_SLOTS;

    let h_count_max = axicor_core::ipc::MAX_HANDOVERS_PER_NIGHT;

    // 3. Obtain slices directly from SHM (Zero-Copy)
    let (weights, targets, flags, handovers) = unsafe {
        let w_ptr = shm_ptr.add(w_off) as *mut i32;
        let t_ptr = shm_ptr.add(t_off) as *mut u32;
        let f_ptr = shm_ptr.add(f_off) as *const u8;
        let h_ptr = shm_ptr.add(h_off) as *mut axicor_core::ipc::AxonHandoverEvent;
        (
            std::slice::from_raw_parts_mut(w_ptr, slot_n),
            std::slice::from_raw_parts_mut(t_ptr, slot_n),
            std::slice::from_raw_parts(f_ptr, padded_n),
            std::slice::from_raw_parts_mut(h_ptr, h_count_max),
        )
    };

    // 4. CPU Sprouting & Living Axons (Zero-Copy)
    let (_new_synapses, generated_handovers, acks) = if let Some(ctx) = ctx.as_deref_mut() {
        // Cast header and compute offsets
        let paths_hdr =
            unsafe { &*(ctx._paths_mmap.as_ptr() as *const axicor_core::layout::PathsFileHeader) };
        let paths_total_axons = paths_hdr.total_axons as usize;

        let lengths_slice = unsafe {
            std::slice::from_raw_parts_mut(ctx._paths_mmap.as_mut_ptr().add(16), paths_total_axons)
        };

        let matrix_offset = axicor_core::layout::calculate_paths_matrix_offset(paths_total_axons);
        let paths_slice = unsafe {
            std::slice::from_raw_parts_mut(
                ctx._paths_mmap.as_mut_ptr().add(matrix_offset) as *mut u32,
                paths_total_axons * 256,
            )
        };

        let soma_positions =
            unsafe { std::slice::from_raw_parts(ctx._pos_mmap.as_ptr() as *const u32, padded_n) };

        // Extract tips and dirs from _geom_mmap
        let geom_total_axons = ctx._total_axons_max as usize;
        let tips_slice = unsafe {
            std::slice::from_raw_parts_mut(
                ctx._geom_mmap.as_mut_ptr() as *mut u32,
                geom_total_axons,
            )
        };
        let dirs_slice = unsafe {
            std::slice::from_raw_parts_mut(
                ctx._geom_mmap.as_mut_ptr().add(geom_total_axons * 4) as *mut u32,
                geom_total_axons,
            )
        };

        axicor_baker::bake::sprouting::run_sprouting_pass(
            targets,
            weights,
            flags,
            &ghost_origins, // NEW: Origin Tracking
            handovers,      // NEW: pass queue
            h_count,        // NEW PARAMETER
            tips_slice,     // From MmapMut
            dirs_slice,     // From MmapMut
            &ctx._soma_to_axon,
            padded_n,
            ctx._total_ghosts as usize, // NEW
            ctx._virtual_axons as usize, // [DOD FIX] Pass layout offset
            ctx._max_x,                 // NEW
            ctx._max_y,                 // NEW
            ctx._max_z,                 // NEW
            blueprints,
            hdr.epoch,
            lengths_slice,    // NEW
            paths_slice,      // NEW
            soma_positions,   // NEW
            ctx._master_seed, // <--- [DOD FIX] Entropy forwarding
            _zone_hash,
            req.max_sprouts,
            req.prune_threshold, // [DOD FIX] For initial weight protection
            shm_ptr,             // NEW: For prune writing
        )
    } else {
        (0, 0, vec![])
    };

    // Update count of generated handovers
    unsafe {
        (*hdr_ptr).handovers_count = generated_handovers as u32;
    }

    if let Some(ctx) = ctx {
        // [DOD FIX] Asynchronous dirty page flush to SSD (Crash Tolerance).
        // Does not block the thread; OS will sync data to disk in background.
        let _ = ctx._geom_mmap.flush_async();
        let _ = ctx._paths_mmap.flush_async();
    }

    // 5. Send ACK
    let ack_magic = axicor_core::ipc::BAKE_READY_MAGIC;
    stream.write_all(&ack_magic.to_le_bytes())?;

    // Send count of generated ACK responses
    let ack_count = acks.len() as u32;
    stream.write_all(&ack_count.to_le_bytes())?;

    // Send ACKs (Lock-Free Memory Layout)
    if ack_count > 0 {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                acks.as_ptr() as *const u8,
                acks.len() * std::mem::size_of::<axicor_core::ipc::AxonHandoverAck>(),
            )
        };
        stream.write_all(bytes)?;
    }

    stream.flush()?;
    Ok(())
}
