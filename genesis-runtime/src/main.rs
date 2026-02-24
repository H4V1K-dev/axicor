use anyhow::{Context, Result};
use clap::Parser;
use genesis_runtime::config::{parse_shard_config, parse_simulation_config, parse_blueprints_config};
use genesis_runtime::memory::VramState;
use genesis_runtime::{GenesisConstantMemory, VariantParameters, Runtime};
use genesis_runtime::orchestrator::night_phase::NightPhase;
use genesis_runtime::network::bsp::BspBarrier;
use genesis_runtime::network::router::SpikeRouter;

use genesis_runtime::network::geometry_client::GeometryServer;
use genesis_runtime::network::socket::NodeSocket;

use std::ffi::c_void;
use std::path::PathBuf;
use tokio::time::Instant;
use genesis_runtime::network::telemetry::TelemetryServer;

#[derive(Parser, Debug)]
#[command(
    name = "genesis-node",
    about = "Distributed Genesis Brain Node Daemon",
    version
)]
struct Cli {
    /// Path to the shard configuration (e.g. shard_04.toml)
    #[arg(short, long)]
    config: PathBuf,

    /// Local port to bind the Node to (TCP = port+1, UDP = port)
    #[arg(short, long, default_value = "8000")]
    port: u16,

    /// Path to the global simulation configuration (e.g. simulation.toml)
    #[arg(short, long, default_value = "config/simulation.toml")]
    simulation: PathBuf,

    /// Directory containing baked binary blocks (.state, .axons)
    #[arg(short = 'b', long, default_value = "baked/")]
    baked_dir: PathBuf,

    /// Path to the blueprints configuration (neuron type LUT)
    #[arg(long, default_value = "zones/V1/blueprints.toml")]
    blueprints: PathBuf,

    /// Actively injects a sweeping signal into Virtual Axons each sync_batch
    #[arg(long)]
    mock_retina: bool,

    /// Unix socket path for genesis-baker-daemon (optional).
    /// If not provided, Night Phase Sprouting is skipped (Sort & Prune still runs).
    #[arg(long)]
    baker_socket: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse CLI
    let cli = Cli::parse();
    println!("[Node] Starting Genesis Distributed Daemon...");

    // 2. Load Instance Config
    let config = parse_shard_config(&cli.config)
        .with_context(|| format!("Failed to load config: {:?}", cli.config))?;
    
    let sim_config = parse_simulation_config(&cli.simulation)
        .with_context(|| format!("Failed to load simulation config: {:?}", cli.simulation))?;

    println!("[Node] Target Zone: {}", config.zone_id);
    println!(
        "[Node] World Offset: [{}, {}, {}]",
        config.world_offset.x, config.world_offset.y, config.world_offset.z
    );

    // 3. Load Baked Geometry (Zero-Copy to VRAM)
    // We expect baker to produce shard_{zone_id}.state or similar. 
    // Here we'll just look for a generic shard.state / shard.axons for the MVP.
    let state_path = cli.baked_dir.join("shard.state");
    let axons_path = cli.baked_dir.join("shard.axons");

    println!("[Node] Loading VRAM payload from {:?}...", cli.baked_dir);
    let state_bytes = std::fs::read(&state_path).context("Missing shard.state")?;
    let axons_bytes = std::fs::read(&axons_path).context("Missing shard.axons")?;

    let num_virtual = sim_config.simulation.num_virtual_axons.unwrap_or(0);
    let vram = VramState::load_shard(&state_bytes, &axons_bytes, num_virtual)
        .context("Failed to push shard data to GPU VRAM")?;

    println!("[Node] VRAM Load Complete. {} neurons, {} total axons ({} virtual)", 
             vram.padded_n, vram.total_axons, vram.num_virtual);

    // 4. Initialize Network
    // - UDP Fast Path Router
    // - TCP Geometry Server Server (Slow Path)
    
    // Setup Geometry Server on TCP (port + 1)
    let tcp_port = cli.port + 1;
    let geo_addr = format!("0.0.0.0:{}", tcp_port).parse().unwrap();
    let geo_server = GeometryServer::bind(geo_addr).await
        .context("Failed to bind TCP Geometry Server")?;
    println!("[Node] Bound TCP Geometry Server on {}", geo_addr);
    
    // The orchestrator will listen to these requests and handle them during Night Phase
    let _geo_receiver = geo_server.spawn();

    // Setup UDP Router (routing table built dynamically based on InstanceConfig neighbors)
    let mut router = SpikeRouter::new();
    
    // Setup Telemetry WebSockets Server
    let telemetry_port = cli.port + 2; // e.g. 8002
    let telemetry_tx = TelemetryServer::start(telemetry_port).await;

    // 4.1 Fast Path: Bind UDP Socket
    let udp_addr = format!("0.0.0.0:{}", cli.port);
    let socket = NodeSocket::bind(&udp_addr).await
        .context("Failed to bind UDP NodeSocket")?;
    println!("[Node] Bound UDP Fast Path on {}", udp_addr);

    // Setup BSP Barrier configured to the simulation rules
    let sync_batch_ticks = sim_config.simulation.sync_batch_ticks as usize;
    let mut barrier = BspBarrier::new(sync_batch_ticks);
    barrier.socket = Some(socket);
    
    // Parse InstanceConfig to establish peer mappings
    // (Assuming simple logical Shard IDs for now: X_Plus = 1, X_Minus = 2, Y_Plus = 3, Y_Minus = 4)
    let default_neighbors = [
        (1, &config.neighbors.x_plus),
        (2, &config.neighbors.x_minus),
        (3, &config.neighbors.y_plus),
        (4, &config.neighbors.y_minus),
    ];

    for (logical_id, neighbor_opt) in default_neighbors {
        if let Some(target_str) = neighbor_opt {
            if target_str != "Self" && !target_str.is_empty() {
                if let Ok(addr) = target_str.parse::<std::net::SocketAddr>() {
                    barrier.peer_addresses.insert(logical_id, addr);
                    println!("[Node] Discovered Peer Shard {}: {}", logical_id, addr);
                } else {
                    eprintln!("[Node] Warning: Could not parse neighbor address: {}", target_str);
                }
            }
        }
    }

    let mut gpu_schedule_buffer = vec![0u8; sync_batch_ticks * 1024 * 4];

    // 5. Load Blueprints and Upload Constant Memory to GPU
    let blueprints = parse_blueprints_config(&cli.blueprints)
        .with_context(|| format!("Failed to load blueprints: {:?}", cli.blueprints))?;

    let mut const_mem = GenesisConstantMemory::default();
    // Fill up to 4 variants from blueprints (indices 0..3 = Variant bits 2-3)
    for (i, nt) in blueprints.neuron_types.iter().take(4).enumerate() {
        const_mem.variants[i] = VariantParameters {
            threshold:            nt.threshold,
            rest_potential:       nt.rest_potential,
            leak:                 nt.leak_rate,
            homeostasis_penalty:  nt.homeostasis_penalty,
            homeostasis_decay:    nt.homeostasis_decay,
            gsop_potentiation:    nt.gsop_potentiation,
            gsop_depression:      nt.gsop_depression,
            refractory_period:    nt.refractory_period,
            synapse_refractory:   nt.synapse_refractory_period,
            slot_decay_ltm:       nt.slot_decay_ltm,
            slot_decay_wm:        nt.slot_decay_wm,
            propagation_length:   nt.signal_propagation_length as u8,
            _padding:             [0; 3],
        };
    }
    // Inertia LUT: linearly decreasing resistance — stronger weights change slower
    // Rank 0 (weak, abs<2048): inertia=128 (full delta), rank 15 (strong): inertia=8
    for rank in 0..16usize {
        const_mem.inertia_lut[rank] = (128u32.saturating_sub(rank as u32 * 8)) as u8;
    }

    if !Runtime::init_constants(&const_mem) {
        anyhow::bail!("Failed to upload GenesisConstantMemory to GPU");
    }
    println!("[Node] Constant memory uploaded ({} neuron types).", blueprints.neuron_types.len().min(4));

    // 5.1 Initialize Engine Runtime
    let v_seg = (sim_config.simulation.signal_speed_um_tick / sim_config.simulation.voxel_size_um) as u32;
    let master_seed = genesis_core::seed::MasterSeed::from_str(&sim_config.simulation.master_seed);
    let mut runtime = Runtime::new(
        vram,
        v_seg,
        master_seed.raw(),
        Some(cli.baked_dir.clone()),
    );

    // 5.2 Connect to baker daemon (optional)
    if let Some(ref socket_path) = cli.baker_socket {
        let zone_u16 = config.zone_id.parse::<u16>().unwrap_or_else(|_| {
            eprintln!("[Node] Warning: zone_id '{}' is not numeric, using 0 for SHM name", config.zone_id);
            0
        });
        match genesis_runtime::ipc::BakerClient::connect(zone_u16, socket_path) {
            Ok(client) => {
                println!("[Node] Baker daemon connected via {:?}", socket_path);
                runtime.baker_client = Some(client);
            }
            Err(e) => {
                eprintln!("[Node] Warning: cannot connect to baker daemon: {e}. Sprouting disabled.");
            }
        }
    } else {
        println!("[Node] No --baker-socket provided. Night Phase Sprouting disabled.");
    }

    // 6. Enter the Ephemeral Loop
    let mut current_tick = 0u64;
    let night_interval = sim_config.simulation.night_interval_ticks;
    let prune_threshold: i16 = blueprints.neuron_types.first()
        .map(|nt| nt.prune_threshold as i16)
        .unwrap_or(15);
    println!("[Node] Night Phase: interval={} ticks, prune_threshold={}",
        if night_interval == 0 { "DISABLED".to_string() } else { night_interval.to_string() },
        prune_threshold);

    
    println!("[Node] Engine Online. Starting Ephemeral Loop.");

    loop {
        let _loop_start = Instant::now();

        // 6.1 Mock Retina Injection
        if cli.mock_retina && num_virtual > 0 {
            let side = (num_virtual as f32).sqrt().ceil() as u32;
            // Двигаем полосу каждые 2 батча
            let sweep_x = (current_tick as u32 / 2) % side; 
            let mut bitmask = vec![0u32; (num_virtual as usize + 31) / 32];
            for i in 0..num_virtual {
                let ix = i % side;
                if ix == sweep_x || ix == (sweep_x + 1) % side { // Линия толщиной 2 вокселя
                    bitmask[i as usize / 32] |= 1 << (i % 32);
                }
            }
            runtime.vram.upload_input_bitmask(&bitmask).context("Mock Retina upload failed")?;
        }

        // 6.2 Day Phase: GPU Execution & Network Sync
        // We utilize the GPU async execution cycle.
        genesis_runtime::orchestrator::day_phase::DayPhase::run_batch(
            &mut runtime,
            &mut barrier,
            &mut router,
            gpu_schedule_buffer.as_mut_ptr() as *mut c_void,
            current_tick as u32,
            Some(&telemetry_tx)
        ).await.context("Day Phase execution failed")?;

        current_tick += 1;
        let real_ticks_completed = current_tick * sync_batch_ticks as u64;

        // 6.2 Night Phase Check (trigger from config, not hardcoded)
        let is_night = NightPhase::check_and_run(&mut runtime, 0, night_interval, real_ticks_completed, prune_threshold);
        if is_night {
            // Re-sync local states if needed
            println!("[Node] Night Phase concluded at tick {}.", real_ticks_completed);
        }

        // Throttle to simulated real time (e.g. 10ms network barrier limits display speed)
        // Here we just print per batch completed.
        if current_tick % 10 == 0 {
            let elapsed = _loop_start.elapsed().as_millis();
            println!("[Node] Processed Batch {}. Simulated {} Ticks. Wall Clock: {} ms", 
                     current_tick, 
                     current_tick * sync_batch_ticks as u64,
                     elapsed);
        }
    }
}
