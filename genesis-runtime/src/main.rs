use anyhow::{Context, Result};
use clap::Parser;
use genesis_runtime::config::{parse_shard_config, parse_simulation_config, parse_blueprints_config};
use genesis_core::config::brain::parse_brain_config;
use genesis_runtime::memory::VramState;
use genesis_runtime::Runtime;
use genesis_runtime::zone_runtime::ZoneRuntime;
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
    /// Path to the brain manifest (e.g. brain.toml)
    #[arg(short, long, default_value = "config/brain.toml")]
    brain: PathBuf,

    /// Local port to bind the Node to (TCP = port+1, UDP = port)
    #[arg(short, long, default_value = "8000")]
    port: u16,

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

    // 2. Load Brain Config
    let brain_config = parse_brain_config(&cli.brain)
        .map_err(|e| anyhow::anyhow!("Failed to load brain config: {:?}, err: {}", cli.brain, e))?;
    
    let sim_config = parse_simulation_config(&brain_config.simulation.config)
        .with_context(|| format!("Failed to load simulation config: {:?}", brain_config.simulation.config))?;

    let num_virtual = sim_config.simulation.num_virtual_axons.unwrap_or(0);
    let sync_batch_ticks = sim_config.simulation.sync_batch_ticks as usize;

    println!("[Node] Brain Manifest Loaded. {} zones configured.", brain_config.zones.len());

    // 3. Initialize Shared Network Components
    let tcp_port = cli.port + 1;
    let geo_addr = format!("0.0.0.0:{}", tcp_port).parse().unwrap();
    let geo_server = GeometryServer::bind(geo_addr).await
        .context("Failed to bind TCP Geometry Server")?;
    println!("[Node] Bound TCP Geometry Server on {}", geo_addr);
    let _geo_receiver = geo_server.spawn();

    let mut router = SpikeRouter::new();
    let telemetry_port = cli.port + 2; // e.g. 8002
    let telemetry_tx = TelemetryServer::start(telemetry_port).await;

    let udp_addr = format!("0.0.0.0:{}", cli.port);
    let socket = NodeSocket::bind(&udp_addr).await
        .context("Failed to bind UDP NodeSocket")?;
    println!("[Node] Bound UDP Fast Path on {}", udp_addr);

    let mut barrier = BspBarrier::new(sync_batch_ticks);
    barrier.socket = Some(socket);

    // 4. Load Zones
    let mut zones: Vec<ZoneRuntime> = Vec::new();
    for zone_entry in &brain_config.zones {
        println!("[Node] Loading Zone: {}", zone_entry.name);
        
        // 4.1 Parse Configs
        let blueprints = parse_blueprints_config(&zone_entry.blueprints)
            .with_context(|| format!("Failed to load blueprints for zone {}", zone_entry.name))?;
            
        let shard_toml_path = zone_entry.baked_dir.join("shard.toml");
        let shard_config = if shard_toml_path.exists() {
            parse_shard_config(&shard_toml_path).unwrap_or_else(|_| panic!("Failed to parse shard config {:?}", shard_toml_path))
        } else {
            println!("[Node] Warning: No shard.toml found at {:?}, using fallback.", shard_toml_path);
            genesis_core::config::instance::InstanceConfig {
                zone_id: "0".to_string(),
                world_offset: genesis_core::config::instance::Coordinate { x: 0, y: 0, z: 0 },
                dimensions: genesis_core::config::instance::Dimensions { w: 1000, d: 1000, h: 1000 },
                neighbors: genesis_core::config::instance::Neighbors { x_plus: None, x_minus: None, y_plus: None, y_minus: None },
            }
        };

        let default_neighbors = [
            (1, &shard_config.neighbors.x_plus),
            (2, &shard_config.neighbors.x_minus),
            (3, &shard_config.neighbors.y_plus),
            (4, &shard_config.neighbors.y_minus),
        ];

        for (logical_id, neighbor_opt) in default_neighbors {
            if let Some(target_string) = neighbor_opt {
                let target_str = target_string.as_str();
                if target_str != "Self" && !target_str.is_empty() {
                    if let Ok(addr) = target_str.parse::<std::net::SocketAddr>() {
                        barrier.peer_addresses.insert(logical_id, addr);
                    }
                }
            }
        }

        // 4.2 Load VRAM
        let state_path = zone_entry.baked_dir.join("shard.state");
        let axons_path = zone_entry.baked_dir.join("shard.axons");
        let state_bytes = std::fs::read(&state_path).context(format!("Missing {:?} (did you run baker?)", state_path))?;
        let axons_bytes = std::fs::read(&axons_path).context(format!("Missing {:?}", axons_path))?;

        let vram = VramState::load_shard(&state_bytes, &axons_bytes, num_virtual)
            .context("Failed to push shard data to GPU VRAM")?;

        println!("       VRAM Load Complete. {} neurons, {} total axons", vram.padded_n, vram.total_axons);

        // 4.3 Setup ZoneRuntime
        let const_mem = ZoneRuntime::build_constant_memory(&blueprints);
        let prune_threshold = blueprints.neuron_types.first().map(|n| n.prune_threshold as i16).unwrap_or(15);
        let v_seg = (sim_config.simulation.signal_speed_um_tick / sim_config.simulation.voxel_size_um) as u32;
        let master_seed = genesis_core::seed::MasterSeed::from_str(&sim_config.simulation.master_seed);

        let mut runtime = Runtime::new(vram, v_seg, master_seed.raw(), Some(zone_entry.baked_dir.clone()));

        if let Some(ref socket_path) = cli.baker_socket {
            let zone_u16 = shard_config.zone_id.parse::<u16>().unwrap_or(0);
            if let Ok(client) = genesis_runtime::ipc::BakerClient::connect(zone_u16, socket_path) {
                println!("       Baker daemon connected.");
                runtime.baker_client = Some(client);
            }
        }

        zones.push(ZoneRuntime {
            name: zone_entry.name.clone(),
            runtime,
            const_mem,
            config: shard_config,
            prune_threshold,
        });
    }

    if zones.is_empty() {
        anyhow::bail!("No zones configured in brain.toml!");
    }

    let mut gpu_schedule_buffer = vec![0u8; sync_batch_ticks * 1024 * 4];

    // 6. Enter the Ephemeral Loop
    let mut current_tick = 0u64;
    let night_interval = sim_config.simulation.night_interval_ticks;
    
    println!("[Node] Engine Online. Loaded {} zones. Starting Ephemeral Loop.", zones.len());

    loop {
        let _loop_start = Instant::now();

        // 6.1 Mock Retina Injection (only for the first zone with virtual axons)
        if cli.mock_retina && num_virtual > 0 {
            if let Some(zone) = zones.first_mut() {
                let side = (num_virtual as f32).sqrt().ceil() as u32;
                let sweep_x = (current_tick as u32 / 2) % side; 
                let mut bitmask = vec![0u32; (num_virtual as usize + 31) / 32];
                for i in 0..num_virtual {
                    let ix = i % side;
                    if ix == sweep_x || ix == (sweep_x + 1) % side { 
                        bitmask[i as usize / 32] |= 1 << (i % 32);
                    }
                }
                let _ = zone.runtime.vram.upload_input_bitmask(&bitmask);
            }
        }

        // 6.2 Day Phase: GPU Execution & Network Sync for ALL Zones
        genesis_runtime::orchestrator::day_phase::DayPhase::run_batch(
            &mut zones,
            &mut barrier,
            &mut router,
            gpu_schedule_buffer.as_mut_ptr() as *mut c_void,
            current_tick as u32,
            Some(&telemetry_tx)
        ).await.context("Day Phase execution failed")?;

        current_tick += 1;
        let real_ticks_completed = current_tick * sync_batch_ticks as u64;

        // 6.3 Night Phase Check (trigger from config, not hardcoded)
        for zone in &mut zones {
            let is_night = NightPhase::check_and_run(&mut zone.runtime, 0, night_interval, real_ticks_completed, zone.prune_threshold);
            if is_night {
                println!("[Node] Night Phase for Zone '{}' concluded.", zone.name);
            }
        }

        // Throttle to simulated real time (e.g. 10ms network barrier limits display speed)
        if current_tick % 10 == 0 {
            let elapsed = _loop_start.elapsed().as_millis();
            println!("[Node] Processed Batch {}. Simulated {} Ticks. Wall Clock: {} ms", 
                     current_tick, 
                     real_ticks_completed,
                     elapsed);
        }
    }
}
