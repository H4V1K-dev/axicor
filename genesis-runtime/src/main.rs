use anyhow::{Context, Result};
use clap::Parser;
use genesis_runtime::config::{parse_shard_config, parse_simulation_config};
use genesis_runtime::memory::VramState;
use genesis_runtime::orchestrator::night_phase::NightPhase;
use genesis_runtime::network::bsp::BspBarrier;
use genesis_runtime::network::router::{SpikeRouter, GhostTarget};
use genesis_runtime::network::geometry_client::GeometryServer;
use genesis_runtime::network::socket::NodeSocket;
use genesis_runtime::Runtime;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Arc;
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
    #[arg(short, long, default_value = "simulation.toml")]
    simulation: PathBuf,

    /// Directory containing baked binary blocks (.state, .axons)
    #[arg(short = 'b', long, default_value = "baked/")]
    baked_dir: PathBuf,
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

    let vram = VramState::load_shard(&state_bytes, &axons_bytes)
        .context("Failed to push shard data to GPU VRAM")?;

    println!("[Node] VRAM Load Complete. {} neurons, {} axons", vram.padded_n, vram.total_axons);

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

    // 5. Initialize Engine Runtime
    // (We pass empty CPU lists for neurons/axons because the Baker hasn't supplied them in this simplified harness)
    let v_seg = (sim_config.simulation.signal_speed_um_tick / sim_config.simulation.voxel_size_um) as u32;
    let mut runtime = Runtime::new(
        vram,
        v_seg,
        Arc::new(vec![]),
        Arc::new(vec![]),
        Arc::new(vec![]),
        0, // master seed
    );

    // 6. Enter the Ephemeral Loop
    let mut current_tick = 0u64;
    let night_interval = 10000; // placeholder
    
    println!("[Node] Engine Online. Starting Ephemeral Loop.");

    loop {
        let _loop_start = Instant::now();

        // 6.1 Night Phase Check
        let is_night = NightPhase::check_and_run(&mut runtime, 0, night_interval, current_tick);

        if is_night {
            // Re-sync local states if needed
            println!("[Node] Night Phase concluded at tick {}.", current_tick);
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
