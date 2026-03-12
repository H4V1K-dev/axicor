#![deny(warnings)]
#![deny(unused_variables)]
#![deny(dead_code)]
pub mod boot;
pub mod network;
pub mod node;
pub mod orchestrator;
pub mod zone_runtime;
pub mod config;
pub mod ipc;
pub mod input;
pub mod output;
pub mod tui;
pub mod sentinel;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Builder;
use crate::boot::Bootloader;

#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
pub enum CpuProfile {
    #[default]
    Aggressive, // 100% CPU, 0ms latency (spin_loop)
    Balanced,   // High CPU, OS scheduler yielding (yield_now)
    Eco,        // Low CPU, sleep polling (sleep 1ms)
}

#[derive(Parser, Debug)]
#[command(
    name = "genesis-node",
    about = "Distributed Genesis Brain Node Daemon",
    version
)]
struct Cli {
    /// Путь к файлу brain.toml для автоматического запуска всех зон
    #[arg(long)]
    pub brain: Option<PathBuf>,

    /// Пути к манифестам зон (можно использовать вместе или вместо --brain)
    #[arg(long = "manifest")]
    pub manifests: Vec<PathBuf>,

    #[arg(long, default_value = "9000")]
    pub fast_path_port: u16,

    #[arg(long)]
    pub peer: Vec<String>,

    #[arg(long, default_value = "aggressive")]
    pub cpu_profile: CpuProfile,

    #[arg(long)]
    pub log: bool,
}

fn main() -> Result<()> {
    // 1. Initialize dedicated Tokio Runtime for I/O (2 threads max)
    let rt = Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Fatal: Failed to build Tokio runtime");

    rt.block_on(async {
        let cli = Cli::parse();
        
        let mut manifest_paths = cli.manifests.clone();

        // Если указан --brain, читаем топологию и добавляем все манифесты автоматически
        if let Some(brain_arg) = cli.brain {
            // Умный резолвер путей (Convention over Configuration)
            let brain_path = if brain_arg.exists() || brain_arg.to_string_lossy().ends_with(".toml") {
                brain_arg // Это явный путь к файлу
            } else {
                // Это просто имя модели (например, "mouse_agent")
                let models_root = std::env::var("GENESIS_MODELS_PATH")
                    .unwrap_or_else(|_| "Genesis_Models".to_string());
                std::path::PathBuf::from(models_root).join(brain_arg).join("brain.toml")
            };

            println!("🧠 Loading cluster topology from: {:?}", brain_path);
            match genesis_core::config::brain::parse_brain_config(&brain_path) {
                Ok(brain_cfg) => {
                    for zone in brain_cfg.zones {
                        let manifest_path = zone.baked_dir.join("manifest.toml");
                        println!("  + Discovered zone: {} -> {:?}", zone.name, manifest_path);
                        manifest_paths.push(manifest_path);
                    }
                }
                Err(e) => {
                    eprintln!("❌ FATAL: Failed to parse brain config: {}", e);
                    std::process::exit(1);
                }
            }
        }

        if manifest_paths.is_empty() {
            eprintln!("❌ FATAL: No manifests provided. Use --brain <path/to/brain.toml> or --manifest <path>");
            std::process::exit(1);
        }

        // 0. Initialize CLI Monitor (TUI or Log)
        let telemetry = Arc::new(crate::tui::state::LockFreeTelemetry::default());
        let log_mode = cli.log;

        println!("[Node] Starting Genesis Distributed Daemon...");
        
        // 2-5. Execution of the 5-Component Fail-Fast Boot Sequence
        let boot_result = Bootloader::boot_node_with_profile(&manifest_paths, telemetry.clone(), cli.cpu_profile).await
            .context("Node Bootstrap Failed")?;

        // [DOD FIX] Immediate Cluster Join
        for &local_hash in boot_result.node_runtime.compute_dispatchers.keys() {
            boot_result.node_runtime.broadcast_route_update(local_hash).await;
            println!("📡 [Node] Cluster joined. Route announced for 0x{:08X}", local_hash);
        }

        println!("[Node] Bootstrap Successful. Hands-off to NodeRuntime.");

        // Spawn IO Receiver Loop
        let io_server = boot_result.node_runtime.services.io_server.clone();
        tokio::spawn(async move {
            io_server.run_rx_loop().await;
        });

        // Spawn Geometry Server
        boot_result.geometry_server.spawn(boot_result.geometry_data);

        // 6. Spawn Dedicated OS Thread for Lock-Free UDP Egress
        let worker_pool = boot_result.egress_pool.clone();
        let profile = cli.cpu_profile;
        std::thread::Builder::new()
            .name("genesis-egress-tx".into())
            .spawn(move || {
                let socket = std::net::UdpSocket::bind("0.0.0.0:0").unwrap();
                use std::sync::atomic::{AtomicU32, Ordering};
                static EGRESS_LOG_COUNT: AtomicU32 = AtomicU32::new(0);
                loop {
                    if let Some(msg) = worker_pool.ready_queue.pop() {
                        if msg.target.port() == 8092 {
                            let n = EGRESS_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
                            if n % 100 == 0 {
                                println!("📤 [Egress Thread] Sending {} bytes to {} ({} packets)", msg.size, msg.target, n + 1);
                            }
                        }
                        let _ = socket.send_to(&msg.buffer[..msg.size], msg.target);
                        worker_pool.free_queue.push(msg).unwrap();
                    } else {
                        match profile {
                            CpuProfile::Aggressive => std::hint::spin_loop(),
                            CpuProfile::Balanced => std::thread::yield_now(),
                            CpuProfile::Eco => std::thread::sleep(std::time::Duration::from_millis(1)),
                        }
                    }
                }
            }).expect("Fatal: Failed to spawn egress worker thread");

        // 7. Enter the high-performance Node Loop (Synchronous GPU / Asynchronous IO)
        // [Architectural Invariant] This loop dispatches work to dedicated OS threads.
        let mut node = boot_result.node_runtime;
        std::thread::Builder::new()
            .name("genesis-orchestrator".into())
            .spawn(move || {
                let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
                unsafe { libc::CPU_SET(0, &mut cpuset) };
                let res = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset) };
                if res == 0 {
                    println!("🚀 [Core] Orchestrator locked to OS Thread Core 0");
                }
                node.run_node_loop();
            })
            .expect("Fatal: Failed to spawn orchestrator OS thread");

        // 8. Run Dashboard UI (blocks main thread)
        // [DOD FIX] TUI now uses the lock-free telemetry bridge and maintains its own local state
        crate::tui::run_tui(telemetry, log_mode).unwrap_or_else(|e| {
            eprintln!("UI error: {:?}", e);
        });
        Ok(())
    })
}
