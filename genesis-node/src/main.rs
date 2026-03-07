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
pub mod simple_reporter;
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
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long, default_value = "9000")]
    fast_path_port: u16,

    #[arg(long)]
    peer: Vec<String>,

    #[arg(long, default_value = "100")]
    batch_size: u32,

    #[arg(long, default_value = "aggressive")]
    cpu_profile: CpuProfile,
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
        
        // 0. Initialize CLI Monitor
        let reporter = Arc::new(crate::simple_reporter::SimpleReporter::new());
        let reporter_clone = reporter.clone();

        // CLI Monitor Thread
        std::thread::Builder::new()
            .name("genesis-cli-monitor".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    reporter_clone.print_status();
                }
            }).expect("Fatal: Failed to spawn monitor thread");

        println!("[Node] Starting Genesis Distributed Daemon...");
        
        // 2-5. Execution of the 5-Component Fail-Fast Boot Sequence
        let boot_result = Bootloader::boot_node_with_profile(&cli.manifest, reporter.clone(), cli.cpu_profile).await
            .context("Node Bootstrap Failed")?;

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
        let node = boot_result.node_runtime;
        std::thread::Builder::new()
            .name("genesis-orchestrator".into())
            .spawn(move || {
                node.run_node_loop(cli.batch_size);
            })
            .expect("Fatal: Failed to spawn orchestrator OS thread");

        // 8. Park the Tokio main thread so daemon tasks stay alive
        std::thread::park();
        Ok(())
    })
}
