use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Builder;
use genesis_runtime::boot::Bootloader;

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
        
        // 0. Initialize CLI Dashboard
        let dashboard = Arc::new(genesis_runtime::tui::DashboardState::new(true));
        let app = genesis_runtime::tui::app::DashboardApp::new(dashboard.clone());
        app.spawn();

        println!("[Node] Starting Genesis Distributed Daemon...");
        
        // 2-5. Execution of the 5-Component Fail-Fast Boot Sequence
        let boot_result = Bootloader::boot_node(&cli.manifest, dashboard.clone()).await
            .context("Node Bootstrap Failed")?;

        println!("[Node] Bootstrap Successful. Hands-off to NodeRuntime.");

        // Spawn IO Receiver Loop
        let io_server = boot_result.node_runtime.io_server.clone();
        tokio::spawn(async move {
            io_server.run_rx_loop().await;
        });

        // Spawn Geometry Server (if not already spawned by its own method)
        boot_result.geometry_server.spawn();

        // 6. Enter the high-performance Node Loop (Synchronous GPU / Asynchronous IO)
        // [Architectural Invariant] This loop dispatches work to dedicated OS threads.
        boot_result.node_runtime.run_node_loop(cli.batch_size).await;

        Ok(())
    })
}
