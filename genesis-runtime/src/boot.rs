use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use genesis_core::config::manifest::ZoneManifest;
use crate::memory::VramState;
use crate::zone_runtime::ZoneRuntime;
use crate::network::geometry_client::GeometryServer;
use crate::network::telemetry::TelemetryServer;
use crate::network::io_server::ExternalIoServer;
use crate::network::bsp::{BspBarrier, PingPongSchedule};
use crate::compute::shard::ShardComputeIsland;
use crate::node::NodeRuntime;
use crate::network::router::RoutingTable;
use std::sync::atomic::AtomicBool;
use std::collections::HashMap;
use std::time::Duration;

pub struct Bootloader;

pub struct BootResult {
    pub node_runtime: NodeRuntime,
    pub geometry_server: GeometryServer,
    pub telemetry_swapchain: Arc<crate::network::telemetry::TelemetrySwapchain>,
}

impl Bootloader {
    /// Full node bootstrap sequence.
    /// [Fail-Fast Policy] Panics if any component fails to start or artifacts are invalid.
    pub async fn boot_node(manifest_path: &Path, dashboard: Arc<crate::tui::DashboardState>) -> Result<BootResult> {
        let manifest_toml = std::fs::read_to_string(manifest_path)
            .with_context(|| format!("Failed to read manifest: {:?}", manifest_path))?;
        let manifest: ZoneManifest = toml::from_str(&manifest_toml)
            .with_context(|| format!("Failed to parse manifest: {:?}", manifest_path))?;

        let baked_dir = manifest_path.parent().unwrap_or(std::path::Path::new("."));
        let local_port = manifest.network.fast_path_udp_local;

        // 1. Geometry Server (TCP port+1)
        let geo_port = local_port + 1;
        let geo_addr = format!("0.0.0.0:{}", geo_port).parse()?;
        let geometry_server = GeometryServer::bind(geo_addr).await?;
        println!("[Boot] 1/5 Geometry Server bound to {}", geo_addr);

        // 2. Telemetry Server (WS port+2)
        let tele_port = local_port + 2;
        let telemetry_swapchain = TelemetryServer::start(tele_port).await;
        println!("[Boot] 2/5 Telemetry Server started on WS port {}", tele_port);

        // 3. Zone Artifact Loading (Zero-Copy)
        let (zone, island) = Self::load_zone(manifest_path, dashboard.clone())?;
        println!("[Boot] 3/5 Zone artifacts loaded and verified (Warp Aligned)");

        // 4. Intra-GPU Channel & BSP Barrier
        let mut bsp_barrier = BspBarrier::new();
        let initial_peers = HashMap::new();
        bsp_barrier.add_zone(manifest.zone_hash, zone.ping_pong.clone());
        let bsp_barrier = Arc::new(bsp_barrier);
        let routing_table = Arc::new(RoutingTable::new(initial_peers));

        // 5. External IO Server (UDP)
        let words_per_tick = (zone.runtime.vram.num_pixels + 31) / 32;
        let sync_batch_ticks = 100;
        let input_bytes = (words_per_tick as usize) * sync_batch_ticks * 4;
        
        let pinned_input_ptr = unsafe { crate::ffi::gpu_host_alloc(input_bytes) as *mut u32 };
        let mut io_server = ExternalIoServer::new(
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            1024,
            0,
            0,
            dashboard.clone(),
            routing_table.clone(),
            Arc::new(tokio::net::UdpSocket::bind(&format!("0.0.0.0:{}", manifest.network.external_udp_in)).await.unwrap())
        ).unwrap();

        // Register matrix offsets
        let gxi_path = baked_dir.join("shard.gxi");
        if gxi_path.exists() {
            let gxi = crate::input::GxiFile::load(&gxi_path);
            for m in gxi.matrices {
                io_server.matrix_offsets.insert(m.name_hash, m.offset);
            }
        }
        let io_server = Arc::new(io_server);
        println!("[Boot] 4/5 External IO Server bound to UDP {}", manifest.network.external_udp_in);
        println!("[Boot] 5/5 Intra-GPU synchronization established");

        let node_runtime = NodeRuntime::boot(
            vec![(manifest.zone_hash, island)],
            io_server,
            routing_table,
            bsp_barrier,
            std::net::Ipv4Addr::new(127, 0, 0, 1),
            manifest.network.fast_path_udp_local,
        );

        Ok(BootResult {
            node_runtime,
            geometry_server,
            telemetry_swapchain,
        })
    }

    fn load_zone(manifest_path: &Path, dashboard: Arc<crate::tui::DashboardState>) -> Result<(ZoneRuntime, ShardComputeIsland)> {
        let manifest_toml = std::fs::read_to_string(manifest_path)?;
        let manifest: ZoneManifest = toml::from_str(&manifest_toml)?;
        let baked_dir = manifest_path.parent().unwrap_or(std::path::Path::new("."));
        
        let state_path = baked_dir.join("shard.state");
        let axons_path = baked_dir.join("shard.axons");

        Self::verify_warp_alignment(&state_path)?;
        Self::verify_warp_alignment(&axons_path)?;

        let state_bytes = std::fs::read(&state_path)?;
        let axons_bytes = std::fs::read(&axons_path)?;

        let gxi = Self::try_load_gxi(baked_dir);
        let gxo = Self::try_load_gxo(baked_dir);

        let vram = VramState::load_shard(
            &state_bytes,
            &axons_bytes,
            gxi.as_ref(),
            gxo.as_ref(),
            100,
            manifest.memory.v_seg as u32,
            manifest.memory.ghost_capacity
        )?;

        let island = ShardComputeIsland::new(
            vram.to_layout(),
            vram.num_pixels,
            vram.num_mapped_somas,
        );

        let mut const_mem = [genesis_core::config::manifest::GpuVariantParameters::default(); 16];
        for variant in manifest.variants {
            let idx = variant.id as usize;
            if idx < 16 {
                const_mem[idx] = variant.into_gpu();
            }
        }

        let is_sleeping = Arc::new(AtomicBool::new(false));
        let ping_pong = Arc::new(unsafe {
            PingPongSchedule::new(100, 1024, is_sleeping.clone())
        });

        let zone = ZoneRuntime {
            name: format!("Zone_{:08X}", manifest.zone_hash),
            artifact_dir: baked_dir.to_path_buf(),
            runtime: crate::Runtime::new(vram, manifest.memory.v_seg as u32, 42, Some(baked_dir.to_path_buf())),
            const_mem,
            config: Default::default(),
            prune_threshold: -50,
            is_sleeping,
            sleep_requested: false,
            ping_pong,
            last_night_time: std::time::Instant::now(),
            min_night_delay: std::time::Duration::from_secs(30),
            slow_path_queues: Arc::new(crate::network::slow_path::SlowPathQueues::new()),
            hot_reload_queue: Arc::new(crossbeam::queue::SegQueue::new()),
            inter_node_channels: Vec::new(),
            intra_gpu_channels: Vec::new(),
            spatial_grid: Arc::new(std::sync::Mutex::new(crate::orchestrator::spatial_grid::SpatialGrid::new())),
            dashboard,
        };

        Ok((zone, island))
    }

    fn verify_warp_alignment(path: &Path) -> Result<()> {
        let metadata = std::fs::metadata(path)
            .with_context(|| format!("Missing artifact: {:?}", path))?;
        let size = metadata.len();
        if size % 8 != 0 {
            panic!(
                "FATAL: Alignment Violation in {:?}. Size {} is not a multiple of 8 bytes (Required for u64/double access).",
                path, size
            );
        }
        Ok(())
    }

    fn try_load_gxi(dir: &Path) -> Option<crate::input::GxiFile> {
        let path = dir.join("shard.gxi");
        if path.exists() { Some(crate::input::GxiFile::load(&path)) } else { None }
    }

    fn try_load_gxo(dir: &Path) -> Option<crate::output::GxoFile> {
        let path = dir.join("shard.gxo");
        if path.exists() { Some(crate::output::GxoFile::load(&path)) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_boot_alignment_check() {
        let mut good_file = NamedTempFile::new().unwrap();
        good_file.write_all(&[0u8; 32]).unwrap();
        assert!(Bootloader::verify_warp_alignment(good_file.path()).is_ok());

        let mut bad_file = NamedTempFile::new().unwrap();
        bad_file.write_all(&[0u8; 31]).unwrap();
        
        let result = std::panic::catch_unwind(|| {
            let _ = Bootloader::verify_warp_alignment(bad_file.path());
        });
        assert!(result.is_err(), "Should panic on 31-byte file");
    }
}
