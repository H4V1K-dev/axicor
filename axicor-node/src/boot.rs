use anyhow::{Context, Result};
use crossbeam::queue::SegQueue;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::SystemTime;
use tracing::info;

use crate::input::GxiFile;
use crate::network::bsp::BspBarrier;
use crate::network::geometry_client::GeometryServer;
use crate::network::ghosts::load_ghosts;
use crate::network::intra_gpu::IntraGpuChannel;
use crate::network::io_server::ExternalIoServer;
use crate::network::router::RoutingTable;
use crate::network::telemetry::TelemetryServer;
use crate::node::NodeRuntime;
use crate::output::GxoFile;
use axicor_compute::memory::{calculate_state_blob_size, compute_state_offsets, VramState};
use axicor_compute::ShardEngine;
use axicor_core::config::manifest::ZoneManifest;

type BootShard = crate::node::shard_thread::ShardDescriptor;

pub struct Bootloader;

pub struct BootResult {
    pub node_runtime: NodeRuntime,
    pub geometry_server: GeometryServer,
    pub geometry_data: Vec<u32>,
    pub telemetry_swapchain: Arc<crate::network::telemetry::TelemetrySwapchain>,
    pub egress_pool: Arc<crate::network::egress::EgressPool>,
}

/// Initializes ShardEngine via direct DMA copy from .state and .axons.
/// This method implements O(1) size derivation based on the file contract:
/// - .state: 910 bytes per neuron (SoA)
/// - .axons: 32 bytes per axon (BurstHeads8)
pub fn boot_shard_from_vfs(
    archive: &axicor_core::vfs::AxicArchive,
    zone_name: &str,
    manifest: &ZoneManifest,
    project_name: &str,
    use_gpu: bool,
) -> Result<(ShardEngine, Vec<u32>, PathBuf)> {
    // DOD FIX: ROM / SRAM pattern
    let mem_zone_dir = PathBuf::from("Axicor-Models")
        .join(format!("{}.axic.mem", project_name))
        .join(zone_name);

    std::fs::create_dir_all(&mem_zone_dir)?;

    let state_path = mem_zone_dir.join("shard.state");
    let axons_path = mem_zone_dir.join("shard.axons");

    let (state_blob, axons_blob) = if state_path.exists() && axons_path.exists() {
        info!("[Boot]  Resuming from SRAM: {:?}", mem_zone_dir);
        (std::fs::read(&state_path)?, std::fs::read(&axons_path)?)
    } else {
        info!("[Boot]  Unpacking ROM to SRAM for zone {}", zone_name);

        let state_vfs_path = format!("baked/{}/shard.state", zone_name);
        let axons_vfs_path = format!("baked/{}/shard.axons", zone_name);

        let state = archive
            .get_file(&state_vfs_path)
            .with_context(|| format!("FATAL: Missing {} in archive", state_vfs_path))?
            .to_vec();
        let axons = archive
            .get_file(&axons_vfs_path)
            .with_context(|| format!("FATAL: Missing {} in archive", axons_vfs_path))?
            .to_vec();

        std::fs::write(&state_path, &state)?;
        std::fs::write(&axons_path, &axons)?;

        // Export geometry for Plasticity Daemon
        if let Some(geom) = archive.get_file(&format!("baked/{}/shard.geom", zone_name)) {
            std::fs::write(mem_zone_dir.join("shard.geom"), geom)?;
        }
        if let Some(paths) = archive.get_file(&format!("baked/{}/shard.paths", zone_name)) {
            std::fs::write(mem_zone_dir.join("shard.paths"), paths)?;
        }

        // DOD FIX: Unpack DNA into SRAM for the Daemon
        let brain_dna_dir = mem_zone_dir.join("BrainDNA");
        std::fs::create_dir_all(&brain_dna_dir)?;
        for file in [
            "simulation.toml",
            "blueprints.toml",
            "anatomy.toml",
            "shard.toml",
        ] {
            if let Some(data) = archive.get_file(&format!("baked/{}/BrainDNA/{}", zone_name, file))
            {
                std::fs::write(brain_dna_dir.join(file), data)?;
            }
        }

        (state, axons)
    };

    assert!(
        axons_blob.len() % 32 == 0,
        "C-ABI Alignment Violation: .axons size ({}) is not a multiple of 32 bytes for zone {}",
        axons_blob.len(),
        zone_name
    );

    // [DOD FIX] Hardware pre-allocation for Dynamic Capacity Routing.
    let total_ghosts = manifest.memory.ghost_capacity as u32;
    let file_axons = (axons_blob.len() / 32) as u32;

    // 2. Size derivation (take maximum between config and reality on disk)
    let calc_axons = (manifest.memory.padded_n as u32)
        + (manifest.memory.virtual_axons as u32)
        + manifest.memory.injected_ghosts
        + total_ghosts;
    let total_axons = std::cmp::max(calc_axons, file_axons);
    let total_axons = (total_axons + 31) & !31; // Warp Alignment

    let padded_n = manifest.memory.padded_n as u32;

    // Integrity verification (protection against corrupted files)
    let (_, expected_state_size) = calculate_state_blob_size(padded_n as usize);
    if state_blob.len() != expected_state_size {
        anyhow::bail!(
            "FATAL: .state blob size mismatch for zone {}. Expected {}, got {}. Corruption or version skew!",
            zone_name, expected_state_size, state_blob.len()
        );
    }

    // Extract soma_to_axon for Ghost-axon routing
    let offsets = compute_state_offsets(padded_n as usize);
    let s2a_bytes =
        &state_blob[offsets.soma_to_axon..offsets.soma_to_axon + (padded_n as usize * 4)];
    let soma_to_axon: Vec<u32> = s2a_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // 3. VRAM allocation and DMA upload
    let vram = VramState::allocate(padded_n, total_axons, total_ghosts, use_gpu);
    vram.upload_state(&state_blob);
    vram.upload_axon_heads(&axons_blob);

    Ok((ShardEngine::new(vram), soma_to_axon, mem_zone_dir))
}

impl Bootloader {
    /// Full node bootstrap sequence. Standard "Axicor Sequence" pipeline.
    pub async fn boot_node(
        archive: Arc<axicor_core::vfs::AxicArchive>,
        project_name: &str,
        zone_names: &[String],
    ) -> Result<BootResult> {
        Self::boot_node_with_profile(
            archive,
            project_name,
            zone_names,
            crate::CpuProfile::Aggressive,
            true,
        )
        .await
    }

    // MONOLITH: HIGH — Bootloader::boot_node_with_profile is a massive async procedural block.
    // REFACTOR: Split into discrete boot phases (ROM/SRAM, Hardware, Routing) using a stateful pipeline.
    pub async fn boot_node_with_profile(
        archive: Arc<axicor_core::vfs::AxicArchive>,
        project_name: &str,
        zone_names: &[String],
        cpu_profile: crate::CpuProfile,
        use_gpu: bool,
    ) -> Result<BootResult> {
        // 1. Data/Config Phase: Load brain and simulation configs
        let mut zone_manifests_with_names = Vec::new();
        let mut sim_config = None;
        let mut manifest_metadata = HashMap::new();

        for zone_name in zone_names {
            let (zm, sc) = Self::parse_manifests_from_vfs(&archive, zone_name)?;

            let atomic_settings = Arc::new(crate::node::shard_thread::ShardAtomicSettings {
                night_interval_ticks: std::sync::atomic::AtomicU64::new(
                    zm.settings.night_interval_ticks,
                ),
                save_checkpoints_interval_ticks: std::sync::atomic::AtomicU64::new(
                    zm.settings.save_checkpoints_interval_ticks,
                ),
                prune_threshold: std::sync::atomic::AtomicI16::new(
                    zm.settings.plasticity.prune_threshold,
                ),
                max_sprouts: std::sync::atomic::AtomicU16::new(zm.settings.plasticity.max_sprouts),
            });

            manifest_metadata.insert(
                zm.zone_hash,
                crate::node::ShardMetadata {
                    manifest_path: PathBuf::from(zone_name), // Virtual path
                    last_modified: SystemTime::now(),        // VFS is immutable during runtime
                    atomic_settings,
                },
            );

            zone_manifests_with_names.push((zm.clone(), zone_name.clone()));
            if sim_config.is_none() {
                sim_config = Some(sc);
            }
        }

        let sim_config = sim_config.context("No zones provided")?;

        let sync_batch_ticks = sim_config.simulation.sync_batch_ticks;
        let cluster_secret = axicor_core::seed::seed_from_str(&sim_config.simulation.master_seed); // [DOD FIX]

        // 2. Hardware & VRAM Phase: Allocate weights/targets and flash physics laws
        let (shards, s2a_maps, axon_head_ptrs, io_contexts, all_geo_data, output_routes) =
            Self::load_all_shards_into_vram_vfs(
                &archive,
                project_name,
                &zone_manifests_with_names,
                sync_batch_ticks,
                use_gpu,
            )?;

        let first_manifest = zone_manifests_with_names[0].0.clone();
        unsafe { Self::flash_hardware_physics(&first_manifest, use_gpu)? };

        // 3. Topology Interconnect: Build local and remote routing channels
        let (intra_gpu_channels, inter_node_channels, expected_peers) =
            Self::build_routing_channels_vfs(
                &archive,
                &zone_manifests_with_names,
                &s2a_maps,
                &axon_head_ptrs,
            )?;

        // [DOD FIX] Flash routing table for inter-zone Egress
        let mut initial_routes = HashMap::new();
        let mut logged_routes = std::collections::HashSet::new();

        for (zm, _) in &zone_manifests_with_names {
            for conn in &zm.connections {
                let src_hash = axicor_core::hash::fnv1a_32(conn.from.as_bytes());
                let dst_hash = axicor_core::hash::fnv1a_32(conn.to.as_bytes());

                if src_hash == zm.zone_hash {
                    if let Some(peer_addr) = zm.network.fast_path_peers.get(&conn.to) {
                        let addr = peer_addr
                            .parse::<std::net::SocketAddr>()
                            .expect("FATAL: Invalid peer IP");
                        initial_routes.insert(dst_hash, (addr, 65507));
                        if logged_routes.insert((zm.zone_hash, dst_hash, "Egress")) {
                            info!(
                                "[Boot] Route (Egress): {} (0x{:08X}) -> {}",
                                conn.to, dst_hash, addr
                            );
                        }
                    }
                } else if dst_hash == zm.zone_hash {
                    if let Some(peer_addr) = zm.network.fast_path_peers.get(&conn.from) {
                        let addr = peer_addr
                            .parse::<std::net::SocketAddr>()
                            .expect("FATAL: Invalid peer IP");
                        initial_routes.insert(src_hash, (addr, 65507));
                        if logged_routes.insert((zm.zone_hash, src_hash, "ACK")) {
                            info!(
                                "[Boot] Route (ACK): {} (0x{:08X}) -> {}",
                                conn.from, src_hash, addr
                            );
                        }
                    }
                }
            }
        }
        let shared_acks_queue = Arc::new(crossbeam::queue::SegQueue::new());
        let shared_prunes_queue = Arc::new(crossbeam::queue::SegQueue::new());
        let routing_table = Arc::new(RoutingTable::new(initial_routes));

        // 4. Network Setup: IO, Geometry, and Telemetry servers
        let (io_server, geometry_server, telemetry_swapchain, egress_pool, inter_node_router) =
            Self::setup_networking(
                &first_manifest,
                io_contexts,
                routing_table.clone(),
                shared_acks_queue.clone(),
                shared_prunes_queue.clone(),
                cluster_secret,
            )
            .await?;

        // 5. Orchestrator Assembly: Glue everything into NodeRuntime
        let bsp_barrier = Arc::new(
            BspBarrier::new(
                sim_config.simulation.sync_batch_ticks as usize,
                expected_peers,
            )
            .with_cpu_profile(cpu_profile),
        );
        let node_runtime = NodeRuntime::boot(
            shards,
            io_server,
            routing_table,
            bsp_barrier,
            telemetry_swapchain.clone(),
            std::net::Ipv4Addr::new(127, 0, 0, 1),
            first_manifest.network.fast_path_udp_local,
            output_routes,
            intra_gpu_channels,
            inter_node_channels,
            inter_node_router,
            axon_head_ptrs,
            egress_pool.clone(),
            manifest_metadata,
            shared_acks_queue,
            shared_prunes_queue,
            sync_batch_ticks,
            cluster_secret,
        );

        Ok(BootResult {
            node_runtime,
            geometry_server,
            geometry_data: all_geo_data,
            telemetry_swapchain,
            egress_pool,
        })
    }

    fn parse_manifests_from_vfs(
        archive: &axicor_core::vfs::AxicArchive,
        zone_name: &str,
    ) -> Result<(ZoneManifest, axicor_core::config::SimulationConfig)> {
        let manifest_vfs_path = format!("baked/{}/manifest.toml", zone_name);
        let manifest_bytes = archive.get_file(&manifest_vfs_path).with_context(|| {
            format!(
                "Failed to find manifest for zone {} at {}",
                zone_name, manifest_vfs_path
            )
        })?;

        let zone_manifest: ZoneManifest = toml::from_str(std::str::from_utf8(manifest_bytes)?)
            .with_context(|| format!("Failed to parse zone manifest for {}", zone_name))?;

        let sim_ref = zone_manifest
            .simulation
            .as_ref()
            .context("ZoneManifest missing simulation reference")?;
        // In the archive, simulation.toml is in the root or via manifest path.
        // Baker places it in BrainDNA/simulation.toml inside the zone folder.
        let sim_vfs_path = format!("baked/{}/{}", zone_name, sim_ref.config.to_string_lossy());

        let sim_bytes = archive
            .get_file(&sim_vfs_path)
            .with_context(|| format!("Failed to read simulation.toml at {}", sim_vfs_path))?;
        let sim_config: axicor_core::config::SimulationConfig =
            toml::from_str(std::str::from_utf8(sim_bytes)?)
                .context("Failed to parse simulation.toml")?;

        Ok((zone_manifest, sim_config))
    }

    unsafe fn flash_hardware_physics(first_manifest: &ZoneManifest, use_gpu: bool) -> Result<()> {
        let mut gpu_variants = [axicor_core::layout::VariantParameters::default(); 16];
        for v in &first_manifest.variants {
            if (v.id as usize) < 16 {
                gpu_variants[v.id as usize] = v.clone().into_gpu();
            }
        }
        if use_gpu {
            let err = axicor_compute::ffi::cu_upload_constant_memory(
                gpu_variants.as_ptr()
            );
            if err != 0 {
                anyhow::bail!("FATAL: cu_upload_constant_memory failed with {}", err);
            }
        } else {
            axicor_compute::bindings::cpu_upload_constant_memory(
                gpu_variants.as_ptr()
            );
        }
        info!("[Boot] Hardware physics parameters flashed.");
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn load_all_shards_into_vram_vfs(
        archive: &axicor_core::vfs::AxicArchive,
        project_name: &str,
        zone_manifests_with_names: &[(ZoneManifest, String)],
        sync_batch_ticks: u32,
        use_gpu: bool,
    ) -> Result<(
        Vec<BootShard>,
        HashMap<u32, Vec<u32>>,
        HashMap<u32, *mut axicor_core::layout::BurstHeads8>,
        Vec<(u32, crate::network::io_server::ZoneIoContext)>,
        Vec<u32>,
        HashMap<u32, Vec<(String, u32, usize, usize)>>,
    )> {
        // [DOD FIX] Explicitly bind device context to main thread before loading!
        if use_gpu {
            unsafe {
                axicor_compute::ffi::gpu_set_device(0);
            }
        }

        let mut engines = Vec::new();
        let mut io_contexts = Vec::new();
        let mut all_geo_data = Vec::new();
        let mut output_routes: HashMap<u32, Vec<(String, u32, usize, usize)>> = HashMap::new();
        let mut axon_head_ptrs = HashMap::new();
        let mut s2a_maps = HashMap::new();

        for (zone_manifest, zone_name) in zone_manifests_with_names {
            let zone_hash = zone_manifest.zone_hash;

            info!(
                "[Boot] Loading Local Zone {} from VFS (GPU={})",
                zone_name, use_gpu
            );
            let (engine, s2a, mem_zone_dir) =
                boot_shard_from_vfs(archive, zone_name, zone_manifest, project_name, use_gpu)?;

            let ptrs = match engine {
                ShardEngine::Gpu(ref gpu) => &gpu.vram.ptrs,
                ShardEngine::Cpu(ref cpu) => &cpu.vram.ptrs,
            };
            axon_head_ptrs.insert(zone_hash, ptrs.axon_heads);
            s2a_maps.insert(zone_hash, s2a);

            let io_vfs_path = format!("baked/{}/BrainDNA/io.toml", zone_name);
            let mut expected_inputs = false;
            let mut expected_outputs = false;
            let mut matrix_offsets = HashMap::new();
            if let Some(io_bytes) = archive.get_file(&io_vfs_path) {
                let io_str = std::str::from_utf8(io_bytes)?;
                if let Ok(io_config) = toml::from_str::<axicor_core::config::io::IoConfig>(io_str) {
                    expected_inputs = !io_config.input.is_empty();
                    expected_outputs = !io_config.output.is_empty();

                    let mut current_bit_offset = 0u32;
                    for matrix in &io_config.input {
                        for pin in &matrix.pin {
                            let hash = axicor_core::hash::fnv1a_32(pin.name.as_bytes());
                            matrix_offsets.insert(hash, current_bit_offset / 8);
                            current_bit_offset += pin.width * pin.height;
                            current_bit_offset = (current_bit_offset + 31) & !31;
                        }
                    }

                    let mut current_pixel_offset = 0usize;
                    for matrix in &io_config.output {
                        for pin in &matrix.pin {
                            let hash = axicor_core::hash::fnv1a_32(pin.name.as_bytes());
                            let chunk_pixels = (pin.width * pin.height) as usize;

                            let target = zone_manifest
                                .network
                                .external_udp_out_target
                                .clone()
                                .unwrap_or_else(|| "127.0.0.1:8092".to_string());

                            output_routes
                                .entry(zone_hash)
                                .or_default()
                                .push((target.clone(), hash, current_pixel_offset, chunk_pixels));
                            info!(
                                "[Boot] Registered Output Route: {} (0x{:08X}) -> {}",
                                pin.name, hash, target
                            );

                            current_pixel_offset += chunk_pixels;
                        }
                    }
                }
            }

            // [DOD FIX] virtual_offset must be valid even for zones without external I/O
            let virtual_offset = match engine {
                ShardEngine::Gpu(ref gpu) => gpu.vram.virtual_offset(),
                ShardEngine::Cpu(ref cpu) => cpu.vram.virtual_offset(),
            };

            let num_virtual_axons = if expected_inputs {
                let gxi_vfs_path = format!("baked/{}/shard.gxi", zone_name);
                let gxi_bytes = archive.get_file(&gxi_vfs_path).with_context(|| {
                    format!(
                        "FATAL: Zone {} expects inputs but {} is missing from archive!",
                        zone_name, gxi_vfs_path
                    )
                })?;
                let gxi = GxiFile::load_from_bytes(gxi_bytes);
                gxi.total_pixels
            } else {
                0
            };

            let (num_outputs, mapped_soma_ids_host) = if expected_outputs {
                let gxo_vfs_path = format!("baked/{}/shard.gxo", zone_name);
                let gxo_bytes = archive.get_file(&gxo_vfs_path).with_context(|| {
                    format!(
                        "FATAL: Zone {} expects outputs but {} is missing from archive!",
                        zone_name, gxo_vfs_path
                    )
                })?;
                let gxo = GxoFile::load_from_bytes(gxo_bytes);
                (gxo.total_pixels, Some(gxo.soma_ids))
            } else {
                (0, None)
            };

            let pos_vfs_path = format!("baked/{}/shard.pos", zone_name);
            let pos_bytes = archive
                .get_file(&pos_vfs_path)
                .with_context(|| format!("Failed to read {} from archive", pos_vfs_path))?;
            let geo_data: Vec<u32> = pos_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            all_geo_data.extend(geo_data);

            // [DOD FIX] Hardcode removed: let sync_batch_ticks = 100u32;
            #[allow(clippy::manual_div_ceil)]
            let input_words_per_tick = (num_virtual_axons + 63) / 64 * 2;
            let input_capacity_bytes = (input_words_per_tick * sync_batch_ticks * 4) as usize;

            let io_ctx = crate::network::io_server::ZoneIoContext {
                swapchain: std::sync::Arc::new(
                    crate::network::io_server::InputSwapchain::new(input_capacity_bytes)
                        .expect("FATAL: Failed to allocate Pinned RAM for Input"),
                ),
                matrix_offsets,
            };

            let shard_toml_vfs_path = format!("baked/{}/BrainDNA/shard.toml", zone_name);
            let shard_toml_bytes = archive
                .get_file(&shard_toml_vfs_path)
                .with_context(|| format!("Failed to load {} from archive", shard_toml_vfs_path))?;
            let instance_config =
                axicor_core::config::InstanceConfig::parse(std::str::from_utf8(shard_toml_bytes)?)
                    .map_err(anyhow::Error::msg)?;

            let incoming_grow = Arc::new(SegQueue::new());

            let v_seg = zone_manifest.memory.v_seg as u32;

            engines.push(BootShard {
                hash: zone_hash,
                engine,
                num_virtual_axons,
                virtual_offset,
                num_outputs,
                mapped_soma_ids_host,
                baked_dir: mem_zone_dir, // Using SRAM path
                config: instance_config,
                v_seg,
                incoming_grow,
            });
            io_contexts.push((zone_hash, io_ctx));
        }

        Ok((
            engines,
            s2a_maps,
            axon_head_ptrs,
            io_contexts,
            all_geo_data,
            output_routes,
        ))
    }

    #[allow(clippy::type_complexity)]
    fn build_routing_channels_vfs(
        archive: &axicor_core::vfs::AxicArchive,
        zone_manifests_with_names: &[(ZoneManifest, String)],
        s2a_maps: &HashMap<u32, Vec<u32>>,
        axon_head_ptrs: &HashMap<u32, *mut axicor_core::layout::BurstHeads8>,
    ) -> Result<(
        Vec<(
            *mut axicor_core::layout::BurstHeads8,
            *mut axicor_core::layout::BurstHeads8,
            IntraGpuChannel,
        )>,
        Vec<(
            *mut axicor_core::layout::BurstHeads8,
            crate::network::inter_node::InterNodeChannel,
        )>,
        usize, // expected_peers
    )> {
        let mut intra_gpu = Vec::new();
        let inter_node = Vec::new();
        let mut expected_peers = 0;

        // [DOD FIX] Epic 5: Bootstrap Idempotence.
        // ZoneManifests redundantly store the global connectome. We MUST deduplicate
        // to prevent VRAM memory leaks and fatal Data Races in cu_ghost_sync_kernel.
        let mut unique_connections = std::collections::HashMap::new();
        let mut receiver_manifests: HashMap<u32, &ZoneManifest> = HashMap::new();
        for (zm, _) in zone_manifests_with_names {
            receiver_manifests.insert(zm.zone_hash, zm);
            for conn in &zm.connections {
                unique_connections.insert((conn.from.clone(), conn.to.clone()), conn.clone());
            }
        }

        let all_connections: Vec<_> = unique_connections.into_values().collect();

        for conn in &all_connections {
            let src_hash = axicor_core::hash::fnv1a_32(conn.from.as_bytes());
            let dst_hash = axicor_core::hash::fnv1a_32(conn.to.as_bytes());

            let is_src_local = axon_head_ptrs.contains_key(&src_hash);
            let is_dst_local = axon_head_ptrs.contains_key(&dst_hash);

            if !is_src_local && is_dst_local {
                info!("[Boot] Peer detected (Ingress): expecting fast-path data from remote zone 0x{:08X}", src_hash);
                expected_peers += 1;
            }
            if is_src_local && !is_dst_local {
                info!("[Boot] Peer detected (Egress): expecting fast-path ACK from remote zone 0x{:08X}", dst_hash);
                expected_peers += 1;
            }

            if !is_src_local {
                continue;
            } // Outbound routing from remote source doesn't concern us

            // [DOD FIX] Ghost file lives in RECEIVER's baked_dir, not sender's.
            if !is_dst_local {
                info!("[Boot] Outbound connection {} -> {} (routing only, no local ghost file needed)", conn.from, conn.to);
                continue;
            }

            // We are the RECEIVER (is_dst_local). Ghost file must be in OUR baked_dir in archive.
            let receiver_manifest = receiver_manifests.get(&dst_hash).unwrap();
            let capacity = receiver_manifest.memory.ghost_capacity as u32;
            let ghosts_vfs_path = format!("baked/{}/{}_{}.ghosts", conn.to, conn.from, conn.to);

            if let Some(ghosts_bytes) = archive.get_file(&ghosts_vfs_path) {
                let (src_somas, dst_ghosts) = load_ghosts(ghosts_bytes);
                info!(
                    "[Ghosts] Successfully loaded {} links from archive: {}",
                    src_somas.len(),
                    ghosts_vfs_path
                );
                let s2a = s2a_maps
                    .get(&src_hash)
                    .context("S2A map missing for source zone")?;

                let mut src_axons = Vec::with_capacity(src_somas.len());
                for &soma in &src_somas {
                    let mut axon_id = axicor_core::constants::AXON_SENTINEL;
                    if soma != axicor_core::ipc::EMPTY_PIXEL {
                        axon_id = s2a[soma as usize];
                    }
                    src_axons.push(axon_id);
                }

                let src_ptr = *axon_head_ptrs.get(&src_hash).unwrap();
                let dst_ptr = *axon_head_ptrs.get(&dst_hash).unwrap();
                let receiver_manifest = receiver_manifests.get(&dst_hash).unwrap();
                let dst_total_axons = receiver_manifest.memory.padded_n 
                    + receiver_manifest.memory.virtual_axons 
                    + receiver_manifest.memory.injected_ghosts as usize
                    + receiver_manifest.memory.ghost_capacity;

                let channel = unsafe {
                    IntraGpuChannel::from_slices(
                        src_hash,
                        dst_hash,
                        &src_axons,
                        &dst_ghosts,
                        capacity,
                        dst_total_axons as u32, // [DOD FIX] Pass layout limit
                    )
                };
                intra_gpu.push((src_ptr, dst_ptr, channel));
                info!(
                    "[Boot] Built IntraGpuChannel: {} -> {} ({} links, capacity: {})",
                    conn.from,
                    conn.to,
                    src_axons.len(),
                    capacity
                );
            } else {
                panic!(
                    "CRITICAL TOPOLOGY ERROR: Incoming ghost file not found in archive: {}",
                    ghosts_vfs_path
                );
            }
        }
        Ok((intra_gpu, inter_node, expected_peers))
    }

    async fn setup_networking(
        first_manifest: &ZoneManifest,
        io_contexts: Vec<(u32, crate::network::io_server::ZoneIoContext)>,
        routing_table: Arc<RoutingTable>,
        shared_acks_queue: Arc<crossbeam::queue::SegQueue<axicor_core::ipc::AxonHandoverAck>>,
        shared_prunes_queue: Arc<crossbeam::queue::SegQueue<axicor_core::ipc::AxonHandoverPrune>>,
        cluster_secret: u64, // [DOD FIX]
    ) -> Result<(
        Arc<ExternalIoServer>,
        GeometryServer,
        Arc<crate::network::telemetry::TelemetrySwapchain>,
        Arc<crate::network::egress::EgressPool>,
        Arc<crate::network::router::InterNodeRouter>,
    )> {
        let local_port = first_manifest.network.fast_path_udp_local;
        let udp_in = first_manifest.network.external_udp_in;

        let io_socket = tokio::net::UdpSocket::bind(&format!("127.0.0.1:{}", udp_in)).await
            .with_context(|| format!("Failed to bind UDP external_udp_in (port {}). Port in use? Kill any running axicor-node: Get-Process axicor-node -EA 0 | Stop-Process -Force", udp_in))?;
        let io_server = Arc::new(ExternalIoServer::new(
            Arc::new(AtomicBool::new(false)),
            io_contexts,
            routing_table.clone(),
            Arc::new(io_socket),
            cluster_secret,
        )?);

        let geo_port = local_port + 1;
        let geo_addr = format!("127.0.0.1:{}", geo_port).parse()?;
        let geometry_server = GeometryServer::bind(geo_addr, shared_acks_queue, shared_prunes_queue).await
            .with_context(|| format!("Failed to bind Geometry Server (TCP port {}). Port in use? Kill any running axicor-node: Get-Process axicor-node -EA 0 | Stop-Process -Force", geo_port))?;
        let telemetry_port = local_port + 2;
        let telemetry_swapchain = TelemetryServer::start(telemetry_port).await;

        let egress_socket = Arc::new(tokio::net::UdpSocket::bind("0.0.0.0:0").await?);
        let inter_node_router = Arc::new(crate::network::router::InterNodeRouter::new(
            egress_socket,
            routing_table,
        ));
        let egress_pool = Arc::new(crate::network::egress::EgressPool::new(1024));

        Ok((
            io_server,
            geometry_server,
            telemetry_swapchain,
            egress_pool,
            inter_node_router,
        ))
    }
}
