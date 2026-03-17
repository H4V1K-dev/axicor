use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::collections::HashMap;
use crossbeam::queue::SegQueue;

use genesis_core::config::manifest::ZoneManifest;
use genesis_compute::memory::{VramState, calculate_state_blob_size, compute_state_offsets};
use genesis_compute::ShardEngine;
use crate::network::geometry_client::GeometryServer;
use crate::network::telemetry::TelemetryServer;
use crate::network::io_server::ExternalIoServer;
use crate::network::bsp::BspBarrier;
use crate::node::NodeRuntime;
use crate::network::router::RoutingTable;
use crate::input::GxiFile;
use crate::output::GxoFile;
use crate::network::ghosts::load_ghosts;
use crate::network::intra_gpu::IntraGpuChannel;

type BootShard = crate::node::shard_thread::ShardDescriptor;

pub struct Bootloader;

pub struct BootResult {
    pub node_runtime: NodeRuntime,
    pub geometry_server: GeometryServer,
    pub geometry_data: Vec<u32>,
    pub telemetry_swapchain: Arc<crate::network::telemetry::TelemetrySwapchain>,
    pub egress_pool: Arc<crate::network::egress::EgressPool>,
}

/// Инициализирует ShardEngine прямым DMA-копированием из .state и .axons.
/// Этот метод реализует O(1) деривацию размеров на основе файлового контракта:
/// - .state: 910 байта на нейрон (SoA)
/// - .axons: 32 байта на аксон (BurstHeads8)
pub fn boot_shard_from_disk(baked_dir: &Path, manifest: &ZoneManifest) -> Result<(ShardEngine, Vec<u32>)> {
    let chk_state = baked_dir.join("checkpoint.state");
    let chk_axons = baked_dir.join("checkpoint.axons");
    let base_state = baked_dir.join("shard.state");
    let base_axons = baked_dir.join("shard.axons");

    // [DOD FIX] Strict Causal Continuity: Загружаем только синхронизированную пару
    let (state_path, axons_path) = if chk_state.exists() && chk_axons.exists() {
        println!("[Boot] 💾 Resuming from matched checkpoints (State + Active Tails)");
        (chk_state, chk_axons)
    } else {
        (base_state, base_axons)
    };

    let state_blob = std::fs::read(&state_path)
        .with_context(|| format!("FATAL: Missing .state file at {:?}", state_path))?;
    let axons_blob = std::fs::read(&axons_path)
        .with_context(|| format!("FATAL: Missing .axons file at {:?}", axons_path))?;

    assert!(
        axons_blob.len() % 32 == 0,
        "C-ABI Alignment Violation: .axons file size ({}) is not a multiple of 32 bytes. Path: {:?}",
        axons_blob.len(), axons_path
    );

    // 1. Сначала сканируем baked_dir на наличие входящих связей
    let mut total_ghosts = 0;

    for connection in &manifest.connections {
        let to_hash = genesis_core::hash::fnv1a_32(connection.to.as_bytes());
        if to_hash == manifest.zone_hash {
            let ghost_file_name = format!("{}_{}.ghosts", connection.from, connection.to);
            let ghost_path = baked_dir.join(ghost_file_name);
            
            if ghost_path.exists() {
                let (src_somas, _) = load_ghosts(&ghost_path);
                total_ghosts += src_somas.len() as u32;
                println!("[Boot] Prepared {} ghost axons from {}", src_somas.len(), connection.from);
            }
        }
    }

    // 2. Деривация размеров
    let total_axons = (manifest.memory.padded_n as u32) + total_ghosts + (manifest.memory.virtual_axons as u32);
    let padded_n = manifest.memory.padded_n as u32;

    // Верификация целостности (защита от битых файлов)
    let (_, expected_state_size) = calculate_state_blob_size(padded_n as usize);
    if state_blob.len() != expected_state_size {
        anyhow::bail!(
            "FATAL: .state blob size mismatch for {:?}. Expected {}, got {}. Corruption or version skew!",
            state_path, expected_state_size, state_blob.len()
        );
    }

    // Извлекаем soma_to_axon для маршрутизации Ghost-аксонов
    let offsets = compute_state_offsets(padded_n as usize);
    let s2a_bytes = &state_blob[offsets.soma_to_axon .. offsets.soma_to_axon + (padded_n as usize * 4)];
    let soma_to_axon: Vec<u32> = s2a_bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // 3. Выделение VRAM и DMA-заливка
    let vram = VramState::allocate(padded_n, total_axons, total_ghosts);
    vram.upload_state(&state_blob);
    vram.upload_axon_heads(&axons_blob);

    Ok((ShardEngine::new(vram), soma_to_axon))
}

impl Bootloader {
    /// Full node bootstrap sequence. Standard "Genesis Sequence" pipeline.
    pub async fn boot_node(manifest_paths: &[PathBuf], telemetry: Arc<crate::tui::state::LockFreeTelemetry>) -> Result<BootResult> {
        Self::boot_node_with_profile(manifest_paths, telemetry, crate::CpuProfile::Aggressive).await
    }

    pub async fn boot_node_with_profile(manifest_paths: &[PathBuf], telemetry: Arc<crate::tui::state::LockFreeTelemetry>, cpu_profile: crate::CpuProfile) -> Result<BootResult> {
        // 1. Data/Config Phase: Load brain and simulation configs
        let mut zone_manifests_with_paths = Vec::new();
        let mut sim_config = None;
        let mut manifest_metadata = HashMap::new();
        
        for path in manifest_paths {
            let root_dir: &Path = path.parent().unwrap_or(Path::new("."));
            let (zm, sc) = Self::parse_manifests(path, root_dir)?;
            
            // [DOD FIX] Initialize Hot-Reload metadata
            let last_modified = std::fs::metadata(path)
                .and_then(|m| m.modified())
                .unwrap_or_else(|_| SystemTime::now());
            
            let atomic_settings = Arc::new(crate::node::shard_thread::ShardAtomicSettings {
                night_interval_ticks: std::sync::atomic::AtomicU64::new(zm.settings.night_interval_ticks),
                save_checkpoints_interval_ticks: std::sync::atomic::AtomicU64::new(zm.settings.save_checkpoints_interval_ticks),
                prune_threshold: std::sync::atomic::AtomicI16::new(zm.settings.plasticity.prune_threshold),
                max_sprouts: std::sync::atomic::AtomicU16::new(zm.settings.plasticity.max_sprouts),
            });

            manifest_metadata.insert(zm.zone_hash, crate::node::ShardMetadata {
                manifest_path: path.clone(),
                last_modified,
                atomic_settings,
            });

            zone_manifests_with_paths.push((zm.clone(), root_dir.to_path_buf()));
            if sim_config.is_none() {
                sim_config = Some(sc);
            }

            // [TUI] Initialize ZoneMetrics
            // Note: In the new system, Shards report spikes directly to the telemetry via hash.
            // DashboardState (the rendering view) will likely need to be updated to match.
            // For now, we only need to ensure the telemetry object is shared correctly.
        }
        
        let sim_config = sim_config.context("No manifests provided")?;

        let sync_batch_ticks = sim_config.simulation.sync_batch_ticks;
        let cluster_secret = genesis_core::seed::seed_from_str(&sim_config.simulation.master_seed); // [DOD FIX]

        // 2. Hardware & VRAM Phase: Allocate weights/targets and flash physics laws
        let (shards, s2a_maps, axon_head_ptrs, io_contexts, all_geo_data, output_routes) = 
            Self::load_all_shards_into_vram(&zone_manifests_with_paths, sync_batch_ticks)?;

        let first_manifest = zone_manifests_with_paths[0].0.clone();
        unsafe { Self::flash_hardware_physics(&first_manifest)? };

        // 3. Topology Interconnect: Build local and remote routing channels
        let (intra_gpu_channels, inter_node_channels, expected_peers) = 
            Self::build_routing_channels(&zone_manifests_with_paths, &s2a_maps, &axon_head_ptrs)?;

        // [DOD FIX] Прошиваем таблицу маршрутизации для межзонального Egress
        let mut initial_routes = HashMap::new();
        for (zm, _) in &zone_manifests_with_paths {
            for conn in &zm.connections {
                let src_hash = genesis_core::hash::fnv1a_32(conn.from.as_bytes());
                let dst_hash = genesis_core::hash::fnv1a_32(conn.to.as_bytes());

                if src_hash == zm.zone_hash {
                    if let Some(peer_addr) = zm.network.fast_path_peers.get(&conn.to) {
                        let addr = peer_addr.parse::<std::net::SocketAddr>().expect("FATAL: Invalid peer IP");
                        initial_routes.insert(dst_hash, addr);
                        println!("[Boot] Route (Egress): {} (0x{:08X}) -> {}", conn.to, dst_hash, addr);
                    }
                } else if dst_hash == zm.zone_hash {
                    if let Some(peer_addr) = zm.network.fast_path_peers.get(&conn.from) {
                        let addr = peer_addr.parse::<std::net::SocketAddr>().expect("FATAL: Invalid peer IP");
                        initial_routes.insert(src_hash, addr);
                        println!("[Boot] Route (ACK): {} (0x{:08X}) -> {}", conn.from, src_hash, addr);
                    }
                }
            }
        }
        let shared_acks_queue = Arc::new(crossbeam::queue::SegQueue::new());
        let routing_table = Arc::new(RoutingTable::new(initial_routes));

        // 4. Network Setup: IO, Geometry, and Telemetry servers
        let (io_server, geometry_server, telemetry_swapchain, egress_pool, inter_node_router) = 
            Self::setup_networking(&first_manifest, io_contexts, routing_table.clone(), shared_acks_queue.clone(), telemetry.clone(), cluster_secret).await?;

        // 5. Orchestrator Assembly: Glue everything into NodeRuntime
        let bsp_barrier = Arc::new(BspBarrier::new(sim_config.simulation.sync_batch_ticks as usize, expected_peers).with_cpu_profile(cpu_profile));

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
            telemetry,
            shared_acks_queue,
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

    fn parse_manifests(manifest_path: &Path, _root_dir: &Path) -> Result<(ZoneManifest, genesis_core::config::SimulationConfig)> {
        let manifest_toml = std::fs::read_to_string(manifest_path).map_err(|e| {
            let hint = if e.kind() == std::io::ErrorKind::NotFound {
                "\nHint: Run 'cargo run --release -p genesis-baker --bin baker -- --brain config/brain.toml' first to create baked artifacts."
            } else {
                ""
            };
            anyhow::anyhow!("Failed to read manifest: {:?}{}", manifest_path, hint)
        })?;
        
        let zone_manifest: ZoneManifest = toml::from_str(&manifest_toml)
            .with_context(|| format!("Failed to parse zone manifest: {:?}", manifest_path))?;

        let sim_ref = zone_manifest.simulation.as_ref().context("ZoneManifest missing simulation reference")?;
        let sim_path = manifest_path.parent().unwrap().join(&sim_ref.config);
        
        println!("[Boot] Loading Simulation Config from: {:?}", sim_path);
        let sim_toml = std::fs::read_to_string(&sim_path)
            .with_context(|| format!("Failed to read simulation.toml at {:?}", sim_path))?;
        let sim_config: genesis_core::config::SimulationConfig = toml::from_str(&sim_toml)
            .context("Failed to parse simulation.toml")?;
        
        Ok((zone_manifest, sim_config))
    }

    fn _load_zone_manifest(root_dir: &Path, zones: &[genesis_core::config::brain::ZoneEntry]) -> Result<ZoneManifest> {
        let first_manifest_path = root_dir.join(&zones[0].baked_dir).join("manifest.toml");
        let manifest_toml = std::fs::read_to_string(&first_manifest_path)
            .with_context(|| format!("Failed to read zone manifest: {:?}", first_manifest_path))?;
        let manifest: ZoneManifest = toml::from_str(&manifest_toml)?;
        Ok(manifest)
    }

    unsafe fn flash_hardware_physics(first_manifest: &ZoneManifest) -> Result<()> {
        let mut gpu_variants = [genesis_core::layout::VariantParameters::default(); 16];
        for v in &first_manifest.variants {
            if (v.id as usize) < 16 {
                gpu_variants[v.id as usize] = v.clone().into_gpu();
            }
        }
        let err = genesis_compute::ffi::cu_upload_constant_memory(
            gpu_variants.as_ptr() as *const genesis_core::layout::VariantParameters
        );
        if err != 0 {
            anyhow::bail!("FATAL: cu_upload_constant_memory failed with {}", err);
        }
        println!("[Boot] Hardware physics parameters flashed to Constant Memory.");
        Ok(())
    }

    fn load_all_shards_into_vram(zone_manifests_with_paths: &[(ZoneManifest, PathBuf)], sync_batch_ticks: u32) 
        -> Result<(Vec<BootShard>, HashMap<u32, Vec<u32>>, HashMap<u32, *mut genesis_core::layout::BurstHeads8>, Vec<(u32, crate::network::io_server::ZoneIoContext)>, Vec<u32>, HashMap<u32, Vec<(String, u32)>>)> 
    {
        // [DOD FIX] Явно биндим контекст устройства к главному потоку перед загрузкой!
        unsafe { genesis_compute::ffi::gpu_set_device(0); }

        let mut engines = Vec::new();
        let mut io_contexts = Vec::new();
        let mut all_geo_data = Vec::new();
        let mut output_routes: HashMap<u32, Vec<(String, u32)>> = HashMap::new();
        let mut axon_head_ptrs = HashMap::new();
        let mut s2a_maps = HashMap::new();

        for (zone_manifest, root_dir) in zone_manifests_with_paths {
            let baked_dir = root_dir.clone();
            let zone_hash = zone_manifest.zone_hash;

            println!("[Boot] Loading Local Zone at {:?}", baked_dir);
            let (engine, s2a) = boot_shard_from_disk(&baked_dir, zone_manifest)?;

            axon_head_ptrs.insert(zone_hash, engine.vram.ptrs.axon_heads);
            s2a_maps.insert(zone_hash, s2a);

            let dna_dir = baked_dir.join("BrainDNA");
            let io_config_path = dna_dir.join("io.toml");

            let mut expected_inputs = false;
            let mut expected_outputs = false;
            let mut matrix_offsets = HashMap::new();

            if io_config_path.exists() {
                if let Ok(io_config) = genesis_core::config::io::IoConfig::load(&io_config_path) {
                    expected_inputs = !io_config.inputs.is_empty();
                    expected_outputs = !io_config.outputs.is_empty();

                    let mut current_bit_offset = 0u32;
                    for input in &io_config.inputs {
                        let hash = genesis_core::hash::fnv1a_32(input.name.as_bytes());
                        matrix_offsets.insert(hash, (current_bit_offset / 8) as u32);
                        current_bit_offset += input.width * input.height;
                        current_bit_offset = (current_bit_offset + 31) & !31;
                    }
                    
                    for output in &io_config.outputs {
                        let hash = genesis_core::hash::fnv1a_32(output.name.as_bytes());
                        let source_hash = genesis_core::hash::fnv1a_32(output.source_zone.as_bytes());
                        
                        if source_hash == zone_hash {
                            let target = zone_manifest.network.external_udp_out_target
                                .clone()
                                .unwrap_or_else(|| "127.0.0.1:8092".to_string());
                            
                            output_routes.entry(zone_hash).or_insert_with(Vec::new)
                                .push((target.clone(), hash));
                            println!("[Boot] Registered Output Route: {} (0x{:08X}) -> {}", output.name, hash, target);
                        }
                    }
                }
            }

            // [DOD FIX] virtual_offset must be valid even for zones without external I/O
            let virtual_offset = engine.vram.virtual_offset();
            let gxi_path = baked_dir.join("shard.gxi");
            let num_virtual_axons = if expected_inputs {
                if !gxi_path.exists() {
                    anyhow::bail!("FATAL: Zone expects inputs but {:?} is missing!", gxi_path);
                }
                let gxi = GxiFile::load(&gxi_path);
                gxi.total_pixels
            } else {
                0
            };

            let gxo_path = baked_dir.join("shard.gxo");
            let (num_outputs, mapped_soma_ids_host) = if expected_outputs {
                if !gxo_path.exists() {
                    anyhow::bail!("FATAL: Zone expects outputs but {:?} is missing!", gxo_path);
                }
                let gxo = GxoFile::load(&gxo_path);
                (gxo.total_pixels, Some(gxo.soma_ids))
            } else {
                (0, None)
            };

            let pos_path = baked_dir.join("shard.pos");
            let pos_blob = std::fs::read(&pos_path)
                .with_context(|| format!("Failed to read shard.pos: {:?}", pos_path))?;
            let geo_data: Vec<u32> = pos_blob.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            all_geo_data.extend(geo_data);

            // [DOD FIX] Hardcode removed: let sync_batch_ticks = 100u32;
            let input_words_per_tick = (num_virtual_axons + 31) / 32;
            let input_capacity_bytes = (input_words_per_tick * sync_batch_ticks * 4) as usize;

            let io_ctx = crate::network::io_server::ZoneIoContext {
                swapchain: std::sync::Arc::new(
                    crate::network::io_server::InputSwapchain::new(input_capacity_bytes)
                        .expect("FATAL: Failed to allocate Pinned RAM for Input"),
                ),
                matrix_offsets,
            };

            let instance_path = dna_dir.join("shard.toml");
            let instance_config = genesis_core::config::InstanceConfig::load(&instance_path)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("Failed to load InstanceConfig from {:?}", instance_path))?;

            let incoming_grow = Arc::new(SegQueue::new());

            let v_seg = zone_manifest.memory.v_seg as u32;

            engines.push(BootShard {
                hash: zone_hash,
                engine,
                num_virtual_axons,
                virtual_offset,
                num_outputs,
                mapped_soma_ids_host,
                baked_dir,
                config: instance_config,
                fast_path_udp_local: zone_manifest.network.fast_path_udp_local,
                v_seg,
                incoming_grow,
            });
            io_contexts.push((zone_hash, io_ctx));
        }

        Ok((engines, s2a_maps, axon_head_ptrs, io_contexts, all_geo_data, output_routes))
    }

    fn build_routing_channels(
        zone_manifests_with_paths: &[(ZoneManifest, PathBuf)],
        s2a_maps: &HashMap<u32, Vec<u32>>,
        axon_head_ptrs: &HashMap<u32, *mut genesis_core::layout::BurstHeads8>
    ) -> Result<(
        Vec<(*mut genesis_core::layout::BurstHeads8, *mut genesis_core::layout::BurstHeads8, IntraGpuChannel)>, 
        Vec<(*mut genesis_core::layout::BurstHeads8, crate::network::inter_node::InterNodeChannel)>, 
        usize // expected_peers
    )> {
        let mut intra_gpu = Vec::new();
        let inter_node = Vec::new();
        let mut expected_peers = 0;

        let mut all_connections = Vec::new();
        let mut receiver_dirs: HashMap<u32, PathBuf> = HashMap::new();
        let mut receiver_manifests: HashMap<u32, &ZoneManifest> = HashMap::new();
        for (zm, path) in zone_manifests_with_paths {
            receiver_dirs.insert(zm.zone_hash, path.clone());
            receiver_manifests.insert(zm.zone_hash, zm);
            all_connections.extend(zm.connections.clone());
        }

        for conn in &all_connections {
            let src_hash = genesis_core::hash::fnv1a_32(conn.from.as_bytes());
            let dst_hash = genesis_core::hash::fnv1a_32(conn.to.as_bytes());

            let is_src_local = axon_head_ptrs.contains_key(&src_hash);
            let is_dst_local = axon_head_ptrs.contains_key(&dst_hash);

            if !is_src_local && is_dst_local {
                println!("[Boot] Peer detected (Ingress): expecting fast-path data from remote zone 0x{:08X}", src_hash);
                expected_peers += 1;
            }
            if is_src_local && !is_dst_local {
                println!("[Boot] Peer detected (Egress): expecting fast-path ACK from remote zone 0x{:08X}", dst_hash);
                expected_peers += 1;
            }

            if !is_src_local { continue; } // Outbound routing from remote source doesn't concern us

            // [DOD FIX] Ghost file lives in RECEIVER's baked_dir, not sender's.
            // For outbound connections we don't need ghost data — just the routing table.
            if !is_dst_local {
                println!("[Boot] Outbound connection {} -> {} (routing only, no local ghost file needed)", conn.from, conn.to);
                continue;
            }

            // We are the RECEIVER (is_dst_local). Ghost file must be in OUR baked_dir.
            let root_dir = receiver_dirs.get(&dst_hash).unwrap();
            let receiver_manifest = receiver_manifests.get(&dst_hash).unwrap();
            let capacity = receiver_manifest.memory.ghost_capacity as u32;
            let ghosts_path = root_dir.join(format!("{}_{}.ghosts", conn.from, conn.to));

            if ghosts_path.exists() {
                let (src_somas, dst_ghosts) = load_ghosts(&ghosts_path);
                println!("[Ghosts] Successfully loaded {} links from {:?}", src_somas.len(), ghosts_path);
                let s2a = s2a_maps.get(&src_hash).context("S2A map missing for source zone")?;

                let mut src_axons = Vec::with_capacity(src_somas.len());
                for &soma in &src_somas {
                    let mut axon_id = genesis_core::constants::AXON_SENTINEL;
                    if soma != genesis_core::ipc::EMPTY_PIXEL {
                        axon_id = s2a[soma as usize];
                    }
                    src_axons.push(axon_id);
                }

                let src_ptr = *axon_head_ptrs.get(&src_hash).unwrap();
                let dst_ptr = *axon_head_ptrs.get(&dst_hash).unwrap();
                let channel = unsafe { IntraGpuChannel::from_slices(src_hash, dst_hash, &src_axons, &dst_ghosts, capacity) };
                intra_gpu.push((src_ptr, dst_ptr, channel));
                println!("[Boot] Built IntraGpuChannel: {} -> {} ({} links, capacity: {})", conn.from, conn.to, src_axons.len(), capacity);
            } else {
                // В режиме Ant-v4 это критично!
                panic!("CRITICAL TOPOLOGY ERROR: Incoming ghost file not found: {:?}", ghosts_path);
            }
        }
        Ok((intra_gpu, inter_node, expected_peers))
    }

    async fn setup_networking(
        first_manifest: &ZoneManifest,
        io_contexts: Vec<(u32, crate::network::io_server::ZoneIoContext)>,
        routing_table: Arc<RoutingTable>,
        shared_acks_queue: Arc<crossbeam::queue::SegQueue<genesis_core::ipc::AxonHandoverAck>>,
        telemetry: Arc<crate::tui::state::LockFreeTelemetry>,
        cluster_secret: u64, // [DOD FIX]
    ) -> Result<(Arc<ExternalIoServer>, GeometryServer, Arc<crate::network::telemetry::TelemetrySwapchain>, Arc<crate::network::egress::EgressPool>, Arc<crate::network::inter_node::InterNodeRouter>)> {
        let local_port = first_manifest.network.fast_path_udp_local;
        let udp_in = first_manifest.network.external_udp_in;

        let io_socket = tokio::net::UdpSocket::bind(&format!("0.0.0.0:{}", udp_in)).await
            .with_context(|| format!("Failed to bind UDP external_udp_in (port {}). Port in use? Kill any running genesis-node: Get-Process genesis-node -EA 0 | Stop-Process -Force", udp_in))?;
        let io_server = Arc::new(ExternalIoServer::new(
            Arc::new(AtomicBool::new(false)),
            io_contexts,
            routing_table.clone(),
            Arc::new(io_socket),
            telemetry.clone(),
            cluster_secret,
        )?);

        let geo_port = local_port + 1;
        let geo_addr = format!("0.0.0.0:{}", geo_port).parse()?;
        let geometry_server = GeometryServer::bind(geo_addr, shared_acks_queue).await
            .with_context(|| format!("Failed to bind Geometry Server (TCP port {}). Port in use? Kill any running genesis-node: Get-Process genesis-node -EA 0 | Stop-Process -Force", geo_port))?;
        let telemetry_port = local_port + 2;
        let telemetry_swapchain = TelemetryServer::start(telemetry_port).await;

        let egress_socket = Arc::new(tokio::net::UdpSocket::bind("0.0.0.0:0").await?);
        let inter_node_router = Arc::new(crate::network::inter_node::InterNodeRouter::new(egress_socket, routing_table));
        let egress_pool = Arc::new(crate::network::egress::EgressPool::new(1024));

        Ok((io_server, geometry_server, telemetry_swapchain, egress_pool, inter_node_router))
    }
}
