use anyhow::{Context, Result};
use std::path::Path;
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
use crate::network::replication::ReplicationServer;
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
pub fn boot_shard_from_disk(baked_dir: &Path) -> Result<(ShardEngine, Vec<u32>)> {
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

    // 1. Zero-Parsing загрузка сырых байт в RAM
    let state_blob = std::fs::read(&state_path)
        .with_context(|| format!("FATAL: Missing .state file at {:?}", state_path))?;
    let axons_blob = std::fs::read(&axons_path)
        .with_context(|| format!("FATAL: Missing .axons file at {:?}", axons_path))?;

    // 2. Деривация размеров за O(1) (910 байт/нейрон)
    let padded_n = (state_blob.len() / 910) as u32;
    let total_axons = (axons_blob.len() / 32) as u32;

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
    let soma_to_axon: Vec<u32> = bytemuck::cast_slice(s2a_bytes).to_vec();

    // 3. Выделение VRAM и DMA-заливка
    let vram = VramState::allocate(padded_n, total_axons);
    vram.upload_state(&state_blob);
    vram.upload_axon_heads(&axons_blob);

    Ok((ShardEngine::new(vram), soma_to_axon))
}

impl Bootloader {
    /// Full node bootstrap sequence. Standard "Genesis Sequence" pipeline.
    pub async fn boot_node(manifest_path: &Path, reporter: Arc<crate::simple_reporter::SimpleReporter>) -> Result<BootResult> {
        let root_dir = manifest_path.parent().unwrap_or(Path::new("."));
        
        // 1. Data/Config Phase: Load brain and simulation configs
        let (zone_manifest, sim_config) = Self::parse_manifests(manifest_path, root_dir)?;
        let night_interval = sim_config.simulation.night_interval_ticks as u64;

        // 2. Hardware & VRAM Phase: Allocate weights/targets and flash physics laws
        let (mut shards, s2a_maps, axon_head_ptrs, io_contexts, all_geo_data, output_routes) = 
            Self::load_shards_into_vram(&zone_manifest, root_dir)?;

        let first_manifest = zone_manifest.clone();
        unsafe { Self::flash_hardware_physics(&first_manifest)? };

        // 3. Topology Interconnect: Build local and remote routing channels
        let (intra_gpu_channels, inter_node_channels, expected_peer_hashes) = 
            Self::build_routing_channels(&zone_manifest, root_dir, &s2a_maps, &axon_head_ptrs)?;

        // [DOD FIX] Прошиваем таблицу маршрутизации для межзонального Egress
        let mut initial_routes = HashMap::new();
        for conn in &first_manifest.connections {
            let src_hash = genesis_core::hash::fnv1a_32(conn.from.as_bytes());
            let dst_hash = genesis_core::hash::fnv1a_32(conn.to.as_bytes());

            if src_hash == first_manifest.zone_hash {
                // Мы Отправитель. Запоминаем куда слать данные.
                if let Some(peer_addr) = first_manifest.network.fast_path_peers.first() {
                    let addr: std::net::SocketAddr = peer_addr.parse().expect("FATAL: Invalid peer IP");
                    initial_routes.insert(dst_hash, addr);
                    println!("[Boot] Route (Egress): 0x{:08X} -> {}", dst_hash, addr);
                }
            } else if dst_hash == first_manifest.zone_hash {
                // Мы Получатель. Запоминаем куда слать ACK.
                if let Some(peer_addr) = first_manifest.network.fast_path_peers.first() {
                    let addr: std::net::SocketAddr = peer_addr.parse().expect("FATAL: Invalid peer IP");
                    initial_routes.insert(src_hash, addr);
                    println!("[Boot] Route (ACK): 0x{:08X} -> {}", src_hash, addr);
                }
            }
        }
        let routing_table = Arc::new(RoutingTable::new(initial_routes));

        // 4. Network Setup: IO, Geometry, and Telemetry servers
        let queues = Arc::new(crate::network::slow_path::SlowPathQueues::new());
        // Link first shard to these queues (MVP: one node = one set of queues for now)
        if let Some(shard) = shards.get_mut(0) {
            shard.incoming_grow = queues.incoming_grow.clone();
            shard.incoming_prune = queues.incoming_prune.clone();
        }

        let (io_server, geometry_server, telemetry_swapchain, egress_pool, inter_node_router) = 
            Self::setup_networking(&first_manifest, io_contexts, routing_table.clone(), queues).await?;

        let mut shard_padded_ns = HashMap::new();
        for shard in &shards {
            shard_padded_ns.insert(shard.hash, shard.engine.vram.padded_n as usize);
        }
 
        let bsp_barrier = Arc::new(BspBarrier::new(sim_config.simulation.sync_batch_ticks as usize, expected_peer_hashes));

        let mut replication_targets = HashMap::new();
        for shard in &shards {
            let mut targets = Vec::new();
            if let Some(ref addr) = shard.config.neighbors.x_plus { targets.push(addr.clone()); }
            if let Some(ref addr) = shard.config.neighbors.x_minus { targets.push(addr.clone()); }
            if let Some(ref addr) = shard.config.neighbors.y_plus { targets.push(addr.clone()); }
            if let Some(ref addr) = shard.config.neighbors.y_minus { targets.push(addr.clone()); }
            if let Some(ref addr) = shard.config.neighbors.z_plus { targets.push(addr.clone()); }
            if let Some(ref addr) = shard.config.neighbors.z_minus { targets.push(addr.clone()); }
            
            // Convert Neighbor IP:FastPathPort to IP:ReplicationPort (FastPath + 3)
            let mut repl_targets = Vec::new();
            for addr_str in targets {
                if let Ok(mut addr) = addr_str.parse::<std::net::SocketAddr>() {
                    addr.set_port(addr.port() + 3);
                    repl_targets.push(addr.to_string());
                }
            }
            replication_targets.insert(shard.hash, repl_targets);
        }
 
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
            night_interval,
            reporter,
            tokio::runtime::Handle::current(),
            replication_targets,
            shard_padded_ns,
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
        let manifest_toml = std::fs::read_to_string(manifest_path)
            .with_context(|| format!("Failed to read manifest: {:?}", manifest_path))?;
        
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

    /// [DOD] Строгий барьер: прошивка законов физики в Constant Memory до запуска любых вычислительных ядер
    unsafe fn flash_hardware_physics(first_manifest: &ZoneManifest) -> Result<()> {
        let mut gpu_variants = [genesis_core::config::manifest::GpuVariantParameters::default(); 16];
        for v in &first_manifest.variants {
            if (v.id as usize) < 16 {
                gpu_variants[v.id as usize] = v.clone().into_gpu();
            }
        }
        let err = genesis_compute::ffi::cu_upload_constant_memory(
            gpu_variants.as_ptr() as *const genesis_compute::ffi::VariantParameters
        );
        if err != 0 {
            anyhow::bail!("FATAL: cu_upload_constant_memory failed with {}", err);
        }
        println!("[Boot] Hardware physics parameters flashed to Constant Memory.");
        Ok(())
    }

    fn load_shards_into_vram(zone_manifest: &ZoneManifest, root_dir: &Path) 
        -> Result<(Vec<BootShard>, HashMap<u32, Vec<u32>>, HashMap<u32, *mut genesis_core::layout::BurstHeads8>, Vec<(u32, crate::network::io_server::ZoneIoContext)>, Vec<u32>, HashMap<u32, Vec<(String, u32)>>)> 
    {
        let mut engines = Vec::new();
        let mut io_contexts = Vec::new();
        let mut all_geo_data = Vec::new();
        let mut output_routes: HashMap<u32, Vec<(String, u32)>> = HashMap::new();
        let mut axon_head_ptrs = HashMap::new();
        let mut s2a_maps = HashMap::new();

        // Since we are booting from a ZoneManifest, it represents ONE local zone.
        let baked_dir = root_dir.to_path_buf();
        let zone_hash = zone_manifest.zone_hash;

        println!("[Boot] Loading Local Zone at {:?}", baked_dir);
        let (engine, s2a) = boot_shard_from_disk(&baked_dir)?;

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
                        // For CartPole E2E, we route all motor outputs to the Python client on port 8092
                        output_routes.entry(zone_hash).or_insert_with(Vec::new)
                            .push(("127.0.0.1:8092".to_string(), hash));
                        println!("[Boot] Registered Output Route: {} (0x{:08X}) -> 127.0.0.1:8092", output.name, hash);
                    }
                }
            }
        }

        let gxi_path = baked_dir.join("shard.gxi");
        let (num_virtual_axons, virtual_offset) = if expected_inputs {
            if !gxi_path.exists() {
                anyhow::bail!("FATAL: Zone expects inputs but {:?} is missing!", gxi_path);
            }
            let gxi = GxiFile::load(&gxi_path);
            // [DOD FIX] Берем абсолютный смещенный индекс первого виртуального аксона
            let offset = gxi.axon_ids.first().copied().unwrap_or(0);
            (gxi.total_pixels, offset)
        } else {
            (0, 0)
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
        let geo_data: Vec<u32> = bytemuck::cast_slice(&pos_blob).to_vec();
        all_geo_data.extend(geo_data);

        let sync_batch_ticks = 100u32;
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
            v_seg,
            incoming_grow: incoming_grow.clone(),
            incoming_prune: Arc::new(SegQueue::new()), // Each shard gets its own prune queue
        });
        io_contexts.push((zone_hash, io_ctx));

        Ok((engines, s2a_maps, axon_head_ptrs, io_contexts, all_geo_data, output_routes))
    }

    fn build_routing_channels(
        zone_manifest: &ZoneManifest,
        root_dir: &Path,
        s2a_maps: &HashMap<u32, Vec<u32>>,
        axon_head_ptrs: &HashMap<u32, *mut genesis_core::layout::BurstHeads8>
    ) -> Result<(
        Vec<(*mut genesis_core::layout::BurstHeads8, *mut genesis_core::layout::BurstHeads8, IntraGpuChannel)>, 
        Vec<(*mut genesis_core::layout::BurstHeads8, crate::network::inter_node::InterNodeChannel)>, 
        Vec<u32> // expected_peer_hashes
    )> {
        let mut intra_gpu = Vec::new();
        let mut inter_node = Vec::new();
        let mut expected_peers = std::collections::HashSet::new();

        for conn in &zone_manifest.connections {
            let src_hash = genesis_core::hash::fnv1a_32(conn.from.as_bytes());
            let dst_hash = genesis_core::hash::fnv1a_32(conn.to.as_bytes());

            let is_src_local = axon_head_ptrs.contains_key(&src_hash);
            let is_dst_local = axon_head_ptrs.contains_key(&dst_hash);

            if !is_src_local && is_dst_local {
                println!("[Boot] Peer detected (Ingress): expecting fast-path data from remote zone 0x{:08X}", src_hash);
                expected_peers.insert(src_hash);
            }
            if is_src_local && !is_dst_local {
                println!("[Boot] Peer detected (Egress): expecting fast-path ACK from remote zone 0x{:08X}", dst_hash);
                expected_peers.insert(dst_hash);
            }

            if !is_src_local { continue; } // Outbound routing from remote source doesn't concern us

            let src_baked_dir = root_dir.to_path_buf();
            let ghosts_path = src_baked_dir.join(format!("{}_{}.ghosts", conn.from, conn.to));

            if ghosts_path.exists() {
                let (src_somas, dst_ghosts) = load_ghosts(&ghosts_path);
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

                if is_dst_local {
                    // Local Zero-Copy
                    let dst_ptr = *axon_head_ptrs.get(&dst_hash).unwrap();
                    let channel = unsafe { IntraGpuChannel::from_slices(&src_axons, &dst_ghosts) };
                    intra_gpu.push((src_ptr, dst_ptr, channel));
                    println!("[Boot] Built IntraGpuChannel: {} -> {} ({} links)", conn.from, conn.to, src_axons.len());
                } else {
                    // Network Fast Path Egress
                    let channel = unsafe { crate::network::inter_node::InterNodeChannel::new(src_hash, dst_hash, &src_axons, &dst_ghosts) };
                    inter_node.push((src_ptr, channel));
                    println!("[Boot] Built InterNodeChannel: {} -> Remote(0x{:08X}) ({} links)", conn.from, dst_hash, src_axons.len());
                }
            }
        }
        Ok((intra_gpu, inter_node, expected_peers.into_iter().collect()))
    }

    async fn setup_networking(
        first_manifest: &ZoneManifest,
        io_contexts: Vec<(u32, crate::network::io_server::ZoneIoContext)>,
        routing_table: Arc<RoutingTable>,
        queues: Arc<crate::network::slow_path::SlowPathQueues>,
    ) -> Result<(Arc<ExternalIoServer>, GeometryServer, Arc<crate::network::telemetry::TelemetrySwapchain>, Arc<crate::network::egress::EgressPool>, Arc<crate::network::inter_node::InterNodeRouter>)> {
        let local_port = first_manifest.network.fast_path_udp_local;

        let io_socket = tokio::net::UdpSocket::bind(&format!("0.0.0.0:{}", first_manifest.network.external_udp_in)).await?;
        let io_server = Arc::new(ExternalIoServer::new(
            Arc::new(AtomicBool::new(false)),
            io_contexts,
            routing_table.clone(),
            Arc::new(io_socket)
        )?);

        let geo_addr = format!("0.0.0.0:{}", local_port + 1).parse()?;
        let geometry_server = GeometryServer::bind(geo_addr, queues).await?;
        let telemetry_swapchain = TelemetryServer::start(local_port + 2).await;

        let egress_socket = Arc::new(tokio::net::UdpSocket::bind("0.0.0.0:0").await?);
        let inter_node_router = Arc::new(crate::network::inter_node::InterNodeRouter::new(egress_socket, routing_table));
        let egress_pool = Arc::new(crate::network::egress::EgressPool::new(1024));

        let repl_server = ReplicationServer::new(&format!("0.0.0.0:{}", local_port + 3));
        tokio::spawn(async move {
            if let Err(e) = repl_server.run().await {
                eprintln!("[Boot] ReplicationServer failed: {}", e);
            }
        });
 
        Ok((io_server, geometry_server, telemetry_swapchain, egress_pool, inter_node_router))
    }
}
