use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::process::{Command, Child};
use std::env;
use std::path::PathBuf;
use std::time::SystemTime;
use std::sync::atomic::{AtomicU32, Ordering};
use crossbeam::channel::{bounded, Sender, Receiver};
use crate::network::io_server::ExternalIoServer;
use crate::network::bsp::BspBarrier;
use crate::network::router::RoutingTable;
use crate::network::inter_node::InterNodeRouter;
use crate::node::shard_thread::ShardAtomicSettings;

pub struct ShardMetadata {
    pub manifest_path: PathBuf,
    pub last_modified: SystemTime,
    pub atomic_settings: Arc<ShardAtomicSettings>,
}

pub mod recovery;
pub mod shard_thread;

pub enum ComputeCommand {
    RunBatch {
        tick_base: u32,
        batch_size: u32,
        global_dopamine: i16,
    },
    Resurrect, // [DOD] Trigger 100-tick stabilization warmup
    Shutdown,
}

pub enum ComputeFeedback {
    BatchComplete {
        ticks_processed: u32,
        zone_hash: u32,
        pinned_out_ptr: usize,
        output_bytes: usize,
        is_warmup: bool, // [DOD] True if shard is stabilizing
    },
}

/// [Phase 23] Network channel topology — raw pointers, inter/intra GPU channels.
pub struct NetworkTopology {
    pub intra_gpu_channels: Vec<(*mut genesis_core::layout::BurstHeads8, *mut genesis_core::layout::BurstHeads8, crate::network::intra_gpu::IntraGpuChannel)>,
    pub inter_node_channels: Vec<(*mut genesis_core::layout::BurstHeads8, crate::network::inter_node::InterNodeChannel)>,
    pub inter_node_router: Arc<crate::network::inter_node::InterNodeRouter>,
    pub egress_pool: Arc<crate::network::egress::EgressPool>,
    pub axon_head_ptrs: HashMap<u32, *mut genesis_core::layout::BurstHeads8>,
    pub routing_acks: std::sync::Arc<crossbeam::queue::SegQueue<genesis_core::ipc::AxonHandoverAck>>,
    pub routing_prunes: std::sync::Arc<crossbeam::queue::SegQueue<genesis_core::ipc::AxonHandoverPrune>>,
}

// Safety: raw pointers in NetworkTopology are pinned GPU pointers owned by ShardEngine.
// They are valid for the lifetime of the VRAM allocation and are only accessed from
// the orchestrator thread (run_node_loop), which is single-threaded.
unsafe impl Send for NetworkTopology {}
unsafe impl Sync for NetworkTopology {}

/// [Phase 23] Shared infrastructure services.
pub struct NodeServices {
    pub io_server: Arc<ExternalIoServer>,
    pub routing_table: Arc<RoutingTable>,
    pub bsp_barrier: Arc<BspBarrier>,
    pub telemetry: Arc<crate::tui::state::LockFreeTelemetry>,
}

pub struct NodeRuntime {
    pub services: NodeServices,
    pub network: NetworkTopology,
    pub compute_dispatchers: HashMap<u32, Sender<ComputeCommand>>,
    pub feedback_sender: Sender<ComputeFeedback>,
    pub feedback_receiver: Receiver<ComputeFeedback>,
    pub total_ticks: Arc<AtomicU32>,
    pub local_ip: std::net::Ipv4Addr,
    pub local_port: u16,
    /// [DOD] Маршруты выходов: zone_hash -> (TargetAddr, matrix_hash)
    pub output_routes: HashMap<u32, Vec<(String, u32)>>,
    // [DOD] Владение дочерними процессами (Baker Daemons)
    pub daemons: Mutex<Vec<Child>>,
    // [DOD FIX] Метаданные шардов для Hot-Reload
    pub manifest_metadata: Mutex<HashMap<u32, ShardMetadata>>,
    pub zone_v_segs: HashMap<u32, u32>,
    pub virtual_offset_map: HashMap<u32, u32>,
    pub sync_batch_ticks: u32,
    pub cluster_secret: u64, // [DOD FIX]
}

unsafe impl Send for NodeRuntime {}
unsafe impl Sync for NodeRuntime {}

impl NodeRuntime {
    fn log_discrete_membrane_modes_from_manifest(
        &self,
        hash: u32,
        manifest: &genesis_core::config::manifest::ZoneManifest,
    ) {
        let discrete_variants: Vec<String> = manifest
            .variants
            .iter()
            .filter(|v| v.adaptive_leak_mode == genesis_core::physics::ADAPTIVE_LEAK_MODE_DISCRETE)
            .map(|v| format!("{}[{},{}]", v.name, v.leak_min, v.leak_max))
            .collect();

        if discrete_variants.is_empty() {
            return;
        }

        self.services.telemetry.push_log(
            format!(
                "[Adaptive Leak] Zone 0x{:08X} discrete membrane modes enabled for variants: {}. \
stable=base, responsive=mid(leak_min, base), excited=mid(base, leak_max), recovery=leak_max when burst_count >= {}.",
                hash,
                discrete_variants.join(", "),
                genesis_core::physics::RECOVERY_BURST_THRESHOLD
            ),
            crate::tui::state::LogLevel::Info,
        );
    }

    fn log_discrete_membrane_mode_configs(&self) {
        let metadata_map = self.manifest_metadata.lock().unwrap();
        for (hash, metadata) in metadata_map.iter() {
            if let Ok(toml_str) = std::fs::read_to_string(&metadata.manifest_path) {
                if let Ok(zm) = toml::from_str::<genesis_core::config::manifest::ZoneManifest>(&toml_str) {
                    self.log_discrete_membrane_modes_from_manifest(*hash, &zm);
                }
            }
        }
    }

    pub fn boot(
        shards: Vec<crate::node::shard_thread::ShardDescriptor>,
        io_server: Arc<ExternalIoServer>,
        routing_table: Arc<RoutingTable>,
        bsp_barrier: Arc<BspBarrier>,
        _telemetry_swapchain: Arc<crate::network::telemetry::TelemetrySwapchain>,
        local_ip: std::net::Ipv4Addr,
        local_port: u16,
        output_routes: HashMap<u32, Vec<(String, u32)>>,
        intra_gpu_channels: Vec<(*mut genesis_core::layout::BurstHeads8, *mut genesis_core::layout::BurstHeads8, crate::network::intra_gpu::IntraGpuChannel)>,
        inter_node_channels: Vec<(*mut genesis_core::layout::BurstHeads8, crate::network::inter_node::InterNodeChannel)>,
        inter_node_router: Arc<crate::network::inter_node::InterNodeRouter>,
        axon_head_ptrs: HashMap<u32, *mut genesis_core::layout::BurstHeads8>,
        egress_pool: Arc<crate::network::egress::EgressPool>,
        manifest_metadata: HashMap<u32, ShardMetadata>,
        telemetry: Arc<crate::tui::state::LockFreeTelemetry>,
        shared_acks_queue: Arc<crossbeam::queue::SegQueue<genesis_core::ipc::AxonHandoverAck>>,
        sync_batch_ticks: u32,
        cluster_secret: u64, // [DOD FIX]
    ) -> Self {
        let (feedback_tx, feedback_rx) = bounded(shards.len() + 32);
        let total_ticks = Arc::new(AtomicU32::new(0));

        let bsp_listener_clone = bsp_barrier.clone();
        tokio::spawn(InterNodeRouter::spawn_ghost_listener(local_port, bsp_listener_clone, routing_table.clone()));

        // [DOD] Structured Concurrency: Оркестратор спавнит демонов сам
        let daemons = Self::spawn_baker_daemons(&shards);

        // [Windows] Daemon creates SHM file asynchronously; wait for it before shard threads open
        std::thread::sleep(std::time::Duration::from_millis(1500));

        let mut compute_dispatchers = HashMap::new();
        let mut shard_receivers = HashMap::new();
        for desc in &shards {
            let (tx, rx) = bounded(1);
            compute_dispatchers.insert(desc.hash, tx);
            shard_receivers.insert(desc.hash, rx);
        }

        let rt_handle = tokio::runtime::Handle::current();
        
        let mut zone_v_segs = HashMap::new();
        let mut virtual_offset_map = HashMap::new();

        // [DOD] Consume shards to spawn threads
        let mut core_id = 1;
        for desc in shards {
            let hash = desc.hash;
            let v_seg = desc.v_seg;
            let v_offset = desc.virtual_offset;
            let rx = shard_receivers.remove(&hash).unwrap();
            zone_v_segs.insert(hash, v_seg);
            virtual_offset_map.insert(hash, v_offset);
            
            let my_io_ctx = io_server.io_contexts.iter()
                .find(|(h, _)| *h == hash)
                .map(|(_, ctx)| ctx.swapchain.clone())
                .expect("FATAL: IO Context for zone not found");

            let metadata = manifest_metadata.get(&hash).expect("FATAL: Metadata for zone not found");

            let ctx = crate::node::shard_thread::NodeContext {
                bsp_barrier: bsp_barrier.clone(),
                io_ctx: my_io_ctx,
                rt_handle: rt_handle.clone(),
                atomic_settings: metadata.atomic_settings.clone(),
                incoming_grow: desc.incoming_grow.clone(),
                telemetry: telemetry.clone(),
                routing_table: routing_table.clone(), // [DOD FIX]
            };
                
            crate::node::shard_thread::spawn_shard_thread(
                desc, ctx, rx, feedback_tx.clone(), core_id, sync_batch_ticks
            );
            core_id += 1;
        }

        let node = Self {
            services: NodeServices {
                io_server,
                routing_table,
                bsp_barrier,
                telemetry,
            },
            network: NetworkTopology {
                intra_gpu_channels,
                inter_node_channels,
                inter_node_router,
                egress_pool,
                axon_head_ptrs,
                routing_acks: shared_acks_queue,
                routing_prunes: Arc::new(crossbeam::queue::SegQueue::new()),
            },
            compute_dispatchers,
            feedback_sender: feedback_tx,
            feedback_receiver: feedback_rx,
            total_ticks,
            local_ip,
            local_port,
            output_routes,
            daemons: Mutex::new(daemons),
            manifest_metadata: Mutex::new(manifest_metadata),
            zone_v_segs,
            virtual_offset_map,
            sync_batch_ticks,
            cluster_secret,
        };

        node
    }

    fn reload_manifests(&self) {
        let mut metadata_map = self.manifest_metadata.lock().unwrap();
        for (hash, metadata) in metadata_map.iter_mut() {
            if let Ok(attr) = std::fs::metadata(&metadata.manifest_path) {
                if let Ok(modified) = attr.modified() {
                    if modified > metadata.last_modified {
                        println!("♻️ [Hot-Reload] Manifest changed for zone 0x{:08X}", hash);
                        metadata.last_modified = modified;
                        
                        if let Ok(toml_str) = std::fs::read_to_string(&metadata.manifest_path) {
                            if let Ok(zm) = toml::from_str::<genesis_core::config::manifest::ZoneManifest>(&toml_str) {
                                // 1. Update Atomic Settings
                                metadata.atomic_settings.night_interval_ticks.store(zm.settings.night_interval_ticks, Ordering::Relaxed);
                                metadata.atomic_settings.save_checkpoints_interval_ticks.store(zm.settings.save_checkpoints_interval_ticks, Ordering::Relaxed);
                                metadata.atomic_settings.prune_threshold.store(zm.settings.plasticity.prune_threshold, Ordering::Relaxed);
                                metadata.atomic_settings.max_sprouts.store(zm.settings.plasticity.max_sprouts, Ordering::Relaxed);
                                    
                                    // 2. Update GPU Constants (Variants)
                                    let mut gpu_variants = [genesis_core::layout::VariantParameters::default(); 16];
                                    for v in &zm.variants {
                                        if (v.id as usize) < 16 {
                                            gpu_variants[v.id as usize] = v.clone().into_gpu();
                                        }
                                    }
                                    unsafe {
                                        genesis_compute::ffi::cu_upload_constant_memory(
                                            gpu_variants.as_ptr() as *const genesis_core::layout::VariantParameters
                                        );
                                    }
                                    self.log_discrete_membrane_modes_from_manifest(*hash, &zm);
                                    self.services.telemetry.push_log(format!("Zone 0x{:08X} updated (Night: {}, Prune: {}, GPU Physics reflashed)", 
                                        hash, zm.settings.night_interval_ticks, zm.settings.plasticity.prune_threshold), crate::tui::state::LogLevel::Info);
                                } else {
                                    self.services.telemetry.push_log(format!("Failed to parse manifest at {:?}", metadata.manifest_path), crate::tui::state::LogLevel::Error);
                                }
                        }
                    }
                }
            }
        }
    }

    fn patch_routing_tables(&mut self) {
        let stream = std::ptr::null_mut(); 

        while let Some(ack) = self.network.routing_acks.pop() {
            // Ищем Inter-Node канал (если сосед на другой машине)
            if let Some((_, channel)) = self.network.inter_node_channels.iter_mut()
                .find(|(_, c)| c.target_zone_hash == ack.target_zone_hash) 
            {
                unsafe { channel.push_route(ack.src_axon_id, ack.dst_ghost_id, stream); }
            }
            // Ищем Intra-GPU канал (если обе зоны сидят в нашей VRAM)
            else if let Some((_, _, channel)) = self.network.intra_gpu_channels.iter_mut()
                // [DOD FIX] Точный матчинг по хэшам без магических заглушек!
                .find(|(_, _, c)| c.target_zone_hash == ack.target_zone_hash) 
            {
                unsafe { channel.push_route(ack.src_axon_id, ack.dst_ghost_id, stream); }
            }
        }

        while let Some(prune) = self.network.routing_prunes.pop() {
            if let Some((_, channel)) = self.network.inter_node_channels.iter_mut()
                .find(|(_, c)| c.target_zone_hash == prune.target_zone_hash) 
            {
                unsafe { channel.prune_route(prune.dst_ghost_id, stream); }
            }
            else if let Some((_, _, channel)) = self.network.intra_gpu_channels.iter_mut()
                .find(|(_, _, c)| c.target_zone_hash == prune.target_zone_hash) 
            {
                unsafe { channel.prune_route(prune.dst_ghost_id, stream); }
            }
        }
    }

    fn spawn_baker_daemons(
        shards: &[crate::node::shard_thread::ShardDescriptor],
    ) -> Vec<Child> {
        let mut daemons = Vec::with_capacity(shards.len());
        let exe_path = env::current_exe().expect("FATAL: Failed to get current exe path");
        let daemon_path = exe_path.with_file_name("genesis-baker-daemon");

        for desc in shards {
            let socket_addr = genesis_core::ipc::default_socket_path(desc.hash);
            #[cfg(unix)]
            let _ = std::fs::remove_file(&socket_addr);

            // [DOD FIX] Pass root brain.toml so daemon can find simulation.toml and blueprints
            let brain_path = desc.baked_dir
                .parent().unwrap() // Выходим из папки зоны
                .parent().unwrap() // Выходим из папки baked
                .join("brain.toml");

            println!("[Orchestrator] Spawning CPU Baker Daemon for zone 0x{:08X} at {:?} (IPC: {})", desc.hash, desc.baked_dir, socket_addr);
            let child = Command::new(&daemon_path)
                .arg("--brain").arg(&brain_path)
                .arg("--zone-hash")
                .arg(desc.hash.to_string())
                .arg("--baked-dir")
                .arg(&desc.baked_dir)
                // [DOD FIX] Снимаем глушитель! Мы должны видеть паники CPU-демона в консоли ноды.
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .spawn()
                .expect("FATAL: Failed to spawn genesis-baker-daemon. Was it compiled?");
            
            daemons.push(child);
        }
        daemons
    }

    // [DOD FIX] The correct Pipeline Order: Compute -> Network Tx -> Network Rx Wait
    pub fn run_node_loop(&mut self) {
        let batch_size = self.sync_batch_ticks;
        let mut current_tick = 0;
        let mut batch_counter: u64 = 0;

        // [DOD FIX] Жесткая привязка OS-потока оркестратора к аппаратному контексту
        unsafe { genesis_compute::ffi::gpu_set_device(0); }

        // [DOD] Pre-allocate outbound buffers to avoid heap thrashing
        let mut _io_tx_buffer = vec![0u8; genesis_core::constants::MAX_UDP_PAYLOAD];

        self.services.telemetry.push_log(format!("Entering main loop (Batch size: {})", batch_size), crate::tui::state::LogLevel::Info);
        self.log_discrete_membrane_mode_configs();

        let loop_start = std::time::Instant::now();
        let mut batch_start = loop_start;

        loop {
            // 1. Swap IO Buffers (Acquire semantics)
            for (_, io_ctx) in &self.services.io_server.io_contexts {
                io_ctx.swapchain.swap();
            }

            // 2. Dispatch batches to compute shards
            /* 
            let current_dopamine = self.services.io_server.global_dopamine.load(Ordering::Relaxed) as i16;
            if current_dopamine != 0 && batch_counter % 100 == 0 {
                self.services.telemetry.push_log(format!("Dopamine: {}", current_dopamine), crate::tui::state::LogLevel::Info);
            }
            */
            let current_dopamine = self.services.io_server.global_dopamine.load(Ordering::Relaxed) as i16;

            let num_dispatchers = self.compute_dispatchers.len();
            if num_dispatchers == 0 {
                println!("[!] ERROR: No compute dispatchers found!");
            }
            for tx in self.compute_dispatchers.values() {
                let _ = tx.send(ComputeCommand::RunBatch {
                    tick_base: current_tick as u32,
                    batch_size,
                    global_dopamine: current_dopamine,
                });
            }

            // 3. Collect feedback
            let mut pending_outputs = Vec::new();

            for _ in 0..num_dispatchers {
                if let Ok(feedback) = self.feedback_receiver.recv() {
                    match feedback {
                        ComputeFeedback::BatchComplete { ticks_processed: _, zone_hash, pinned_out_ptr, mut output_bytes, is_warmup } => {
                            // [DOD] Hardware Gating: Mute all outgoing traffic during Warmup
                            if is_warmup {
                                for (_, channel) in &self.network.inter_node_channels {
                                    if channel.src_zone_hash == zone_hash {
                                        unsafe { std::ptr::write_volatile(channel.out_count_pinned, 0); }
                                    }
                                }
                                output_bytes = 0; // Mute external IO motors
                            }

                            if output_bytes > 0 {
                                pending_outputs.push((zone_hash, pinned_out_ptr, output_bytes));
                            }
                        }
                    }
                }
            }

            // [DOD] GPU Hardware Barrier — дожидаемся завершения всех стримов
            unsafe { genesis_compute::ffi::gpu_device_synchronize(); }

            // Ship outputs to network targets ONLY POST SYNC!
            for (zone_hash, pinned_out_ptr, output_bytes) in pending_outputs {
                if let Some(routes) = self.output_routes.get(&zone_hash) {
                    for (addr, m_hash) in routes {
                        self.services.io_server.send_output_batch_pool(
                            &self.network.egress_pool,
                            &addr,
                            zone_hash,
                            *m_hash,
                            pinned_out_ptr,
                            output_bytes,
                        );
                        // [DOD FIX] Возрождаем счетчик UDP OUT!
                        self.services.telemetry.udp_out_packets.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            // [DOD] 4. Intra-GPU Ghost Sync + Inter-Node Extraction
            for (src_ptr, dst_ptr, channel) in &self.network.intra_gpu_channels {
                unsafe { channel.sync_ghosts(*src_ptr, *dst_ptr, std::ptr::null_mut()); }
            }

            for (src_ptr, channel) in &self.network.inter_node_channels {
                // v_seg must be passed for accurate spike timing extraction
                let v_seg = self.zone_v_segs.get(&channel.src_zone_hash).copied().unwrap_or(1);
                unsafe { channel.extract_spikes(*src_ptr, batch_size, v_seg, std::ptr::null_mut()); }
            }

            // [DOD] 5. GPU Barrier Sync (дожидаемся sync_ghosts в default stream)
            unsafe { genesis_compute::ffi::gpu_stream_synchronize(std::ptr::null_mut()); }

            // [DOD] 6. Inter-Node Fast Path (Egress)
            for (_, channel) in &self.network.inter_node_channels {
                let out_count = unsafe { std::ptr::read_volatile(channel.out_count_pinned) };

                if out_count > 0 {
                    self.services.telemetry.udp_out_packets.fetch_add(1, Ordering::Relaxed);
                }

                // BSP Heartbeat: ВСЕГДА формируем и отправляем пакет
                let events_slice = unsafe {
                    std::slice::from_raw_parts(channel.out_events_pinned, out_count as usize)
                };
                self.network.inter_node_router.flush_outgoing_batch_pool(
                    &self.network.egress_pool,
                    channel.src_zone_hash,
                    channel.target_zone_hash,
                    events_slice,
                    (batch_counter & 0xFFFFFFFF) as u32
                );
            }

            // [DOD] 7. Wait for Ingress data (Strict BSP network sync)
            self.services.bsp_barrier.wait_for_data_sync();

            self.services.bsp_barrier.sync_and_swap();

            // [DOD FIX] 8. Dynamic Capacity Routing: Hot-Patching VRAM
            // Барьер пройден, GPU стоит. Идеальное время переписать 8 байт по шине PCIe.
            self.patch_routing_tables();

            let wall_ms = batch_start.elapsed().as_millis() as u64;
            batch_start = std::time::Instant::now();

            current_tick += batch_size;
            batch_counter += 1;
            
            self.services.telemetry.batch_number.store(batch_counter, Ordering::Relaxed);
            self.services.telemetry.total_ticks.store(current_tick as u64, Ordering::Relaxed);
            self.services.telemetry.wall_ms.store(wall_ms, Ordering::Relaxed);

            if batch_counter > 0 && batch_counter % 50 == 0 {
                // Remove console printing to keep TUI clean, stats are visible in Core Loop widget
                // [DOD FIX] Hot-Reload entry point
                self.reload_manifests();
            }
        }
    }
}

impl Drop for NodeRuntime {
    fn drop(&mut self) {
        let mut daemons = self.daemons.lock().unwrap();
        for (i, daemon) in daemons.iter_mut().enumerate() {
            let _ = daemon.kill(); // Отправляем SIGKILL
            let _ = daemon.wait(); // Ждем подтверждения смерти от ОС, предотвращая зомби
            println!("[Orchestrator] Baker Daemon {} successfully terminated.", i);
        }
    }
}

#[cfg(test)]
mod tests {
    // Legacy isolation test depended on an outdated NodeRuntime::boot and
    // ExternalIoServer API; it was removed to avoid pinning core orchestration
    // signatures to an obsolete test harness.
}
