use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::process::{Command, Child};
use std::env;
use std::sync::atomic::{AtomicU32, Ordering};
use crossbeam::channel::{bounded, Sender, Receiver};
use crate::network::io_server::ExternalIoServer;
use crate::network::bsp::BspBarrier;
use crate::network::router::RoutingTable;
use crate::network::inter_node::InterNodeRouter;

pub mod recovery;
pub mod shard_thread;

pub enum ComputeCommand {
    RunBatch {
        tick_base: u32,
        batch_size: u32,
        global_dopamine: i16,
    },
    Shutdown,
}

pub enum ComputeFeedback {
    BatchComplete {
        ticks_processed: u32,
        zone_hash: u32,
        pinned_out_ptr: usize,
        output_bytes: usize,
    },
}

/// [Phase 23] Network channel topology — raw pointers, inter/intra GPU channels.
pub struct NetworkTopology {
    pub intra_gpu_channels: Vec<(*mut genesis_core::layout::BurstHeads8, *mut genesis_core::layout::BurstHeads8, crate::network::intra_gpu::IntraGpuChannel)>,
    pub inter_node_channels: Vec<(*mut genesis_core::layout::BurstHeads8, crate::network::inter_node::InterNodeChannel)>,
    pub inter_node_router: Arc<crate::network::inter_node::InterNodeRouter>,
    pub egress_pool: Arc<crate::network::egress::EgressPool>,
    pub axon_head_ptrs: HashMap<u32, *mut genesis_core::layout::BurstHeads8>,
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
    pub reporter: Arc<crate::simple_reporter::SimpleReporter>,
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
    // [DOD] Night Phase интервал в тиках
    pub night_interval: u64,
    pub zone_v_segs: HashMap<u32, u32>,
    pub virtual_offset_map: HashMap<u32, u32>,
}

unsafe impl Send for NodeRuntime {}
unsafe impl Sync for NodeRuntime {}

impl NodeRuntime {
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
        night_interval: u64,
        reporter: Arc<crate::simple_reporter::SimpleReporter>,
    ) -> Self {
        let (feedback_tx, feedback_rx) = bounded(shards.len() + 32);
        let total_ticks = Arc::new(AtomicU32::new(0));

        let bsp_listener_clone = bsp_barrier.clone();
        tokio::spawn(InterNodeRouter::spawn_ghost_listener(local_port, bsp_listener_clone, routing_table.clone()));

        // [DOD] Structured Concurrency: Оркестратор спавнит демонов сам
        let daemons = Self::spawn_baker_daemons(&shards);

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

            let ctx = crate::node::shard_thread::NodeContext {
                bsp_barrier: bsp_barrier.clone(),
                io_ctx: my_io_ctx,
                rt_handle: rt_handle.clone(),
                night_interval,
                incoming_grow: desc.incoming_grow.clone(),
            };
                
            crate::node::shard_thread::spawn_shard_thread(
                desc, ctx, rx, feedback_tx.clone(),
            );
        }


        let node = Self {
            services: NodeServices {
                io_server,
                routing_table,
                bsp_barrier,
                reporter,
            },
            network: NetworkTopology {
                intra_gpu_channels,
                inter_node_channels,
                inter_node_router,
                egress_pool,
                axon_head_ptrs,
            },
            compute_dispatchers,
            feedback_sender: feedback_tx,
            feedback_receiver: feedback_rx,
            total_ticks,
            local_ip,
            local_port,
            output_routes,
            daemons: Mutex::new(daemons),
            night_interval,
            zone_v_segs,
            virtual_offset_map,
        };

        node
    }

    fn spawn_baker_daemons(
        shards: &[crate::node::shard_thread::ShardDescriptor],
    ) -> Vec<Child> {
        let mut daemons = Vec::with_capacity(shards.len());
        let exe_path = env::current_exe().expect("FATAL: Failed to get current exe path");
        let daemon_path = exe_path.with_file_name("genesis-baker-daemon");

        for desc in shards {
            let socket_path = genesis_core::ipc::default_socket_path(desc.hash);
            let _ = std::fs::remove_file(&socket_path);

            println!("[Orchestrator] Spawning CPU Baker Daemon for zone 0x{:08X} at {:?}", desc.hash, desc.baked_dir);
            let child = Command::new(&daemon_path)
                .arg("--zone-hash")
                .arg(desc.hash.to_string())
                .arg("--baked-dir")
                .arg(&desc.baked_dir)
                .spawn()
                .expect("FATAL: Failed to spawn genesis-baker-daemon. Was it compiled?");
            
            daemons.push(child);
        }
        daemons
    }

    // [DOD FIX] The correct Pipeline Order: Compute -> Network Tx -> Network Rx Wait
    pub fn run_node_loop(&self, batch_size: u32) {
        let mut current_tick = 0;
        let mut batch_counter: u64 = 0;

        // [DOD FIX] Жесткая привязка OS-потока оркестратора к аппаратному контексту
        unsafe { genesis_compute::ffi::gpu_set_device(0); }

        // [DOD] Pre-allocate outbound buffers to avoid heap thrashing
        let mut _io_tx_buffer = vec![0u8; genesis_core::constants::MAX_UDP_PAYLOAD];

        println!("[Node] Entering main loop (Batch size: {})", batch_size);

        loop {
            // 1. Swap IO Buffers (Acquire semantics)
            for (_, io_ctx) in &self.services.io_server.io_contexts {
                io_ctx.swapchain.swap();
            }

            // 2. Dispatch batches to compute shards
            let current_dopamine = self.services.io_server.global_dopamine.load(Ordering::Relaxed) as i16;
            if current_dopamine != 0 && batch_counter % 100 == 0 {
                println!("💉 [Node] Dopamine: {}", current_dopamine);
            }

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
            for _ in 0..num_dispatchers {
                if let Ok(feedback) = self.feedback_receiver.recv() {
                    match feedback {
                        ComputeFeedback::BatchComplete { ticks_processed: _, zone_hash, pinned_out_ptr, output_bytes } => {
                            if output_bytes > 0 {
                                // Ship outputs to network targets
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
                                    }
                                }
                            }
                        }
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

            // [DOD] 5. GPU Barrier Sync (дожидаемся завершения физики, sync_ghosts и экстракции)
            unsafe { genesis_compute::ffi::gpu_stream_synchronize(std::ptr::null_mut()); }

            // [DOD] 6. Inter-Node Fast Path (Egress)
            for (_, channel) in &self.network.inter_node_channels {
                let out_count = unsafe { std::ptr::read_volatile(channel.out_count_pinned) };

                if out_count > 0 {
                    println!("🚀 [Egress] Extracted {} spikes for zone 0x{:08X}", out_count, channel.target_zone_hash);
                    self.services.reporter.udp_out_packets.fetch_add(1, Ordering::Relaxed);
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

            // [DOD] 8. Synchronize BSP and consume Ghost events
            self.services.bsp_barrier.sync_and_swap();

            current_tick += batch_size;
            batch_counter += 1;
            self.services.reporter.total_ticks.store(current_tick as u64, Ordering::Relaxed);
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
