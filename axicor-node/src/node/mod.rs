use crate::network::bsp::BspBarrier;
use crate::network::io_server::ExternalIoServer;
use crate::network::router::InterNodeRouter;
use crate::network::router::RoutingTable;
use crate::node::shard_thread::ShardAtomicSettings;
use crossbeam::channel::{bounded, Receiver, Sender};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use tracing::{error, info, warn};

pub struct ShardMetadata {
    pub manifest_path: PathBuf, // Still used for reporting/display
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

/// [Phase 23] Network channel topology ~ raw pointers, inter/intra GPU channels.
pub struct NetworkTopology {
    pub intra_gpu_channels: Vec<(
        *mut axicor_core::layout::BurstHeads8,
        *mut axicor_core::layout::BurstHeads8,
        crate::network::intra_gpu::IntraGpuChannel,
    )>,
    pub inter_node_channels: Vec<(
        *mut axicor_core::layout::BurstHeads8,
        crate::network::inter_node::InterNodeChannel,
    )>,
    pub inter_node_router: Arc<crate::network::router::InterNodeRouter>,
    pub egress_pool: Arc<crate::network::egress::EgressPool>,
    pub axon_head_ptrs: HashMap<u32, *mut axicor_core::layout::BurstHeads8>,
    pub routing_acks: std::sync::Arc<crossbeam::queue::SegQueue<axicor_core::ipc::AxonHandoverAck>>,
    pub routing_prunes:
        std::sync::Arc<crossbeam::queue::SegQueue<axicor_core::ipc::AxonHandoverPrune>>,
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
}

// MONOLITH: HIGH — NodeRuntime is a god-object managing lifecycle, compute dispatch, and network topology.
// REFACTOR: Decompose into LifecycleManager, ComputeOrchestrator, and NetworkRegistry.
pub struct NodeRuntime {
    pub services: NodeServices,
    pub network: NetworkTopology,
    pub compute_dispatchers: HashMap<u32, Sender<ComputeCommand>>,
    pub feedback_sender: Sender<ComputeFeedback>,
    pub feedback_receiver: Receiver<ComputeFeedback>,
    pub total_ticks: Arc<AtomicU32>,
    pub local_ip: std::net::Ipv4Addr,
    pub local_port: u16,
    /// [DOD] Output routes: zone_hash -> (TargetAddr, MatrixHash, PixelOffset, ChunkPixels)
    pub output_routes: HashMap<u32, Vec<(String, u32, usize, usize)>>,
    pub compute_thread_handles: Option<Vec<std::thread::JoinHandle<()>>>,
    // [DOD] Ownership of child processes (Baker Daemons)
    pub daemons: Mutex<Vec<Child>>,
    // [DOD FIX] Shard metadata for Hot-Reload
    pub manifest_metadata: Mutex<HashMap<u32, ShardMetadata>>,
    pub zone_v_segs: HashMap<u32, u32>,
    pub virtual_offset_map: HashMap<u32, u32>,
    pub sync_batch_ticks: u32,
    pub cluster_secret: u64, // [DOD FIX]
    // [DOD FIX] Preallocated buffer for output transposition without allocations
    pub egress_transpose_buffer: Vec<u8>,
}

unsafe impl Send for NodeRuntime {}
unsafe impl Sync for NodeRuntime {}

impl NodeRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn boot(
        shards: Vec<crate::node::shard_thread::ShardDescriptor>,
        io_server: Arc<ExternalIoServer>,
        routing_table: Arc<RoutingTable>,
        bsp_barrier: Arc<BspBarrier>,
        _telemetry_swapchain: Arc<crate::network::telemetry::TelemetrySwapchain>,
        local_ip: std::net::Ipv4Addr,
        local_port: u16,
        output_routes: HashMap<u32, Vec<(String, u32, usize, usize)>>,
        intra_gpu_channels: Vec<(
            *mut axicor_core::layout::BurstHeads8,
            *mut axicor_core::layout::BurstHeads8,
            crate::network::intra_gpu::IntraGpuChannel,
        )>,
        inter_node_channels: Vec<(
            *mut axicor_core::layout::BurstHeads8,
            crate::network::inter_node::InterNodeChannel,
        )>,
        inter_node_router: Arc<crate::network::router::InterNodeRouter>,
        axon_head_ptrs: HashMap<u32, *mut axicor_core::layout::BurstHeads8>,
        egress_pool: Arc<crate::network::egress::EgressPool>,
        manifest_metadata: HashMap<u32, ShardMetadata>,
        shared_acks_queue: Arc<crossbeam::queue::SegQueue<axicor_core::ipc::AxonHandoverAck>>,
        shared_prunes_queue: Arc<crossbeam::queue::SegQueue<axicor_core::ipc::AxonHandoverPrune>>,
        sync_batch_ticks: u32,
        cluster_secret: u64, // [DOD FIX]
    ) -> Self {
        let (feedback_tx, feedback_rx) = bounded(shards.len() + 32);
        let total_ticks = Arc::new(AtomicU32::new(0));

        let bsp_listener_clone = bsp_barrier.clone();
        tokio::spawn(InterNodeRouter::spawn_ghost_listener(
            local_port,
            bsp_listener_clone,
            routing_table.clone(),
            cluster_secret,
        ));

        // [DOD] Structured Concurrency: Orchestrator spawns daemons itself
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

        let mut thread_handles = Vec::new();

        // [DOD] Consume shards to spawn threads
        for (core_id, desc) in (1..).zip(shards) {
            let hash = desc.hash;
            let v_seg = desc.v_seg;
            let v_offset = desc.virtual_offset;
            let rx = shard_receivers.remove(&hash).unwrap();
            zone_v_segs.insert(hash, v_seg);
            virtual_offset_map.insert(hash, v_offset);

            let my_io_ctx = io_server
                .io_contexts
                .iter()
                .find(|(h, _)| *h == hash)
                .map(|(_, ctx)| ctx.swapchain.clone())
                .expect("FATAL: IO Context for zone not found");

            let metadata = manifest_metadata
                .get(&hash)
                .expect("FATAL: Metadata for zone not found");

            let ctx = crate::node::shard_thread::NodeContext {
                bsp_barrier: bsp_barrier.clone(),
                io_ctx: my_io_ctx,
                rt_handle: rt_handle.clone(),
                atomic_settings: metadata.atomic_settings.clone(),
                incoming_grow: desc.incoming_grow.clone(),
                routing_table: routing_table.clone(), // [DOD FIX]
            };

            let handle = crate::node::shard_thread::spawn_shard_thread(
                desc,
                ctx,
                rx,
                feedback_tx.clone(),
                core_id,
                sync_batch_ticks,
            );
            thread_handles.push(handle);
        }

        Self {
            services: NodeServices {
                io_server,
                routing_table,
                bsp_barrier,
            },
            network: NetworkTopology {
                intra_gpu_channels,
                inter_node_channels,
                inter_node_router,
                egress_pool,
                axon_head_ptrs,
                routing_acks: shared_acks_queue,
                routing_prunes: shared_prunes_queue,
            },
            compute_dispatchers,
            feedback_sender: feedback_tx,
            feedback_receiver: feedback_rx,
            total_ticks,
            local_ip,
            local_port,
            output_routes,
            compute_thread_handles: Some(thread_handles),
            daemons: Mutex::new(daemons),
            manifest_metadata: Mutex::new(manifest_metadata),
            zone_v_segs,
            virtual_offset_map,
            sync_batch_ticks,
            cluster_secret,
            egress_transpose_buffer: Vec::with_capacity(1024 * 1024), // 1MB reserve
        }
    }

    fn patch_routing_tables(&mut self) {
        let stream = std::ptr::null_mut();

        while let Some(ack) = self.network.routing_acks.pop() {
            if let Some((_, channel)) = self
                .network
                .inter_node_channels
                .iter_mut()
                .find(|(_, c)| c.src_zone_hash == ack.target_zone_hash && c.target_zone_hash == ack.receiver_zone_hash)
            {
                unsafe { channel.push_route(ack.src_axon_id, ack.dst_ghost_id, stream); }
            }
            else if let Some((_, _, channel)) = self
                .network
                .intra_gpu_channels
                .iter_mut()
                .find(|(_, _, c)| c.src_zone_hash == ack.target_zone_hash && c.target_zone_hash == ack.receiver_zone_hash)
            {
                unsafe { channel.push_route(ack.src_axon_id, ack.dst_ghost_id, stream); }
            }
        }

        while let Some(prune) = self.network.routing_prunes.pop() {
            if let Some((_, channel)) = self
                .network
                .inter_node_channels
                .iter_mut()
                .find(|(_, c)| c.src_zone_hash == prune.target_zone_hash && c.target_zone_hash == prune.receiver_zone_hash)
            {
                unsafe { channel.prune_route(prune.dst_ghost_id, stream); }
            } else if let Some((_, _, channel)) = self
                .network
                .intra_gpu_channels
                .iter_mut()
                .find(|(_, _, c)| c.src_zone_hash == prune.target_zone_hash && c.target_zone_hash == prune.receiver_zone_hash)
            {
                unsafe { channel.prune_route(prune.dst_ghost_id, stream); }
            }
        }
    }

    fn spawn_baker_daemons(shards: &[crate::node::shard_thread::ShardDescriptor]) -> Vec<Child> {
        let mut daemons = Vec::with_capacity(shards.len());
        let exe_path = env::current_exe().expect("FATAL: Failed to get current exe path");

        #[cfg(target_os = "windows")]
        let daemon_path = exe_path.with_file_name("axicor-baker-daemon.exe");
        #[cfg(not(target_os = "windows"))]
        let daemon_path = exe_path.with_file_name("axicor-baker-daemon");

        for desc in shards {
            let socket_addr = axicor_core::ipc::default_socket_path(desc.hash);
            #[cfg(unix)]
            let _ = std::fs::remove_file(&socket_addr);

            // [DOD FIX] Shard threads use manifests from /dev/shm
            let manifest_shm_path = axicor_core::ipc::manifest_shm_path(desc.hash);

            info!(
                "[Orchestrator] Spawning CPU Baker Daemon for zone 0x{:08X} (IPC: {})",
                desc.hash, socket_addr
            );
            let child = Command::new(&daemon_path)
                .arg("--manifest")
                .arg(&manifest_shm_path)
                .arg("--zone-hash")
                .arg(desc.hash.to_string())
                .arg("--baked-dir")
                .arg(&desc.baked_dir)
                // [DOD FIX] Remove silencer! We must see CPU-daemon panics in node console.
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .spawn()
                .expect("FATAL: Failed to spawn axicor-baker-daemon. Was it compiled?");

            daemons.push(child);
        }
        daemons
    }

    // [DOD FIX] The correct Pipeline Order: Compute -> Network Tx -> Network Rx Wait
    pub fn run_node_loop(&mut self, is_running: Arc<std::sync::atomic::AtomicBool>) {
        let batch_size = self.sync_batch_ticks;
        let mut current_tick = 0;
        let mut batch_counter: u64 = 0;

        // [DOD FIX] Strict binding of orchestrator OS thread to hardware context
        unsafe {
            axicor_compute::ffi::gpu_set_device(0);
        }

        // [DOD] Pre-allocate outbound buffers to avoid heap thrashing
        let mut _io_tx_buffer = vec![0u8; axicor_core::constants::MAX_UDP_PAYLOAD];

        info!("Entering main loop (Batch size: {})", batch_size);

        let loop_start = std::time::Instant::now();
        let mut batch_start = loop_start;

        while is_running.load(Ordering::Acquire) {
            // 1. Swap IO Buffers (Acquire semantics)
            for (_, io_ctx) in &self.services.io_server.io_contexts {
                io_ctx.swapchain.swap();
            }

            // 2. Dispatch batches to compute shards
            let current_dopamine = self
                .services
                .io_server
                .global_dopamine
                .load(Ordering::Relaxed) as i16;

            let num_dispatchers = self.compute_dispatchers.len();
            if num_dispatchers == 0 {
                error!("No compute dispatchers found!");
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
                        ComputeFeedback::BatchComplete {
                            ticks_processed: _,
                            zone_hash,
                            pinned_out_ptr,
                            mut output_bytes,
                            is_warmup,
                        } => {
                            // [DOD] Hardware Gating: Mute all outgoing traffic during Warmup
                            if is_warmup {
                                for (_, channel) in &self.network.inter_node_channels {
                                    if channel.src_zone_hash == zone_hash {
                                        unsafe {
                                            std::ptr::write_volatile(channel.out_count_pinned, 0);
                                        }
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

            // [DOD] GPU Hardware Barrier ~ wait for all streams to finish
            unsafe {
                axicor_compute::ffi::gpu_device_synchronize();
            }

            // Ship outputs to network targets ONLY POST SYNC!
            for (zone_hash, pinned_out_ptr, output_bytes) in pending_outputs {
                let batch_size_usize = batch_size as usize;
                let total_pixels = output_bytes / batch_size_usize;
                let pinned_out_slice = unsafe {
                    std::slice::from_raw_parts(pinned_out_ptr as *const u8, output_bytes)
                };

                // 160 KB copied in L1/L2 processor cache in nanoseconds. No allocations and memset!
                // [DOD FIX] Safe reallocation prevents STATUS_HEAP_CORRUPTION
                self.egress_transpose_buffer.resize(output_bytes, 0);

                for t in 0..batch_size_usize {
                    for p in 0..total_pixels {
                        self.egress_transpose_buffer[p * batch_size_usize + t] =
                            pinned_out_slice[t * total_pixels + p];
                    }
                }

                // 2. Slice and send strictly by L7 chunks
                if let Some(routes) = self.output_routes.get(&zone_hash) {
                    for (addr, m_hash, pixel_offset, chunk_pixels) in routes {
                        let byte_offset = pixel_offset * batch_size_usize;
                        let byte_size = chunk_pixels * batch_size_usize;
                        let payload =
                            &self.egress_transpose_buffer[byte_offset..byte_offset + byte_size];

                        self.services.io_server.send_output_batch_pool(
                            &self.network.egress_pool,
                            addr,
                            zone_hash,
                            *m_hash,
                            payload,
                        );
                        // [DOD FIX] Revive UDP OUT counter!
                    }
                }
            }

            // [DOD] 4. Intra-GPU Ghost Sync
            for (src_ptr, dst_ptr, channel) in &self.network.intra_gpu_channels {
                unsafe {
                    let v_seg = self
                        .zone_v_segs
                        .get(&channel.src_zone_hash)
                        .copied()
                        .unwrap_or(1);
                    channel.sync_ghosts(
                        *src_ptr,
                        *dst_ptr,
                        batch_size,
                        v_seg,
                        std::ptr::null_mut(),
                    );
                }
            }

            for (src_ptr, channel) in &self.network.inter_node_channels {
                // v_seg must be passed for accurate spike timing extraction
                let v_seg = self
                    .zone_v_segs
                    .get(&channel.src_zone_hash)
                    .copied()
                    .unwrap_or(1);
                unsafe {
                    channel.extract_spikes(*src_ptr, batch_size, v_seg, std::ptr::null_mut());
                }
            }

            // [DOD] 5. GPU Barrier Sync (wait for sync_ghosts in default stream)
            unsafe {
                axicor_compute::ffi::gpu_stream_synchronize(std::ptr::null_mut());
            }

            // [DOD] 6. Inter-Node Fast Path (Egress)
            for (_, channel) in &self.network.inter_node_channels {
                let out_count = unsafe { std::ptr::read_volatile(channel.out_count_pinned) };

                // BSP Heartbeat: ALWAYS build and send packet
                let events_slice = unsafe {
                    std::slice::from_raw_parts(channel.out_events_pinned, out_count as usize)
                };
                self.network.inter_node_router.flush_outgoing_batch_pool(
                    &self.network.egress_pool,
                    channel.src_zone_hash,
                    channel.target_zone_hash,
                    events_slice,
                    (batch_counter & 0xFFFFFFFF) as u32,
                );
            }

            // [DOD] 7. Wait for Ingress data (Strict BSP network sync)
            self.services.bsp_barrier.wait_for_data_sync();

            if let Err(actual_epoch) = self
                .services
                .bsp_barrier
                .sync_and_swap((batch_counter & 0xFFFFFFFF) as u32)
            {
                let delta = actual_epoch.saturating_sub((batch_counter & 0xFFFFFFFF) as u32);
                if delta > 0 {
                    warn!(
                        "[AEP Barrier] Desync! Fast-forwarding local orchestrator by {} batches",
                        delta
                    );

                    // Hardware Fast-Forward
                    batch_counter += delta as u64;
                    current_tick += (delta * batch_size) as u64;

                    // Mandatory cleanup of hanging read_schedule, otherwise GPU
                    // will read ghosts from distant past on next tick.
                    self.services.bsp_barrier.get_read_schedule().clear();

                    // Skip remainder of cycle (routing patching etc.) and start a clean new batch
                    continue;
                }
            }

            // [DOD FIX] 8. Dynamic Capacity Routing: Hot-Patching VRAM
            // Barrier passed, GPU idle. Ideal window for PCIe bus.
            while let Some(prune) = self.network.routing_prunes.pop() {
                // Find owner channel and remove route
                for (_, channel) in &mut self.network.inter_node_channels {
                    unsafe {
                        channel.prune_route(prune.dst_ghost_id, std::ptr::null_mut());
                    }
                }
                for (_, _, channel) in &mut self.network.intra_gpu_channels {
                    unsafe {
                        channel.prune_route(prune.dst_ghost_id, std::ptr::null_mut());
                    }
                }
            }
            self.patch_routing_tables();

            let _wall_ms = batch_start.elapsed().as_millis() as u64;
            batch_start = std::time::Instant::now();

            batch_counter += 1;
            // [DOD FIX] Restore time progression! Without this GPU will get stuck in first N ticks.
            current_tick += batch_size as u64;
        }
    }

    pub fn teardown(&mut self) {
        tracing::info!("[Orchestrator] Executing synchronous C-ABI teardown...");
        
        // 1. Send shutdown to all shards
        for tx in self.compute_dispatchers.values() {
            let _ = tx.send(ComputeCommand::Shutdown);
        }
        
        // 2. Drop dispatchers to close channels
        self.compute_dispatchers.clear();
        
        // 3. Await all OS threads to gracefully clean up VRAM
        if let Some(handles) = self.compute_thread_handles.take() {
            for handle in handles {
                let _ = handle.join();
            }
        }
        tracing::info!("[Orchestrator] All compute threads terminated. VRAM cleared.");
    }
}

impl Drop for NodeRuntime {
    fn drop(&mut self) {
        let mut daemons = self.daemons.lock().unwrap();
        for (i, daemon) in daemons.iter_mut().enumerate() {
            let _ = daemon.kill(); // Send SIGKILL
            let _ = daemon.wait(); // Wait for OS death confirmation, preventing zombies
            info!("[Orchestrator] Baker Daemon {} successfully terminated.", i);
        }
    }
}
