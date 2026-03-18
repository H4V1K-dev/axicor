use std::sync::Arc;
use std::thread;
use std::ptr;
use crossbeam::channel::Receiver;
use genesis_compute::ShardEngine;
use genesis_core::config::InstanceConfig;
use genesis_core::ipc::{AxonHandoverEvent, shm_size, ShmHeader};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use crate::network::bsp::BspBarrier;
use crate::network::io_server::InputSwapchain;
use super::{ComputeCommand, ComputeFeedback};

/// [Phase 23] Static shard geometry/physics — owns all per-shard data.
pub struct ShardDescriptor {
    pub hash: u32,
    pub engine: ShardEngine,
    pub num_virtual_axons: u32,
    pub virtual_offset: u32,
    pub num_outputs: u32,
    pub mapped_soma_ids_host: Option<Vec<u32>>,
    pub baked_dir: std::path::PathBuf,
    pub config: InstanceConfig,
    pub v_seg: u32,
    pub incoming_grow: Arc<crossbeam::queue::SegQueue<AxonHandoverEvent>>,
}

use std::sync::atomic::{AtomicU64, AtomicI16, AtomicU16, Ordering};

// TODO: Найти идеальный баланс для линейного стабильного роста, а потом аппроксимировать и 
// заложить расчет нейрогенеза для каждого шарда автоматически на основе типов внутри.
pub struct ShardAtomicSettings {
    pub night_interval_ticks: AtomicU64,
    pub save_checkpoints_interval_ticks: AtomicU64, // ticks counter
    pub prune_threshold: AtomicI16,
    pub max_sprouts: AtomicU16,
}

/// [Phase 23] Shared orchestrator resources — cheap Clone via Arc.
#[derive(Clone)]
pub struct NodeContext {
    pub bsp_barrier: Arc<BspBarrier>,
    pub io_ctx: Arc<InputSwapchain>,
    pub rt_handle: tokio::runtime::Handle,
    pub atomic_settings: Arc<ShardAtomicSettings>,
    pub incoming_grow: Arc<crossbeam::queue::SegQueue<AxonHandoverEvent>>,
    // [DOD FIX] Удален Mutex<DashboardState>, используем Lock-Free структуру
    pub telemetry: Arc<crate::tui::state::LockFreeTelemetry>, 
    pub routing_table: Arc<crate::network::router::RoutingTable>, // [DOD FIX]
}
pub struct ThreadWorkspace {
    pub weights_offset: usize,
    pub targets_offset: usize,
    pub handovers_offset: usize,
    pub flags_offset: usize,
    pub shm_buffer: MmapMut,
    pub checkpoint_state_buffer: Vec<u8>,
    pub checkpoint_axons_buffer: Vec<u8>, // [DOD FIX] Буфер для Active Tails
    pub ghost_origins: Vec<u32>, // [DOD FIX] O(1) Origin Tracking
}

impl ThreadWorkspace {
    pub fn new(zone_hash: u32, padded_n: usize, total_ghosts: usize) -> Self {
        let path = genesis_core::ipc::shm_file_path(zone_hash);
        let total_size = shm_size(padded_n);

        // Daemon creates the file; retry until valid header (race: daemon may not be ready yet)
        for _ in 0..30 {
            let file = match OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
            {
                Ok(f) => f,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            };
            if file.metadata().map(|m| m.len()).unwrap_or(0) < total_size as u64 {
                let _ = file.set_len(total_size as u64);
            }
            let mmap = match unsafe { MmapMut::map_mut(&file) } {
                Ok(m) => m,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            };
            let header = unsafe { std::ptr::read(mmap.as_ptr() as *const ShmHeader) };
            if header.validate().is_ok() {
                return Self {
                    weights_offset: header.weights_offset as usize,
                    targets_offset: header.targets_offset as usize,
                    handovers_offset: header.handovers_offset as usize,
                    flags_offset: header.flags_offset as usize, // [DOD FIX]
                    shm_buffer: mmap,
                    checkpoint_state_buffer: vec![0u8; 0],
                    checkpoint_axons_buffer: vec![0u8; 0],
                    ghost_origins: vec![0u32; total_ghosts],
                };
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        panic!("SHM file {:?} not ready after 3s (is genesis-baker-daemon running?)", path);
    }

    pub fn weights_slice_mut(&mut self, padded_n: usize) -> &mut [i16] {
        let len = padded_n * 128;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.shm_buffer.as_mut_ptr().add(self.weights_offset) as *mut i16,
                len,
            )
        }
    }

    pub fn targets_slice_mut(&mut self, padded_n: usize) -> &mut [u32] {
        let len = padded_n * 128;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.shm_buffer.as_mut_ptr().add(self.targets_offset) as *mut u32,
                len,
            )
        }
    }

    pub fn flags_slice_mut(&mut self, padded_n: usize) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.shm_buffer.as_mut_ptr().add(self.flags_offset) as *mut u8,
                padded_n,
            )
        }
    }
}

// ФАЗА 1: Выполнение GPU батча (Day Phase)
#[inline(always)]
fn execute_day_phase(
    shard: &mut ShardEngine,
    batch_size: u32,
    global_dopamine: i16,
    bsp_barrier: &Arc<BspBarrier>,
    my_io_ctx: &Arc<InputSwapchain>,
    io_buffers: &genesis_compute::compute::shard::IoDeviceBuffers,
    virtual_offset: u32,
    num_virtual_axons: u32,
    mapped_soma_ids: *const u32,
    v_seg: u32,
    _batch_counter: u64,
    tick_base: u32, // <--- ADD
) {
    let _sync_batch_ticks = 100u32;
    let input_words_per_tick = (num_virtual_axons + 31) / 32;

    let schedule = bsp_barrier.get_read_schedule();
    
    // [DOD FIX] Zero-cost, 100% безопасное взятие слайсов из Pinned RAM
    let incoming_slice = schedule.ghost_ids.as_slice();
    let counts_slice_atomic = schedule.counts.as_slice();
    let counts_slice = unsafe {
        // AtomicU32 and u32 have identical memory layout.
        std::slice::from_raw_parts(counts_slice_atomic.as_ptr() as *const u32, counts_slice_atomic.len())
    };

    let input_ptr = my_io_ctx.consume_for_gpu();
    let input_slice = if !input_ptr.is_null() && input_words_per_tick > 0 {
        unsafe {
            Some(std::slice::from_raw_parts(
                input_ptr as *const u32,
                (input_words_per_tick * batch_size) as usize,
            ))
        }
    } else {
        None
    };

    /* 
    if batch_counter % 100 == 0 {
        println!("🔍 [Shard I/O] v_offset: {}, v_axons: {}, v_seg: {}", virtual_offset, num_virtual_axons, v_seg);
        if let Some(slice) = input_slice {
            let active = slice.iter().any(|&w| w != 0);
            if active {
                println!("🔥 [Shard I/O] Input active!");
            }
        }
    }
    */

    shard.step_day_phase_batch(
        batch_size,
        io_buffers,
        input_slice,
        Some(incoming_slice),
        counts_slice,
        virtual_offset,
        num_virtual_axons,
        mapped_soma_ids,
        v_seg,
        global_dopamine,
        tick_base,
    );
}

fn download_outputs(
    num_outputs: u32,
    pinned_out: &mut genesis_compute::memory::PinnedBuffer<u8>,
    io_buffers: &genesis_compute::compute::shard::IoDeviceBuffers,
    output_bytes: usize,
    stream: genesis_compute::ffi::CudaStream,
) {
    if num_outputs > 0 {
        unsafe {
            genesis_compute::ffi::cu_dma_d2h_io(
                pinned_out.as_mut_ptr(),
                io_buffers.d_output_history,
                output_bytes as u32,
                stream,
            );
            // Synchronize ONLY our stream before CPU reads the PinnedBuffer
            genesis_compute::ffi::gpu_stream_synchronize(stream);
        }
    }
}

// ФАЗА 3: Периодический сброс на диск (I/O)
#[inline(always)]
fn save_hot_checkpoint(
    shard: &ShardEngine, 
    _hash: u32, 
    baked_dir: &std::path::Path, 
    state_buf: &mut [u8], 
    axons_buf: &mut [u8]
) {
    unsafe {
        // 1. Асинхронный захват обоих массивов
        genesis_compute::ffi::gpu_memcpy_device_to_host(
            state_buf.as_mut_ptr() as *mut _,
            shard.vram.ptrs.soma_voltage as *const _,
            state_buf.len(),
        );
        genesis_compute::ffi::gpu_memcpy_device_to_host(
            axons_buf.as_mut_ptr() as *mut _,
            shard.vram.ptrs.axon_heads as *const _,
            axons_buf.len(),
        );
        // Ждем завершения DMA перед записью на диск
        genesis_compute::ffi::gpu_device_synchronize(); 
    }

    let chk_state = baked_dir.join("checkpoint.state");
    let tmp_state = baked_dir.join("checkpoint.state.tmp");
    let chk_axons = baked_dir.join("checkpoint.axons");
    let tmp_axons = baked_dir.join("checkpoint.axons.tmp");

    // 2. Атомарная запись на диск
    if std::fs::write(&tmp_state, state_buf).is_ok() && std::fs::write(&tmp_axons, axons_buf).is_ok() {
        let _ = std::fs::rename(&tmp_state, &chk_state);
        let _ = std::fs::rename(&tmp_axons, &chk_axons);
        // println!("💾 [Shard {:08X}] State & Axons checkpoint saved", hash);
    }
}

// ФАЗА 4: Обслуживание графа (Night Phase)
#[inline(always)]
fn execute_night_phase(
    shard: &mut ShardEngine,
    hash: u32,
    socket_path: &std::path::Path,
    baker_client: &mut Option<crate::ipc::BakerClient>,
    incoming_grow: &Arc<crossbeam::queue::SegQueue<AxonHandoverEvent>>,
    shard_config: &InstanceConfig,
    rt_handle: &tokio::runtime::Handle,
    workspace: &mut ThreadWorkspace,
    prune_threshold: i16, 
    max_sprouts: u16,
    routing_table: &Arc<crate::network::router::RoutingTable>,
    telemetry: &Arc<crate::tui::state::LockFreeTelemetry>,
) {
    let padded_n = shard.vram.padded_n as usize;
    let dendrites_count = padded_n * genesis_core::constants::MAX_DENDRITE_SLOTS;

    unsafe {
        // 0. DMA: soma_flags to SHM [Capture spikes BEFORE they are cleared by pruning kernel]
        genesis_compute::ffi::gpu_memcpy_device_to_host(
            workspace.flags_slice_mut(padded_n).as_mut_ptr() as *mut _,
            shard.vram.ptrs.soma_flags as *const _,
            padded_n * std::mem::size_of::<u8>(),
        );

        // 1. GPU Defragmentation & Prune
        genesis_compute::ffi::launch_sort_and_prune(
            &shard.vram.ptrs,
            shard.vram.padded_n,
            prune_threshold,
        );
        genesis_compute::ffi::gpu_device_synchronize();

        // 2. D2H: Dendrite matrices DIRECTLY to SHM
        genesis_compute::ffi::gpu_memcpy_device_to_host(
            workspace.weights_slice_mut(padded_n).as_mut_ptr() as *mut _,
            shard.vram.ptrs.dendrite_weights as *const _,
            dendrites_count * std::mem::size_of::<i16>(),
        );
        genesis_compute::ffi::gpu_memcpy_device_to_host(
            workspace.targets_slice_mut(padded_n).as_mut_ptr() as *mut _,
            shard.vram.ptrs.dendrite_targets as *const _,
            dendrites_count * std::mem::size_of::<u32>(),
        );
    }

    // 3. Sprouting (Late Binding)
    if baker_client.is_none() {
        *baker_client = crate::ipc::BakerClient::connect(hash, socket_path).ok();
    }

    if let Some(client) = baker_client.as_mut() {
        let mut incoming_handovers = Vec::new();
        while let Some(ev) = incoming_grow.pop() {
            incoming_handovers.push(ev);
            if incoming_handovers.len() >= genesis_core::ipc::MAX_HANDOVERS_PER_NIGHT {
                break;
            }
        }

        match client.run_night(
            &incoming_handovers,
            &workspace.ghost_origins,
            padded_n,
            std::time::Duration::from_secs(10),
            prune_threshold,
            max_sprouts,
        ) {
            Ok(acks) => {
                unsafe {
                    genesis_compute::ffi::gpu_memcpy_host_to_device(
                        shard.vram.ptrs.dendrite_targets as *mut _,
                        workspace.targets_slice_mut(padded_n).as_ptr() as *const _,
                        dendrites_count * std::mem::size_of::<u32>(),
                    );
                    // [DOD FIX] Синхронизируем веса новых синапсов (Закон Дейла)
                    genesis_compute::ffi::gpu_memcpy_host_to_device(
                        shard.vram.ptrs.dendrite_weights as *mut _,
                        workspace.weights_slice_mut(padded_n).as_ptr() as *const _,
                        dendrites_count * std::mem::size_of::<i16>(),
                    );
                    genesis_compute::ffi::gpu_device_synchronize();
                }

                dispatch_handovers(client, shard_config, rt_handle);
                
                // [DOD FIX] Наполняем карту владельцев призраков (Origin Tracking)
                for ack in &acks {
                    let idx = (ack.dst_ghost_id as usize).saturating_sub(padded_n);
                    if idx < workspace.ghost_origins.len() {
                        workspace.ghost_origins[idx] = ack.target_zone_hash;
                    }
                }
                
                dispatch_acks(acks, rt_handle, routing_table); 
                
                // [DOD FIX] Чтение GC-очисток из SHM и маршрутизация смерти
                dispatch_prunes(client, &workspace.ghost_origins, padded_n, rt_handle, routing_table);

                telemetry.push_log(format!("🌅 [Shard {:08X}] Night Phase complete. Waking up.", hash), crate::tui::state::LogLevel::Night);
            }
            Err(e) => {
                telemetry.push_log(format!("❌ [Shard {:08X}] Sprouting failed: {}", hash, e), crate::tui::state::LogLevel::Error);
                *baker_client = None;
            }
        }
    } else {
        telemetry.push_log(format!("⚠️ [Shard {:08X}] Skipping Sprouting (Daemon not connected). Will retry next night.", hash), crate::tui::state::LogLevel::Warning);
    }
}

// Абстракция для отправки Spatial Geometry Routing внутри Night Phase
#[inline(always)]
fn dispatch_handovers(
    client: &crate::ipc::BakerClient,
    shard_config: &InstanceConfig,
    rt_handle: &tokio::runtime::Handle,
) {
    let shm_hdr = unsafe { std::ptr::read(client.shm_ptr as *const genesis_core::ipc::ShmHeader) };
    let handovers_slice = unsafe {
        std::slice::from_raw_parts(
            client.shm_ptr.add(shm_hdr.handovers_offset as usize) as *const crate::network::slow_path::AxonHandoverEvent,
            shm_hdr.handovers_count as usize,
        )
    };

    let mut x_plus = Vec::new();
    let mut x_minus = Vec::new();
    let mut y_plus = Vec::new();
    let mut y_minus = Vec::new();

    let max_x = shard_config.dimensions.w as u16;
    let max_y = shard_config.dimensions.d as u16;

    for ev in handovers_slice {
        if ev.entry_x >= max_x {
            x_plus.push(*ev);
        } else if ev.entry_x == 0 {
            x_minus.push(*ev);
        } else if ev.entry_y >= max_y {
            y_plus.push(*ev);
        } else if ev.entry_y == 0 {
            y_minus.push(*ev);
        }
    }

    let mut routes_to_execute: Vec<(String, Vec<crate::network::slow_path::AxonHandoverEvent>)> = Vec::new();
    let neighbors = &shard_config.neighbors;
    if !x_plus.is_empty() {
        if let Some(ref addr) = neighbors.x_plus {
            routes_to_execute.push((addr.clone(), x_plus));
        }
    }
    if !x_minus.is_empty() {
        if let Some(ref addr) = neighbors.x_minus {
            routes_to_execute.push((addr.clone(), x_minus));
        }
    }
    if !y_plus.is_empty() {
        if let Some(ref addr) = neighbors.y_plus {
            routes_to_execute.push((addr.clone(), y_plus));
        }
    }
    if !y_minus.is_empty() {
        if let Some(ref addr) = neighbors.y_minus {
            routes_to_execute.push((addr.clone(), y_minus));
        }
    }

    for (addr_str, batch) in routes_to_execute {
        rt_handle.spawn(async move {
            if let Ok(addr) = addr_str.parse::<std::net::SocketAddr>() {
                let req = crate::network::slow_path::GeometryRequest::BulkHandover(batch);
                let _ = crate::network::geometry_client::send_geometry_request(addr, &req).await;
            }
        });
    }
}

fn dispatch_acks(
    acks: Vec<genesis_core::ipc::AxonHandoverAck>, 
    rt_handle: &tokio::runtime::Handle,
    routing_table: &Arc<crate::network::router::RoutingTable>
) {
    if acks.is_empty() { return; }

    // Группируем по target_zone_hash (чтобы отправить одним TCP пакетом на ноду)
    let mut grouped: std::collections::HashMap<u32, Vec<genesis_core::ipc::AxonHandoverAck>> = std::collections::HashMap::new();
    for ack in acks {
        grouped.entry(ack.target_zone_hash).or_default().push(ack);
    }

    for (target_hash, batch) in grouped {
        if let Some((addr, _mtu)) = routing_table.get_address(target_hash) {
            rt_handle.spawn(async move {
                let req = crate::network::slow_path::GeometryRequest::BulkAck(batch);
                let _ = crate::network::geometry_client::send_geometry_request(addr, &req).await;
            });
        }
    }
}

fn dispatch_prunes(
    client: &crate::ipc::BakerClient,
    ghost_origins: &[u32],
    padded_n: usize,
    rt_handle: &tokio::runtime::Handle,
    routing_table: &Arc<crate::network::router::RoutingTable>,
) {
    let shm_hdr = unsafe { std::ptr::read(client.shm_ptr as *const ShmHeader) };
    if shm_hdr.prunes_count == 0 { return; }

    let prunes_slice = unsafe {
        std::slice::from_raw_parts(
            client.shm_ptr.add(shm_hdr.prunes_offset as usize) as *const genesis_core::ipc::AxonHandoverPrune,
            shm_hdr.prunes_count as usize,
        )
    };

    for prune in prunes_slice {
        let ghost_id = prune.dst_ghost_id as usize;
        let idx = ghost_id.saturating_sub(padded_n);
        
        if idx < ghost_origins.len() {
            let target_hash = ghost_origins[idx];
            if target_hash != 0 {
                if let Some((addr, _mtu)) = routing_table.get_address(target_hash) {
                    let gid = prune.dst_ghost_id;
                    rt_handle.spawn(async move {
                        let req = crate::network::slow_path::GeometryRequest::Prune(gid);
                        let _ = crate::network::geometry_client::send_geometry_request(addr, &req).await;
                    });
                }
            }
        }
    }
}

// Инициализация VRAM буферов
fn init_io_buffers(
    num_virtual_axons: u32,
    max_spikes_per_tick: u32,
    num_outputs: u32,
    sync_batch_ticks: u32,
) -> genesis_compute::compute::shard::IoDeviceBuffers {
    let input_words_per_tick = (num_virtual_axons + 31) / 32;
    let mut d_input = ptr::null_mut();
    let mut d_spikes = ptr::null_mut();
    let mut d_output = ptr::null_mut();
    unsafe {
        genesis_compute::ffi::cu_allocate_io_buffers(
            input_words_per_tick * sync_batch_ticks,
            max_spikes_per_tick * sync_batch_ticks,
            num_outputs * sync_batch_ticks,
            &mut d_input,
            &mut d_spikes,
            &mut d_output,
        );
    }
    genesis_compute::compute::shard::IoDeviceBuffers {
        d_input_bitmask: d_input,
        d_incoming_spikes: d_spikes,
        d_output_history: d_output,
        max_spikes_per_tick,
        input_words_per_tick,
        num_outputs,
    }
}

pub fn spawn_shard_thread(
    mut desc: ShardDescriptor,
    ctx: NodeContext,
    rx: Receiver<ComputeCommand>,
    f_tx: crossbeam::channel::Sender<ComputeFeedback>,
    core_id: usize,
    sync_batch_ticks: u32,
) {
    let max_spikes_per_tick = 100_000u32;
    let output_bytes = (desc.num_outputs * sync_batch_ticks) as usize;

    let hash = desc.hash;

    thread::Builder::new()
        .name(format!("compute-{}", hash))
        .spawn(move || {
            // [DOD FIX] Hardware Sympathy: Pinning compute thread
            let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            unsafe { libc::CPU_SET(core_id, &mut cpuset) };
            let res = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset) };
            if res != 0 {
                eprintln!("Warning: Failed to set thread affinity to core {}", core_id);
            } else {
                println!("🚀 [Core] Shard 0x{:08X} compute locked to OS Thread Core {}", hash, core_id);
            }

            // 1. Инициализация аппаратного контекста
            unsafe { genesis_compute::ffi::gpu_set_device(0); }

            let mapped_soma_ids: *const u32 = if let Some(ref host_ids) = desc.mapped_soma_ids_host {
                let bytes = host_ids.len() * 4;
                let ptr = unsafe { genesis_compute::ffi::gpu_malloc(bytes) } as *mut u32;
                unsafe {
                    genesis_compute::ffi::gpu_memcpy_host_to_device(
                        ptr as *mut _,
                        host_ids.as_ptr() as *const _,
                        bytes,
                    );
                }
                ptr as *const u32
            } else {
                std::ptr::null()
            };

            let io_buffers = init_io_buffers(desc.num_virtual_axons, max_spikes_per_tick, desc.num_outputs, sync_batch_ticks);
            let mut pinned_out = genesis_compute::memory::PinnedBuffer::<u8>::new(output_bytes).unwrap();
            let mut baker_client: Option<crate::ipc::BakerClient> = None;
            let socket_path = genesis_core::ipc::default_socket_path(hash);
            
            let padded_n = desc.engine.vram.padded_n as usize;
            let _dendrites_count = padded_n * genesis_core::constants::MAX_DENDRITE_SLOTS;
            let (_, state_size) = genesis_compute::memory::calculate_state_blob_size(padded_n);

            // [DOD FIX] Выравниваем размер payload'а с ожиданиями Baker Daemon
            let mut workspace = ThreadWorkspace::new(hash, padded_n, desc.engine.vram.total_ghosts as usize);
            let axons_size = desc.engine.vram.total_axons as usize * std::mem::size_of::<genesis_core::layout::BurstHeads8>();
            workspace.checkpoint_state_buffer = vec![0u8; state_size];
            workspace.checkpoint_axons_buffer = vec![0u8; axons_size];

            let mut batch_counter: u64 = 0;
            let mut warmup_ticks_remaining: u32 = 2000;

            // 2. Плоский горячий цикл
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    ComputeCommand::Resurrect => {
                        warmup_ticks_remaining = 100;
                        ctx.telemetry.push_log(format!("Entering Warmup Phase (100 ticks) for 0x{:08X}", hash), crate::tui::state::LogLevel::Info);
                    }
                    ComputeCommand::RunBatch { tick_base, batch_size, global_dopamine } => {
                        let is_warmup = warmup_ticks_remaining > 0;
                        
                        // ФАЗА 1: Выполнение GPU батча (Day Phase)
                        execute_day_phase(
                            &mut desc.engine, batch_size, global_dopamine, &ctx.bsp_barrier,
                            &ctx.io_ctx, &io_buffers, desc.virtual_offset, desc.num_virtual_axons, mapped_soma_ids, desc.v_seg, batch_counter, tick_base
                        );

                        // --- ФАЗА 2: Чтение выходов (Асинхронно в своем стриме) ---
                        if desc.num_outputs > 0 {
                            download_outputs(desc.num_outputs, &mut pinned_out, &io_buffers, output_bytes, desc.engine.stream);
                        }
                        if is_warmup {
                            warmup_ticks_remaining = warmup_ticks_remaining.saturating_sub(batch_size);
                            if warmup_ticks_remaining == 0 {
                                ctx.telemetry.push_log(format!("Warmup complete for 0x{:08X}. Voltage stabilized.", hash), crate::tui::state::LogLevel::Info);
                            }
                        }

                        // ФАЗА 3: Периодический сброс на диск (I/O)
                        let cp_interval_ticks = ctx.atomic_settings.save_checkpoints_interval_ticks.load(Ordering::Relaxed);
                        let cp_interval = (cp_interval_ticks as u32 / batch_size).max(1);
                        if batch_counter > 0 && batch_counter % cp_interval as u64 == 0 {
                            save_hot_checkpoint(
                                &desc.engine, 
                                hash, 
                                &desc.baked_dir, 
                                &mut workspace.checkpoint_state_buffer,
                                &mut workspace.checkpoint_axons_buffer
                            );
                        }

                        // ФАЗА 4: Обслуживание графа (Night Phase)
                        let current_tick_count = (batch_counter + 1) * batch_size as u64;
                        let n_interval = ctx.atomic_settings.night_interval_ticks.load(Ordering::Relaxed);
                        if n_interval > 0 && current_tick_count % n_interval == 0 {
                            let current_prune_threshold = ctx.atomic_settings.prune_threshold.load(Ordering::Relaxed);
                            let current_max_sprouts = ctx.atomic_settings.max_sprouts.load(Ordering::Relaxed);
                            let night_start = std::time::Instant::now();
                             execute_night_phase(
                                &mut desc.engine, hash, std::path::Path::new(&socket_path), &mut baker_client,
                                &ctx.incoming_grow, &desc.config, &ctx.rt_handle, &mut workspace,
                                current_prune_threshold, current_max_sprouts, &ctx.routing_table, &ctx.telemetry
                            );
                            let elapsed_ns = night_start.elapsed().as_nanos();
                            ctx.telemetry.push_log(format!("🌙 [Shard {:08X}] Night Phase completed in {} ns", hash, elapsed_ns), crate::tui::state::LogLevel::Night);
                        }

                        // Update spikes in UI
                        {
                            let actual_spikes = pinned_out.as_slice().iter().filter(|&&x| x != 0).count() as u32;

                            // [DOD FIX] Zero-cost обновление через атомик, никакого лока планировщика
                            ctx.telemetry.update_zone_spikes(hash, actual_spikes);
                        }

                        // Отправка отчета оркестратору
                        if f_tx.send(ComputeFeedback::BatchComplete {
                            ticks_processed: batch_size,
                            zone_hash: hash,
                            pinned_out_ptr: pinned_out.as_ptr() as usize,
                            output_bytes,
                            is_warmup, // [DOD FIX] Tell orchestrator to mute network
                        }).is_err() { break; }

                        batch_counter += 1;
                    }
                    ComputeCommand::Shutdown => break,
                }
            }
        }).expect("Failed to spawn compute thread");
}

