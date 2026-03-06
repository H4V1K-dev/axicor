use std::sync::Arc;
use std::thread;
use std::ptr;
use crossbeam::channel::Receiver;
use genesis_compute::ShardEngine;
use genesis_core::config::InstanceConfig;
use genesis_core::ipc::{AxonHandoverEvent, shm_name, shm_size, ShmHeader};
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
    pub incoming_prune: Arc<crossbeam::queue::SegQueue<crate::network::slow_path::AxonHandoverPrune>>,
}

/// [Phase 23] Shared orchestrator resources — cheap Clone via Arc.
#[derive(Clone)]
pub struct NodeContext {
    pub bsp_barrier: Arc<BspBarrier>,
    pub io_ctx: Arc<InputSwapchain>,
    pub rt_handle: tokio::runtime::Handle,
    pub night_interval: u64,
    pub incoming_grow: Arc<crossbeam::queue::SegQueue<AxonHandoverEvent>>,
}


pub struct ThreadWorkspace {
    pub shm_buffer: MmapMut,
    pub weights_offset: usize,
    pub targets_offset: usize,
    pub s2a_offset: usize,
    pub gxo_offset: usize,
    pub handovers_offset: usize,
    pub checkpoint_state_buffer: Vec<u8>,
    pub checkpoint_axons_buffer: Vec<u8>, // [DOD FIX] Буфер для Active Tails
}

impl ThreadWorkspace {
    pub fn new(zone_hash: u32, padded_n: usize) -> Self {
        let name = shm_name(zone_hash);
        let path = format!("/dev/shm{}", name);
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .expect("Failed to create SHM file");
        
        let total_size = shm_size(padded_n);
        file.set_len(total_size as u64).expect("Failed to set SHM size");
        
        let mut mmap = unsafe { MmapMut::map_mut(&file).expect("Failed to mmap SHM") };
        
        // Инициализируем заголовок контракта (Node владеет жизненным циклом)
        let header = ShmHeader::new(zone_hash, padded_n as u32, 0); 
        unsafe {
            std::ptr::write(mmap.as_mut_ptr() as *mut ShmHeader, header);
        }

        Self {
            weights_offset: header.weights_offset as usize,
            targets_offset: header.targets_offset as usize,
            s2a_offset: header.s2a_offset as usize,
            gxo_offset: header.gxo_offset as usize,
            handovers_offset: header.handovers_offset as usize,
            shm_buffer: mmap,
            checkpoint_state_buffer: vec![0u8; 0], // Will be resized in spawn
            checkpoint_axons_buffer: vec![0u8; 0], // Will be resized in spawn
        }
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
    batch_counter: u64,
) {
    let _sync_batch_ticks = 100u32;
    let input_words_per_tick = (num_virtual_axons + 31) / 32;

    unsafe {
        genesis_compute::ffi::update_global_dopamine(
            global_dopamine, 
            std::ptr::null_mut()
        );
    }

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

    if batch_counter % 10 == 0 {
        println!("🔍 [Shard I/O] v_offset: {}, v_axons: {}, v_seg: {}", virtual_offset, num_virtual_axons, v_seg);
        if let Some(slice) = input_slice {
            let active = slice.iter().any(|&w| w != 0);
            if active {
                println!("🔥 [Shard I/O] Input active!");
            }
        }
    }

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
    );
}

// ФАЗА 2: Чтение выходов
#[inline(always)]
fn download_outputs(
    num_outputs: u32,
    pinned_out: &mut genesis_compute::memory::PinnedBuffer<u8>,
    io_buffers: &genesis_compute::compute::shard::IoDeviceBuffers,
    output_bytes: usize,
) {
    if num_outputs > 0 {
        unsafe {
            genesis_compute::ffi::gpu_memcpy_device_to_host(
                pinned_out.as_mut_ptr() as *mut _,
                io_buffers.d_output_history as *const _,
                output_bytes,
            );
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
    baked_dir: &std::path::Path,
    incoming_prune: &Arc<crossbeam::queue::SegQueue<crate::network::slow_path::AxonHandoverPrune>>,
    mapped_soma_ids: *const u32,
) {
    let padded_n = shard.vram.padded_n as usize;
    let dendrites_count = padded_n * genesis_core::constants::MAX_DENDRITE_SLOTS;

    unsafe {
        // 1. GPU Defragmentation & Prune
        genesis_compute::ffi::launch_sort_and_prune(
            &shard.vram.ptrs,
            shard.vram.padded_n,
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

        let hdr_ptr = workspace.shm_buffer.as_mut_ptr() as *mut genesis_core::ipc::ShmHeader;
        let hdr = &mut *hdr_ptr;

        // 2.1 Flags D2H (Required for Night Phase Axon Reset)
        genesis_compute::ffi::gpu_memcpy_device_to_host(
            workspace.shm_buffer.as_mut_ptr().add(hdr.flags_offset as usize) as *mut _,
            shard.vram.ptrs.soma_flags as *const _,
            padded_n * std::mem::size_of::<u8>(),
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

        let mut incoming_prunes = Vec::new();
        while let Some(prune) = incoming_prune.pop() { 
            incoming_prunes.push(prune.ghost_id);
            if incoming_prunes.len() >= genesis_core::ipc::MAX_HANDOVERS_PER_NIGHT {
                break;
            }
        }

        match client.run_night(
            &incoming_handovers,
            &incoming_prunes,
            padded_n,
            std::time::Duration::from_secs(10),
            0, 
        ) {
            Ok(()) => {
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
                    
                    // [DOD Stage 48.2] S2A Zero-Copy Sync
                    genesis_compute::ffi::gpu_memcpy_host_to_device(
                        shard.vram.ptrs.soma_to_axon as *mut _,
                        workspace.shm_buffer.as_ptr().add(workspace.s2a_offset) as *const _,
                        padded_n * std::mem::size_of::<u32>(),
                    );

                    // [DOD Stage 48.3] GXO (Readout) Zero-Copy Sync
                    if let Some(gxo_ptr) = (mapped_soma_ids as *mut u32).as_mut() {
                        let shm_hdr = &*(workspace.shm_buffer.as_ptr() as *const genesis_core::ipc::ShmHeader);
                        if shm_hdr.gxo_count > 0 {
                            genesis_compute::ffi::gpu_memcpy_host_to_device(
                                gxo_ptr as *mut _ as *mut _,
                                workspace.shm_buffer.as_ptr().add(workspace.gxo_offset) as *const _,
                                shm_hdr.gxo_count as usize * std::mem::size_of::<u32>(),
                            );
                        }
                    }
                    
                    // [DOD] Stage 44.3: Hot Reload Refresh (Only .axons as they are not in SHM yet)
                    if let Ok(axons_blob) = std::fs::read(baked_dir.join("shard.axons")) {
                        shard.vram.upload_axon_heads(&axons_blob);
                    }
                    
                    genesis_compute::ffi::gpu_device_synchronize();
                }

                dispatch_handovers(client, shard_config, rt_handle);
                dispatch_prunes(client, shard_config, rt_handle);
                // println!("🌅 [Shard {:08X}] Night Phase complete. Waking up.", hash);
            }
            Err(e) => {
                println!("❌ [Shard {:08X}] Sprouting failed: {}", hash, e);
                *baker_client = None;
            }
        }
    } else {
        println!("⚠️ [Shard {:08X}] Skipping Sprouting (Daemon not connected). Will retry next night.", hash);
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
    let mut z_plus = Vec::new();
    let mut z_minus = Vec::new();

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
        } else if ev.vector_z > 0 {
            z_plus.push(*ev);
        } else if ev.vector_z < 0 {
            z_minus.push(*ev);
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
    if !z_plus.is_empty() {
        if let Some(ref addr) = neighbors.z_plus {
            routes_to_execute.push((addr.clone(), z_plus));
        }
    }
    if !z_minus.is_empty() {
        if let Some(ref addr) = neighbors.z_minus {
            routes_to_execute.push((addr.clone(), z_minus));
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

#[inline(always)]
fn dispatch_prunes(
    client: &crate::ipc::BakerClient,
    _shard_config: &InstanceConfig,
    rt_handle: &tokio::runtime::Handle,
) {
    let shm_hdr = unsafe { std::ptr::read(client.shm_ptr as *const genesis_core::ipc::ShmHeader) };
    if shm_hdr.prunes_count == 0 { return; }

    let prunes_slice = unsafe {
        std::slice::from_raw_parts(
            client.shm_ptr.add(shm_hdr.prunes_offset as usize) as *const u32,
            shm_hdr.prunes_count as usize,
        )
    };

    // [Stage 47.2] Prunes are sent to ALL neighbors because we don't track 
    // which neighbor has which ghost for a local axon (Routing is dynamic).
    let mut targets = Vec::new();
    if let Some(ref addr) = _shard_config.neighbors.x_plus { targets.push(addr.clone()); }
    if let Some(ref addr) = _shard_config.neighbors.x_minus { targets.push(addr.clone()); }
    if let Some(ref addr) = _shard_config.neighbors.y_plus { targets.push(addr.clone()); }
    if let Some(ref addr) = _shard_config.neighbors.y_minus { targets.push(addr.clone()); }
    if let Some(ref addr) = _shard_config.neighbors.z_plus { targets.push(addr.clone()); }
    if let Some(ref addr) = _shard_config.neighbors.z_minus { targets.push(addr.clone()); }

    for axon_id in prunes_slice {
        let axon_id = *axon_id;
        for addr_str in &targets {
            let addr_str = addr_str.clone();
            rt_handle.spawn(async move {
                if let Ok(addr) = addr_str.parse::<std::net::SocketAddr>() {
                    let req = crate::network::slow_path::GeometryRequest::Prune(axon_id);
                    let _ = crate::network::geometry_client::send_geometry_request(addr, &req).await;
                }
            });
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
) {
    let sync_batch_ticks = 100u32;
    let max_spikes_per_tick = 100_000u32;
    let output_bytes = (desc.num_outputs * sync_batch_ticks) as usize;

    let hash = desc.hash;

    thread::Builder::new()
        .name(format!("compute-{}", hash))
        .spawn(move || {
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

            // [DOD] ЕДИНСТВЕННАЯ аллокация на весь жизненный цикл потока
            let mut workspace = ThreadWorkspace::new(hash, padded_n);
            let axons_size = desc.engine.vram.total_axons as usize * std::mem::size_of::<u32>();
            workspace.checkpoint_state_buffer = vec![0u8; state_size];
            workspace.checkpoint_axons_buffer = vec![0u8; axons_size];

            let mut batch_counter: u64 = 0;

            // 2. Плоский горячий цикл
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    ComputeCommand::Resurrect { zone_hash } => {
                        println!("⚡ [Shard {:08X}] Resurrecting from shadow replica...", zone_hash);
                        
                        let shadow_path = format!("/dev/shm/0x{:08X}.shadow", zone_hash);
                        if let Ok(file) = std::fs::File::open(&shadow_path) {
                            let mmap = unsafe { memmap2::Mmap::map(&file).expect("Mmap failed") };
                            let _header = unsafe { &*(mmap.as_ptr() as *const genesis_core::ipc::ShardStateHeader) };
                            
                            // Load weights into GPU (weights start 32 bytes after ShardStateHeader)
                            let current_padded_n = desc.engine.vram.padded_n as usize;
                            let weights_ptr = unsafe { mmap.as_ptr().add(32) as *const i16 };
                            
                            unsafe {
                                genesis_compute::ffi::gpu_memcpy_host_to_device(
                                    desc.engine.vram.ptrs.dendrite_weights as *mut _,
                                    weights_ptr as *const _,
                                    current_padded_n * 128 * std::mem::size_of::<i16>(),
                                );
                            }

                            // [NEW Phase 50] High-Precision Warmup
                            // Выполняем 100 тиков "тихой" симуляции (без внешних выходов)
                            for _ in 0..100 {
                                execute_day_phase(
                                    &mut desc.engine,
                                    1, // tick by tick
                                    0, // global_dopamine
                                    &ctx.bsp_barrier,
                                    &ctx.io_ctx,
                                    &io_buffers,
                                    desc.virtual_offset,
                                    desc.num_virtual_axons,
                                    mapped_soma_ids,
                                    desc.v_seg,
                                    batch_counter
                                );
                            }
                            println!("✅ [Shard {:08X}] Warmup complete. Shard is LIVE.", zone_hash);
                        } else {
                            eprintln!("🔴 [Shard {:08X}] Shadow file NOT FOUND at {}", zone_hash, shadow_path);
                        }
                    }
                    ComputeCommand::RunBatch { tick_base: _, batch_size, global_dopamine, telemetry_ptrs } => {
                        // ФАЗА 1: Выполнение GPU батча (Day Phase)
                        execute_day_phase(
                            &mut desc.engine, batch_size, global_dopamine, &ctx.bsp_barrier,
                            &ctx.io_ctx, &io_buffers, desc.virtual_offset, desc.num_virtual_axons, mapped_soma_ids, desc.v_seg, batch_counter
                        );

                        if let Some((ids_ptr, count_ptr)) = telemetry_ptrs {
                            unsafe {
                                std::ptr::write_volatile(count_ptr as *mut u32, 0);
                                genesis_compute::ffi::launch_extract_telemetry(
                                    &desc.engine.vram.ptrs,
                                    desc.engine.vram.padded_n,
                                    ids_ptr as *mut u32,
                                    count_ptr as *mut u32,
                                    std::ptr::null_mut(), // stream 0
                                );
                            }
                        }

                        // ФАЗА 2: Чтение выходов
                        download_outputs(desc.num_outputs, &mut pinned_out, &io_buffers, output_bytes);

                        // ФАЗА 3: Периодический сброс на диск (I/O)
                        if batch_counter > 0 && batch_counter % 500 == 0 {
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
                        if ctx.night_interval > 0 && current_tick_count % ctx.night_interval == 0 {
                            execute_night_phase(
                                &mut desc.engine, hash, std::path::Path::new(&socket_path), &mut baker_client,
                                &ctx.incoming_grow, &desc.config, &ctx.rt_handle, &mut workspace, &desc.baked_dir,
                                &desc.incoming_prune, mapped_soma_ids,
                            );
                        }

                        // Отправка отчета оркестратору
                        if f_tx.send(ComputeFeedback::BatchComplete {
                            ticks_processed: batch_size,
                            zone_hash: hash,
                            pinned_out_ptr: pinned_out.as_ptr() as usize,
                            output_bytes,
                        }).is_err() { break; }

                        batch_counter += 1;
                    }
                    ComputeCommand::Shutdown => break,
                }
            }
        }).expect("Failed to spawn compute thread");
}

