use super::{ComputeCommand, ComputeFeedback};
use crate::network::bsp::BspBarrier;
use crate::network::io_server::InputSwapchain;
use axicor_compute::ShardEngine;
use axicor_core::config::InstanceConfig;
use axicor_core::ipc::{shm_size, AxonHandoverEvent, ShmHeader};
use crossbeam::channel::Receiver;
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::ptr;
use std::sync::Arc;
use std::thread;
use tracing::{error, info, warn};

/// [Phase 23] Static shard geometry/physics owns all per-shard data.
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

use std::sync::atomic::{AtomicI16, AtomicU16, AtomicU64, Ordering};

// TODO: Find ideal balance for linear stable growth, then approximate and
// embed neurogenesis calculation for each shard automatically based on internal types.
pub struct ShardAtomicSettings {
    pub night_interval_ticks: AtomicU64,
    pub save_checkpoints_interval_ticks: AtomicU64, // ticks counter
    pub prune_threshold: AtomicI16,
    pub max_sprouts: AtomicU16,
}

/// [Phase 23] Shared orchestrator resources  cheap Clone via Arc.
#[derive(Clone)]
pub struct NodeContext {
    pub bsp_barrier: Arc<BspBarrier>,
    pub io_ctx: Arc<InputSwapchain>,
    pub rt_handle: tokio::runtime::Handle,
    pub atomic_settings: Arc<ShardAtomicSettings>,
    pub incoming_grow: Arc<crossbeam::queue::SegQueue<AxonHandoverEvent>>,
    pub routing_table: Arc<crate::network::router::RoutingTable>, // [DOD FIX]
}
pub struct ThreadWorkspace {
    pub weights_offset: usize,
    pub targets_offset: usize,
    pub handovers_offset: usize,
    pub flags_offset: usize,
    pub voltage_offset: usize,
    pub threshold_offset_offset: usize,
    pub timers_offset: usize,
    pub shm_buffer: MmapMut,
    pub checkpoint_state_buffer: Vec<u8>,
    pub checkpoint_axons_buffer: Vec<u8>, 
    pub ghost_origins: Vec<u32>,          
    // [DOD FIX] Ephys GPU Pointers and SHM Mapping
    pub ephys_mmap: Option<memmap2::MmapMut>,
    pub d_ephys_tids: *mut u32,
    pub d_ephys_uvs: *mut i32,
    pub d_ephys_trace: *mut i32,
    pub cpu_ephys_tids: Vec<u32>,
    pub cpu_ephys_uvs: Vec<i32>,
    pub cpu_ephys_trace: Vec<i32>,
    pub use_gpu: bool,
}

impl ThreadWorkspace {
    pub fn new(zone_hash: u32, padded_n: usize, total_ghosts: usize, use_gpu: bool) -> Self {
        let path = axicor_core::ipc::shm_file_path(zone_hash);
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
                let mut res = Self {
                    weights_offset: header.weights_offset as usize,
                    targets_offset: header.targets_offset as usize,
                    handovers_offset: header.handovers_offset as usize,
                    flags_offset: header.flags_offset as usize,
                    voltage_offset: header.voltage_offset as usize,
                    threshold_offset_offset: header.threshold_offset_offset as usize,
                    timers_offset: header.timers_offset as usize,
                    shm_buffer: mmap,
                    checkpoint_state_buffer: vec![0u8; 0],
                    checkpoint_axons_buffer: vec![0u8; 0],
                    ghost_origins: vec![0u32; total_ghosts],
                    ephys_mmap: {
                        let path = axicor_core::ipc::ephys_shm_path(zone_hash);
                        if let Ok(file) = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path) {
                            let size = std::mem::size_of::<axicor_core::ipc::EphysShm>() as u64;
                            if file.metadata().map(|m| m.len()).unwrap_or(0) < size {
                                let _ = file.set_len(size);
                            }
                            unsafe { memmap2::MmapMut::map_mut(&file).ok() }
                        } else { None }
                    },
                    d_ephys_tids: std::ptr::null_mut(),
                    d_ephys_uvs: std::ptr::null_mut(),
                    d_ephys_trace: std::ptr::null_mut(),
                    cpu_ephys_tids: vec![0u32; 16],
                    cpu_ephys_uvs: vec![0i32; 16],
                    cpu_ephys_trace: vec![0i32; 16 * 10000],
                    use_gpu,
                };
                if use_gpu {
                    res.d_ephys_tids = unsafe { axicor_compute::ffi::gpu_malloc(16 * 4) as *mut u32 };
                    res.d_ephys_uvs = unsafe { axicor_compute::ffi::gpu_malloc(16 * 4) as *mut i32 };
                    res.d_ephys_trace = unsafe { axicor_compute::ffi::gpu_malloc(16 * 10000 * 4) as *mut i32 };
                } else {
                    res.d_ephys_tids = res.cpu_ephys_tids.as_mut_ptr();
                    res.d_ephys_uvs = res.cpu_ephys_uvs.as_mut_ptr();
                    res.d_ephys_trace = res.cpu_ephys_trace.as_mut_ptr();
                }
                return res;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        panic!(
            "SHM file {:?} not ready after 3s (is axicor-baker-daemon running?)",
            path
        );
    }

    pub fn weights_slice_mut(&mut self, padded_n: usize) -> &mut [i32] {
        // [DOD FIX] i32
        let len = padded_n * 128;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.shm_buffer.as_mut_ptr().add(self.weights_offset) as *mut i32, // [DOD FIX] i32
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
                self.shm_buffer.as_mut_ptr().add(self.flags_offset),
                padded_n,
            )
        }
    }

    pub fn voltage_slice_mut(&mut self, padded_n: usize) -> &mut [i32] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.shm_buffer.as_mut_ptr().add(self.voltage_offset) as *mut i32,
                padded_n,
            )
        }
    }

    pub fn threshold_offset_slice_mut(&mut self, padded_n: usize) -> &mut [i32] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.shm_buffer
                    .as_mut_ptr()
                    .add(self.threshold_offset_offset) as *mut i32,
                padded_n,
            )
        }
    }

    pub fn timers_slice_mut(&mut self, padded_n: usize) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.shm_buffer.as_mut_ptr().add(self.timers_offset),
                padded_n,
            )
        }
    }
}

// PHASE 1: GPU batch execution (Day Phase)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_div_ceil)]
#[inline(always)]
fn execute_day_phase(
    shard: &mut ShardEngine,
    batch_size: u32,
    global_dopamine: i16,
    bsp_barrier: &Arc<BspBarrier>,
    my_io_ctx: &Arc<InputSwapchain>,
    io_buffers: &mut axicor_compute::compute::shard::IoBuffers,
    virtual_offset: u32,
    num_virtual_axons: u32,
    mapped_soma_ids: *const u32,
    v_seg: u32,
    _batch_counter: u64,
    tick_base: u32,
    workspace: &mut ThreadWorkspace, // [DOD FIX]
) {
    let _sync_batch_ticks = 100u32;
    let input_words_per_tick = (num_virtual_axons + 63) / 64 * 2;

    let schedule = bsp_barrier.get_read_schedule();

    // [DOD FIX] Zero-cost, 100% safe slice extraction from Pinned RAM
    let incoming_slice = schedule.ghost_ids.as_slice();
    let counts_slice_atomic = schedule.counts.as_slice();
    let counts_slice = unsafe {
        // AtomicU32 and u32 have identical memory layout.
        std::slice::from_raw_parts(
            counts_slice_atomic.as_ptr() as *const u32,
            counts_slice_atomic.len(),
        )
    };

    // [DOD FIX] Ephys SHM Orchestration (Zero-Copy Host-to-Device)
    if let Some(mmap) = &mut workspace.ephys_mmap {
        let ephys = unsafe { &mut *(mmap.as_mut_ptr() as *mut axicor_core::ipc::EphysShm) };
        if ephys.magic == axicor_core::ipc::EPHYS_MAGIC && ephys.state == 1 { // Trigger command from Python SDK
            let count = ephys.count.min(16);
            match shard {
                ShardEngine::Gpu(_) => unsafe {
                    axicor_compute::ffi::gpu_memcpy_host_to_device(workspace.d_ephys_tids as *mut _, ephys.target_tids.as_ptr() as *const _, (count * 4) as usize);
                    axicor_compute::ffi::gpu_memcpy_host_to_device(workspace.d_ephys_uvs as *mut _, ephys.injection_uv.as_ptr() as *const _, (count * 4) as usize);
                },
                ShardEngine::Cpu(_) => unsafe {
                    std::ptr::copy_nonoverlapping(ephys.target_tids.as_ptr(), workspace.d_ephys_tids, count as usize);
                    std::ptr::copy_nonoverlapping(ephys.injection_uv.as_ptr(), workspace.d_ephys_uvs, count as usize);
                }
            }
            shard.set_ephys_state(Some(axicor_compute::compute::shard::EphysState {
                tids_d: workspace.d_ephys_tids,
                uvs_d: workspace.d_ephys_uvs,
                trace_d: workspace.d_ephys_trace,
                count,
                max_ticks: ephys.max_ticks.min(10000),
                current_tick: 0,
            }));
            ephys.state = 2; // Mark Busy
        }
    }

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

    unsafe {
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

    // [DOD FIX] Ephys SHM Output (Zero-Copy Device-to-Host)
    if let Some(mmap) = &mut workspace.ephys_mmap {
        let ephys = unsafe { &mut *(mmap.as_mut_ptr() as *mut axicor_core::ipc::EphysShm) };
        if ephys.state == 2 {
            if let Some(es) = shard.get_ephys_state() {
                ephys.current_tick = es.current_tick;
                if es.current_tick >= es.max_ticks {
                    let bytes = (es.count * es.max_ticks * 4) as usize;
                    match shard {
                        ShardEngine::Gpu(_) => unsafe {
                            axicor_compute::ffi::gpu_memcpy_device_to_host(
                                ephys.out_trace.as_mut_ptr() as *mut _,
                                workspace.d_ephys_trace as *const _,
                                bytes
                            );
                        },
                        ShardEngine::Cpu(_) => unsafe {
                            std::ptr::copy_nonoverlapping(
                                workspace.d_ephys_trace as *const _,
                                ephys.out_trace.as_mut_ptr() as *mut _,
                                (es.count * es.max_ticks) as usize
                            );
                        }
                    }
                    ephys.state = 3; // Mark Done
                    shard.set_ephys_state(None);
                }
            }
        }
    }
}

fn download_outputs(
    num_outputs: u32,
    pinned_out: &mut axicor_compute::memory::PinnedBuffer<u8>,
    io_buffers: &axicor_compute::compute::shard::IoBuffers,
    output_bytes: usize,
    engine: &axicor_compute::ShardEngine,
) {
    if num_outputs > 0 {
        match engine {
            axicor_compute::ShardEngine::Gpu(gpu) => {
                if let axicor_compute::compute::shard::IoBackend::Gpu(ref b) = io_buffers.backend {
                    unsafe {
                        axicor_compute::ffi::cu_dma_d2h_io(
                            pinned_out.as_mut_ptr(),
                            b.d_output_history,
                            output_bytes as u32,
                            gpu.stream,
                        );
                        axicor_compute::ffi::gpu_stream_synchronize(gpu.stream);
                    }
                }
            }
            axicor_compute::ShardEngine::Cpu(_) => {
                if let axicor_compute::compute::shard::IoBackend::Cpu(ref b) = io_buffers.backend {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            b.h_output_history.as_ptr(),
                            pinned_out.as_mut_ptr(),
                            output_bytes,
                        );
                    }
                }
            }
        }
    }
}



// PHASE 4: Graph maintenance (Night Phase)
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn execute_night_phase(
    desc: &mut ShardDescriptor,
    hash: u32,
    socket_path: &std::path::Path,
    baker_client: &mut Option<crate::ipc::BakerClient>,
    rt_handle: &tokio::runtime::Handle,
    workspace: &mut ThreadWorkspace,
    prune_threshold: i16,
    max_sprouts: u16,
    routing_table: &Arc<crate::network::router::RoutingTable>,
) {
    let vram = match &desc.engine {
        ShardEngine::Gpu(gpu) => &gpu.vram,
        ShardEngine::Cpu(cpu) => &cpu.vram,
    };
    let padded_n = vram.padded_n as usize;
    let dendrites_count = padded_n * axicor_core::constants::MAX_DENDRITE_SLOTS;

    match &mut desc.engine {
        ShardEngine::Gpu(ref mut gpu) => unsafe {
            axicor_compute::ffi::gpu_memcpy_device_to_host(
                workspace.flags_slice_mut(padded_n).as_mut_ptr() as *mut _,
                gpu.vram.ptrs.soma_flags as *const _,
                padded_n,
            );
            axicor_compute::ffi::gpu_memcpy_device_to_host(
                workspace.voltage_slice_mut(padded_n).as_mut_ptr() as *mut _,
                gpu.vram.ptrs.soma_voltage as *const _,
                padded_n * 4,
            );
            axicor_compute::ffi::gpu_memcpy_device_to_host(
                workspace.threshold_offset_slice_mut(padded_n).as_mut_ptr() as *mut _,
                gpu.vram.ptrs.threshold_offset as *const _,
                padded_n * 4,
            );
            axicor_compute::ffi::gpu_memcpy_device_to_host(
                workspace.timers_slice_mut(padded_n).as_mut_ptr() as *mut _,
                gpu.vram.ptrs.timers as *const _,
                padded_n,
            );

            axicor_compute::ffi::launch_sort_and_prune(
                &gpu.vram.ptrs,
                gpu.vram.padded_n,
                prune_threshold,
            );
            axicor_compute::ffi::gpu_device_synchronize();

            axicor_compute::ffi::gpu_memcpy_device_to_host(
                workspace.weights_slice_mut(padded_n).as_mut_ptr() as *mut _,
                gpu.vram.ptrs.dendrite_weights as *const _,
                dendrites_count * 4,
            );
            axicor_compute::ffi::gpu_memcpy_device_to_host(
                workspace.targets_slice_mut(padded_n).as_mut_ptr() as *mut _,
                gpu.vram.ptrs.dendrite_targets as *const _,
                dendrites_count * 4,
            );
        },
        ShardEngine::Cpu(ref mut _cpu) => unsafe {
            axicor_compute::ffi::launch_sort_and_prune(
                &_cpu.vram.ptrs,
                padded_n as u32,
                prune_threshold,
            );
            std::ptr::copy_nonoverlapping(
                _cpu.vram.ptrs.soma_flags,
                workspace.flags_slice_mut(padded_n).as_mut_ptr(),
                padded_n,
            );
            std::ptr::copy_nonoverlapping(
                _cpu.vram.ptrs.soma_voltage as *const u8,
                workspace.voltage_slice_mut(padded_n).as_mut_ptr() as *mut u8,
                padded_n * 4,
            );
            std::ptr::copy_nonoverlapping(
                _cpu.vram.ptrs.threshold_offset as *const u8,
                workspace.threshold_offset_slice_mut(padded_n).as_mut_ptr() as *mut u8,
                padded_n * 4,
            );
            std::ptr::copy_nonoverlapping(
                _cpu.vram.ptrs.timers,
                workspace.timers_slice_mut(padded_n).as_mut_ptr(),
                padded_n,
            );

            std::ptr::copy_nonoverlapping(
                _cpu.vram.ptrs.dendrite_weights as *const u8,
                workspace.weights_slice_mut(padded_n).as_mut_ptr() as *mut u8,
                dendrites_count * 4,
            );
            std::ptr::copy_nonoverlapping(
                _cpu.vram.ptrs.dendrite_targets as *const u8,
                workspace.targets_slice_mut(padded_n).as_mut_ptr() as *mut u8,
                dendrites_count * 4,
            );
        },
    }

    // 3. Sprouting (Late Binding)
    if baker_client.is_none() {
        *baker_client = crate::ipc::BakerClient::connect(hash, socket_path).ok();
    }

    if let Some(client) = baker_client.as_mut() {
        let mut incoming_handovers = Vec::new();
        while let Some(ev) = desc.incoming_grow.pop() {
            incoming_handovers.push(ev);
            if incoming_handovers.len() >= axicor_core::ipc::MAX_HANDOVERS_PER_NIGHT {
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
                match &mut desc.engine {
                    ShardEngine::Gpu(gpu) => unsafe {
                        axicor_compute::ffi::gpu_memcpy_host_to_device(
                            gpu.vram.ptrs.dendrite_targets as *mut _,
                            workspace.targets_slice_mut(padded_n).as_ptr() as *const _,
                            dendrites_count * std::mem::size_of::<u32>(),
                        );
                        axicor_compute::ffi::gpu_memcpy_host_to_device(
                            gpu.vram.ptrs.dendrite_weights as *mut _,
                            workspace.weights_slice_mut(padded_n).as_ptr() as *const _,
                            dendrites_count * std::mem::size_of::<i32>(), // Mass Domain Law (i32)
                        );
                        axicor_compute::ffi::gpu_memcpy_host_to_device(
                            gpu.vram.ptrs.soma_voltage as *mut _,
                            workspace.voltage_slice_mut(padded_n).as_ptr() as *const _,
                            padded_n * std::mem::size_of::<i32>(),
                        );
                        axicor_compute::ffi::gpu_memcpy_host_to_device(
                            gpu.vram.ptrs.soma_flags as *mut _,
                            workspace.flags_slice_mut(padded_n).as_ptr() as *const _,
                            padded_n * std::mem::size_of::<u8>(),
                        );
                        axicor_compute::ffi::gpu_memcpy_host_to_device(
                            gpu.vram.ptrs.threshold_offset as *mut _,
                            workspace.threshold_offset_slice_mut(padded_n).as_ptr() as *const _,
                            padded_n * std::mem::size_of::<i32>(),
                        );
                        axicor_compute::ffi::gpu_memcpy_host_to_device(
                            gpu.vram.ptrs.timers as *mut _,
                            workspace.timers_slice_mut(padded_n).as_ptr() as *const _,
                            padded_n * std::mem::size_of::<u8>(),
                        );
                        axicor_compute::ffi::gpu_device_synchronize();
                    },
                    ShardEngine::Cpu(cpu) => unsafe {
                        std::ptr::copy_nonoverlapping(
                            workspace.targets_slice_mut(padded_n).as_ptr(),
                            cpu.vram.ptrs.dendrite_targets,
                            dendrites_count,
                        );
                        std::ptr::copy_nonoverlapping(
                            workspace.weights_slice_mut(padded_n).as_ptr(),
                            cpu.vram.ptrs.dendrite_weights,
                            dendrites_count,
                        );
                        std::ptr::copy_nonoverlapping(
                            workspace.voltage_slice_mut(padded_n).as_ptr(),
                            cpu.vram.ptrs.soma_voltage,
                            padded_n,
                        );
                        std::ptr::copy_nonoverlapping(
                            workspace.flags_slice_mut(padded_n).as_ptr(),
                            cpu.vram.ptrs.soma_flags,
                            padded_n,
                        );
                        std::ptr::copy_nonoverlapping(
                            workspace.threshold_offset_slice_mut(padded_n).as_ptr(),
                            cpu.vram.ptrs.threshold_offset,
                            padded_n,
                        );
                        std::ptr::copy_nonoverlapping(
                            workspace.timers_slice_mut(padded_n).as_ptr(),
                            cpu.vram.ptrs.timers,
                            padded_n,
                        );
                    },
                }

                dispatch_handovers(client, &desc.config, rt_handle);

                // [DOD FIX] Correct offset calculation for Ghost Origin Tracking
                for ack in &acks {
                    let idx = (ack.dst_ghost_id as usize).saturating_sub(padded_n + desc.num_virtual_axons as usize);
                    if idx < workspace.ghost_origins.len() {
                        workspace.ghost_origins[idx] = ack.target_zone_hash;
                    }
                }

                dispatch_acks(acks, rt_handle, routing_table);

                // [DOD FIX] Read GC cleans from SHM and route deaths
                dispatch_prunes(
                    &desc.engine,
                    client,
                    &workspace.ghost_origins,
                    padded_n,
                    rt_handle,
                    routing_table,
                );

                info!(" [Shard {:08X}] Night Phase complete. Waking up.", hash);
            }
            Err(e) => {
                error!("[ERROR] [Shard {:08X}] Sprouting failed: {}", hash, e);
                *baker_client = None;
            }
        }
    } else {
        warn!("[WARN] [Shard {:08X}] Skipping Sprouting (Daemon not connected). Will retry next night.", hash);
    }
}

// Abstraction to send Spatial Geometry Routing within Night Phase
#[inline(always)]
fn dispatch_handovers(
    client: &crate::ipc::BakerClient,
    shard_config: &InstanceConfig,
    rt_handle: &tokio::runtime::Handle,
) {
    let shm_hdr = unsafe { std::ptr::read(client.shm_ptr as *const axicor_core::ipc::ShmHeader) };
    let handovers_slice = unsafe {
        std::slice::from_raw_parts(
            client.shm_ptr.add(shm_hdr.handovers_offset as usize)
                as *const crate::network::slow_path::AxonHandoverEvent,
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
    let max_z = shard_config.dimensions.h as u8;

    for ev in handovers_slice {
        if ev.entry_x >= max_x {
            x_plus.push(*ev);
        } else if ev.entry_x == 0 {
            x_minus.push(*ev);
        } else if ev.entry_y >= max_y {
            y_plus.push(*ev);
        } else if ev.entry_y == 0 {
            y_minus.push(*ev);
        } else if ev.entry_z >= max_z {
            z_plus.push(*ev);
        } else if ev.entry_z == 0 {
            z_minus.push(*ev);
        }
    }

    let mut routes_to_execute: Vec<(String, Vec<crate::network::slow_path::AxonHandoverEvent>)> =
        Vec::new();
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

fn dispatch_acks(
    acks: Vec<axicor_core::ipc::AxonHandoverAck>,
    rt_handle: &tokio::runtime::Handle,
    routing_table: &Arc<crate::network::router::RoutingTable>,
) {
    if acks.is_empty() {
        return;
    }

    let mut grouped: std::collections::HashMap<u32, Vec<axicor_core::ipc::AxonHandoverAck>> =
        std::collections::HashMap::new();
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
    shard: &axicor_compute::ShardEngine,
    client: &crate::ipc::BakerClient,
    ghost_origins: &[u32],
    padded_n: usize,
    rt_handle: &tokio::runtime::Handle,
    routing_table: &Arc<crate::network::router::RoutingTable>,
) {
    let shm_hdr = unsafe { std::ptr::read(client.shm_ptr as *const ShmHeader) };
    if shm_hdr.prunes_count == 0 {
        return;
    }

    let prunes_slice = unsafe {
        std::slice::from_raw_parts(
            client.shm_ptr.add(shm_hdr.prunes_offset as usize)
                as *const axicor_core::ipc::AxonHandoverPrune,
            shm_hdr.prunes_count as usize,
        )
    };

    for prune in prunes_slice {
        let ghost_id = prune.dst_ghost_id as usize;
        let idx = ghost_id.saturating_sub(padded_n);

        let empty_burst =
            axicor_core::layout::BurstHeads8::empty(axicor_core::constants::AXON_SENTINEL);
        match shard {
            ShardEngine::Gpu(gpu) => unsafe {
                axicor_compute::ffi::gpu_memcpy_host_to_device(
                    gpu.vram.ptrs.axon_heads.add(ghost_id) as *mut _,
                    &empty_burst as *const _ as *const _,
                    32,
                );
            },
            ShardEngine::Cpu(cpu) => unsafe {
                *cpu.vram.ptrs.axon_heads.add(ghost_id) = empty_burst;
            },
        }

        if idx < ghost_origins.len() {
            let target_hash = ghost_origins[idx];
            if target_hash != 0 {
                if let Some((addr, _mtu)) = routing_table.get_address(target_hash) {
                    let gid = prune.dst_ghost_id;
                    rt_handle.spawn(async move {
                        let req = crate::network::slow_path::GeometryRequest::Prune(gid);
                        let _ = crate::network::geometry_client::send_geometry_request(addr, &req)
                            .await;
                    });
                }
            }
        }
    }
}
// VRAM buffers initialization
#[allow(clippy::manual_div_ceil)]
fn init_io_buffers(
    num_virtual_axons: u32,
    max_spikes_per_tick: u32,
    num_outputs: u32,
    sync_batch_ticks: u32,
    use_gpu: bool,
) -> axicor_compute::compute::shard::IoBuffers {
    let input_words_per_tick = (num_virtual_axons + 63) / 64 * 2;
    if use_gpu {
        let mut d_input = ptr::null_mut();
        let mut d_spikes = ptr::null_mut();
        let mut d_output = ptr::null_mut();
        unsafe {
            axicor_compute::ffi::cu_allocate_io_buffers(
                input_words_per_tick * sync_batch_ticks,
                max_spikes_per_tick * sync_batch_ticks,
                num_outputs * sync_batch_ticks,
                &mut d_input,
                &mut d_spikes,
                &mut d_output,
            );
        }
        axicor_compute::compute::shard::IoBuffers {
            backend: axicor_compute::compute::shard::IoBackend::Gpu(
                axicor_compute::compute::shard::GpuIoBuffers {
                    d_input_bitmask: d_input,
                    d_incoming_spikes: d_spikes,
                    d_output_history: d_output,
                },
            ),
            max_spikes_per_tick,
            input_words_per_tick,
            num_outputs,
        }
    } else {
        axicor_compute::compute::shard::IoBuffers {
            backend: axicor_compute::compute::shard::IoBackend::Cpu(
                axicor_compute::compute::shard::CpuIoBuffers {
                    h_input_bitmask: vec![0u32; (input_words_per_tick * sync_batch_ticks) as usize],
                    h_incoming_spikes: vec![
                        0u32;
                        (max_spikes_per_tick * sync_batch_ticks) as usize
                    ],
                    h_output_history: vec![0u8; (num_outputs * sync_batch_ticks) as usize],
                },
            ),
            max_spikes_per_tick,
            input_words_per_tick,
            num_outputs,
        }
    }
}

#[allow(clippy::uninit_vec)]
pub fn spawn_shard_thread(
    mut desc: ShardDescriptor,
    ctx: NodeContext,
    rx: Receiver<ComputeCommand>,
    f_tx: crossbeam::channel::Sender<ComputeFeedback>,
    _core_id: usize,
    sync_batch_ticks: u32,
) -> std::thread::JoinHandle<()> {
    let max_spikes_per_tick = 100_000u32;
    let output_bytes = (desc.num_outputs * sync_batch_ticks) as usize;

    let hash = desc.hash;

    thread::Builder::new()
        .name(format!("compute-{}", hash))
        .spawn(move || {
            // [DOD FIX] Hardware Sympathy: Pinning compute thread
            #[cfg(target_os = "linux")]
            {
                let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
                unsafe { libc::CPU_SET(_core_id, &mut cpuset) };
                let res = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset) };
                if res != 0 {
                    warn!("Failed to set thread affinity to core {}", _core_id);
                } else {
                    info!(" [Core] Shard 0x{:08X} compute locked to OS Thread Core {}", hash, _core_id);
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                info!(" [Core] Shard 0x{:08X} compute thread spawned (affinity not supported on this OS)", hash);
            }

            // 1. Hardware context initialization
            let use_gpu = matches!(desc.engine, ShardEngine::Gpu(_));
            if use_gpu {
                unsafe { axicor_compute::ffi::gpu_set_device(0); }
            }

            let (mapped_soma_ids, _gpu_ptr) = if let Some(ref host_ids) = desc.mapped_soma_ids_host {
                if use_gpu {
                    let bytes = host_ids.len() * 4;
                    let ptr = unsafe { axicor_compute::ffi::gpu_malloc(bytes) } as *mut u32;
                    unsafe {
                        axicor_compute::ffi::gpu_memcpy_host_to_device(
                            ptr as *mut _,
                            host_ids.as_ptr() as *const _,
                            bytes,
                        );
                    }
                    (ptr as *const u32, Some(ptr))
                } else {
                    (host_ids.as_ptr(), None)
                }
            } else {
                (std::ptr::null(), None)
            };

            let mut io_buffers = init_io_buffers(desc.num_virtual_axons, max_spikes_per_tick, desc.num_outputs, sync_batch_ticks, use_gpu);
            let mut pinned_out = axicor_compute::memory::PinnedBuffer::<u8>::new(output_bytes).unwrap();
            let mut baker_client: Option<crate::ipc::BakerClient> = None;
            let socket_path = axicor_core::ipc::default_socket_path(hash);
            
            let vram = match desc.engine {
                ShardEngine::Gpu(ref gpu) => &gpu.vram,
                ShardEngine::Cpu(ref cpu) => &cpu.vram,
            };
            let padded_n = vram.padded_n as usize;
            let _dendrites_count = padded_n * axicor_core::constants::MAX_DENDRITE_SLOTS;
            let (_, state_size) = axicor_compute::memory::calculate_state_blob_size(padded_n);

            // [DOD FIX] Align payload size with Baker Daemon expectations
            let mut workspace = ThreadWorkspace::new(hash, padded_n, vram.total_ghosts as usize, use_gpu);
            let axons_size = vram.total_axons as usize * std::mem::size_of::<axicor_core::layout::BurstHeads8>();
            workspace.checkpoint_state_buffer = vec![0u8; state_size];
            workspace.checkpoint_axons_buffer = vec![0u8; axons_size];

            let mut batch_counter: u64 = 0;
            let mut warmup_ticks_remaining: u32 = 2000;

            // 2. Flat hot loop
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    ComputeCommand::Resurrect => {
                        warmup_ticks_remaining = 100;
                        info!("Entering Warmup Phase (100 ticks) for 0x{:08X}", hash);
                    }
                    ComputeCommand::RunBatch { tick_base, batch_size, global_dopamine } => {
                        let is_warmup = warmup_ticks_remaining > 0;
                        
                        // PHASE 1: GPU batch execution (Day Phase)
                        execute_day_phase(
                            &mut desc.engine, batch_size, global_dopamine, &ctx.bsp_barrier,
                            &ctx.io_ctx, &mut io_buffers, desc.virtual_offset, desc.num_virtual_axons, mapped_soma_ids, desc.v_seg, batch_counter, tick_base, &mut workspace
                        );

                        // --- PHASE 2: Read outputs ---
                        if desc.num_outputs > 0 {
                            download_outputs(desc.num_outputs, &mut pinned_out, &io_buffers, output_bytes, &desc.engine);
                        }

                        if is_warmup {
                            warmup_ticks_remaining = warmup_ticks_remaining.saturating_sub(batch_size);
                            if warmup_ticks_remaining == 0 {
                                info!("Warmup complete for 0x{:08X}. Voltage stabilized.", hash);
                            }
                        }

                        // PHASE 3: Periodic disk flush (I/O)
                        let cp_interval_ticks = ctx.atomic_settings.save_checkpoints_interval_ticks.load(Ordering::Relaxed);
                        let cp_interval = (cp_interval_ticks as u32 / batch_size).max(1);
                        if batch_counter > 0 && batch_counter.is_multiple_of(cp_interval as u64) {
                            match &desc.engine {
                                axicor_compute::ShardEngine::Gpu(gpu) => unsafe {
                                    axicor_compute::ffi::gpu_memcpy_device_to_host(
                                        workspace.checkpoint_state_buffer.as_mut_ptr() as *mut _,
                                        gpu.vram.ptrs.soma_voltage as *const _,
                                        workspace.checkpoint_state_buffer.len(),
                                    );
                                    axicor_compute::ffi::gpu_memcpy_device_to_host(
                                        workspace.checkpoint_axons_buffer.as_mut_ptr() as *mut _,
                                        gpu.vram.ptrs.axon_heads as *const _,
                                        workspace.checkpoint_axons_buffer.len(),
                                    );
                                    axicor_compute::ffi::gpu_device_synchronize();
                                },
                                axicor_compute::ShardEngine::Cpu(cpu) => unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        cpu.vram.ptrs.soma_voltage as *const u8,
                                        workspace.checkpoint_state_buffer.as_mut_ptr(),
                                        workspace.checkpoint_state_buffer.len(),
                                    );
                                    std::ptr::copy_nonoverlapping(
                                        cpu.vram.ptrs.axon_heads as *const u8,
                                        workspace.checkpoint_axons_buffer.as_mut_ptr(),
                                        workspace.checkpoint_axons_buffer.len(),
                                    );
                                },
                            }

                            // Take filled vectors (Ownership Move), and put empty vectors into Workspace
                            // with already allocated capacity, to avoid reallocations as they fill up.
                            let state_size = workspace.checkpoint_state_buffer.capacity();
                            let axons_size = workspace.checkpoint_axons_buffer.capacity();

                            let mut new_state = Vec::with_capacity(state_size);
                            unsafe { new_state.set_len(state_size); }
                            let mut new_axons = Vec::with_capacity(axons_size);
                            unsafe { new_axons.set_len(axons_size); }

                            let state_data = std::mem::replace(
                                &mut workspace.checkpoint_state_buffer, 
                                new_state
                            );
                            let axons_data = std::mem::replace(
                                &mut workspace.checkpoint_axons_buffer, 
                                new_axons
                            );

                            let payload = crate::node::storage::CheckpointPayload {
                                zone_hash: hash,
                                baked_dir: desc.baked_dir.clone(),
                                state_data,
                                axons_data,
                            };

                            crate::node::storage::dispatch_checkpoint(payload, &ctx.rt_handle);
                        }

                        // PHASE 4: Graph maintenance (Night Phase)
                        let current_tick_count = (batch_counter + 1) * batch_size as u64;
                        let n_interval = ctx.atomic_settings.night_interval_ticks.load(Ordering::Relaxed);
                        if n_interval > 0 && current_tick_count.is_multiple_of(n_interval) {
                            let current_prune_threshold = ctx.atomic_settings.prune_threshold.load(Ordering::Relaxed);
                            let current_max_sprouts = ctx.atomic_settings.max_sprouts.load(Ordering::Relaxed);
                            let night_start = std::time::Instant::now();
                             execute_night_phase(
                                &mut desc, hash, std::path::Path::new(&socket_path), &mut baker_client,
                                &ctx.rt_handle, &mut workspace,
                                current_prune_threshold, current_max_sprouts, &ctx.routing_table
                            );
                            crate::node::storage::flush_mmap_geometry(workspace.ephys_mmap.as_mut());
                            let elapsed_ns = night_start.elapsed().as_nanos();
                            info!(" [Shard {:08X}] Night Phase completed in {} ns", hash, elapsed_ns);
                        }

                        // Send feedback to orchestrator
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
        }).expect("Failed to spawn compute thread")
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_zero_copy_buffer_swap_invariant() {
        let required_capacity = 1024 * 1024; // 1 MB
        
        // Simulate a workspace buffer that has already completed one cycle
        let mut active_buffer: Vec<u8> = Vec::with_capacity(required_capacity);
        unsafe { active_buffer.set_len(required_capacity); }
        active_buffer.fill(0xAA); // Garbage from previous dump
        
        let initial_capacity = active_buffer.capacity();
        assert!(initial_capacity >= required_capacity);

        // --- HFT cycle logic ---
        let current_cap = active_buffer.capacity();
        let mut new_empty_buffer = Vec::with_capacity(current_cap);
        
        // [DOD] Bypass Memset. unsafe is legal here, because DMA will overwrite 
        // these bytes in hardware. Reading them before DMA is strictly prohibited.
        unsafe { new_empty_buffer.set_len(current_cap); }
        
        let extracted_payload = std::mem::replace(&mut active_buffer, new_empty_buffer);
        // --- End of logic ---

        // 1. Check that extracted payload contains old data and was not copied O(N)
        assert_eq!(extracted_payload.len(), required_capacity);
        assert_eq!(extracted_payload[0], 0xAA);
        
        // 2. Check Ownership Invariant: new buffer is ready for the next DMA
        assert_eq!(active_buffer.len(), current_cap, "Fatal: Next DMA will crash due to 0 length!");
        assert_eq!(active_buffer.capacity(), current_cap, "Fatal: Capacity drifted, allocation triggered!");
    }
}
