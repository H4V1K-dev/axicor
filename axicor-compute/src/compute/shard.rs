use crate::memory::VramState;
use std::ptr;

#[derive(Clone, Copy)]
pub struct EphysState {
    pub tids_d: *const u32,
    pub uvs_d: *const i32,
    pub trace_d: *mut i32,
    pub count: u32,
    pub max_ticks: u32,
    pub current_tick: u32,
}

pub struct GpuIoBuffers {
    pub d_input_bitmask: *mut u32,
    pub d_incoming_spikes: *mut u32,
    pub d_output_history: *mut u8,
}

pub struct CpuIoBuffers {
    pub h_input_bitmask: Vec<u32>,
    pub h_incoming_spikes: Vec<u32>,
    pub h_output_history: Vec<u8>,
}

pub enum IoBackend {
    Gpu(GpuIoBuffers),
    Cpu(CpuIoBuffers),
}

pub struct IoBuffers {
    pub backend: IoBackend,
    pub max_spikes_per_tick: u32,
    pub input_words_per_tick: u32,
    pub num_outputs: u32,
}

unsafe impl Send for IoBuffers {}
unsafe impl Sync for IoBuffers {}

impl Drop for IoBuffers {
    fn drop(&mut self) {
        match &mut self.backend {
            IoBackend::Gpu(b) => unsafe {
                //   VRAM-.   .
                crate::ffi::cu_free_io_buffers(
                    b.d_input_bitmask,
                    b.d_incoming_spikes,
                    b.d_output_history,
                );
            },
            IoBackend::Cpu(_) => {
                //  CPU    (Rust Drop)
            }
        }
    }
}

pub struct GpuEngine {
    pub vram: VramState,
    pub stream: crate::ffi::CudaStream,
    pub telemetry_ids_d: *mut u32,
    pub telemetry_count_d: *mut u32,
    pub telemetry_count_pinned_h: *mut u32,
    pub ephys_state: Option<EphysState>, // [DOD FIX] Debug Harness State
}

pub struct CpuEngine {
    pub vram: VramState,
    pub telemetry_ids: Vec<u32>,
    pub telemetry_count: u32,
    pub ephys_state: Option<EphysState>, // [DOD FIX]
}

pub enum ShardEngine {
    Gpu(GpuEngine),
    Cpu(CpuEngine),
}

unsafe impl Send for ShardEngine {}
unsafe impl Sync for ShardEngine {}

impl ShardEngine {
    pub fn new(vram: VramState) -> Self {
        let n = vram.padded_n as usize;
        if vram.use_gpu {
            #[cfg_attr(feature = "mock-gpu", allow(unused_mut))]
            let mut stream = std::ptr::null_mut();
            #[cfg(not(feature = "mock-gpu"))]
            {
                let err = unsafe { crate::ffi::gpu_stream_create(&mut stream) };
                assert_eq!(err, 0, "FATAL: cudaStreamCreate failed");
            }

            let telemetry_ids_d: *mut u32;
            let telemetry_count_d: *mut u32;
            let telemetry_count_pinned_h: *mut u32;

            unsafe {
                telemetry_ids_d = crate::ffi::gpu_malloc(n * 4) as *mut u32;
                telemetry_count_d = crate::ffi::gpu_malloc(4) as *mut u32;
                telemetry_count_pinned_h = crate::ffi::gpu_host_alloc(4) as *mut u32;
            }

            Self::Gpu(GpuEngine {
                vram,
                stream,
                telemetry_ids_d,
                telemetry_count_d,
                telemetry_count_pinned_h,
                ephys_state: None,
            })
        } else {
            Self::Cpu(CpuEngine {
                vram,
                telemetry_ids: vec![0u32; n],
                telemetry_count: 0,
                ephys_state: None,
            })
        }
    }

    pub fn set_ephys_state(&mut self, state: Option<EphysState>) {
        match self {
            Self::Gpu(gpu) => gpu.ephys_state = state,
            Self::Cpu(cpu) => cpu.ephys_state = state,
        }
    }

    pub fn get_ephys_state(&self) -> Option<EphysState> {
        match self {
            Self::Gpu(gpu) => gpu.ephys_state,
            Self::Cpu(cpu) => cpu.ephys_state,
        }
    }

    /// # Safety
    /// Caller must ensure device pointers and buffer boundaries are valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn step_day_phase_batch(
        &mut self,
        sync_batch_ticks: u32,
        io_buffers: &mut IoBuffers,
        h_input_bitmask: Option<&[u32]>,
        h_incoming_spikes: Option<&[u32]>,
        h_spike_counts: &[u32],
        virtual_offset: u32,
        num_virtual_axons: u32,
        mapped_soma_ids_device: *const u32, // Device ptr for GPU, Host ptr for CPU
        v_seg: u32,
        dopamine: i16,
        tick_base: u32,
    ) {
        match self {
            Self::Gpu(ref mut gpu) => {
                let total_input_words = io_buffers.input_words_per_tick * sync_batch_ticks;
                let total_schedule_capacity = io_buffers.max_spikes_per_tick * sync_batch_ticks;

                if let IoBackend::Gpu(ref mut b) = io_buffers.backend {
                    unsafe {
                        crate::ffi::cu_dma_h2d_io(
                            b.d_input_bitmask,
                            h_input_bitmask.map_or(ptr::null(), |s| s.as_ptr()),
                            if h_input_bitmask.is_some() {
                                total_input_words
                            } else {
                                0
                            },
                            b.d_incoming_spikes,
                            h_incoming_spikes.map_or(ptr::null(), |s| s.as_ptr()),
                            if h_incoming_spikes.is_some() {
                                total_schedule_capacity
                            } else {
                                0
                            },
                            gpu.stream,
                        );

                        crate::ffi::cu_reset_burst_counters(
                            &gpu.vram.ptrs,
                            gpu.vram.padded_n,
                            gpu.stream,
                        );
                        crate::ffi::gpu_reset_telemetry_count(gpu.telemetry_count_d, gpu.stream);

                        for tick in 0..sync_batch_ticks {
                            let global_tick = tick_base + tick;
                            let tick_input_ptr = if b.d_input_bitmask.is_null() {
                                ptr::null()
                            } else {
                                b.d_input_bitmask
                                    .add((tick * io_buffers.input_words_per_tick) as usize)
                            };
                            let tick_spikes_ptr = if b.d_incoming_spikes.is_null() {
                                ptr::null()
                            } else {
                                b.d_incoming_spikes
                                    .add((tick * io_buffers.max_spikes_per_tick) as usize)
                            };
                            let tick_output_ptr = if b.d_output_history.is_null() {
                                ptr::null_mut()
                            } else {
                                b.d_output_history
                                    .add((tick * io_buffers.num_outputs) as usize)
                            };

                            // [DOD FIX] Ephys Current Injection
                            if let Some(ephys) = &gpu.ephys_state {
                                crate::ffi::launch_debug_inject_current(
                                    gpu.vram.ptrs.soma_voltage,
                                    ephys.tids_d,
                                    ephys.uvs_d,
                                    ephys.count,
                                    gpu.stream,
                                );
                            }

                            crate::ffi::cu_step_day_phase(
                                &gpu.vram.ptrs,
                                gpu.vram.padded_n,
                                gpu.vram.total_axons,
                                v_seg,
                                global_tick,
                                tick_input_ptr,
                                virtual_offset,
                                num_virtual_axons,
                                tick_spikes_ptr,
                                h_spike_counts[tick as usize],
                                mapped_soma_ids_device,
                                tick_output_ptr,
                                io_buffers.num_outputs,
                                dopamine,
                                gpu.stream,
                            );

                            // [DOD FIX] Ephys V(t) Recording
                            if let Some(ephys) = &mut gpu.ephys_state {
                                crate::ffi::launch_debug_record_v(
                                    gpu.vram.ptrs.soma_voltage,
                                    ephys.tids_d,
                                    ephys.trace_d,
                                    ephys.current_tick,
                                    ephys.count,
                                    ephys.max_ticks,
                                    gpu.stream,
                                );
                                ephys.current_tick += 1;
                            }

                            crate::ffi::launch_extract_telemetry(
                                gpu.vram.ptrs.soma_flags,
                                gpu.telemetry_ids_d,
                                gpu.telemetry_count_d,
                                gpu.vram.padded_n,
                                gpu.stream,
                            );

                            // [DOD FIX] Zero-Cost Async DMA to Pinned Host Memory
                            crate::ffi::gpu_memcpy_device_to_host_async(
                                gpu.telemetry_count_pinned_h as *mut std::ffi::c_void,
                                gpu.telemetry_count_d as *const std::ffi::c_void,
                                4,
                                gpu.stream,
                            );
                        }
                    }
                }
            }
            Self::Cpu(ref mut cpu) => {
                if let IoBackend::Cpu(ref mut b) = io_buffers.backend {
                    // 1. STRICT BSP INVARIANTS (Pre-loop validation)
                    //       .    -    (Data-Oriented  ),
                    //    .
                    if let Some(mask) = h_input_bitmask {
                        let expected_mask_len =
                            (io_buffers.input_words_per_tick * sync_batch_ticks) as usize;
                        assert_eq!(
                            mask.len(),
                            expected_mask_len,
                            "FATAL: Input bitmask size violation"
                        );
                    }
                    if let Some(spikes) = h_incoming_spikes {
                        let expected_spikes_len =
                            (io_buffers.max_spikes_per_tick * sync_batch_ticks) as usize;
                        assert_eq!(
                            spikes.len(),
                            expected_spikes_len,
                            "FATAL: Incoming spikes capacity violation"
                        );
                    }
                    assert_eq!(
                        h_spike_counts.len(),
                        sync_batch_ticks as usize,
                        "FATAL: Spike counts size violation"
                    );

                    // 1.5. Hardware Integrity Check: ,       capacity
                    let max_spikes = io_buffers.max_spikes_per_tick;
                    for &c in h_spike_counts {
                        assert!(
                            c <= max_spikes,
                            "FATAL: Network provided spike count exceeding tick capacity"
                        );
                    }

                    for tick in 0..sync_batch_ticks {
                        let global_tick = tick_base + tick;
                        let tick_idx = tick as usize;

                        // 0. Inject Virtual Axons (Sensors) - ZERO COST SLICING
                        if let Some(mask) = h_input_bitmask {
                            let start = tick_idx * io_buffers.input_words_per_tick as usize;
                            let len = io_buffers.input_words_per_tick as usize;
                            unsafe {
                                // bounds   -,  raw pointer math
                                let tick_mask =
                                    std::slice::from_raw_parts(mask.as_ptr().add(start), len);
                                let axon_heads = std::slice::from_raw_parts_mut(
                                    cpu.vram.ptrs.axon_heads,
                                    cpu.vram.total_axons as usize,
                                );
                                crate::cpu::physics::cpu_inject_inputs(
                                    axon_heads,
                                    tick_mask,
                                    virtual_offset,
                                    num_virtual_axons,
                                    v_seg,
                                );
                            }
                        }

                        // 1. Inject Network Spikes - ZERO COST SLICING
                        if let Some(spikes) = h_incoming_spikes {
                            let start = tick_idx * io_buffers.max_spikes_per_tick as usize;
                            unsafe {
                                // get_unchecked
                                let count = *h_spike_counts.get_unchecked(tick_idx) as usize;
                                debug_assert!(count <= io_buffers.max_spikes_per_tick as usize);

                                let tick_spikes =
                                    std::slice::from_raw_parts(spikes.as_ptr().add(start), count);
                                let axon_heads = std::slice::from_raw_parts_mut(
                                    cpu.vram.ptrs.axon_heads,
                                    cpu.vram.total_axons as usize,
                                );
                                crate::cpu::physics::cpu_apply_spike_batch(
                                    axon_heads,
                                    tick_spikes,
                                    v_seg,
                                );
                            }
                        }

                        unsafe {
                            let axon_heads = std::slice::from_raw_parts_mut(
                                cpu.vram.ptrs.axon_heads,
                                cpu.vram.total_axons as usize,
                            );
                            crate::cpu::physics::cpu_propagate_axons(axon_heads, v_seg);
                        }

                        // [DOD FIX] Ephys Current Injection
                        if let Some(ephys) = &cpu.ephys_state {
                            unsafe {
                                crate::cpu::physics::cpu_debug_inject_current(
                                    cpu.vram.ptrs.soma_voltage,
                                    std::slice::from_raw_parts(ephys.tids_d, ephys.count as usize),
                                    std::slice::from_raw_parts(ephys.uvs_d, ephys.count as usize),
                                );
                            }
                        }

                        // 2. GLIF Physics
                        unsafe {
                            crate::cpu::physics::cpu_update_neurons(
                                &cpu.vram.ptrs,
                                cpu.vram.padded_n,
                                cpu.vram.total_axons,
                                global_tick,
                                v_seg,
                            );
                        }

                        // 3. Plasticity
                        unsafe {
                            crate::cpu::physics::cpu_apply_gsop(
                                &cpu.vram.ptrs,
                                cpu.vram.padded_n,
                                cpu.vram.total_axons,
                                dopamine,
                            );
                        }

                        // [DOD FIX] Ephys V(t) Recording
                        if let Some(ephys) = &mut cpu.ephys_state {
                            unsafe {
                                crate::cpu::physics::cpu_debug_record_v(
                                    cpu.vram.ptrs.soma_voltage,
                                    std::slice::from_raw_parts(ephys.tids_d, ephys.count as usize),
                                    std::slice::from_raw_parts_mut(ephys.trace_d, (ephys.count * ephys.max_ticks) as usize),
                                    ephys.current_tick,
                                    ephys.max_ticks,
                                );
                            }
                            ephys.current_tick += 1;
                        }

                        // 4. Output History
                        let soma_flags = unsafe {
                            std::slice::from_raw_parts(
                                cpu.vram.ptrs.soma_flags,
                                cpu.vram.padded_n as usize,
                            )
                        };
                        let mapped_ids = unsafe {
                            std::slice::from_raw_parts(
                                mapped_soma_ids_device,
                                io_buffers.num_outputs as usize,
                            )
                        };
                        crate::cpu::physics::cpu_record_outputs(
                            soma_flags,
                            mapped_ids,
                            &mut b.h_output_history,
                            tick,
                            io_buffers.num_outputs,
                        );

                        // 5. Telemetry
                        cpu.telemetry_count = crate::cpu::physics::cpu_extract_telemetry(
                            soma_flags,
                            &mut cpu.telemetry_ids,
                        );
                    }
                }
            }
        }
    }
}

impl Drop for ShardEngine {
    #[allow(unused_variables)]
    fn drop(&mut self) {
        if let Self::Gpu(gpu) = self {
            #[cfg(not(feature = "mock-gpu"))]
            unsafe {
                crate::ffi::gpu_stream_destroy(gpu.stream);
                crate::ffi::gpu_free(gpu.telemetry_ids_d as *mut _);
                crate::ffi::gpu_free(gpu.telemetry_count_d as *mut _);
                crate::ffi::gpu_host_free(gpu.telemetry_count_pinned_h as *mut _);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::ShardVramPtrs;

    #[test]
    fn test_shard_allocation() {
        //    (CPU)
        let vram = VramState {
            padded_n: 32,
            total_axons: 64,
            total_ghosts: 0,
            use_gpu: false,
            ptrs: ShardVramPtrs {
                soma_voltage: std::ptr::null_mut(),
                soma_flags: std::ptr::null_mut(),
                threshold_offset: std::ptr::null_mut(),
                timers: std::ptr::null_mut(),
                soma_to_axon: std::ptr::null_mut(),
                dendrite_targets: std::ptr::null_mut(),
                dendrite_weights: std::ptr::null_mut(),
                dendrite_timers: std::ptr::null_mut(),
                axon_heads: std::ptr::null_mut(),
            },
        };
        let engine = ShardEngine::new(vram);
        drop(engine);
    }

    #[test]
    fn test_io_buffers_allocation_mock_gpu() {
        //    Mock GPU C-ABI
        let mut d_input = std::ptr::null_mut();
        let mut d_spikes = std::ptr::null_mut();
        let mut d_output = std::ptr::null_mut();

        unsafe {
            let err = crate::ffi::cu_allocate_io_buffers(
                10,
                20,
                30,
                &mut d_input,
                &mut d_spikes,
                &mut d_output,
            );
            assert_eq!(err, 0);
            assert!(!d_input.is_null());
            assert!(!d_spikes.is_null());
            assert!(!d_output.is_null());

            crate::ffi::cu_free_io_buffers(d_input, d_spikes, d_output);
        }
    }

    #[test]
    #[should_panic(expected = "FATAL: Network provided spike count exceeding tick capacity")]
    fn test_shard_step_crash_on_exceeded_spikes() {
        let vram = VramState {
            padded_n: 32,
            total_axons: 64,
            total_ghosts: 0,
            use_gpu: false,
            ptrs: ShardVramPtrs {
                soma_voltage: std::ptr::null_mut(),
                soma_flags: std::ptr::null_mut(),
                threshold_offset: std::ptr::null_mut(),
                timers: std::ptr::null_mut(),
                soma_to_axon: std::ptr::null_mut(),
                dendrite_targets: std::ptr::null_mut(),
                dendrite_weights: std::ptr::null_mut(),
                dendrite_timers: std::ptr::null_mut(),
                axon_heads: std::ptr::null_mut(),
            },
        };
        let mut engine = ShardEngine::new(vram);
        let mut io_buffers = IoBuffers {
            backend: IoBackend::Cpu(CpuIoBuffers {
                h_input_bitmask: vec![],
                h_incoming_spikes: vec![],
                h_output_history: vec![],
            }),
            max_spikes_per_tick: 1000,
            input_words_per_tick: 0,
            num_outputs: 0,
        };

        let spike_counts = vec![999999];

        unsafe {
            engine.step_day_phase_batch(
                1,
                &mut io_buffers,
                None,
                None,
                &spike_counts,
                0,
                0,
                std::ptr::null(),
                0,
                0,
                0,
            );
        }
    }

    #[test]
    #[should_panic(expected = "FATAL: Input bitmask size violation")]
    fn test_shard_step_crash_on_mismatched_mask() {
        let vram = VramState {
            padded_n: 32,
            total_axons: 64,
            total_ghosts: 0,
            use_gpu: false,
            ptrs: ShardVramPtrs {
                soma_voltage: std::ptr::null_mut(),
                soma_flags: std::ptr::null_mut(),
                threshold_offset: std::ptr::null_mut(),
                timers: std::ptr::null_mut(),
                soma_to_axon: std::ptr::null_mut(),
                dendrite_targets: std::ptr::null_mut(),
                dendrite_weights: std::ptr::null_mut(),
                dendrite_timers: std::ptr::null_mut(),
                axon_heads: std::ptr::null_mut(),
            },
        };
        let mut engine = ShardEngine::new(vram);
        let mut io_buffers = IoBuffers {
            backend: IoBackend::Cpu(CpuIoBuffers {
                h_input_bitmask: vec![],
                h_incoming_spikes: vec![],
                h_output_history: vec![],
            }),
            max_spikes_per_tick: 10,
            input_words_per_tick: 2,
            num_outputs: 0,
        };

        // sync_batch_ticks = 2. Expected mask len = 2 * 2 = 4.
        // We provide mask len = 3.
        let bad_mask = vec![0u32; 3];
        let spike_counts = vec![0u32; 2];

        unsafe {
            engine.step_day_phase_batch(
                2,
                &mut io_buffers,
                Some(&bad_mask),
                None,
                &spike_counts,
                0,
                0,
                std::ptr::null(),
                0,
                0,
                0,
            );
        }
    }
}
