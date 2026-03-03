use genesis_core::layout::VramState;
use genesis_core::ipc::SpikeEvent;
use crate::ffi::*;
use crate::zone_runtime::ZoneRuntime;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// DayPhaseRunner orchestrates the "Hot Loop" (6 CUDA kernels per tick).
/// It ensures strict execution order and maximizes GPU occupancy by avoiding
/// per-tick synchronization.
pub struct DayPhaseRunner {
    pub state: VramState,
    pub constants_ptr: *const std::ffi::c_void, // Pointer to 1024B in __constant__ VRAM (host mirror)
    pub mapped_soma_ids: *const u32,            // Pixel mapping array for RecordReadout (GXO)
    pub num_inputs: u32,                        // Count of virtual axons
    pub num_outputs: u32,                       // Count of mapped output somas
}

impl DayPhaseRunner {
    /// Executes a batch of simulation ticks.
    ///
    /// # Contract (spec §1.0):
    /// 1. Mandatory 1-6 kernel sequence per tick.
    /// 2. No per-tick synchronization (Async kernel launches only).
    /// 3. Zero-copy DMA: bitmask and history buffers must be Pinned RAM.
    /// 4. Final barrier (gpu_synchronize) only at the end of the batch.
    #[inline(never)]
    pub fn run_batch(
        &mut self,
        sync_batch_ticks: u32,
        base_tick: u32,
        v_seg: u32,
        input_bitmask: *const u32,        // From PinnedBuffer
        output_history: *mut u8,          // From PinnedBuffer
        schedule: *const SpikeEvent,      // From Ring Buffer
        spikes_per_tick: &[u32],          // Length = sync_batch_ticks
    ) {
        let mut schedule_offset = 0;

        for tick_offset in 0..sync_batch_ticks {
            let current_tick = base_tick + tick_offset;
            let tick_spikes_count = spikes_per_tick[tick_offset as usize];
            
            unsafe {
                // 1. Входы (Virtual Axons)
                if self.num_inputs > 0 {
                    launch_inject_inputs(self.state, input_bitmask, current_tick, self.num_inputs);
                }
                
                // 2. Сеть (Ghost Axons)
                if tick_spikes_count > 0 {
                    let tick_schedule = schedule.add(schedule_offset);
                    launch_apply_spike_batch(self.state, tick_schedule, tick_spikes_count);
                    schedule_offset += tick_spikes_count as usize;
                }
                
                // 3. Пропагация (Сдвиг поездов)
                launch_propagate_axons(self.state, v_seg);
                
                // 4. Физика сомы и интеграция дендритов
                launch_update_neurons(self.state, self.constants_ptr, current_tick);
                
                // 5. STDP Пластичность (GSOP)
                launch_apply_gsop(self.state);
                
                // 6. Выходы (Readout)
                if self.num_outputs > 0 {
                    launch_record_readout(
                        self.state, 
                        self.mapped_soma_ids, 
                        output_history, 
                        current_tick, 
                        self.num_outputs
                    );
                }
            }
        }
        
        // Final synchronization barrier (BSP Barrier).
        #[cfg(not(feature = "mock-gpu"))]
        unsafe {
            crate::ffi::gpu_synchronize(); 
        }
    }
}

// The TelemetrySwapchain struct and its new function are assumed to be in `crate::network::telemetry`
// and are modified there. For completeness, here's how they would look if they were in this file:
/*
pub struct TelemetrySwapchain {
    /// Host-mapped pointer to the buffer that is ready for export (A or B).
    pub ready_for_export: AtomicPtr<u32>,
    /// Host-mapped pointer to the buffer that is currently being filled by the GPU (A or B).
    pub back_buffer: AtomicPtr<u32>,
    /// Host-mapped pointer for the count of fired IDs (size = 1 u32).
    pub count_buffer: PinnedBuffer<u32>,
    /// Number of spikes in the ready buffer.
    pub ready_count: AtomicUsize,
    /// Number of active clients connected to the telemetry stream.
    pub active_clients: AtomicUsize,
    /// Last tick when the ready buffer was swapped.
    pub last_swap_tick: AtomicU64,
}

impl TelemetrySwapchain {
    pub fn new(capacity: usize) -> Result<Self, String> {
        let buffer_a = PinnedBuffer::new(capacity)?;
        let buffer_b = PinnedBuffer::new(capacity)?;
        let count_buffer = PinnedBuffer::new(1)?;
        
        Ok(Self {
            ready_for_export: AtomicPtr::new(buffer_a.as_ptr() as *mut u32),
            back_buffer: AtomicPtr::new(buffer_b.as_ptr() as *mut u32),
            active_clients: AtomicUsize::new(0),
            count_buffer,
            ready_count: AtomicUsize::new(0),
            last_swap_tick: AtomicU64::new(0),
        })
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_ffi::{clear_call_log, get_call_log};

    fn dummy_vram() -> VramState {
        VramState {
            padded_n: 32,
            total_axons: 64,
            voltage: std::ptr::null_mut(),
            flags: std::ptr::null_mut(),
            threshold_offset: std::ptr::null_mut(),
            refractory_timer: std::ptr::null_mut(),
            soma_to_axon: std::ptr::null_mut(),
            dendrite_targets: std::ptr::null_mut(),
            dendrite_weights: std::ptr::null_mut(),
            dendrite_timers: std::ptr::null_mut(),
            axon_heads: std::ptr::null_mut(),
            input_bitmask: std::ptr::null_mut(),
            output_history: std::ptr::null_mut(),
            telemetry_count: std::ptr::null_mut(),
            telemetry_spikes: std::ptr::null_mut(),
        }
    }

    #[test]
    #[cfg(feature = "mock-gpu")]
    fn test_day_phase_order() {
        clear_call_log();
        let mut runner = DayPhaseRunner {
            state: dummy_vram(),
            constants_ptr: std::ptr::null(),
            mapped_soma_ids: std::ptr::null(),
            num_inputs: 64,
            num_outputs: 64,
        };

        runner.run_batch(2, 0, 1, std::ptr::null(), std::ptr::null_mut(), std::ptr::null(), &[1, 0]);

        let log = get_call_log();
        assert_eq!(log.len(), 11);
        assert_eq!(log[0].0, "InjectInputs");
        assert_eq!(log[1].0, "ApplySpikeBatch");
        assert_eq!(log[6].0, "InjectInputs");
    }

    #[test]
    #[cfg(feature = "mock-gpu")]
    fn test_schedule_pointer_math() {
        clear_call_log();
        let mut runner = DayPhaseRunner {
            state: dummy_vram(),
            constants_ptr: std::ptr::null(),
            mapped_soma_ids: std::ptr::null(),
            num_inputs: 0,
            num_outputs: 0,
        };

        let base_schedule = 0x1000 as *const SpikeEvent;
        runner.run_batch(2, 0, 1, std::ptr::null(), std::ptr::null_mut(), base_schedule, &[2, 3]);

        let log = get_call_log();
        let apply_calls: Vec<_> = log.iter().filter(|(name, _)| name == "ApplySpikeBatch").collect();
        assert_eq!(apply_calls.len(), 2);

        let spike_event_size = std::mem::size_of::<SpikeEvent>();
        assert_eq!(apply_calls[0].1, 0x1000);
        assert_eq!(apply_calls[1].1, 0x1000 + 2 * spike_event_size);
    }
}
