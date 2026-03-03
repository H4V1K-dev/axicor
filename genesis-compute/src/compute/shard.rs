use genesis_core::layout::VramState;
use genesis_core::ipc::SpikeEvent;
use crate::ffi::*;

/// The Compute Island: A strictly isolated, synchronous container for GPU-resident shard data.
/// It represents the "Execution Core" and has no knowledge of network, I/O, or async state.
pub struct ShardEngine {
    pub state: VramState,
    pub constants_ptr: *const std::ffi::c_void,
    pub mapped_soma_ids: *const u32,
    pub num_inputs: u32,
    pub num_outputs: u32,
}

unsafe impl Send for ShardEngine {}
unsafe impl Sync for ShardEngine {}

impl ShardEngine {
    pub fn new(
        state: VramState,
        constants_ptr: *const std::ffi::c_void,
        mapped_soma_ids: *const u32,
        num_inputs: u32,
        num_outputs: u32,
    ) -> Self {
        Self {
            state,
            constants_ptr,
            mapped_soma_ids,
            num_inputs,
            num_outputs,
        }
    }

    /// Executes a batch of simulation ticks.
    ///
    /// # Contract (spec §1.0):
    /// 1. Mandatory 1-6 kernel sequence per tick.
    /// 2. No per-tick synchronization (Async kernel launches only).
    /// 3. Zero-copy DMA: bitmask and history buffers must be Pinned RAM.
    /// 4. Final barrier (gpu_synchronize) only at the end of the batch.
    #[inline(never)]
    pub fn step_day_phase(
        &mut self,
        sync_batch_ticks: u32,
        base_tick: u32,
        v_seg: u32,
        input_bitmask: *const u32,
        output_history: *mut u8,
        schedule: *const SpikeEvent,
        spikes_per_tick: &[u32],
    ) {
        let mut schedule_offset = 0;

        for tick_offset in 0..sync_batch_ticks {
            let current_tick = base_tick + tick_offset;
            let tick_spikes_count = spikes_per_tick[tick_offset as usize];
            
            unsafe {
                // 1. Входы (Virtual Axons)
                if self.num_inputs > 0 {
                    launch_inject_inputs(self.state.clone(), input_bitmask, current_tick, self.num_inputs);
                }
                
                // 2. Сеть (Ghost Axons)
                if tick_spikes_count > 0 {
                    let tick_schedule = schedule.add(schedule_offset);
                    launch_apply_spike_batch(self.state.clone(), tick_schedule, tick_spikes_count);
                    schedule_offset += tick_spikes_count as usize;
                }
                
                // 3. Пропагация (Сдвиг поездов)
                launch_propagate_axons(self.state.clone(), v_seg);
                
                // 4. Физика сомы и интеграция дендритов
                launch_update_neurons(self.state.clone(), self.constants_ptr, current_tick);
                
                // 5. STDP Пластичность (GSOP)
                launch_apply_gsop(self.state.clone());
                
                // 6. Выходы (Readout)
                if self.num_outputs > 0 {
                    launch_record_readout(
                        self.state.clone(), 
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
