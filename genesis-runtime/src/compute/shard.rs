use genesis_core::layout::VramState;
use genesis_core::ipc::SpikeEvent;
use crate::orchestrator::day_phase::DayPhaseRunner;
use std::ptr;

/// The Compute Island: A strictly isolated, synchronous container for GPU-resident shard data.
/// It represents the "Execution Core" and has no knowledge of network, I/O, or async state.
pub struct ShardComputeIsland {
    pub state: VramState,
    pub num_inputs: u32,
    pub num_outputs: u32,
}

unsafe impl Send for ShardComputeIsland {}
unsafe impl Sync for ShardComputeIsland {}

impl ShardComputeIsland {
    pub fn new(state: VramState, num_inputs: u32, num_outputs: u32) -> Self {
        Self {
            state,
            num_inputs,
            num_outputs,
        }
    }

    /// Executes the Day Phase batch (Step 1-6 kernels).
    /// Contract: Only synchronous FFI calls and raw pointer arithmetic.
    /// No Arc, No Mutex, No Tokio.
    pub fn execute_day_batch(
        &mut self,
        sync_batch_ticks: u32,
        base_tick: u32,
        v_seg: u32,
        input_bitmask: *const u32,
        output_history: *mut u8,
        schedule: *const SpikeEvent,
        spikes_per_tick: &[u32],
    ) {
        let mut runner = DayPhaseRunner {
            state: self.state,
            constants_ptr: ptr::null(),
            mapped_soma_ids: ptr::null(), // Needs to be passed if GXO is active
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
        };

        runner.run_batch(
            sync_batch_ticks,
            base_tick,
            v_seg,
            input_bitmask,
            output_history,
            schedule,
            spikes_per_tick,
        );
    }
}
