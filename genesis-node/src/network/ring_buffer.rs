const MAX_SPIKES_PER_TICK: usize = 1024; // configurable upper bound

/// Ring buffer for incoming spikes, configured for the size of a Sync Batch.
/// Ping-Pong Double Buffering: One is read by the GPU, the other written by the Network.
pub struct SpikeSchedule {
    pub sync_batch_ticks: usize,
    /// Flat array of `sync_batch_ticks * MAX_SPIKES_PER_TICK`
    /// We use a flat 1D array to allow a single `cudaMemcpy` to VRAM.
    pub buffer: Vec<u32>,
    /// Number of spikes registered for each tick `0..sync_batch_ticks`
    pub counts: Vec<u32>,
}

impl SpikeSchedule {
    pub fn new(sync_batch_ticks: usize) -> Self {
        Self {
            sync_batch_ticks,
            buffer: vec![0; sync_batch_ticks * MAX_SPIKES_PER_TICK],
            counts: vec![0; sync_batch_ticks],
        }
    }

    /// Map Phase: Enqueue an incoming ghost_id to fire on a specific tick offset.
    pub fn schedule_spike(&mut self, ghost_id: u32, tick_offset: u32) -> Result<(), &'static str> {
        let tick = tick_offset as usize;
        if tick >= self.sync_batch_ticks {
            return Err("tick_offset exceeds sync_batch_ticks");
        }

        let count = self.counts[tick] as usize;
        if count >= MAX_SPIKES_PER_TICK {
            // Buffer overflow. In real scenario: log a warning and drop the spike
            return Err("MAX_SPIKES_PER_TICK exceeded");
        }

        let idx = tick * MAX_SPIKES_PER_TICK + count;
        self.buffer[idx] = ghost_id;
        self.counts[tick] += 1;

        Ok(())
    }

    /// O(1) reset for the next batch.
    pub fn clear(&mut self) {
        // We only need to reset the counts. The buffer data will be overwritten.
        self.counts.fill(0);
    }
}
