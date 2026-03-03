use genesis_core::layout::VramState;
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use crate::ffi::*;
use crate::memory::PinnedBuffer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// NightPhaseRunner orchestrates the structural reorganisation (Maintenance Cycle).
pub struct NightPhaseRunner {
    pub state: VramState,
}

impl NightPhaseRunner {
    /// Step 1: GPU Sort & Prune (In-place).
    /// Step 2: D2H Download (Only weights/targets).
    pub fn download_maintenance_data(
        &mut self,
        prune_threshold: i16,
        pinned_weights: &mut PinnedBuffer<i16>,
        pinned_targets: &mut PinnedBuffer<u32>,
    ) {
        unsafe {
            // 1. GPU Sort & Prune
            launch_sort_and_prune(self.state, prune_threshold);
            gpu_synchronize();

            // 2. D2H Download (Minimal DMA: only weights and targets)
            let dc = MAX_DENDRITE_SLOTS * self.state.padded_n as usize;
            gpu_memcpy_device_to_host(
                pinned_targets.as_mut_ptr() as *mut _,
                self.state.dendrite_targets as *const _,
                dc * 4
            );
            gpu_memcpy_device_to_host(
                pinned_weights.as_mut_ptr() as *mut _,
                self.state.dendrite_weights as *const _,
                dc * 2
            );
        }
    }

    /// Step 5: H2D Upload (Updated data after Baker Daemon work).
    pub fn upload_maintenance_data(
        &mut self,
        pinned_weights: &PinnedBuffer<i16>,
        pinned_targets: &PinnedBuffer<u32>,
    ) {
        let dc = MAX_DENDRITE_SLOTS * self.state.padded_n as usize;
        unsafe {
            gpu_memcpy_host_to_device(
                self.state.dendrite_targets as *mut _,
                pinned_targets.as_ptr() as *const _,
                dc * 4
            );
            gpu_memcpy_host_to_device(
                self.state.dendrite_weights as *mut _,
                pinned_weights.as_ptr() as *const _,
                dc * 2
            );
        }
    }

    /// Hot Checkpoint (Periodic Dump).
    /// Contract: Writes weights and targets directly to NVMe through Zero-Copy.
    pub fn dump_checkpoint(
        &self,
        artifact_dir: &Path,
        pinned_weights: &PinnedBuffer<i16>,
        pinned_targets: &PinnedBuffer<u32>,
    ) -> std::io::Result<()> {
        let path = artifact_dir.join("checkpoint_weights.bin");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        
        let mut writer = std::io::BufWriter::new(file);

        // Binary Contract: [Targets][Weights]
        let targets_bytes = unsafe {
            std::slice::from_raw_parts(
                pinned_targets.as_ptr() as *const u8,
                pinned_targets.len() * 4
            )
        };
        let weights_bytes = unsafe {
            std::slice::from_raw_parts(
                pinned_weights.as_ptr() as *const u8,
                pinned_weights.len() * 2
            )
        };

        writer.write_all(targets_bytes)?;
        writer.write_all(weights_bytes)?;
        writer.flush()?;

        Ok(())
    }
}

/// Compatibility wrapper for main loop.
/// Spawns the night phase maintenance thread.
#[allow(unused_variables)]
/// Compatibility wrapper for main loop.
/// Performs synchronous maintenance (GPU part).
pub fn trigger_night_phase(
    artifact_dir: std::path::PathBuf,
    vram_ptr: *mut crate::memory::VramState,
    padded_n: u32,
    prune_threshold: i16,
    zone_hash: u32,
    tick: u32,
    baker_client: &crate::orchestrator::baker::BakerClient,
) {
    // In a real implementation, we'd spawn a thread to avoid blocking the hot loop.
    // For this step, we perform a synchronous maintenance cycle.
    let vram = unsafe { &mut *vram_ptr };
    let mut runner = NightPhaseRunner {
        state: vram.to_layout(),
    };

    // Maintenance requires pinned buffers for zero-copy DMA.
    let dc = genesis_core::constants::MAX_DENDRITE_SLOTS * padded_n as usize;
    let mut pinned_weights = PinnedBuffer::<i16>::new(dc).expect("Failed to allocate pinned weights");
    let mut pinned_targets = PinnedBuffer::<u32>::new(dc).expect("Failed to allocate pinned targets");

    // 1. GPU -> Host
    runner.download_maintenance_data(prune_threshold, &mut pinned_weights, &mut pinned_targets);

    // 2. Host -> Disk (Zero-Copy mmap file logic)
    if let Err(e) = runner.dump_checkpoint(&artifact_dir, &pinned_weights, &pinned_targets) {
        eprintln!("[Night Phase] Checkpoint failed: {}", e);
        return;
    }

    // 3. Trigger Baker Daemon (UDS RPC)
    baker_client.trigger_baker(zone_hash, tick, prune_threshold);

    // 4. Host -> GPU (Fresh weights from Baker)
    runner.upload_maintenance_data(&pinned_weights, &pinned_targets);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::bsp::PingPongSchedule;
    use crate::network::SpikeEvent;
    #[cfg(feature = "mock-gpu")]
    use crate::mock_ffi::*;

    #[test]
    #[cfg(feature = "mock-gpu")]
    fn test_sleep_spike_drop() {
        let is_sleeping = Arc::new(AtomicBool::new(false));
        let schedule = unsafe { PingPongSchedule::new(100, 1024, is_sleeping.clone()) };
        
        let event = SpikeEvent {
            ghost_axon_id: 123,
            tick_offset: 0,
        };

        // 1. Awake: spike should be ingested
        unsafe { schedule.ingest_spike(&event) };
        unsafe {
            let count = std::ptr::read_volatile(schedule.counts_b); // Since reading_from_a is true, writes to B
            assert_eq!(count, 1, "Spike should be ingested when awake");
        }

        // 2. Sleeping: spike should be dropped
        is_sleeping.store(true, Ordering::Release);
        unsafe { schedule.clear_write_buffer() }; // Reset count B
        unsafe { schedule.ingest_spike(&event) };
        unsafe {
            let count = std::ptr::read_volatile(schedule.counts_b);
            assert_eq!(count, 0, "Spike MUST be dropped when is_sleeping is true");
        }
    }

    #[test]
    #[cfg(feature = "mock-gpu")]
    fn test_checkpoint_zero_copy() {
        let temp_dir = tempfile::tempdir().unwrap();
        let artifact_dir = temp_dir.path();
        
        let mut weights = PinnedBuffer::<i16>::new(1024).unwrap();
        let mut targets = PinnedBuffer::<u32>::new(1024).unwrap();
        
        // Fill with pattern
        weights.as_mut_slice().fill(0x55);
        targets.as_mut_slice().fill(0xAA);

        let mut runner = NightPhaseRunner {
            state: unsafe { std::mem::zeroed() },
        };

        // Complete cycle manually since trigger_night_phase needs a UDS server
        runner.download_maintenance_data(10, &mut weights, &mut targets);
        runner.dump_checkpoint(artifact_dir, &weights, &targets).unwrap();
        runner.upload_maintenance_data(&weights, &targets);

        // Verify dump
        let dump_path = artifact_dir.join("checkpoint_weights.bin");
        let dump_data = std::fs::read(dump_path).unwrap();
        
        assert_eq!(dump_data.len(), 1024 * 4 + 1024 * 2);
        // First part: targets (0xAA)
        assert_eq!(dump_data[0], 0xAA);
        // Second part: weights (0x55)
        assert_eq!(dump_data[1024 * 4], 0x55);
    }
}
