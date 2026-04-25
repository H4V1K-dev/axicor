use crate::network::ring_buffer::SpikeSchedule;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

/// Max wait for peer data before forcing epoch advance. Prevents deadlock if sender dies.
/// 500ms allows ~20 batches slack when sender at ~4k TPS (batch=25ms).
const BSP_SYNC_TIMEOUT_MS: u64 = 500;

/// BSP Barrier for network and compute synchronization (Latency Hiding).
/// We use Ping-Pong Double Buffering: while GPU reads from A, network writes to B.
pub struct BspBarrier {
    pub schedule_a: SpikeSchedule,
    pub schedule_b: SpikeSchedule,
    /// If true, UDP server writes to B, and GPU reads from A.
    pub writing_to_b: AtomicBool,
    // [DOD] Network synchronization
    pub expected_peers: usize,
    pub current_epoch: AtomicU32,         // [DOD] Global Sync Clock
    pub completed_peers: AtomicUsize,     // [DOD] Count of is_last flags
    pub timeout_log_counter: AtomicU32,   // Throttle: log every 100th timeout
    pub self_heal_log_counter: AtomicU32, // Throttle: log every 100th self-heal
    pub cpu_profile: crate::CpuProfile,
}

impl BspBarrier {
    pub fn new(sync_batch_ticks: usize, expected_peers: usize) -> Self {
        Self {
            schedule_a: SpikeSchedule::new(sync_batch_ticks),
            schedule_b: SpikeSchedule::new(sync_batch_ticks),
            writing_to_b: AtomicBool::new(true),
            expected_peers,
            current_epoch: AtomicU32::new(0),
            completed_peers: AtomicUsize::new(0),
            timeout_log_counter: AtomicU32::new(0),
            self_heal_log_counter: AtomicU32::new(0),
            cpu_profile: crate::CpuProfile::Aggressive,
        }
    }

    pub fn with_cpu_profile(mut self, cpu_profile: crate::CpuProfile) -> Self {
        self.cpu_profile = cpu_profile;
        self
    }

    #[inline(always)]
    pub fn apply_wait_strategy(&self) {
        match self.cpu_profile {
            crate::CpuProfile::Aggressive => std::hint::spin_loop(),
            crate::CpuProfile::Balanced => std::thread::yield_now(),
            crate::CpuProfile::Eco => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }

    /// Wait for all peers to deliver their epoch data. Timeout prevents deadlock if sender dies.
    /// [Refactor] 500ms timeout: 50ms was too aggressive when receiver (MotorCortex) runs faster
    /// than sender (SensoryCortex). At ~4k TPS, batch = 25ms; 500ms allows ~20 batches of slack.
    pub fn wait_for_data_sync(&self) {
        let start = std::time::Instant::now();

        while self.completed_peers.load(Ordering::Acquire) < self.expected_peers {
            if start.elapsed() > std::time::Duration::from_millis(BSP_SYNC_TIMEOUT_MS) {
                let _n = self.timeout_log_counter.fetch_add(1, Ordering::Relaxed);
                if _n.is_multiple_of(100) {
                    tracing::warn!("[BSP] Timeout! Forcing epoch advance ({} timeouts). Dropped data will be filtered out.", _n + 1);
                }
                break;
            }
            // [DOD FIX] Dynamic wait strategy
            self.apply_wait_strategy();
        }
    }

    pub fn sync_and_swap(&self, expected_epoch: u32) -> Result<(), u32> {
        let next_epoch = expected_epoch + 1;
        match self.current_epoch.compare_exchange(
            expected_epoch,
            next_epoch,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                self.completed_peers.store(0, Ordering::Release);

                let was_b = self.writing_to_b.fetch_xor(true, Ordering::SeqCst);
                if was_b {
                    self.schedule_a.clear();
                } else {
                    self.schedule_b.clear();
                }
                Ok(())
            }
            Err(actual) => Err(actual),
        }
    }
    /// Returns reference to buffer where network (Tokio) should currently write.
    pub fn get_write_schedule(&self) -> &SpikeSchedule {
        if self.writing_to_b.load(Ordering::Acquire) {
            &self.schedule_b
        } else {
            &self.schedule_a
        }
    }

    /// Returns reference to buffer from which GPU should currently read.
    pub fn get_read_schedule(&self) -> &SpikeSchedule {
        if self.writing_to_b.load(Ordering::Acquire) {
            &self.schedule_a
        } else {
            &self.schedule_b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuProfile;
    use std::time::Instant;

    #[test]
    fn test_bsp_barrier_cpu_profile_builder() {
        let b1 = BspBarrier::new(100, 1);
        assert!(matches!(b1.cpu_profile, CpuProfile::Aggressive));

        let b2 = BspBarrier::new(100, 1).with_cpu_profile(CpuProfile::Balanced);
        assert!(matches!(b2.cpu_profile, CpuProfile::Balanced));

        let b3 = BspBarrier::new(100, 1).with_cpu_profile(CpuProfile::Eco);
        assert!(matches!(b3.cpu_profile, CpuProfile::Eco));
    }

    #[test]
    fn test_bsp_barrier_apply_wait_strategy() {
        // Just verify that these do not panic
        let barrier = BspBarrier::new(100, 1);

        let b = barrier.with_cpu_profile(CpuProfile::Aggressive);
        b.apply_wait_strategy();

        let b2 = BspBarrier::new(100, 1).with_cpu_profile(CpuProfile::Balanced);
        b2.apply_wait_strategy();

        let b3 = BspBarrier::new(100, 1).with_cpu_profile(CpuProfile::Eco);
        b3.apply_wait_strategy();
    }

    #[test]
    fn test_bsp_barrier_wait_timeout() {
        let barrier = BspBarrier::new(100, 1).with_cpu_profile(CpuProfile::Eco);
        let start = Instant::now();
        barrier.wait_for_data_sync();
        let elapsed = start.elapsed();
        // Should wait at least BSP_SYNC_TIMEOUT_MS, but return eventually
        assert!(elapsed.as_millis() >= super::BSP_SYNC_TIMEOUT_MS as u128);
    }
}
