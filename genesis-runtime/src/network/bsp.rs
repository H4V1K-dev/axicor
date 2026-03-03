use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use genesis_core::ipc::SpikeEvent;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct PingPongSchedule {
    pub buffer_a: *mut u32, // Pinned Host RAM
    pub buffer_b: *mut u32,
    pub counts_a: *mut u32, // Number of spikes per tick
    pub counts_b: *mut u32,
    pub reading_from_a: AtomicBool, // State flag
    pub is_sleeping: Arc<AtomicBool>, // Biological sleep flag
    pub batch_ticks: usize,
    pub max_spikes_per_tick: usize,
    pub packets_received: AtomicUsize, // BSP Barrier counter
}

impl PingPongSchedule {
    pub unsafe fn new(batch_ticks: usize, max_spikes_per_tick: usize, is_sleeping: Arc<AtomicBool>) -> Self {
        let buf_size = batch_ticks * max_spikes_per_tick * 4; // 32-bit ghost IDs
        let counts_size = batch_ticks * 4;
        
        let s = Self {
            buffer_a: crate::ffi::gpu_host_alloc(buf_size) as *mut u32,
            buffer_b: crate::ffi::gpu_host_alloc(buf_size) as *mut u32,
            counts_a: crate::ffi::gpu_host_alloc(counts_size) as *mut u32,
            counts_b: crate::ffi::gpu_host_alloc(counts_size) as *mut u32,
            reading_from_a: AtomicBool::new(true),
            is_sleeping,
            batch_ticks,
            max_spikes_per_tick,
            packets_received: AtomicUsize::new(0),
        };

        // Initialize counts to 0 to avoid garbage in ingest_spike
        std::ptr::write_bytes(s.counts_a as *mut u8, 0, counts_size);
        std::ptr::write_bytes(s.counts_b as *mut u8, 0, counts_size);
        
        s
    }

    /// Wait for data with a timeout.
    pub fn wait_for_data(&self, last_count: usize, timeout: Duration) -> Result<usize, ()> {
        let start = Instant::now();
        loop {
            let current = self.packets_received.load(Ordering::Acquire);
            if current > last_count {
                return Ok(current);
            }
            if start.elapsed() >= timeout {
                return Err(());
            }
            std::hint::spin_loop();
        }
    }

    /// Executed by the background network thread (Map Phase)
    pub unsafe fn ingest_spike(&self, event: &SpikeEvent) {
        // [Contract §1] biological Spike Drop: ignore input if sleeping.
        if self.is_sleeping.load(Ordering::Acquire) {
            return;
        }

        let is_reading_a = self.reading_from_a.load(Ordering::Relaxed);
        
        // Write to the buffer that is currently NOT being read by the GPU
        let (write_buf, write_counts) = if is_reading_a {
            (self.buffer_b, self.counts_b)
        } else {
            (self.buffer_a, self.counts_a)
        };

        let tick = event.tick_offset as usize;
        if tick >= self.batch_ticks { return; }

        let current_count = std::ptr::read_volatile(write_counts.add(tick));
        if current_count < self.max_spikes_per_tick as u32 {
            let offset = tick * self.max_spikes_per_tick + (current_count as usize);
            std::ptr::write_volatile(write_buf.add(offset), event.ghost_axon_id);
            std::ptr::write_volatile(write_counts.add(tick), current_count + 1);
        }
    }

    /// BSP Barrier: O(1) swap of the active buffer
    pub fn sync_and_swap(&self) -> (*mut u32, *mut u32) {
        let current = self.reading_from_a.load(Ordering::Acquire);
        self.reading_from_a.store(!current, Ordering::Release);
        
        if !current {
            (self.buffer_a, self.counts_a)
        } else {
            (self.buffer_b, self.counts_b)
        }
    }

    pub unsafe fn clear_write_buffer(&self) {
        let is_reading_a = self.reading_from_a.load(Ordering::Relaxed);
        let counts_to_clear = if is_reading_a { self.counts_b } else { self.counts_a };
        std::ptr::write_bytes(counts_to_clear as *mut u8, 0, self.batch_ticks * 4);
    }
}

/// Strict BSP Barrier for Cluster Scaling.
/// Isolates network ingest from GPU processing via Double Buffering.
pub struct BspBarrier {
    pub writing_to_b: AtomicBool,
    pub schedules: HashMap<u32, Arc<PingPongSchedule>>,
    pub dead_zones: Arc<std::sync::Mutex<Vec<u32>>>,
}

impl BspBarrier {
    pub fn new() -> Self {
        Self {
            writing_to_b: AtomicBool::new(true),
            schedules: HashMap::new(),
            dead_zones: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn add_zone(&mut self, hash: u32, schedule: Arc<PingPongSchedule>) {
        self.schedules.insert(hash, schedule);
    }

    /// Wait for all active zones to deliver their spikes.
    /// Returns Ok(()) if all zones checked in, or Err(Vec<u32>) if some timed out.
    pub fn sync_and_swap(&self, last_packet_counts: &HashMap<u32, usize>, timeout: Duration) -> Result<(), Vec<u32>> {
        let mut dead = Vec::new();
        let current_dead = self.dead_zones.lock().unwrap().clone();

        for (&hash, schedule) in &self.schedules {
            if current_dead.contains(&hash) {
                continue; // Ignore already dead zones
            }

            let last_count = last_packet_counts.get(&hash).copied().unwrap_or(0);
            if let Err(_) = schedule.wait_for_data(last_count, timeout) {
                dead.push(hash);
            }
        }

        if !dead.is_empty() {
            let mut guard = self.dead_zones.lock().unwrap();
            for &h in &dead {
                if !guard.contains(&h) {
                    guard.push(h);
                }
            }
            return Err(dead);
        }

        // [Contract §13.2] Swap active buffer
        let current = self.writing_to_b.load(Ordering::Acquire);
        self.writing_to_b.store(!current, Ordering::Release);
        
        Ok(())
    }

    /// Zero-Copy deserialization for Inter-Node spikes.
    pub unsafe fn ingest_spike_batch(payload: &[u8]) -> &[SpikeEvent] {
        if payload.len() < 8 {
            return &[];
        }
        let events_ptr = payload.as_ptr().add(8) as *const SpikeEvent;
        let count = (payload.len() - 8) / std::mem::size_of::<SpikeEvent>();
        std::slice::from_raw_parts(events_ptr, count)
    }
}

// Ensure Thread-Safety for shared Schedule wrapper
unsafe impl Send for PingPongSchedule {}
unsafe impl Sync for PingPongSchedule {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use genesis_core::ipc::SpikeBatchHeader;

    #[test]
    fn test_bsp_timeout_recovery() {
        let is_sleeping = Arc::new(AtomicBool::new(false));
        let mut barrier = BspBarrier::new();
        
        // Setup two zones
        let s1 = Arc::new(unsafe { PingPongSchedule::new(10, 10, is_sleeping.clone()) });
        let s2 = Arc::new(unsafe { PingPongSchedule::new(10, 10, is_sleeping.clone()) });
        
        barrier.add_zone(0x1, s1.clone());
        barrier.add_zone(0x2, s2.clone());

        // Simulate s1 sending data
        s1.packets_received.store(1, Ordering::SeqCst);
        
        // Wait with short timeout
        let last_counts = HashMap::new(); // everything at 0
        let result = barrier.sync_and_swap(&last_counts, Duration::from_millis(10));
        
        assert!(result.is_err());
        let dead = result.unwrap_err();
        assert_eq!(dead, vec![0x2]); // Zone 2 timed out

        // Verify Zone 2 is now in dead list
        assert!(barrier.dead_zones.lock().unwrap().contains(&0x2));

        // Next sync should skip Zone 2 and pass
        let mut last_counts_updated = HashMap::new();
        last_counts_updated.insert(0x1, 1);
        s1.packets_received.store(2, Ordering::SeqCst);
        
        let result_retry = barrier.sync_and_swap(&last_counts_updated, Duration::from_millis(10));
        assert!(result_retry.is_ok());
    }

    #[test]
    fn test_zero_copy_ingest() {
        let header = SpikeBatchHeader { magic: 0x5350494B, batch_id: 42 };
        let event1 = SpikeEvent { ghost_axon_id: 100, tick_offset: 5 };
        let event2 = SpikeEvent { ghost_axon_id: 200, tick_offset: 10 };

        let mut raw = Vec::new();
        unsafe {
            let h_bytes = std::slice::from_raw_parts(&header as *const _ as *const u8, 8);
            raw.extend_from_slice(h_bytes);
            let e1_bytes = std::slice::from_raw_parts(&event1 as *const _ as *const u8, 8);
            raw.extend_from_slice(e1_bytes);
            let e2_bytes = std::slice::from_raw_parts(&event2 as *const _ as *const u8, 8);
            raw.extend_from_slice(e2_bytes);
        }

        let events = unsafe { BspBarrier::ingest_spike_batch(&raw) };
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].ghost_axon_id, 100);
        assert_eq!(events[1].tick_offset, 10);
    }

    #[test]
    fn test_bsp_ping_pong_basic() {
        let barrier = BspBarrier::new();
        assert!(barrier.writing_to_b.load(Ordering::Relaxed));
        
        // Fake empty sync
        let last_counts = HashMap::new();
        let _ = barrier.sync_and_swap(&last_counts, Duration::from_millis(1));
        assert!(!barrier.writing_to_b.load(Ordering::Relaxed));
    }
}
