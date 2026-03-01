use std::sync::atomic::{AtomicBool, Ordering};
use crate::network::SpikeEvent;

pub struct PingPongSchedule {
    pub buffer_a: *mut u32, // Pinned Host RAM
    pub buffer_b: *mut u32,
    pub counts_a: *mut u32, // Number of spikes per tick
    pub counts_b: *mut u32,
    pub reading_from_a: AtomicBool, // State flag
    pub batch_ticks: usize,
    pub max_spikes_per_tick: usize,
}

impl PingPongSchedule {
    pub unsafe fn new(batch_ticks: usize, max_spikes_per_tick: usize) -> Self {
        let buf_size = batch_ticks * max_spikes_per_tick * 4; // 32-bit ghost IDs
        let counts_size = batch_ticks * 4;
        
        Self {
            buffer_a: crate::ffi::gpu_host_alloc(buf_size) as *mut u32,
            buffer_b: crate::ffi::gpu_host_alloc(buf_size) as *mut u32,
            counts_a: crate::ffi::gpu_host_alloc(counts_size) as *mut u32,
            counts_b: crate::ffi::gpu_host_alloc(counts_size) as *mut u32,
            reading_from_a: AtomicBool::new(true),
            batch_ticks,
            max_spikes_per_tick,
        }
    }

    /// Executed by the background network thread (Map Phase)
    pub unsafe fn ingest_spike(&self, event: &SpikeEvent) {
        let is_reading_a = self.reading_from_a.load(Ordering::Relaxed);
        
        // Write to the buffer that is currently NOT being read by the GPU
        let (write_buf, write_counts) = if is_reading_a {
            // If reading A, write to B
            (self.buffer_b, self.counts_b)
        } else {
            // If reading B, write to A
            (self.buffer_a, self.counts_a)
        };

        let tick = event.tick_offset as usize;
        if tick >= self.batch_ticks { return; } // Out of bounds drop

        // Atomic increment of the counter for this specific tick.
        // For actual lock-free, we'd use AtomicU32 over the pointer, but in
        // pure BSP, the network thread is the only writer to the non-active buffer.
        let current_count = std::ptr::read_volatile(write_counts.add(tick));
        if current_count < self.max_spikes_per_tick as u32 {
            let offset = tick * self.max_spikes_per_tick + (current_count as usize);
            std::ptr::write_volatile(write_buf.add(offset), event.ghost_id);
            std::ptr::write_volatile(write_counts.add(tick), current_count + 1);
        }
    }

    /// BSP Barrier: O(1) swap of the active buffer
    pub fn sync_and_swap(&self) -> (*mut u32, *mut u32) {
        let current = self.reading_from_a.load(Ordering::Acquire);
        self.reading_from_a.store(!current, Ordering::Release);
        
        // Return pointers to the newly "active" buffer for DMA to VRAM
        if !current {
            // Switched from false -> true. GPU will now read from A.
            (self.buffer_a, self.counts_a)
        } else {
            // Switched from true -> false. GPU will now read from B.
            (self.buffer_b, self.counts_b)
        }
    }

    /// Zero out the counts of the buffer we are about to start writing into
    pub unsafe fn clear_write_buffer(&self) {
        let is_reading_a = self.reading_from_a.load(Ordering::Relaxed);
        let counts_to_clear = if is_reading_a { self.counts_b } else { self.counts_a };
        std::ptr::write_bytes(counts_to_clear as *mut u8, 0, self.batch_ticks * 4);
    }
}

// Ensure Thread-Safety for shared Schedule wrapper
unsafe impl Send for PingPongSchedule {}
unsafe impl Sync for PingPongSchedule {}

