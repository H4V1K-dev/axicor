use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::collections::VecDeque;

pub mod app;

/// Global Engine Dashboard State.
/// Metrics are stored as atomics for lock-free updates from the Hot Loop.
/// Events use a lightweight Mutex + VecDeque (limited to 200 items).
pub struct DashboardState {
    // --- Hot Loop Metrics (Lock-Free) ---
    pub total_ticks: AtomicU64,
    pub udp_in_packets: AtomicU32,
    pub udp_out_packets: AtomicU32,
    pub oversized_skips: AtomicU32,
    pub throughput_batches_sec: AtomicU32,
    
    // --- Event Log (Short-lived Lock) ---
    pub night_events: Mutex<VecDeque<String>>,

    /// If true, the engine should suppress println! and use the dashboard instead.
    pub use_tui: bool,
}

impl DashboardState {
    pub fn new(use_tui: bool) -> Self {
        Self {
            total_ticks: AtomicU64::new(0),
            udp_in_packets: AtomicU32::new(0),
            udp_out_packets: AtomicU32::new(0),
            oversized_skips: AtomicU32::new(0),
            throughput_batches_sec: AtomicU32::new(0),
            night_events: Mutex::new(VecDeque::with_capacity(200)),
            use_tui,
        }
    }

    /// Pushes a new event message. 
    /// Thread-safe and maintains a maximum of 200 items.
    pub fn push_event(&self, msg: String) {
        if !self.use_tui {
            // Fallback to plain-text stdout with timestamp
            let now = chrono::Local::now().format("%H:%M:%S%.3f");
            println!("[{}] {}", now, msg);
            return;
        }

        let mut log = self.night_events.lock().unwrap();
        if log.len() >= 200 {
            log.pop_back();
        }
        log.push_front(msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_dashboard_lock_free_counters() {
        let state = Arc::new(DashboardState::new(true));
        let mut threads = vec![];

        for _ in 0..10 {
            let s = state.clone();
            threads.push(thread::spawn(move || {
                for _ in 0..1000 {
                    s.total_ticks.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(state.total_ticks.load(Ordering::SeqCst), 10000);
    }

    #[test]
    fn test_event_log_capacity() {
        let state = DashboardState::new(true);
        for i in 0..250 {
            state.push_event(format!("Event {}", i));
        }

        let log = state.night_events.lock().unwrap();
        assert_eq!(log.len(), 200);
        // Newest should be at the front
        assert_eq!(log[0], "Event 249");
    }
}
