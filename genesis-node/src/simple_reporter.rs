use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct SimpleReporter {
    total_ticks: Arc<AtomicU64>,
    total_spikes: Arc<AtomicUsize>,
    start_time: Instant,
}

impl SimpleReporter {
    pub fn new() -> Self {
        Self {
            total_ticks: Arc::new(AtomicU64::new(0)),
            total_spikes: Arc::new(AtomicUsize::new(0)),
            start_time: Instant::now(),
        }
    }

    pub fn update(&self, ticks: u64, spikes: usize, batch_ms: u64) {
        self.total_ticks.store(ticks, Ordering::Relaxed);
        self.total_spikes.fetch_add(spikes, Ordering::Relaxed);

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let tps = ticks as f64 / elapsed.max(0.001);
        let sps = spikes as f64 / elapsed.max(0.001);

        eprint!(
            "\r[Genesis] Ticks: {:<8} | TPS: {:<7.0} | Spikes: {:<10} | SPS: {:<12.0} | Batch: {:<4}ms | Elapsed: {:.1}s",
            ticks, tps, spikes, sps, batch_ms, elapsed
        );
        std::io::Write::flush(&mut std::io::stderr()).ok();
    }

    pub fn finish(&self) {
        eprintln!("\n[Genesis] Simulation complete.");
    }
}
