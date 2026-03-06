use crate::network::ring_buffer::SpikeSchedule;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU32, Ordering};

/// Max wait for peer data before forcing epoch advance. Prevents deadlock if sender dies.
/// 500ms allows ~20 batches slack when sender at ~4k TPS (batch=25ms).
const BSP_SYNC_TIMEOUT_MS: u64 = 500;

/// BSP Барьер для синхронизации сети и вычислителя (Latency Hiding).
/// Мы используем Ping-Pong Double Buffering: пока GPU читает из A, сеть пишет в B.
pub struct BspBarrier {
    pub schedule_a: SpikeSchedule,
    pub schedule_b: SpikeSchedule,
    /// Если true, UDP-сервер пишет в B, а GPU читает из A.
    pub writing_to_b: AtomicBool, 
    // [DOD] Сетевая синхронизация
    pub expected_peers: usize,
    pub current_epoch: AtomicU32,     // [DOD] Global Sync Clock
    pub completed_peers: AtomicUsize, // [DOD] Count of is_last flags
    pub timeout_log_counter: AtomicU32, // Throttle: log every 100th timeout
    pub self_heal_log_counter: AtomicU32, // Throttle: log every 100th self-heal
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
        }
    }

    /// Wait for all peers to deliver their epoch data. Timeout prevents deadlock if sender dies.
    /// [Refactor] 500ms timeout: 50ms was too aggressive when receiver (MotorCortex) runs faster
    /// than sender (SensoryCortex). At ~4k TPS, batch = 25ms; 500ms allows ~20 batches of slack.
    pub fn wait_for_data_sync(&self) {
        let start = std::time::Instant::now();

        while self.completed_peers.load(Ordering::Acquire) < self.expected_peers {
            if start.elapsed() > std::time::Duration::from_millis(BSP_SYNC_TIMEOUT_MS) {
                let n = self.timeout_log_counter.fetch_add(1, Ordering::Relaxed);
                if n % 100 == 0 {
                    println!("⚠️ [BSP] Timeout! Forcing epoch advance ({} timeouts). Dropped data will be filtered out.", n + 1);
                }
                break;
            }
            // Yield instead of spin: receiver often waits for slower sender; burning CPU is wasteful
            std::thread::yield_now();
        }
    }

    /// Вызывается ядром Node в конце батча: меняет буферы местами и инкрементирует эпоху.
    pub fn sync_and_swap(&self) {
        // Сбрасываем барьер для следующей эпохи
        self.current_epoch.fetch_add(1, Ordering::SeqCst);
        self.completed_peers.store(0, Ordering::Release);
        
        let was_b = self.writing_to_b.fetch_xor(true, Ordering::SeqCst);
        if was_b {
            self.schedule_a.clear();
        } else {
            self.schedule_b.clear();
        }
    }

    /// Возвращает ссылку на буфер, в который сейчас должна писать сеть (Tokio).
    pub fn get_write_schedule(&self) -> &SpikeSchedule {
        if self.writing_to_b.load(Ordering::Acquire) {
            &self.schedule_b
        } else {
            &self.schedule_a
        }
    }

    /// Возвращает ссылку на буфер, из которого сейчас должен читать GPU (genesis-compute).
    pub fn get_read_schedule(&self) -> &SpikeSchedule {
        if self.writing_to_b.load(Ordering::Acquire) {
            &self.schedule_a
        } else {
            &self.schedule_b
        }
    }
}
