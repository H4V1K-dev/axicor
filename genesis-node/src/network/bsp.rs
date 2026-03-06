use crate::network::ring_buffer::SpikeSchedule;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU32, Ordering};
use std::time::{Instant, Duration};
use dashmap::DashMap;
use thiserror::Error;
 
#[derive(Debug, Error)]
pub enum BspError {
    #[error("Node Isolated: Network fail or Peer 0x{0:08X} dead")]
    NodeIsolated(u32),
}
 
/// BSP Барьер для синхронизации сети и вычислителя (Latency Hiding).
/// Мы используем Ping-Pong Double Buffering: пока GPU читает из A, сеть пишет в B.
pub struct BspBarrier {
    pub schedule_a: SpikeSchedule,
    pub schedule_b: SpikeSchedule,
    /// Если true, UDP-сервер пишет в B, а GPU читает из A.
    pub writing_to_b: AtomicBool, 
    // [DOD] Сетевая синхронизация
    pub expected_peers: usize,
    pub expected_peers_hashes: Vec<u32>,
    pub current_epoch: AtomicU32,      // [DOD] Global Sync Clock
    pub completed_peers: AtomicUsize,  // [DOD] Count of is_last flags
    pub peer_last_seen: DashMap<u32, Instant>, // [NEW Phase 49]
}
 
impl BspBarrier {
    pub fn new(sync_batch_ticks: usize, peers: Vec<u32>) -> Self {
        let expected_peers = peers.len();
        Self {
            schedule_a: SpikeSchedule::new(sync_batch_ticks),
            schedule_b: SpikeSchedule::new(sync_batch_ticks),
            writing_to_b: AtomicBool::new(true),
            expected_peers,
            expected_peers_hashes: peers,
            current_epoch: AtomicU32::new(0),
            completed_peers: AtomicUsize::new(0),
            peer_last_seen: DashMap::new(),
        }
    }
 
    /// Ожидает данные от соседей с активным мониторингом здоровья (Pulse).
    pub fn wait_for_data_sync(&self) -> Result<(), BspError> {
        let start = Instant::now();
        let timeout = Duration::from_millis(50); // Мягкий таймаут для обычного цикла
        let liveness_timeout = Duration::from_millis(500); // Жесткий таймаут изоляции ноды
 
        // Ждем, пока Ingress UDP-сервер не запишет пакеты от всех соседей
        while self.completed_peers.load(Ordering::Acquire) < self.expected_peers {
            let now = Instant::now();
            
            // Проверка здоровья каждого ожидаемого пира
            for &peer_hash in &self.expected_peers_hashes {
                let last_seen = self.peer_last_seen.get(&peer_hash)
                    .map(|v| *v)
                    .unwrap_or(start); // Если ни разу не видели, считаем от начала барьера
                
                if now.duration_since(last_seen) > liveness_timeout {
                    return Err(BspError::NodeIsolated(peer_hash));
                }
            }
 
            if start.elapsed() > timeout {
                // Если мы еще не превысили liveness timeout, но превысили soft timeout, 
                // мы продолжаем ждать, но можем залогировать задержку.
                // println!("⚠️ [BSP] Soft Timeout! Waiting for peers. Current: {}/{}", self.completed_peers.load(Ordering::Acquire), self.expected_peers);
                std::hint::spin_loop(); // Continue spinning, don't break yet
            }
            
            // [DOD] Выжигаем токены CPU минимально, не отдавая тред ОС
            std::hint::spin_loop();
        }
        Ok(())
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
