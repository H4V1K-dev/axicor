// genesis-runtime/src/tui/mod.rs
pub mod app;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Глобальное состояние метрик.
/// Обновляется из Hot Loop (Day/Night) через lock-free атомики.
/// Читается из потока TUI для рендера.
pub struct DashboardState {
    pub total_ticks: AtomicU64,
    pub night_count: AtomicU64,
    pub udp_in_packets: AtomicUsize,
    pub udp_out_packets: AtomicUsize,
    pub is_night_phase: std::sync::atomic::AtomicBool,
    pub latest_batch_ms: AtomicU64, // Замеряем время батча в мс
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            total_ticks: AtomicU64::new(0),
            night_count: AtomicU64::new(0),
            udp_in_packets: AtomicUsize::new(0),
            udp_out_packets: AtomicUsize::new(0),
            is_night_phase: std::sync::atomic::AtomicBool::new(false),
            latest_batch_ms: AtomicU64::new(0),
        }
    }
}
