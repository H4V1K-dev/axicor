use std::time::Instant;
use std::collections::VecDeque;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase { Day, Night, Sleep }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LogLevel { Info, Warning, Night, Error }

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FocusedPanel {
    ZoneTable,
    EventLog,
}

pub struct LogEntry {
    pub timestamp: String,
    pub message: String,
    pub level: LogLevel,
}

pub struct ZoneMetrics {
    pub hash: u32,
    pub name: String,
    pub short_name: String,
    pub neuron_count: u32,
    pub axon_count: u32,
    pub spikes_last_batch: u32,
    pub spike_rate: f64,
    pub phase: Phase,
    pub night_interval_ticks: u64,
}

pub struct DashboardState {
    pub is_running: bool,
    
    // Core Loop
    pub batch_number: u64,
    pub total_ticks: u64,
    pub uptime: Instant,
    pub wall_ms_history: VecDeque<u64>, // changed from f64 to u64 to suit ratatui sparkline better (or keeping u64 ms is simpler)
    pub ticks_per_sec: f64,

    // Per-Zone
    pub zones: Vec<ZoneMetrics>,

    // I/O
    pub udp_in_packets: u64,
    pub udp_out_packets: u64,
    pub oversized_skips: u64,
    pub connected_clients: u32,

    // VRAM
    pub vram_used_mb: f64,
    pub vram_total_mb: f64,
    pub vram_somas_mb: f64,
    pub vram_dendrites_mb: f64,
    pub vram_axons_mb: f64,

    // Night Phase
    pub night_count: u32,
    pub night_interval_ticks: u64,
    pub global_phase: Phase,

    // Event Log
    pub events: VecDeque<LogEntry>,

    // UI State
    pub focus: FocusedPanel,
    pub zone_scroll: usize,
    pub log_scroll: usize,
    pub selected_zone_idx: Option<usize>,
}

impl DashboardState {
    pub fn new() -> Self {
        Self {
            is_running: true,
            batch_number: 0,
            total_ticks: 0,
            uptime: Instant::now(),
            wall_ms_history: VecDeque::with_capacity(60),
            ticks_per_sec: 0.0,
            zones: Vec::new(),
            udp_in_packets: 0,
            udp_out_packets: 0,
            oversized_skips: 0,
            connected_clients: 0,
            vram_used_mb: 0.0,
            vram_total_mb: 0.0,
            vram_somas_mb: 0.0,
            vram_dendrites_mb: 0.0,
            vram_axons_mb: 0.0,
            night_count: 0,
            night_interval_ticks: 200_000,
            global_phase: Phase::Day,
            events: VecDeque::with_capacity(200),
            focus: FocusedPanel::ZoneTable,
            zone_scroll: 0,
            log_scroll: 0,
            selected_zone_idx: None,
        }
    }

    pub fn push_log(&mut self, message: String, level: LogLevel) {
        let now = chrono::Local::now();
        let timestamp = now.format("%H:%M:%S").to_string();
        
        if self.events.len() >= 200 {
            self.events.pop_front();
        }
        self.events.push_back(LogEntry { timestamp, message, level });
    }

    pub fn push_wall_ms(&mut self, ms: u64) {
        if self.wall_ms_history.len() >= 60 {
            self.wall_ms_history.pop_front();
        }
        self.wall_ms_history.push_back(ms);
    }
}

use std::sync::atomic::{AtomicU64, AtomicU32, AtomicI16, Ordering};
use crossbeam::queue::SegQueue;

/// Lock-Free мост между HFT-потоками и UI-рендером
pub struct LockFreeTelemetry {
    pub batch_number: AtomicU64,
    pub total_ticks: AtomicU64,
    pub wall_ms: AtomicU64,
    pub udp_out_packets: AtomicU64,
    pub dopamine: AtomicI16,
    pub logs: SegQueue<LogEntry>,
    // Быстрый маппинг для спайков зон (до 16 зон для MVP)
    pub zone_spikes: [AtomicU32; 16],
    pub zone_hashes: [AtomicU32; 16],
}

impl Default for LockFreeTelemetry {
    fn default() -> Self {
        Self {
            batch_number: AtomicU64::new(0),
            total_ticks: AtomicU64::new(0),
            wall_ms: AtomicU64::new(0),
            udp_out_packets: AtomicU64::new(0),
            dopamine: AtomicI16::new(0),
            logs: SegQueue::new(),
            zone_spikes: Default::default(),
            zone_hashes: Default::default(),
        }
    }
}

impl LockFreeTelemetry {
    pub fn push_log(&self, message: String, level: LogLevel) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.logs.push(LogEntry { timestamp, message, level });
    }

    /// O(1) поиск индекса атомика зоны
    pub fn update_zone_spikes(&self, hash: u32, spikes: u32) {
        for i in 0..16 {
            let h = self.zone_hashes[i].load(Ordering::Relaxed);
            if h == hash {
                self.zone_spikes[i].store(spikes, Ordering::Relaxed);
                return;
            } else if h == 0 {
                // Ленивая инициализация слота
                if self.zone_hashes[i].compare_exchange(0, hash, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                    self.zone_spikes[i].store(spikes, Ordering::Relaxed);
                }
                return;
            }
        }
    }
}
