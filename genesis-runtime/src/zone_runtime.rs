use crate::Runtime;
use genesis_core::config::{BlueprintsConfig, InstanceConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::network::bsp::PingPongSchedule;
use std::time::{Instant, Duration};
use std::sync::Arc;
use genesis_core::config::manifest::GpuVariantParameters;

pub struct ZoneRuntime {
    pub const_mem: [GpuVariantParameters; 16],
    pub is_sleeping: Arc<AtomicBool>,
    pub ping_pong: Arc<PingPongSchedule>,
    pub config: InstanceConfig,
    pub sleep_requested: bool,
    pub prune_threshold: i16,
    pub runtime: Runtime,
    pub name: String,
    pub artifact_dir: std::path::PathBuf,
    pub last_night_time: Instant,
    pub min_night_delay: Duration,
    pub slow_path_queues: Arc<crate::network::slow_path::SlowPathQueues>,
    pub hot_reload_queue: Arc<crossbeam::queue::SegQueue<[GpuVariantParameters; 16]>>,
    pub inter_node_channels: Vec<crate::network::inter_node::InterNodeChannel>,
    pub intra_gpu_channels: Vec<crate::network::intra_gpu::IntraGpuChannel>,
    pub spatial_grid: std::sync::Arc<std::sync::Mutex<crate::orchestrator::spatial_grid::SpatialGrid>>,
    pub dashboard: Arc<crate::tui::DashboardState>,
}

