// Removed Runtime import as we only need ZoneRuntime struct itself
use genesis_core::config::InstanceConfig;
use std::sync::atomic::AtomicBool;
// Removed PingPongSchedule as we use BspBarrier now
use std::time::{Instant, Duration};
use std::sync::Arc;
use genesis_core::layout::VariantParameters;

pub struct ZoneRuntime {
    pub const_mem: [VariantParameters; 16],
    pub is_sleeping: Arc<AtomicBool>,
    pub config: InstanceConfig,
    pub sleep_requested: bool,
    pub prune_threshold: i16,
    pub name: String,
    pub artifact_dir: std::path::PathBuf,
    pub last_night_time: Instant,
    pub min_night_delay: Duration,
    pub slow_path_queues: Arc<crate::network::slow_path::SlowPathQueues>,
    pub hot_reload_queue: Arc<crossbeam::queue::SegQueue<[VariantParameters; 16]>>,
    pub inter_node_channels: Vec<crate::network::inter_node::InterNodeChannel>,
    pub intra_gpu_channels: Vec<crate::network::intra_gpu::IntraGpuChannel>,
    pub spatial_grid: std::sync::Arc<std::sync::Mutex<crate::orchestrator::spatial_grid::SpatialGrid>>,
}

