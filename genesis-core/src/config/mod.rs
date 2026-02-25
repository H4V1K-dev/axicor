/// Архитектура конфигурации (Spec 02 §1.1).
///
/// `simulation.toml` — глобальные "Законы Вселенной". 
/// Единственный парсер для движка (baker и runtime используют этот).
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub mod blueprints;
pub mod anatomy;
pub mod io;
pub mod instance;
pub mod brain;

pub use blueprints::{BlueprintsConfig, NeuronType};
pub use anatomy::{AnatomyConfig, LayerConfig};
pub use io::{IoConfig, InputChannel};
pub use instance::InstanceConfig;

/// Полный распарсенный конфиг `simulation.toml`.
#[derive(Debug, Deserialize, Clone)]
pub struct SimulationConfig {
    pub world: WorldConfig,
    pub simulation: SimulationParams,
}

/// Физические размеры пространства (§1.1).
#[derive(Debug, Deserialize, Clone)]
pub struct WorldConfig {
    pub width_um: u32,
    pub depth_um: u32,
    pub height_um: u32,
}

/// Глобальные параметры симуляции (§1.1).
#[derive(Debug, Deserialize, Clone)]
pub struct SimulationParams {
    pub tick_duration_us: u32,
    pub total_ticks: u64,
    pub master_seed: String,
    pub global_density: f32,
    pub voxel_size_um: u32,
    pub signal_speed_um_tick: u32, // Было разночтение: baker (u16), runtime (u32). Приведено к u32.
    pub sync_batch_ticks: u32,
    
    #[serde(default = "default_segment_length")]
    pub segment_length_voxels: u32,
    
    pub num_virtual_axons: Option<u32>,
    
    #[serde(default = "default_max_steps")]
    pub axon_growth_max_steps: u32,
    
    #[serde(default)] // default 0
    pub night_interval_ticks: u32,
}

fn default_segment_length() -> u32 { 5 }
fn default_max_steps() -> u32 { 2000 }

impl SimulationConfig {
    /// Общее число вокселей для заданного размера вокселя (в мкм).
    pub fn total_voxels(&self) -> u64 {
        let v_um = self.simulation.voxel_size_um;
        if v_um == 0 { return 0; }
        let w = (self.world.width_um / v_um) as u64;
        let d = (self.world.depth_um / v_um) as u64;
        let h = (self.world.height_um / v_um) as u64;
        w * d * h
    }

    /// Максимальное число нейронов = total_voxels * global_density.
    pub fn neuron_budget(&self) -> u64 {
        (self.total_voxels() as f64 * self.simulation.global_density as f64) as u64
    }

    /// Парсит конфиг из строки TOML.
    pub fn parse(src: &str) -> Result<Self, String> {
        toml::from_str(src).map_err(|e| format!("TOML parse error: {}", e))
    }

    /// Загружает конфиг с диска.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {:?}: {}", path, e))?;
        Self::parse(&content)
    }
}

#[cfg(test)]
#[path = "test_config.rs"]
mod test_config;
