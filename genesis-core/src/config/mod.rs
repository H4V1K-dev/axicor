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
pub mod manifest;

pub use blueprints::{BlueprintsConfig, NeuronType};
pub use anatomy::{AnatomyConfig, LayerConfig};
pub use io::{IoConfig, InputMap};
pub use instance::InstanceConfig;
pub use manifest::*;

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
    pub voxel_size_um: f32,
    pub signal_speed_m_s: f32,
    pub sync_batch_ticks: u32,
    
    #[serde(default = "default_segment_length")]
    pub segment_length_voxels: u32,
    
    #[serde(default = "default_max_steps")]
    pub axon_growth_max_steps: u32,
}

fn default_segment_length() -> u32 { 5 }
fn default_max_steps() -> u32 { 2000 }

impl SimulationConfig {
    /// Общее число вокселей для заданного размера вокселя (в мкм).
    pub fn total_voxels(&self) -> u64 {
        let v_um = self.simulation.voxel_size_um;
        if v_um <= 0.0 { return 0; }
        let w = (self.world.width_um as f32 / v_um) as u64;
        let d = (self.world.depth_um as f32 / v_um) as u64;
        let h = (self.world.height_um as f32 / v_um) as u64;
        w * d * h
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
