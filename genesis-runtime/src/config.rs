use serde::Deserialize;
use std::path::Path;
use anyhow::{Context, Result};
use std::fs;

/// Configures exactly *what* piece of the brain this node simulates, 
/// and *where* its neighbors are located.
#[derive(Debug, Deserialize, Clone)]
pub struct InstanceConfig {
    /// Reference to the zone folder name (e.g. "V1")
    pub zone_id: String,
    
    /// Offset of this shard in the global brain space (in voxels)
    pub world_offset: Coordinate,
    
    /// Dimensions of this shard (in voxels)
    pub dimensions: Dimensions,
    
    /// Neighborhood topology. "Self" means loopback (toroidal graph mapping),
    /// otherwise an IP:Port string. Left blank if bounded.
    pub neighbors: Neighbors,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Coordinate {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Dimensions {
    pub w: u32,
    pub d: u32,
    pub h: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Neighbors {
    pub x_plus: Option<String>,
    pub x_minus: Option<String>,
    pub y_plus: Option<String>,
    pub y_minus: Option<String>,
}

/// Parses an Instance Config (e.g. `shard_04.toml`) from disk.
pub fn parse_shard_config(path: &Path) -> Result<InstanceConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read InstanceConfig at {:?}", path))?;

    let config: InstanceConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse TOML InstanceConfig from {:?}", path))?;

    Ok(config)
}


// ---- Simulation Configuration (Runtime specifics) ----

pub use genesis_core::config::SimulationConfig;

pub fn parse_simulation_config(path: &Path) -> Result<SimulationConfig> {
    SimulationConfig::load(path).map_err(|e| anyhow::anyhow!(e))
}

// ---- Blueprints Configuration (Neuron Types for GPU LUT) ----

pub use genesis_core::config::blueprints::{BlueprintsConfig, NeuronType as NeuronTypeConfig};

pub fn parse_blueprints_config(path: &Path) -> Result<BlueprintsConfig> {
    BlueprintsConfig::load(path).map_err(|e| anyhow::anyhow!(e))
}
