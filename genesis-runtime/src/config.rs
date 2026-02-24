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

#[derive(Debug, Deserialize, Clone)]
pub struct SimulationConfigRoot {
    pub simulation: SimulationRuntime,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SimulationRuntime {
    pub master_seed: String,
    pub sync_batch_ticks: u32,
    pub voxel_size_um: u32,
    pub signal_speed_um_tick: u32,
    #[serde(default = "default_segment_length_voxels")]
    pub segment_length_voxels: u32,
    /// Night Phase trigger interval in ticks. 0 = never (static zone).
    #[serde(default)]
    pub night_interval_ticks: u32,
    pub num_virtual_axons: Option<u32>,
}

fn default_segment_length_voxels() -> u32 { 5 }

pub fn parse_simulation_config(path: &Path) -> Result<SimulationConfigRoot> {
    let content = fs::read_to_string(path)?;
    let config: SimulationConfigRoot = toml::from_str(&content)?;
    Ok(config)
}

// ---- Blueprints Configuration (Neuron Types for GPU LUT) ----

#[derive(Debug, Deserialize, Clone)]
pub struct NeuronTypeConfig {
    pub name: String,
    pub threshold: i32,
    pub rest_potential: i32,
    pub leak_rate: i32,
    pub refractory_period: u8,
    pub synapse_refractory_period: u8,
    pub homeostasis_penalty: i32,
    pub homeostasis_decay: i32,
    pub slot_decay_ltm: u8,
    pub slot_decay_wm: u8,
    // GSOP: optional (defaults to 74 / 2 if not specified)
    #[serde(default = "default_gsop_pot")]
    pub gsop_potentiation: u16,
    #[serde(default = "default_gsop_dep")]
    pub gsop_depression: u16,
    // Active Tail length per-variant (defaults to 10)
    #[serde(default = "default_propagation_length")]
    pub signal_propagation_length: u16,
    // Pruning threshold for Night Phase Sort & Prune (defaults to 15)
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: u16,
}

fn default_gsop_pot() -> u16 { 74 }
fn default_gsop_dep() -> u16 { 2 }
fn default_propagation_length() -> u16 { 10 }
fn default_prune_threshold() -> u16 { 15 }

#[derive(Debug, Deserialize, Clone)]
pub struct BlueprintsConfig {
    #[serde(rename = "neuron_type")]
    pub neuron_types: Vec<NeuronTypeConfig>,
}

pub fn parse_blueprints_config(path: &Path) -> Result<BlueprintsConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read blueprints at {:?}", path))?;
    let config: BlueprintsConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse blueprints TOML from {:?}", path))?;
    Ok(config)
}
