use std::path::Path;
use anyhow::Result;
pub use genesis_core::config::instance::{InstanceConfig, Coordinate, Dimensions, Neighbors};

/// Parses an Instance Config (e.g. `shard_04.toml`) from disk.
pub fn parse_shard_config(path: &Path) -> Result<InstanceConfig> {
    InstanceConfig::load(path).map_err(|e| anyhow::anyhow!(e))
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
