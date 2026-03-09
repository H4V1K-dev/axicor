use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;

/// Configures exactly *what* piece of the brain this node simulates, 
/// and *where* its neighbors are located.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
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

    /// Shard-specific runtime settings
    #[serde(default)]
    pub settings: ShardSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ShardSettings {
    /// How often to save checkpoint.state (in ticks)
    pub save_checkpoints_interval_ticks: u32,
}

impl Default for ShardSettings {
    fn default() -> Self {
        Self {
            save_checkpoints_interval_ticks: 100_000, // Default: every 1000 batches if batch=100
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct Coordinate {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct Dimensions {
    pub w: u32,
    pub d: u32,
    pub h: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct Neighbors {
    pub x_plus: Option<String>,
    pub x_minus: Option<String>,
    pub y_plus: Option<String>,
    pub y_minus: Option<String>,
}

impl InstanceConfig {
    /// Parses an Instance Config (e.g. `shard_04.toml`) from string.
    pub fn parse(src: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(src)
    }

    /// Loads and parses an Instance Config from a file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read InstanceConfig at {:?}: {}", path, e))?;
        Self::parse(&content)
            .map_err(|e| format!("Failed to parse TOML InstanceConfig from {:?}: {}", path, e))
    }
}

#[cfg(test)]
#[path = "test_instance.rs"]
mod test_instance;
