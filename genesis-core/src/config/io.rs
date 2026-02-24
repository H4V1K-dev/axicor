use serde::Deserialize;

/// Represents external projection connections coming into this shard (White Matter/Atlas).
#[derive(Debug, Deserialize, Clone)]
pub struct IoConfig {
    #[serde(rename = "input")]
    pub inputs: Vec<InputChannel>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct InputChannel {
    /// Friendly name describing the origin of this channel (e.g., "From_LGN")
    pub name: String,
    
    /// The target layer inside this zone where these external axons will spread
    pub target_layer: String,
    
    /// Hard quota of axons arriving from this external source
    pub axon_count: u32,
    
    /// The phenotype mask to assign to these incoming segments
    /// Affects downstream plasticity (GSOP LUT applied by the dendrites)
    pub type_mask: u8,
}

impl IoConfig {
    /// Парсит конфиг из TOML строки.
    pub fn parse(src: &str) -> Result<Self, String> {
        toml::from_str(src).map_err(|e| format!("TOML parse error: {}", e))
    }

    /// Загружает конфиг с диска.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {:?}: {}", path, e))?;
        Self::parse(&content)
    }
}

#[cfg(test)]
#[path = "test_io.rs"]
mod test_io;
