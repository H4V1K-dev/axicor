use serde::Deserialize;

/// Represents external projection connections coming into this shard (White Matter/Atlas).
#[derive(Debug, Deserialize, Clone, Default)]
pub struct IoConfig {
    #[serde(default)]
    #[serde(rename = "input")]
    pub inputs: Vec<InputMap>,

    #[serde(default)]
    #[serde(rename = "output")]
    pub outputs: Vec<OutputMap>,

    /// Количество тиков в одном батче вывода (по умолчанию равно размеру sync_batch_ticks)
    #[serde(default)]
    pub readout_batch_ticks: Option<u32>,
}


#[derive(Debug, Deserialize, Clone, Default)]
pub struct InputMap {
    pub name: String,
    #[serde(alias = "zone")] 
    pub target_zone: String,
    #[serde(default)] 
    pub target_type: String,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct OutputMap {
    pub name: String,
    #[serde(alias = "zone")]
    pub source_zone: String,
    #[serde(default)]
    pub target_type: String,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
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
