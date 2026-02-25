use serde::Deserialize;
use std::path::PathBuf;
use std::fs;

/// Root configuration describing the whole brain (multi-zone setup).
#[derive(Debug, Deserialize)]
pub struct BrainConfig {
    #[serde(default)]
    pub simulation: SimulationConfigRef,
    
    #[serde(rename = "zone", default)]
    pub zones: Vec<ZoneEntry>,

    #[serde(rename = "connection", default)]
    pub connections: Vec<ConnectionEntry>,
}

#[derive(Debug, Deserialize, Default)]
pub struct SimulationConfigRef {
    pub config: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct ZoneEntry {
    pub name: String,
    pub blueprints: PathBuf,
    pub baked_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct ConnectionEntry {
    pub from: String,
    pub to: String,
    // TODO: В будущем здесь будет путь к файлу коннектома (например `baked/connections/v1_v2.ghosts`),
    // содержащему миллионы связей. Пока для MVP используем вложенный массив.
    #[serde(default)]
    pub axon_ids: Vec<u32>,
}

/// Parses the `brain.toml` manifest file.
pub fn parse_brain_config(path: &std::path::Path) -> Result<BrainConfig, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read brain config file at {:?}: {}", path, e))?;

    let config: BrainConfig = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse brain config file {:?}: {}", path, e))?;

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_brain_config() {
        let toml_str = r#"
        [simulation]
        config = "config/simulation.toml"

        [[zone]]
        name = "V1"
        blueprints = "config/zones/V1/blueprints.toml"
        baked_dir = "baked/V1/"

        [[zone]]
        name = "V2"
        blueprints = "config/zones/V2/blueprints.toml"
        baked_dir = "baked/V2/"

        [[connection]]
        from = "V1"
        to = "V2"
        axon_ids = [100, 101, 102]
        "#;

        let config: BrainConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.simulation.config.to_str().unwrap(), "config/simulation.toml");
        assert_eq!(config.zones.len(), 2);
        
        assert_eq!(config.zones[0].name, "V1");
        assert_eq!(config.zones[0].blueprints.to_str().unwrap(), "config/zones/V1/blueprints.toml");
        assert_eq!(config.zones[0].baked_dir.to_str().unwrap(), "baked/V1/");

        assert_eq!(config.zones[1].name, "V2");
        assert_eq!(config.zones[1].blueprints.to_str().unwrap(), "config/zones/V2/blueprints.toml");
        assert_eq!(config.zones[1].baked_dir.to_str().unwrap(), "baked/V2/");

        assert_eq!(config.connections.len(), 1);
        assert_eq!(config.connections[0].from, "V1");
        assert_eq!(config.connections[0].to, "V2");
        assert_eq!(config.connections[0].axon_ids, vec![100, 101, 102]);
    }
}
