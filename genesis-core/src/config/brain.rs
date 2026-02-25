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
    }
}
