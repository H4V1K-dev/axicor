use serde::Deserialize;
use std::collections::HashMap;

/// Полный `anatomy.toml` — список слоёв зоны.
#[derive(Debug, Deserialize, Clone)]
pub struct AnatomyConfig {
    #[serde(rename = "layer")]
    pub layers: Vec<LayerConfig>,
}

/// Один [[layer]] блок из anatomy.toml.
#[derive(Debug, Deserialize, Clone)]
pub struct LayerConfig {
    /// Имя слоя, например "L1", "L4", "Nuclear".
    pub name: String,
    /// Высота слоя как доля от world.height_um (0.0..1.0).
    pub height_pct: f32,
    /// [DOD FIX] Абсолютная плотность вокселей
    pub density: f32,
    /// Жёсткие квоты: {type_name → fraction}. Сумма должна быть = 1.0.
    pub composition: HashMap<String, f32>,
}

impl AnatomyConfig {
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
#[path = "test_anatomy.rs"]
mod test_anatomy;
