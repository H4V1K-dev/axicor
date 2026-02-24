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
    /// Доля от общего нейронного бюджета зоны (0.0..1.0).
    pub population_pct: f32,
    /// Жёсткие квоты: {type_name → fraction}. Сумма должна быть = 1.0.
    pub composition: HashMap<String, f32>,
}

impl AnatomyConfig {
    /// Рассчитывает абсолютное число нейронов каждого типа в каждом слое.
    /// Возвращает: Vec<(layer_name, type_name, count)>
    pub fn neuron_counts(&self, total_budget: u64) -> Vec<(String, String, u64)> {
        let mut result = Vec::new();
        for layer in &self.layers {
            let layer_budget = (total_budget as f64 * layer.population_pct as f64) as u64;
            for (type_name, &quota) in &layer.composition {
                let count = (layer_budget as f64 * quota as f64) as u64;
                result.push((layer.name.clone(), type_name.clone(), count));
            }
        }
        result
    }

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
