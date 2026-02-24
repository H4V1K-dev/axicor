//! Парсер чертежей (делегируется к `genesis_core::config`).

pub use genesis_core::config::{BlueprintsConfig as Blueprints, NeuronType};

/// Парсит `blueprints.toml` из строки.
pub fn parse(src: &str) -> anyhow::Result<Blueprints> {
    Blueprints::parse(src).map_err(|e| anyhow::anyhow!(e))
}

