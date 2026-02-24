//! Парсер анатомии (делегируется к `genesis_core::config`).

pub use genesis_core::config::{AnatomyConfig as Anatomy, LayerConfig as Layer};

/// Парсит `anatomy.toml` из строки.
pub fn parse(src: &str) -> anyhow::Result<Anatomy> {
    Anatomy::parse(src).map_err(|e| anyhow::anyhow!(e))
}

