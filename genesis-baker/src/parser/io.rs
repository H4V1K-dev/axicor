//! Парсер IO (делегируется к `genesis_core::config`).

pub use genesis_core::config::{IoConfig, InputChannel};

/// Парсит `io.toml` из строки.
pub fn parse(src: &str) -> anyhow::Result<IoConfig> {
    IoConfig::parse(src).map_err(|e| anyhow::anyhow!(e))
}
