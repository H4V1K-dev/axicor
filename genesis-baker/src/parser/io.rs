use serde::Deserialize;

/// Represents external projection connections coming into this shard (White Matter/Atlas).
#[derive(Debug, Deserialize)]
pub struct IoConfig {
    pub input: Vec<InputChannel>,
}

#[derive(Debug, Deserialize)]
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

/// Parse `io.toml` from a string
pub fn parse(src: &str) -> anyhow::Result<IoConfig> {
    let io: IoConfig = toml::from_str(src)?;
    Ok(io)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"
[[input]]
name = "From_Thalamus_LGN"
target_layer = "L4"
axon_count = 15000
type_mask = 0x11 # Example: Geo/Sign/Variant

[[input]]
name = "From_Motor_Cortex"
target_layer = "L1"
axon_count = 5000
type_mask = 0x12
"#;

    #[test]
    fn parse_io_example() {
        let io = parse(EXAMPLE).expect("parse failed");
        assert_eq!(io.input.len(), 2);
        
        let lgn = &io.input[0];
        assert_eq!(lgn.name, "From_Thalamus_LGN");
        assert_eq!(lgn.target_layer, "L4");
        assert_eq!(lgn.axon_count, 15000);
        assert_eq!(lgn.type_mask, 0x11);
    }
}
