use super::*;

#[test]
fn test_parse_instance_config() {
    let toml_str = r#"
        zone_id = "V1"

        [world_offset]
        x = 1000
        y = 0
        z = 0

        [dimensions]
        w = 500
        d = 500
        h = 1000

        [neighbors]
        x_minus = "127.0.0.1:8000"
        y_plus = "Self"
		
		[settings]
    "#;

    let config = InstanceConfig::parse(toml_str).expect("Failed to parse");
    assert_eq!(config.zone_id, "V1");
    assert_eq!(config.world_offset.x, 1000);
    assert_eq!(config.dimensions.h, 1000);
    assert_eq!(config.neighbors.x_minus.as_deref(), Some("127.0.0.1:8000"));
    assert_eq!(config.neighbors.y_plus.as_deref(), Some("Self"));
    assert_eq!(config.neighbors.x_plus, None);
}
