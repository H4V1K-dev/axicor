use super::*;

#[test]
fn test_io_parse_basic() {
    let toml = r#"
        [[input]]
        name = "From_Thalamus_LGN"
        target_layer = "L1"
        axon_count = 1000
        type_mask = 0x11
        
        [[input]]
        name = "From_Motor_Cortex"
        target_layer = "L4"
        axon_count = 500
        type_mask = 0x22
    "#;

    let io = IoConfig::parse(toml).unwrap();
    assert_eq!(io.inputs.len(), 2);
    
    assert_eq!(io.inputs[0].name, "From_Thalamus_LGN");
    assert_eq!(io.inputs[0].target_layer, "L1");
    assert_eq!(io.inputs[0].axon_count, 1000);
    assert_eq!(io.inputs[0].type_mask, 0x11);
    
    assert_eq!(io.inputs[1].target_layer, "L4");
    assert_eq!(io.inputs[1].axon_count, 500);
}
