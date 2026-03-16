use super::*;

#[test]
fn test_io_parse_basic() {
    let toml = r#"
        [[input]]
        name = "retina_edges"
        target_zone = "V1"
        target_type = "L4_Stellate"
        width = 64
        height = 64
		stride = 1
        
        [[input]]
        name = "audio_spectrogram"
        target_zone = "A1"
        target_type = "ALL"
        width = 128
        height = 16
		stride = 1
    "#;

    let io = IoConfig::parse(toml).unwrap();
    assert_eq!(io.inputs.len(), 2);
    
    assert_eq!(io.inputs[0].name, "retina_edges");
    assert_eq!(io.inputs[0].target_zone, "V1");
    assert_eq!(io.inputs[0].target_type, "L4_Stellate");
    assert_eq!(io.inputs[0].width, 64);
    assert_eq!(io.inputs[0].height, 64);
    
    assert_eq!(io.inputs[1].name, "audio_spectrogram");
    assert_eq!(io.inputs[1].target_zone, "A1");
    assert_eq!(io.inputs[1].target_type, "ALL");
    assert_eq!(io.inputs[1].width, 128);
    assert_eq!(io.inputs[1].height, 16);
}

#[test]
fn test_io_parse_with_outputs() {
    let toml = r#"
        readout_batch_ticks = 100

        [[output]]
        name = "motor_arm"
        source_zone = "M1"
        target_type = "L5_Pyramidal"
        width = 8
        height = 8
		stride = 1
        
        [[output]]
        name = "motor_leg"
        source_zone = "M1"
        target_type = "ALL"
        width = 4
        height = 4
		stride = 1
    "#;

    let io = IoConfig::parse(toml).unwrap();
    assert_eq!(io.readout_batch_ticks, Some(100));
    assert_eq!(io.outputs.len(), 2);
    
    assert_eq!(io.outputs[0].name, "motor_arm");
    assert_eq!(io.outputs[0].source_zone, "M1");
    assert_eq!(io.outputs[0].target_type, "L5_Pyramidal");
    assert_eq!(io.outputs[0].width, 8);
    assert_eq!(io.outputs[0].height, 8);
    
    assert_eq!(io.outputs[1].name, "motor_leg");
    assert_eq!(io.outputs[1].source_zone, "M1");
    assert_eq!(io.outputs[1].target_type, "ALL");
    assert_eq!(io.outputs[1].width, 4);
    assert_eq!(io.outputs[1].height, 4);
}
