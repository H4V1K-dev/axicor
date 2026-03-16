use super::*;

#[test]
fn test_anatomy_parse_basic() {
    let toml = r#"
        [[layer]]
        name = "L1"
        height_pct = 0.2
        density = 0.1
        composition = { "Excitatory" = 0.8, "Inhibitory" = 0.2 }

        [[layer]]
        name = "L2"
        height_pct = 0.8
        density = 0.9
        composition = { "Excitatory" = 1.0 }
    "#;

    let anatomy = AnatomyConfig::parse(toml).unwrap();
    assert_eq!(anatomy.layers.len(), 2);
    assert_eq!(anatomy.layers[0].name, "L1");
    assert_eq!(anatomy.layers[1].composition.get("Excitatory"), Some(&1.0));
}
