use super::*;

#[test]
fn test_anatomy_parse_basic() {
    let toml = r#"
        [[layer]]
        name = "L1"
        height_pct = 0.2
        population_pct = 0.1
        composition = { "Excitatory" = 0.8, "Inhibitory" = 0.2 }

        [[layer]]
        name = "L2"
        height_pct = 0.8
        population_pct = 0.9
        composition = { "Excitatory" = 1.0 }
    "#;

    let anatomy = AnatomyConfig::parse(toml).unwrap();
    assert_eq!(anatomy.layers.len(), 2);
    assert_eq!(anatomy.layers[0].name, "L1");
    assert_eq!(anatomy.layers[1].composition.get("Excitatory"), Some(&1.0));
}

#[test]
fn test_neuron_counts_calculation() {
    let toml = r#"
        [[layer]]
        name = "L1"
        height_pct = 1.0
        population_pct = 1.0
        composition = { "TypeA" = 0.5, "TypeB" = 0.5 }
    "#;

    let anatomy = AnatomyConfig::parse(toml).unwrap();
    let counts = anatomy.neuron_counts(1000);
    
    // Ожидаем 2 записи по 500
    assert_eq!(counts.len(), 2);
    let type_a = counts.iter().find(|(_, t, _)| t == "TypeA").unwrap();
    let type_b = counts.iter().find(|(_, t, _)| t == "TypeB").unwrap();
    assert_eq!(type_a.2, 500);
    assert_eq!(type_b.2, 500);
}
