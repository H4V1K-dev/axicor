use super::*;

#[test]
fn test_blueprints_parse_full() {
    let toml = r#"
        [[neuron_type]]
        name = "Vertical_Excitatory"
        threshold = 42000
        rest_potential = 10000
        leak_rate = 1200
        refractory_period = 15
        synapse_refractory_period = 15
        conduction_velocity = 200
        signal_propagation_length = 10
        axon_growth_step = 12
        steering_fov_deg = 60.0
        steering_radius_um = 100.0
        steering_weight_inertia = 0.6
        steering_weight_sensor = 0.3
        steering_weight_jitter = 0.1
        homeostasis_penalty = 5000
        homeostasis_decay = 10
        slot_decay_ltm = 160
        slot_decay_wm = 96
        sprouting_weight_distance = 0.5
        sprouting_weight_power = 0.4
        sprouting_weight_explore = 0.1
        gsop_potentiation = 80
        gsop_depression = 3
        prune_threshold = 20
    "#;

    let bp = BlueprintsConfig::parse(toml).unwrap();
    assert_eq!(bp.neuron_types.len(), 1);
    
    let nt = &bp.neuron_types[0];
    assert_eq!(nt.name, "Vertical_Excitatory");
    assert_eq!(nt.threshold, 42000);
    assert_eq!(nt.homeostasis_decay, 10);
    assert_eq!(nt.gsop_potentiation, 80);
    assert_eq!(nt.prune_threshold, 20);
    assert!((nt.sprouting_weight_sum() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_blueprints_parse_minimal_with_defaults() {
    let toml = r#"
        [[neuron_type]]
        name = "Simple"
        threshold = 1000
        rest_potential = 500
        leak_rate = 10
        refractory_period = 5
        synapse_refractory_period = 5
        conduction_velocity = 50
        homeostasis_penalty = 100
        homeostasis_decay = 5
        slot_decay_ltm = 120
        slot_decay_wm = 100
    "#;

    let bp = BlueprintsConfig::parse(toml).unwrap();
    assert_eq!(bp.neuron_types.len(), 1);
    
    let nt = &bp.neuron_types[0];
    assert_eq!(nt.name, "Simple");
    // Проверка default fallbacks
    assert_eq!(nt.signal_propagation_length, 10);
    assert_eq!(nt.axon_growth_step, 12);
    assert_eq!(nt.steering_fov_deg, 60.0);
    assert_eq!(nt.gsop_potentiation, 74);
    assert_eq!(nt.gsop_depression, 2);
    assert_eq!(nt.prune_threshold, 15);
}

#[test]
fn test_blueprints_whitelist_and_initial_weight() {
    let toml = r#"
        [[neuron_type]]
        name = "Excitatory"
        threshold = 1000
        rest_potential = 500
        leak_rate = 10
        refractory_period = 5
        synapse_refractory_period = 5
        conduction_velocity = 50
        homeostasis_penalty = 100
        homeostasis_decay = 5
        slot_decay_ltm = 120
        slot_decay_wm = 100
        dendrite_whitelist = ["Inhibitory", "Relay"]
        initial_synapse_weight = 90
        is_inhibitory = false

        [[neuron_type]]
        name = "Inhibitory"
        threshold = 800
        rest_potential = 400
        leak_rate = 15
        refractory_period = 3
        synapse_refractory_period = 3
        conduction_velocity = 40
        homeostasis_penalty = 80
        homeostasis_decay = 3
        slot_decay_ltm = 100
        slot_decay_wm = 80
        is_inhibitory = true
    "#;

    let bp = BlueprintsConfig::parse(toml).unwrap();
    assert_eq!(bp.neuron_types.len(), 2);

    let e = &bp.neuron_types[0];
    assert_eq!(e.dendrite_whitelist, vec!["Inhibitory", "Relay"]);
    assert_eq!(e.initial_synapse_weight, 90);
    assert!(!e.is_inhibitory);

    let i = &bp.neuron_types[1];
    assert!(i.dendrite_whitelist.is_empty()); // default = []
    assert_eq!(i.initial_synapse_weight, 74); // default
    assert!(i.is_inhibitory);
}
