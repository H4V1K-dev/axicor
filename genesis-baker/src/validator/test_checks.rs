use super::*;
use crate::parser::simulation::{SimulationConfig, SimulationParams, WorldConfig};
use crate::parser::blueprints::{Blueprints, NeuronType};

fn make_sim_config(speed: u32, len_voxels: u32) -> SimulationConfig {
    SimulationConfig {
        world: WorldConfig {
            width_um: 50,
            depth_um: 50,
            height_um: 50,
        },
        simulation: SimulationParams {
            voxel_size_um: 25,
            segment_length_voxels: len_voxels, // -> v_seg = speed / (25 * len_voxels)
            axon_growth_max_steps: 100,
            tick_duration_us: 1000,
            total_ticks: 100_000,
            master_seed: "0".to_string(),
            global_density: 1.0,
            signal_speed_um_tick: speed,
            sync_batch_ticks: 10,
            num_virtual_axons: None,
            night_interval_ticks: 1000,
        },
    }
}

fn make_blueprints_with_prop(prop_len: u16) -> Blueprints {
    let nt = NeuronType {
        name: "TestNeuron".to_string(),
        growth_vertical_bias: 0.0,
        steering_fov_deg: 90.0,
        steering_radius_um: 150.0,
        steering_weight_inertia: 0.1,
        steering_weight_sensor: 0.9,
        steering_weight_jitter: 0.0,
        initial_synapse_weight: 74,
        dendrite_whitelist: vec![],
        is_inhibitory: false,
        sprouting_weight_type: 0.0,
        type_affinity: 1.0,
        inertia_curve: [0; 16],
        ltm_slot_count: 80,
        conduction_velocity: 1,
        signal_propagation_length: prop_len,
        homeostasis_penalty: 1,
        homeostasis_decay: 1,
        slot_decay_ltm: 1,
        slot_decay_wm: 1,
        threshold: 1000,
        rest_potential: 0,
        leak_rate: 10,
        refractory_period: 5,
        synapse_refractory_period: 5,
        axon_growth_step: 10,
        gsop_depression: 10,
        gsop_potentiation: 10,
        sprouting_weight_distance: 1.0,
        sprouting_weight_power: 0.0,
        sprouting_weight_explore: 0.0,
        prune_threshold: 10,
    };

    Blueprints {
        neuron_types: vec![nt],
    }
}

#[test]
fn test_validator_accepts_prop_eq_v_seg() {
    // speed=50, voxel=25, voxels_per_seg=2 -> speed=50, seg_um=50 -> v_seg=1
    let sim = make_sim_config(50, 2);
    // prop_len = 1 >= v_seg (1) -> Ok
    let blueprints = make_blueprints_with_prop(1);

    let res = check_propagation_covers_v_seg(&sim, &blueprints);
    assert!(res.is_ok(), "Validator should accept prop_len >= v_seg");
}

#[test]
fn test_validator_accepts_prop_gt_v_seg() {
    // v_seg = 1
    let sim = make_sim_config(50, 2);
    // prop_len = 5 >= v_seg (1) -> Ok
    let blueprints = make_blueprints_with_prop(5);

    let res = check_propagation_covers_v_seg(&sim, &blueprints);
    assert!(res.is_ok(), "Validator should accept prop_len > v_seg");
}

#[test]
fn test_validator_rejects_prop_lt_v_seg() {
    // speed=100, voxel=25, voxels_per_seg=1 -> speed=100, seg_um=25 -> v_seg=4
    let sim = make_sim_config(100, 1);
    // prop_len = 3 < v_seg (4) -> Error!
    let blueprints = make_blueprints_with_prop(3);

    let res = check_propagation_covers_v_seg(&sim, &blueprints);
    assert!(res.is_err(), "Validator MUST reject prop_len < v_seg");
    
    let err_msg = res.unwrap_err().to_string();
    assert!(err_msg.contains("нарушает §1.1 Invariant"));
    assert!(err_msg.contains("signal_propagation_length (3) < v_seg (4)"));
}
