#[cfg(test)]
mod tests {
    use crate::bake::axon_growth::{grow_axons, LayerZRange, ShardBounds};
    use crate::bake::neuron_placement::PlacedNeuron;
    use genesis_core::config::blueprints::{NeuronType};
    use crate::parser::simulation::{SimulationConfig, SimulationParams, WorldConfig};
    use genesis_core::coords::pack_position;


    fn make_sim_config(w: u32, d: u32, h: u32) -> SimulationConfig {
        SimulationConfig {
            world: WorldConfig {
                width_um: w * 50,
                depth_um: d * 50,
                height_um: h * 50,
            },
            simulation: SimulationParams {
                voxel_size_um: 50.0,
                segment_length_voxels: 1, // 1 voxel step
                axon_growth_max_steps: 100,
                tick_duration_us: 1000,
                total_ticks: 100_000,
                master_seed: "0".to_string(),
                global_density: 1.0,
                signal_speed_m_s: 0.05,
                sync_batch_ticks: 10,
                night_interval_ticks: 1000,
            },
        }
    }

    fn make_neuron(x: u32, y: u32, z: u32, t: u8) -> PlacedNeuron {
        PlacedNeuron {
            position: pack_position(x, y, z, t as u32),
            type_idx: t as usize,
            layer_name: "TestLayer".to_string(),
        }
    }

    fn make_v_type() -> NeuronType {
        NeuronType {
            name: "V_Type".to_string(),
            growth_vertical_bias: 1.0, // Push upward/downward only
            steering_fov_deg: 90.0,
            steering_radius_um: 150.0, // 3 voxels
            steering_weight_inertia: 0.1,
            steering_weight_sensor: 0.9,
            steering_weight_jitter: 0.0, // No noise to make deterministic
            initial_synapse_weight: 100,
            dendrite_whitelist: vec![],
            is_inhibitory: false,
            type_affinity: 1.0, // Follow same type
            inertia_curve: [0; 16],
            ltm_slot_count: 80,
            
            // Dummy values for parsing compliance
            conduction_velocity: 1,
            signal_propagation_length: 5,
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
            sprouting_weight_distance: 0.4,
            sprouting_weight_power: 0.4,
            sprouting_weight_explore: 0.1,
            sprouting_weight_type: 0.1,
            prune_threshold: 10,
        }
    }



    fn setup_env_v(w: u32, d: u32, h: u32) -> (SimulationConfig, Vec<LayerZRange>, Vec<NeuronType>, ShardBounds) {
        let sim = make_sim_config(w, d, h);
        let bounds = ShardBounds::full_world(&sim);
        let layers = vec![
            LayerZRange { name: "L1".to_string(), z_start_vox: 0, z_end_vox: 10 },    // L1 (z: 0..10)
            LayerZRange { name: "L2".to_string(), z_start_vox: 10, z_end_vox: 20 },   // L2 (z: 10..20)
        ];
        let types = vec![make_v_type()];
        (sim, layers, types, bounds)
    }

    #[test]
    fn test_v_neuron_grows_upward() {
        let (sim, layers, types, bounds) = setup_env_v(10, 10, 20);
        let neurons = vec![make_neuron(5, 5, 2, 0)]; // In L1

        let (axons, ghosts) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 42);

        assert_eq!(axons.len(), 1);
        assert!(ghosts.is_empty());

        let axon = &axons[0];
        // Target is L2 (z=10..20). Since soma_rel_z is 2/10 = 0.2, target tip_z = 10 + 0.2*10 = 12.
        assert!(axon.tip_z > 2); // It grew up
        assert_eq!(axon.tip_z, 12); // Reached target plane precisely
    }

    #[test]
    fn test_h_neuron_stays_in_layer() {
        let (sim, layers, mut types, bounds) = setup_env_v(100, 100, 20);
        types[0].growth_vertical_bias = 0.0; // Make it horizontal

        let neurons = vec![make_neuron(50, 50, 5, 0)]; // In L1 (z=0..10)

        let (axons, _ghosts) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 42);
        
        assert_eq!(axons.len(), 1);
        let axon = &axons[0];
        
        // Horizontal growth should stop exactly if it ever leaves the layer.
        // But since bias=0, it moves only in XY and Z=5.
        // It should max out steps or hit stagnation.
        for packed in &axon.segments {
            let z = (packed >> 20) & 0xFF;
            assert!(z <= 10, "Escaped layer! z={}", z);
        }
    }

    #[test]
    fn test_axon_length_nonzero() {
        let (sim, layers, types, bounds) = setup_env_v(5, 5, 5);
        let neurons = vec![make_neuron(2, 2, 2, 0)];
        let (axons, _) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 42);
        
        assert_eq!(axons.len(), 1);
        assert!(axons[0].length_segments >= 1);
    }

    #[test]
    fn test_stagnation_breaks() {
        // Make world tiny so it can't go anywhere
        let (sim, layers, types, bounds) = setup_env_v(1, 1, 1);
        let neurons = vec![make_neuron(0, 0, 0, 0)];

        let (axons, _) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 42);
        
        // Should hit stagnation and break early before max_steps
        assert!(axons[0].length_segments < 100); 
    }

    #[test]
    fn test_segment_packing() {
        let (sim, layers, types, bounds) = setup_env_v(10, 10, 10);
        let neurons = vec![make_neuron(0, 0, 0, 0)]; // t=0
        
        let (axons, _) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 42);
        let last_packed = *axons[0].segments.last().unwrap();
        
        // Unpack:
        let z = (last_packed >> 20) & 0xFF;
        let y = (last_packed >> 10) & 0x3FF;
        let x = last_packed & 0x3FF;
        let t = last_packed >> 28;
        
        assert_eq!(x, axons[0].tip_x);
        assert_eq!(y, axons[0].tip_y);
        assert_eq!(z, axons[0].tip_z);
        assert_eq!(t, 0); // Type 0
    }

    #[test]
    fn test_ghost_packet_on_boundary() {
        let (sim, layers, types, _full_bounds) = setup_env_v(10, 10, 10);
        let mut sim_small = sim.clone();
        sim_small.world.width_um = 3 * 50; 
        sim_small.world.depth_um = 3 * 50;
        let bounds = ShardBounds {
            x_start: 0, x_end: 2,
            y_start: 0, y_end: 2,
            z_start: 0, z_end: 10,
        };

        let neurons = vec![
            make_neuron(1, 1, 5, 0), // Inside bounds
            make_neuron(8, 8, 5, 0), // Way outside bounds
        ];

        let mut h_types = types.clone();
        h_types[0].growth_vertical_bias = 0.0;
        h_types[0].steering_weight_jitter = 0.0; 
        h_types[0].steering_weight_sensor = 1.0; 
        h_types[0].steering_fov_deg = 360.0; // omnidirectional
        h_types[0].steering_radius_um = 1000.0; // global view

        let (_axons, ghosts) = grow_axons(&neurons, &layers, &h_types, &sim_small, &bounds, 42);
        
        // At least Neuron 0 should emit a ghost packet
        assert!(ghosts.len() >= 1);

        let gp = ghosts.iter().find(|g| g.soma_idx == 0).expect("Neuron 0 did not cross boundary!");
        assert!(gp.entry_x >= 2 || gp.entry_y >= 2); 
    }

    #[test]
    fn test_no_ghost_in_full_world() {
        let (sim, layers, types, bounds) = setup_env_v(10, 10, 10);
        let neurons = vec![make_neuron(9, 9, 9, 0)]; // At edge
        // In full_world, boundaries are exactly where max_xyz is anyway, 
        // but crossing check is bypassed via boundaries = u32::MAX
        
        let (_axons, ghosts) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 42);
        assert!(ghosts.is_empty());
    }

    #[test]
    fn test_cone_influence_on_direction() {
        let (sim, layers, types, bounds) = setup_env_v(20, 20, 20);
        
        let neurons = vec![
            make_neuron(5, 5, 5, 0),   // Soma
            make_neuron(15, 15, 5, 0), // Big target Far away (H-neuron path)
        ];

        let mut h_types = types.clone();
        h_types[0].growth_vertical_bias = 0.0; // H-growth
        h_types[0].steering_weight_sensor = 1.0; // Pull heavily towards target
        h_types[0].steering_weight_inertia = 0.0; 
        h_types[0].steering_fov_deg = 360.0;
        h_types[0].steering_radius_um = 1000.0; // Huge radius to see target
        
        let (axons, _) = grow_axons(&neurons, &layers, &h_types, &sim, &bounds, 42);
        let axon = &axons[0];
        
        // Axon should have moved towards (15, 15, 5)
        assert!(axon.tip_x > 5);
        assert!(axon.tip_y > 5);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let (sim, layers, mut types, bounds) = setup_env_v(20, 20, 20);
        types[0].growth_vertical_bias = 0.5;
        types[0].steering_weight_jitter = 1.0; // Max random noise
        
        let neurons = vec![make_neuron(10, 10, 5, 0)];

        let (axons1, _) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 12345);
        let (axons2, _) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 12345);
        let (axons3, _) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 54321);

        // Same seed -> identical
        assert_eq!(axons1[0].tip_x, axons2[0].tip_x);
        assert_eq!(axons1[0].tip_y, axons2[0].tip_y);
        
        // Diff seed -> likely different (chance of exact collision is low but possible, check x != x || y != y)
        assert!(axons1[0].tip_x != axons3[0].tip_x || axons1[0].tip_y != axons3[0].tip_y);
    }

    #[test]
    fn test_coordinates_within_world() {
        let (sim, layers, mut types, bounds) = setup_env_v(10, 10, 10);
        types[0].steering_weight_jitter = 1.0; // Walk randomly
        types[0].growth_vertical_bias = 0.0; // H-Type
        
        let neurons = vec![make_neuron(5, 5, 5, 0)];

        let (axons, _) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 42);
        
        for packed in &axons[0].segments {
            let z = (packed >> 20) & 0xFF;
            let y = (packed >> 10) & 0x3FF;
            let x = packed & 0x3FF;
            
            assert!(x < 10, "X OOB: {}", x);
            assert!(y < 10, "Y OOB: {}", y);
            assert!(z < 10, "Z OOB: {}", z);
        }
    }

    #[test]
    fn test_single_layer_growth() {
        let sim = make_sim_config(40, 40, 20);
        let layers = vec![
            LayerZRange { name: "Single".to_string(), z_start_vox: 0, z_end_vox: 20 },
        ];
        let types = vec![make_v_type()];
        let bounds = ShardBounds::full_world(&sim);
        
        let neuron = PlacedNeuron { position: pack_position(20, 20, 5, 0), layer_name: "Single".to_string(), type_idx: 0 };
        
        let (axons, _) = grow_axons(&[neuron], &layers, &types, &sim, &bounds, 42);
        
        assert!(!axons[0].segments.is_empty(), "Axon in single layer should have segments! tip_z was likely equal to soma_z because target_layer was None");
    }
}
