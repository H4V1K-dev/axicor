// genesis-baker/src/bake/test_dendrite_connect.rs

#[cfg(test)]
mod tests {
    use crate::bake::axon_growth::{GrownAxon, LayerZRange, ShardBounds, grow_axons};
    use crate::bake::dendrite_connect::connect_dendrites;
    use crate::bake::layout::ShardSoA;
    use crate::bake::neuron_placement::PlacedNeuron;
    use crate::parser::simulation::{SimulationConfig, SimulationParams, WorldConfig};
    use genesis_core::config::blueprints::{GenesisConstantMemory, NeuronType};
    use genesis_core::layout::{VariantParameters, unpack_axon_id};
    use genesis_core::coords::{pack_position};
    use genesis_core::constants::MAX_DENDRITE_SLOTS;

    fn make_sim_config(w: u32, d: u32, h: u32) -> SimulationConfig {
        SimulationConfig {
            world: WorldConfig {
                width_um: w * 50,
                depth_um: d * 50,
                height_um: h * 50,
            },
            simulation: SimulationParams {
                voxel_size_um: 50.0,
                segment_length_voxels: 1,
                axon_growth_max_steps: 100,
                tick_duration_us: 100,
                total_ticks: 100_000,
                master_seed: "0".to_string(),
                global_density: 1.0,
                signal_speed_m_s: 0.5,
                sync_batch_ticks: 10,
                night_interval_ticks: 1000,
            },
        }
    }

    fn make_neuron(x: u32, y: u32, z: u32, t: usize) -> PlacedNeuron {
        PlacedNeuron {
            position: pack_position(x, y, z, t as u32),
            type_idx: t,
            layer_name: "TestLayer".to_string(),
        }
    }

    fn make_const_mem() -> GenesisConstantMemory {
        let variant = VariantParameters {
            threshold: 1000,
            rest_potential: 0,
            leak_rate: 10,
            homeostasis_penalty: 1,
            gsop_potentiation: 74,
            gsop_depression: 0,  // positive -> excitatory
            homeostasis_decay: 1,
            signal_propagation_length: 5,
            conduction_velocity: 1,
            slot_decay_ltm: 1,
            slot_decay_wm: 1,
            refractory_period: 5,
            synapse_refractory_period: 5,
            ..Default::default()
        };
        let inhibitory_variant = VariantParameters {
            gsop_depression: -74, // negative -> inhibitory
            gsop_potentiation: 74,
            ..variant
        };
        let mut variants = [variant; 16];
        variants[1] = inhibitory_variant;
        GenesisConstantMemory { variants }
    }

    fn make_types() -> Vec<NeuronType> {
        let mut types = vec![];
        for i in 0..2 {
            types.push(NeuronType {
                name: format!("Type_{}", i),
                is_inhibitory: i == 1,
                growth_vertical_bias: 0.5,
                steering_fov_deg: 90.0,
                steering_radius_um: 150.0,
                steering_weight_inertia: 0.5,
                steering_weight_sensor: 0.5,
                steering_weight_jitter: 0.0,
                signal_propagation_length: 5,
                conduction_velocity: 1,
                ..NeuronType::default()
            });
        }
        types
    }

    fn make_axon(soma_idx: usize, type_idx: usize, segments: Vec<u32>) -> GrownAxon {
        let last = *segments.last().unwrap_or(&0);
        let tip_z = (last >> 20) & 0xFF;
        let tip_y = (last >> 10) & 0x3FF;
        let tip_x = last & 0x3FF;
        GrownAxon {
            soma_idx,
            type_idx,
            tip_x,
            tip_y,
            tip_z,
            length_segments: segments.len() as u32,
            segments,
            last_dir: glam::Vec3::Z,
        }
    }

    fn pack_seg(x: u32, y: u32, z: u32, t: u32) -> u32 {
        (t << 28) | (z << 20) | (y << 10) | x
    }

    #[test]
    fn test_basic_connection() {
        let neurons = vec![
            make_neuron(0, 0, 0, 0),   // A
            make_neuron(10, 10, 0, 0), // B
        ];
        let const_mem = make_const_mem();

        let a_ax = make_axon(0, 0, vec![
            pack_seg(0, 0, 0, 0),
            pack_seg(10, 9, 0, 0), // Very close to B (10, 10, 0)
        ]);

        let mut shard = ShardSoA::new(2, 1);
        connect_dendrites(&mut shard, &neurons, &[a_ax], &const_mem, 42, 30);

        let p_n = shard.padded_n;
        let mut b_connected = false;
        for slot in 0..MAX_DENDRITE_SLOTS {
            let target = shard.dendrite_targets[slot * p_n + 1];
            if target != 0 {
                b_connected = true;
                let weight = shard.dendrite_weights[slot * p_n + 1];
                assert!(weight > 0);
                assert_eq!(unpack_axon_id(target), 0);
            }
        }
        assert!(b_connected);
    }

    #[test]
    fn test_rule_of_uniqueness() {
        let neurons = vec![
            make_neuron(0, 0, 0, 0),
            make_neuron(10, 10, 0, 0),
        ];
        let const_mem = make_const_mem();

        let a_ax = make_axon(0, 0, vec![
            pack_seg(0, 0, 0, 0),
            pack_seg(10, 9, 0, 0),
            pack_seg(9, 10, 0, 0), // Close again
        ]);

        let mut shard = ShardSoA::new(2, 1);
        connect_dendrites(&mut shard, &neurons, &[a_ax], &const_mem, 42, 30);

        let mut connections_count = 0;
        let p_n = shard.padded_n;
        for slot in 0..MAX_DENDRITE_SLOTS {
            if shard.dendrite_targets[slot * p_n + 1] != 0 {
                connections_count += 1;
            }
        }
        assert_eq!(connections_count, 1, "Should have only 1 unique connection per axon_id");
    }

    #[test]
    fn test_visualize_connectivity() {
        let sim = make_sim_config(40, 40, 10);
        let layers = vec![
            LayerZRange { name: "L1".to_string(), z_start_vox: 0, z_end_vox: 5 },
            LayerZRange { name: "L2".to_string(), z_start_vox: 5, z_end_vox: 10 },
        ];
        let const_mem = make_const_mem();
        let bounds = ShardBounds::full_world(&sim);

        let mut neurons = vec![];
        let mut rng_seed = 123;
        for i in 0..15 {
            let x = (rng_seed * 7) % 36 + 2;
            rng_seed = (rng_seed * 13) % 1000;
            let y = (rng_seed * 11) % 36 + 2;
            rng_seed = (rng_seed * 17) % 1000;
            let t = if i % 3 == 0 { 1 } else { 0 };
            neurons.push(make_neuron(x, y, 5, t));
        }

        let types = make_types();
        let (axons, _ghosts) = grow_axons(&neurons, &layers, &types, &sim, &bounds, 42);

        let mut shard = ShardSoA::new(neurons.len(), axons.len());
        connect_dendrites(&mut shard, &neurons, &axons, &const_mem, 42, 30);

        let p_n = shard.padded_n;
        let mut conn_count = 0;
        for i in 0..neurons.len() {
            for slot in 0..MAX_DENDRITE_SLOTS {
                if shard.dendrite_targets[slot * p_n + i] != 0 {
                    conn_count += 1;
                }
            }
        }
        println!("Neurons: {}  Axons: {}  Connections: {}", neurons.len(), axons.len(), conn_count);
        assert!(conn_count > 0, "No connections were formed in visualize test");
    }
}
