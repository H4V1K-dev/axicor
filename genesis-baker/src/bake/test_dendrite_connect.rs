#[cfg(test)]
mod tests {
    use crate::bake::axon_growth::{GrownAxon, LayerZRange, ShardBounds, grow_axons};
    use crate::bake::dendrite_connect::connect_dendrites;
    use crate::bake::layout::ShardSoA;
    use crate::bake::neuron_placement::PlacedNeuron;
    use crate::parser::simulation::{SimulationConfig, SimulationParams, WorldConfig};
    use genesis_core::config::blueprints::{GenesisConstantMemory, VariantParameters};
    use genesis_core::coords::{pack_position, unpack_target};

    fn make_sim_config(w: u32, d: u32, h: u32) -> SimulationConfig {
        SimulationConfig {
            world: WorldConfig {
                width_um: w * 50,
                depth_um: d * 50,
                height_um: h * 50,
            },
            simulation: SimulationParams {
                voxel_size_um: 50,
                segment_length_voxels: 1,
                axon_growth_max_steps: 100,
                tick_duration_us: 1000,
                total_ticks: 100_000,
                master_seed: "0".to_string(),
                global_density: 1.0,
                signal_speed_um_tick: 50,
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
            gsop_depression: 0,  // positive → excitatory
            homeostasis_decay: 1,
            signal_propagation_length: 5,
            conduction_velocity: 1,
            slot_decay_ltm: 1,
            slot_decay_wm: 1,
            refractory_period: 5,
            synapse_refractory_period: 5,
            inertia_curve: [0; 16],
            _reserved: [0; 16],
        };
        let inhibitory_variant = VariantParameters {
            gsop_depression: -74, // negative → inhibitory
            gsop_potentiation: 74,
            ..variant
        };
        let mut variants = [variant; 16];
        variants[1] = inhibitory_variant;
        GenesisConstantMemory { variants }
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
        }
    }

    fn pack_seg(x: u32, y: u32, z: u32, t: u32) -> u32 {
        (t << 28) | (z << 20) | (y << 10) | x
    }

    #[test]
    fn test_basic_connection() {
        let neurons = vec![
            make_neuron(0, 0, 0, 0), // A
            make_neuron(10, 10, 0, 0), // B
        ];
        let const_mem = make_const_mem();

        let a_ax = make_axon(0, 0, vec![
            pack_seg(0, 0, 0, 0),
            pack_seg(5, 5, 0, 0),
            pack_seg(10, 9, 0, 0), // Very close to B (10, 10, 0)
            pack_seg(15, 15, 0, 0),
        ]);

        let mut shard = ShardSoA::new(2, 1);
        connect_dendrites(&mut shard, &neurons, &[a_ax], &const_mem, 42);

        let p_n = shard.padded_n;
        let mut b_connected = false;
        for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
            let target = shard.dendrite_targets[slot * p_n + 1];
            if target != 0 {
                b_connected = true;
                let weight = shard.dendrite_weights[slot * p_n + 1];
                assert!(weight > 0, "Excitatory synapse should be positive, got {}", weight);
            }
        }
        assert!(b_connected, "Neuron B did not connect to Neuron A's axon");

        let mut a_connected = false;
        for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
            if shard.dendrite_targets[slot * p_n + 0] != 0 {
                a_connected = true;
            }
        }
        assert!(!a_connected, "Neuron A connected to its own axon");
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
            pack_seg(9, 10, 0, 0),
            pack_seg(10, 11, 0, 0),
            pack_seg(11, 10, 0, 0),
        ]);

        let mut shard = ShardSoA::new(2, 1);
        connect_dendrites(&mut shard, &neurons, &[a_ax], &const_mem, 42);

        let mut connections_count = 0;
        let p_n = shard.padded_n;
        for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
            if shard.dendrite_targets[slot * p_n + 1] != 0 {
                connections_count += 1;
            }
        }
        assert_eq!(connections_count, 1);
    }

    #[test]
    fn test_inhibitory_weight_sign() {
        let neurons = vec![make_neuron(10, 10, 0, 0)];
        let const_mem = make_const_mem();

        let mut ax_i = make_axon(1, 1, vec![pack_seg(0, 0, 0, 1), pack_seg(10, 10, 0, 1)]);
        ax_i.soma_idx = 1;

        let mut shard = ShardSoA::new(1, 2);
        connect_dendrites(&mut shard, &neurons, &[ax_i], &const_mem, 42);

        let p_n = shard.padded_n;
        let w = shard.dendrite_weights[0 * p_n + 0];
        assert!(w < 0, "Inhibitory synapse should have negative weight, got {}", w);
    }

    #[test]
    fn test_distant_axon_ignored() {
        let neurons = vec![make_neuron(0, 0, 0, 0)];
        let const_mem = make_const_mem();
        let ax = make_axon(1, 0, vec![pack_seg(100, 100, 100, 0)]);

        let mut shard = ShardSoA::new(1, 2);
        connect_dendrites(&mut shard, &neurons, &[ax], &const_mem, 42);

        let p_n = shard.padded_n;
        for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
            assert_eq!(shard.dendrite_targets[slot * p_n + 0], 0);
        }
    }

    #[test]
    fn test_multiple_candidates_sorted() {
        let neurons = vec![make_neuron(5, 5, 0, 0)];
        let const_mem = make_const_mem();

        let ax1 = make_axon(1, 0, vec![pack_seg(0, 0, 0, 0), pack_seg(5, 6, 0, 0)]);
        let ax2 = make_axon(2, 0, vec![pack_seg(0, 0, 0, 0), pack_seg(5, 8, 0, 0)]);
        let ax3 = make_axon(3, 0, vec![pack_seg(0, 0, 0, 0), pack_seg(5, 10, 0, 0)]);

        let mut shard = ShardSoA::new(1, 4);
        connect_dendrites(&mut shard, &neurons, &[ax1, ax2, ax3], &const_mem, 42);

        let p_n = shard.padded_n;
        let (t0, _) = unpack_target(shard.dendrite_targets[0 * p_n + 0]).unwrap_or((999, 999));
        let (t1, _) = unpack_target(shard.dendrite_targets[1 * p_n + 0]).unwrap_or((999, 999));
        let (t2, _) = unpack_target(shard.dendrite_targets[2 * p_n + 0]).unwrap_or((999, 999));

        assert_eq!(t0, 0, "Closest axon should be in slot 0");
        assert_eq!(t1, 1, "Next closest in slot 1");
        assert_eq!(t2, 2, "Farthest in slot 2");
    }

    #[test]
    fn test_empty_world() {
        let const_mem = make_const_mem();
        let mut shard = ShardSoA::new(0, 0);
        connect_dendrites(&mut shard, &[], &[], &const_mem, 42);
        // Should not panic
    }

    #[test]
    #[ignore]
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

        let (axons, _ghosts) = grow_axons(&neurons, &layers, &const_mem, &sim, &bounds, 42);

        let mut shard = ShardSoA::new(neurons.len(), axons.len());
        connect_dendrites(&mut shard, &neurons, &axons, &const_mem, 42);

        let p_n = shard.padded_n;
        let mut conn_count = 0;
        for (i, _n) in neurons.iter().enumerate() {
            for slot in 0..genesis_core::constants::MAX_DENDRITE_SLOTS {
                if shard.dendrite_targets[slot * p_n + i] != 0 {
                    conn_count += 1;
                }
            }
        }
        println!("Neurons: {}  Axons: {}  Connections: {}", neurons.len(), axons.len(), conn_count);
    }
}
