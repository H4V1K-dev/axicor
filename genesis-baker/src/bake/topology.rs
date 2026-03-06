use crate::bake::layout::{ShardSoA, CompiledShard};
use crate::bake::axon_growth::{compute_layer_ranges, ShardBounds, grow_axons, GrownAxon, init_axon_head, inject_ghost_axons};
use crate::bake::neuron_placement::generate_placement_from_config;
use crate::bake::dendrite_connect::connect_dendrites;
use crate::bake::seed::seed_from_str;
use crate::bake::input_map::{build_gxi_mapping};
use crate::bake::output_map::{build_gxo_mapping};
use genesis_core::config::{SimulationConfig, InstanceConfig, blueprints::NeuronType, anatomy::AnatomyConfig, io::IoConfig, blueprints::GenesisConstantMemory};
use genesis_core::types::PackedPosition;
use std::collections::HashMap;

pub fn build_local_topology_internal(
    sim: &SimulationConfig,
    anatomy: &AnatomyConfig,
    shard_cfg: &InstanceConfig,
    neuron_types: &[NeuronType],
    const_mem: &GenesisConstantMemory,
    name_map: &HashMap<String, u8>,
    io: &IoConfig,
    zone_name: &str,
    master_seed: u64,
    ghost_capacity: usize,
) -> (ShardSoA, CompiledShard, u32, usize) {
    println!("[baker] Placing neurons...");
    let mut type_name_pairs: Vec<(&String, &u8)> = name_map.iter().collect();
    type_name_pairs.sort_by_key(|(_, &idx)| idx);
    let type_names: Vec<String> = type_name_pairs.into_iter().map(|(n, _)| n.clone()).collect();

    let positions = generate_placement_from_config(
        anatomy,
        shard_cfg,
        sim.simulation.global_density,
        master_seed,
        &type_names,
    );
    println!("[baker] ✓ Placed {} neurons", positions.len());

    println!("[baker] Growing axons (Cone Tracing)...");
    let layer_ranges = compute_layer_ranges(anatomy, sim);
    let shard_bounds = ShardBounds::from_config(shard_cfg);
    let (mut axons, ghost_packets) = grow_axons(
        &positions,
        &layer_ranges,
        neuron_types,
        sim,
        &shard_bounds,
        master_seed,
    );
    let local_axons_count = axons.len();

    let mut num_virtual = 0;
    if !io.inputs.is_empty() {
        for matrix in &io.inputs {
            let gxi = build_gxi_mapping(
                &matrix.name,
                zone_name,
                matrix.width,
                matrix.height,
                axons.len() as u32,
                matrix.stride as u8,
            );
            num_virtual += gxi.axon_ids.len();
            
            for _ in &gxi.axon_ids {
                axons.push(GrownAxon {
                    soma_idx: usize::MAX,
                    type_idx: 0,
                    tip_x: 0, tip_y: 0, tip_z: 0,
                    length_segments: 0,
                    segments: vec![],
                    last_dir: glam::Vec3::ZERO,
                });
            }
        }
    }

    let packed_positions: Vec<u32> = positions.iter().map(|p| p.0).collect();

    if !ghost_packets.is_empty() {
        let (mut ghost_axons, _) = inject_ghost_axons(
            &ghost_packets,
            &positions,
            const_mem,
            sim,
            &shard_bounds,
            master_seed,
        );
        axons.append(&mut ghost_axons);
    }

    let total_capacity = axons.len() + ghost_capacity;
    let mut shard = ShardSoA::new(positions.len(), total_capacity);

    for (i, pos) in positions.iter().enumerate() {
        if pos.0 == 0 { continue; }
        let type_idx = pos.type_id();
        let variant = &const_mem.variants[type_idx as usize];
        shard.voltage[i] = variant.rest_potential;
        shard.flags[i] = type_idx << 4; 
    }

    for (axon_id, axon) in axons.iter().enumerate() {
        if axon.soma_idx != std::usize::MAX {
            shard.soma_to_axon[axon.soma_idx] = axon_id as u32;
        }
    }

    connect_dendrites(
        &mut shard,
        &positions,
        &axons,
        neuron_types,
        master_seed,
        2, 
    );
    
    let physics = genesis_core::physics::compute_derived_physics(
        sim.simulation.signal_speed_m_s,
        sim.simulation.tick_duration_us,
        sim.simulation.voxel_size_um,
        sim.simulation.segment_length_voxels,
    ).expect("v_seg validation failed");
    let v_seg = physics.v_seg;

    for (i, ax) in axons.iter().enumerate() {
        if i < shard.axon_heads.len() {
            let init_val = init_axon_head(ax.length_segments, v_seg);
            let mut burst = genesis_core::layout::BurstHeads8::empty(genesis_core::constants::AXON_SENTINEL);
            burst.h0 = init_val;
            shard.axon_heads[i] = burst;

            shard.axon_tips_uvw[i] = (ax.tip_z << 20) | (ax.tip_y << 10) | ax.tip_x;
            let dx = (ax.last_dir.x * 127.0).clamp(-127.0, 127.0) as i8 as u32;
            let dy = (ax.last_dir.y * 127.0).clamp(-127.0, 127.0) as i8 as u32;
            let dz = (ax.last_dir.z * 127.0).clamp(-127.0, 127.0) as i8 as u32;
            shard.axon_dirs_xyz[i] = (dz << 16) | (dy << 8) | dx;
        }
    }
    
    shard.soma_positions.copy_from_slice(&packed_positions[..positions.len()]);

    let voxel_um = sim.simulation.voxel_size_um;
    let bounds_voxels = (shard_cfg.dimensions.w, shard_cfg.dimensions.d, shard_cfg.dimensions.h);
    let bounds_um = (shard_cfg.dimensions.w as f32 * voxel_um as f32, shard_cfg.dimensions.d as f32 * voxel_um as f32);

    let compiled_shard = CompiledShard {
        zone_name: zone_name.to_string(),
        local_axons_count,
        soma_to_axon_map: shard.soma_to_axon.clone(),
        packed_positions,
        bounds_voxels,
        bounds_um,
    };

    (shard, compiled_shard, v_seg, num_virtual)
}
