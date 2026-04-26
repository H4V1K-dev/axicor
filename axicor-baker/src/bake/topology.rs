use crate::bake::axon_growth::{
    compute_layer_ranges, grow_axons, init_axon_head, inject_ghost_axons, GrownAxon, ShardBounds,
};
use crate::bake::dendrite_connect::connect_dendrites;
use crate::bake::layout::{CompiledShard, ShardSoA};
use crate::bake::neuron_placement::generate_placement_from_config;
use axicor_core::config::{
    anatomy::AnatomyConfig, blueprints::AxicorConstantMemory, blueprints::NeuronType,
    io::IoConfig, InstanceConfig, SimulationConfig,
};
use axicor_core::types::PackedPosition;
use std::collections::HashMap;
use tracing::info;

// MONOLITH: HIGH — build_local_topology_internal is a massive procedural function handling disparate concerns.
// REFACTOR: Separate into discrete topology stages: PlacementEngine, GrowthOrchestrator, and IoMapper.
#[allow(clippy::too_many_arguments)]
pub fn build_local_topology_internal(
    sim: &SimulationConfig,
    anatomy: &AnatomyConfig,
    shard_cfg: &InstanceConfig,
    neuron_types: &[NeuronType],
    const_mem: &AxicorConstantMemory,
    name_map: &HashMap<String, u8>,
    io: &IoConfig,
    zone_name: &str,
    master_seed: u64,
    ghost_capacity: usize,
) -> (
    ShardSoA,
    CompiledShard,
    u32,
    usize,
    usize, // total_ghosts
    Vec<crate::bake::input_map::BakedGxi>,
    Vec<crate::bake::output_map::BakedGxo>,
) {
    info!("[baker] Placing neurons...");
    let mut type_name_pairs: Vec<(&String, &u8)> = name_map.iter().collect();
    type_name_pairs.sort_by_key(|(_, &idx)| idx);
    let type_names: Vec<String> = type_name_pairs
        .into_iter()
        .map(|(n, _)| n.clone())
        .collect();

    let positions = generate_placement_from_config(anatomy, shard_cfg, master_seed, &type_names);
    info!("[baker]  Placed {} neurons", positions.len());

    info!("[baker] Growing axons (Cone Tracing)...");
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

    // [DOD FIX] Strict VRAM mapping
    let padded_n = positions.len();
    let mut vram_axon_ids = Vec::with_capacity(padded_n * 2);
    for ax in &axons {
        vram_axon_ids.push(ax.soma_idx as u32);
    }

    let mut num_virtual = 0;
    let mut gxi_matrices = Vec::new();
    if !io.input.is_empty() {
        info!("[baker] Processing Input Maps for {}...", zone_name);
        gxi_matrices = crate::bake::input_map::build_gxi_mappings(
            io,
            zone_name,
            (padded_n + num_virtual) as u32,
        );

        for matrix in &io.input {
            let zone_w = shard_cfg.dimensions.w;
            let zone_d = shard_cfg.dimensions.d;
            let zone_h = shard_cfg.dimensions.h;

            for pin in &matrix.pin {
                for py in 0..pin.height {
                    for px in 0..pin.width {
                        let u = px as f32 / pin.width as f32;
                        let v = py as f32 / pin.height as f32;
                        let mapped_u = pin.local_u + u * pin.u_width;
                        let mapped_v = pin.local_v + v * pin.v_height;

                        let start_x =
                            ((mapped_u * zone_w as f32) as u32).min(zone_w.saturating_sub(1));
                        let start_y =
                            ((mapped_v * zone_d as f32) as u32).min(zone_d.saturating_sub(1));

                        let (start_z, z_step, last_dir) = match matrix.entry_z.as_str() {
                            "bottom" => (0, 1i32, glam::Vec3::Z),
                            "mid" => (zone_h / 2, 1i32, glam::Vec3::Z),
                            _ => (zone_h.saturating_sub(1), -1i32, glam::Vec3::NEG_Z), // "top"
                        };

                        let mut segments = Vec::new();
                        let length = zone_h;
                        let mut final_z = start_z;

                        for i in 0..length {
                            let z = (start_z as i32 + i as i32 * z_step)
                                .clamp(0, zone_h.saturating_sub(1) as i32)
                                as u32;
                            final_z = z;
                            segments.push(PackedPosition::pack_raw(start_x, start_y, z, 0).0);
                        }

                        // [DOD FIX] Derived type_idx for Virtual Axons to prevent Neural Silence
                        // If target_type matches a known type, use its index.
                        let virtual_type_idx = name_map.get(&pin.target_type).cloned().unwrap_or(0);

                        axons.push(GrownAxon {
                            soma_idx: usize::MAX,
                            type_idx: virtual_type_idx as usize,
                            tip_x: start_x,
                            tip_y: start_y,
                            tip_z: final_z,
                            length_segments: segments.len() as u32,
                            segments,
                            last_dir,
                        });
                        vram_axon_ids.push((padded_n + num_virtual) as u32);
                        num_virtual += 1;
                    }
                }
            }
        }
        info!(
            "[baker]  Processed {} virtual axons across {} input matrices",
            num_virtual,
            gxi_matrices.len()
        );
    }

    let packed_positions: Vec<u32> = positions.iter().map(|p| p.0).collect();

    let mut gxo_matrices = Vec::new();
    if !io.output.is_empty() {
        info!("[baker] Processing Output Maps for {}...", zone_name);
        gxo_matrices = crate::bake::output_map::build_gxo_mappings(
            io,
            zone_name,
            shard_cfg.dimensions.w,
            shard_cfg.dimensions.d,
            &packed_positions,
            &type_names,
        );
        info!("[baker]  Processed {} output matrices", gxo_matrices.len());
    }

    if !ghost_packets.is_empty() {
        let (mut ghost_axons, _) = inject_ghost_axons(
            &ghost_packets,
            &positions,
            // [DOD FIX] Replaced const_mem with neuron_types
            neuron_types,
            sim,
            &shard_bounds,
            master_seed,
        );
        for (ghost_counter, _) in ghost_axons.iter().enumerate() {
            vram_axon_ids.push((padded_n + num_virtual + ghost_counter) as u32);
        }
        axons.append(&mut ghost_axons);
    }

    let total_ghosts = axons.len() - local_axons_count - num_virtual;
    info!(
        "[baker]  Total Grown: {} axons ({} local, {} virtual, {} ghosts)",
        axons.len(),
        local_axons_count,
        num_virtual,
        total_ghosts
    );

    let total_capacity = axons.len() + ghost_capacity;
    let mut shard = ShardSoA::new(positions.len(), total_capacity);

    for (i, pos) in positions.iter().enumerate() {
        if pos.0 == 0 {
            continue;
        }
        let type_idx = pos.type_id();
        let variant = &const_mem.variants[type_idx as usize];
        shard.voltage[i] = variant.rest_potential;
        shard.flags[i] = type_idx << 4;
    }

    for (axon_id, axon) in axons.iter().enumerate() {
        if axon.soma_idx != usize::MAX {
            shard.soma_to_axon[axon.soma_idx] = vram_axon_ids[axon_id];
        }
    }

    info!("[baker] Connecting dendrites (dynamic per-type radius)...");

    let total_synapses = connect_dendrites(
        &mut shard,
        &positions,
        &axons,
        &vram_axon_ids,
        neuron_types,
        sim.simulation.voxel_size_um, // Pass voxel size
    );
    info!(
        "[baker]  Synapses established: {} (avg: {:.1}/soma)",
        total_synapses,
        total_synapses as f64 / positions.len() as f64
    );

    let physics = axicor_core::physics::compute_derived_physics(
        sim.simulation.signal_speed_m_s,
        sim.simulation.tick_duration_us,
        sim.simulation.voxel_size_um,
        sim.simulation.segment_length_voxels,
    )
    .expect("v_seg validation failed");
    let v_seg = physics.v_seg;

    for (i, ax) in axons.iter().enumerate() {
        let dst_offset = vram_axon_ids[i] as usize;
        if dst_offset >= shard.axon_heads.len() {
            continue;
        }

        let init_val = init_axon_head(ax.length_segments, v_seg);
        let mut burst =
            axicor_core::layout::BurstHeads8::empty(axicor_core::constants::AXON_SENTINEL);
        burst.h0 = init_val;
        shard.axon_heads[dst_offset] = burst;

        // [DOD FIX] Strict 10-10-8-4 PackedPosition layout alignment
        shard.axon_tips_uvw[dst_offset] =
            ((ax.type_idx as u32 & 0x0F) << 28) | (ax.tip_z << 20) | (ax.tip_y << 10) | ax.tip_x;
        let dx = (ax.last_dir.x * 127.0).clamp(-127.0, 127.0) as i8 as u8 as u32;
        let dy = (ax.last_dir.y * 127.0).clamp(-127.0, 127.0) as i8 as u8 as u32;
        let dz = (ax.last_dir.z * 127.0).clamp(-127.0, 127.0) as i8 as u8 as u32;
        shard.axon_dirs_xyz[dst_offset] = (dz << 16) | (dy << 8) | dx;

        // [DOD FIX] Get exact VRAM address, no guessing
        let len = ax.length_segments.min(256) as usize;
        shard.axon_lengths[dst_offset] = len as u8;

        if len > 0 {
            // GrownAxon stores all segments (or only up to the limit)
            let copy_len = ax.segments.len().min(len);
            let path_start = dst_offset * axicor_core::layout::MAX_SEGMENTS_PER_AXON;
            shard.axon_paths[path_start..path_start + copy_len]
                .copy_from_slice(&ax.segments[..copy_len]);
        }
    }
    info!("[baker]  Axon heads initialized (v_seg={})", v_seg);

    shard
        .soma_positions
        .copy_from_slice(&packed_positions[..positions.len()]);

    let voxel_um = sim.simulation.voxel_size_um;
    let bounds_voxels = (
        shard_cfg.dimensions.w,
        shard_cfg.dimensions.d,
        shard_cfg.dimensions.h,
    );
    let bounds_um = (
        shard_cfg.dimensions.w as f32 * voxel_um,
        shard_cfg.dimensions.d as f32 * voxel_um,
    );

    let compiled_shard = CompiledShard {
        _zone_name: zone_name.to_string(),
        local_axons_count,
        // [DOD FIX] Export strict boundaries for Ghost allocation
        ghost_offset_base: (padded_n + num_virtual) as u32,
        ghost_capacity_limit: ghost_capacity as u32,
        soma_to_axon_map: shard.soma_to_axon.clone(),
        packed_positions,
        _bounds_voxels: bounds_voxels,
        bounds_um,
    };

    (
        shard,
        compiled_shard,
        v_seg,
        num_virtual,
        total_ghosts,
        gxi_matrices,
        gxo_matrices,
    )
}

