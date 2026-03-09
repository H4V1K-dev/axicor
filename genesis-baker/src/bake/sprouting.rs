use genesis_core::config::blueprints::NeuronType;
use rand::SeedableRng;



/// Вычисляет привлекательность сомы-кандидата для растущего аксона.
/// Вся математика здесь легально использует f32, так как это Night Phase.
#[inline]
pub fn compute_sprouting_score(
    target_type: &NeuronType,
    distance: f32,
    power_index: f32,
    noise: f32,
) -> f32 {
    let dist_score = 1.0 / (distance + 1.0);
    
    dist_score * target_type.sprouting_weight_distance 
        + power_index * target_type.sprouting_weight_power 
        + noise * target_type.sprouting_weight_explore
}

/// Евклидово расстояние в вокселях между двумя точками.
pub fn voxel_dist(ax: u32, ay: u32, az: u32, bx: u32, by: u32, bz: u32) -> f32 {
    let dx = ax as f32 - bx as f32;
    let dy = ay as f32 - by as f32;
    let dz = az as f32 - bz as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

use genesis_core::config::blueprints::BlueprintsConfig;
use genesis_core::layout::pack_dendrite_target;
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use genesis_core::ipc::AxonHandoverEvent;

use crate::bake::spatial_grid::AxonSegmentGrid;
use genesis_core::types::PackedPosition;

#[inline]
#[allow(clippy::too_many_arguments)]
fn nudge_axon(
    axon_id: usize,
    tips: &mut [u32],
    dirs: &[u32],
    lengths: &mut [u8],
    paths: &mut [u32],
    handovers: &mut [AxonHandoverEvent],
    handovers_count: &mut usize,
    max_x: u32,
    max_y: u32,
) {
    let packed_tip = tips[axon_id];
    if packed_tip == 0 { return; } // Мертвый или улетевший аксон

    // [DOD FIX] 11-11-6-4 Layout
    let tx = packed_tip & 0x7FF;            // 11 bits
    let ty = (packed_tip >> 11) & 0x7FF;     // 11 bits
    let tz = (packed_tip >> 22) & 0x3F;      // 6 bits
    let type_mask = (packed_tip >> 28) & 0x0F;

    let packed_dir = dirs[axon_id];
    let dx = (packed_dir & 0xFF) as i8;
    let dy = ((packed_dir >> 8) & 0xFF) as i8;
    let dz = ((packed_dir >> 16) & 0xFF) as i8;

    let shift_x = if dx > 64 { 1 } else if dx < -64 { -1 } else { 0 };
    let shift_y = if dy > 64 { 1 } else if dy < -64 { -1 } else { 0 };
    let shift_z = if dz > 64 { 1 } else if dz < -64 { -1 } else { 0 };

    let new_tx = tx as i32 + shift_x;
    let new_ty = ty as i32 + shift_y;
    let new_tz = tz as i32 + shift_z;

    // [DOD FIX] Handover Trigger: выход за границы шарда
    if new_tx < 0 || new_tx >= max_x as i32 || new_ty < 0 || new_ty >= max_y as i32 || new_tz < 0 || new_tz > 63 {
        if *handovers_count < genesis_core::ipc::MAX_HANDOVERS_PER_NIGHT {
            let len = lengths[axon_id] as u16;
            handovers[*handovers_count] = AxonHandoverEvent {
                local_axon_id: axon_id as u32,
                entry_x: new_tx.clamp(0, 2047) as u16,
                entry_y: new_ty.clamp(0, 2047) as u16,
                vector_x: dx,
                vector_y: dy,
                vector_z: dz,
                type_mask: type_mask as u8,
                remaining_length: 256u16.saturating_sub(len),
                entry_z: new_tz.clamp(0, 63) as u8,
                _padding: 0,
            };
            *handovers_count += 1;
        }
        // Аксон покинул шард. Обнуляем Tip, чтобы прекратить локальный рост.
        tips[axon_id] = 0;
        return;
    }

    let next_tip = (packed_tip & 0xF0000000) | ((new_tz as u32) << 22) | ((new_ty as u32) << 11) | (new_tx as u32);
    tips[axon_id] = next_tip;

    let len = lengths[axon_id] as usize;
    if len < 256 {
        paths[axon_id * 256 + len] = next_tip;
        lengths[axon_id] = (len + 1) as u8;
    }
}

pub fn run_sprouting_pass(
    targets: &mut [u32],
    weights: &mut [i16],
    flags: &[u8],
    handovers: &mut [AxonHandoverEvent],
    incoming_handovers_count: usize, 
    axon_tips_uvw: &mut [u32],
    axon_dirs_xyz: &mut [u32],
    soma_to_axon: &[u32],
    padded_n: usize,
    total_ghosts: usize,
    max_x: u32,
    max_y: u32,
    blueprints: Option<&BlueprintsConfig>,
    _epoch: u64,
    lengths: &mut [u8],
    paths: &mut [u32],
    soma_positions: &[u32],
) -> (usize, usize) {
    let total_axons = axon_tips_uvw.len();
    let ghost_start = padded_n;
    let ghost_end = padded_n + total_ghosts;

    // 0. Абсорбция входящих Ghost Axons (до перезаписи SHM)
    let mut next_free_ghost = ghost_start;
    for i in 0..incoming_handovers_count {
        let ev = &handovers[i];

        while next_free_ghost < ghost_end && axon_tips_uvw[next_free_ghost] != 0 {
            next_free_ghost += 1;
        }

        if next_free_ghost >= ghost_end {
            println!("WARNING: Ghost capacity exceeded!");
            break;
        }

        let packed_tip = ((ev.type_mask as u32) << 28)
                       | ((ev.entry_z as u32) << 22)
                       | ((ev.entry_y as u32) << 11)
                       | (ev.entry_x as u32);
        
        let packed_dir = ((ev.vector_z as u8 as u32) << 16)
                       | ((ev.vector_y as u8 as u32) << 8)
                       | (ev.vector_x as u8 as u32);

        axon_tips_uvw[next_free_ghost] = packed_tip;
        axon_dirs_xyz[next_free_ghost] = packed_dir;
        lengths[next_free_ghost] = 0; // Сбрасываем длину, он только родился здесь
        paths[next_free_ghost * 256] = packed_tip;

        next_free_ghost += 1;
    }

    let mut handovers_count = 0;

    // 1. Living Axons (Локальные)
    for soma_idx in 0..padded_n {
        if (flags[soma_idx] & 0x01) != 0 {
            let axon_id = soma_to_axon[soma_idx];
            if axon_id != u32::MAX && (axon_id as usize) < total_axons {
                nudge_axon(
                    axon_id as usize, axon_tips_uvw, axon_dirs_xyz, lengths, paths,
                    handovers, &mut handovers_count, max_x, max_y
                );
            }
        }
    }

    // 2. Ghost Axons (Безусловный рост по инерции)
    let ghost_end = padded_n + total_ghosts;
    for axon_id in padded_n..ghost_end {
        nudge_axon(
            axon_id, axon_tips_uvw, axon_dirs_xyz, lengths, paths,
            handovers, &mut handovers_count, max_x, max_y
        );
    }

    // 3. Строим Spatial Grid из путей
    let segment_grid = AxonSegmentGrid::build_from_paths(lengths, paths, total_axons, 2);

    // 4. Synaptogenesis (Zero-Cost Spatial Search)
    let mut new_synapses = 0;

    for i in 0..padded_n {
        let my_pos_raw = soma_positions[i];
        if my_pos_raw == 0 { continue; }

        if (flags[i] & 0x01) == 0 { continue; }

        for slot in (0..MAX_DENDRITE_SLOTS).rev() {
            let col_idx = slot * padded_n + i;
            
            if targets[col_idx] != 0 {
                break; 
            }

            let my_pos = PackedPosition(my_pos_raw);
            let mut candidate = None;

            segment_grid.for_each_in_radius(&my_pos, 2, |seg_ref| {
                if candidate.is_none() && soma_to_axon[i] != seg_ref.axon_id {
                    candidate = Some(*seg_ref);
                }
            });

            if let Some(seg) = candidate {
                let new_target = pack_dendrite_target(seg.axon_id, seg.seg_idx as u32);
                let type_id = seg.type_idx as usize;

                let (is_inhibitory_src, initial_weight) = if let Some(bp) = blueprints {
                    if let Some(nt) = bp.neuron_types.get(type_id) {
                        (nt.is_inhibitory, nt.initial_synapse_weight as i16)
                    } else { (false, 74) }
                } else { (false, 74) };

                targets[col_idx] = new_target;
                weights[col_idx] = if is_inhibitory_src { -initial_weight } else { initial_weight };
                new_synapses += 1;
            }
            break;
        }
    }

    (new_synapses, handovers_count)
}


/// Продолжает рост аксонов, пересёкших границу шарда (Ghost Axons).
pub fn inject_ghost_axons(
    ghost_packets: &[crate::bake::axon_growth::GhostPacket],
    positions: &[PackedPosition],
    _const_mem: &genesis_core::config::blueprints::GenesisConstantMemory,
    sim: &crate::parser::simulation::SimulationConfig,
    shard_bounds: &crate::bake::axon_growth::ShardBounds,
    master_seed: u64,
) -> (Vec<crate::bake::axon_growth::GrownAxon>, Vec<crate::bake::axon_growth::GhostPacket>) {
    let voxel_um = sim.simulation.voxel_size_um;

    let max_search_radius_vox = sim.simulation.segment_length_voxels as f32 * 3.0;
    let spatial_grid = crate::bake::spatial_grid::SpatialGrid::new(positions.to_vec(), f32::max(1.0, max_search_radius_vox.ceil()) as u32);
    let mut grown = Vec::with_capacity(ghost_packets.len());
    let mut outgoing: Vec<crate::bake::axon_growth::GhostPacket> = Vec::new();

    for packet in ghost_packets {
        let fov_cos = (45.0_f32 / 2.0).to_radians().cos();
        let max_search_radius_vox = sim.simulation.segment_length_voxels as f32 * 4.0;

        let current_pos = glam::Vec3::new(
            packet.entry_x as f32,
            packet.entry_y as f32,
            packet.entry_z as f32,
        );
        let current_pos_um = current_pos * voxel_um as f32;
        let forward_dir = packet.entry_dir;

        let ghost_seed = master_seed
            .wrapping_add(packet.soma_idx as u64)
            .wrapping_add(packet.origin_shard_id as u64);
            
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(ghost_seed);

        use crate::bake::cone_tracing::ConeParams;
        let params = ConeParams {
            radius_um: max_search_radius_vox * voxel_um as f32,
            fov_cos,
            owner_type: packet.type_idx as u8,
            type_affinity: 0.5, // Ghost-аксоны: нейтральное сродство
        };
        let weights = crate::bake::axon_growth::SteeringWeights {
            global: 0.6,
            attract: 0.3,
            noise: 0.1,
        };

        let mut ctx = crate::bake::axon_growth::GrowthContext {
            current_pos_um,
            current_pos_vox: current_pos,
            forward_dir,
            target_pos: None, // Ghost-аксоны летят по инерции
            remaining_steps: packet.remaining_steps,
            owner_type_idx: packet.type_idx as u8,
            soma_idx: packet.soma_idx,
            origin_shard_id: packet.origin_shard_id,
        };

        let (segments, maybe_outgoing) = crate::bake::axon_growth::execute_growth_loop(
            &mut ctx,
            &params,
            &weights,
            &spatial_grid,
            sim,
            shard_bounds,
            rng,
        );

        let has_outgoing = maybe_outgoing.is_some();
        if let Some(pkt) = maybe_outgoing {
            outgoing.push(pkt);
        }

        if segments.is_empty() && !has_outgoing {
            continue; 
        }

        let length_segments = segments.len() as u32;
        let (final_x, final_y, final_z) = if let Some(last) = segments.last() {
            ((last & 0x7FF), ((last >> 11) & 0x7FF), ((last >> 22) & 0x3F))
        } else {
            (packet.entry_x, packet.entry_y, packet.entry_z)
        };

        grown.push(crate::bake::axon_growth::GrownAxon {
            soma_idx: usize::MAX, // Ghost — нет локальной сомы
            type_idx: packet.type_idx,
            tip_x: final_x,
            tip_y: final_y,
            tip_z: final_z,
            length_segments,
            segments,
            last_dir: ctx.forward_dir,
        });
    }

    (grown, outgoing)
}
