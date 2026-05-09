use axicor_core::config::blueprints::NeuronType;
use rand::SeedableRng;
use tracing::warn;

/// Calculates the attraction score of a candidate soma for a growing axon.
/// All math here legally uses f32 as this is the Night Phase.
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

/// Euclidean distance in voxels between two points.
pub fn voxel_dist(ax: u32, ay: u32, az: u32, bx: u32, by: u32, bz: u32) -> f32 {
    let dx = ax as f32 - bx as f32;
    let dy = ay as f32 - by as f32;
    let dz = az as f32 - bz as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Calculates soma "power" based on accumulated weights of all incoming synapses.
/// Used for attracting axons toward functionally important neurons.
pub fn compute_power_index(soma_idx: usize, weights: &[i32], padded_n: usize) -> f32 {
    let mut power = 0u64; // [DOD FIX] 128 * 2.14B would overflow u32, using u64
    for slot in 0..MAX_DENDRITE_SLOTS {
        let w = weights[slot * padded_n + soma_idx];
        power += w.unsigned_abs() as u64; // Float-free, branchless
    }
    // Normalization to 0.0..1.0 (128 slots * 2.14B max weight)
    power as f32 / (MAX_DENDRITE_SLOTS as f32 * 2140000000.0)
}

use axicor_core::config::blueprints::BlueprintsConfig;
use axicor_core::constants::MAX_DENDRITE_SLOTS;
use axicor_core::ipc::AxonHandoverEvent;

use crate::bake::spatial_grid::AxonSegmentGrid;
use axicor_core::types::PackedPosition;

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
    max_z: u32,
    zone_hash: u32,
) {
    let packed_tip = tips[axon_id];
    if packed_tip == 0 {
        return;
    } // Dead or departed axon

    let pos = axicor_core::types::PackedPosition(tips[axon_id]);
    let tx = pos.x() as u32;
    let ty = pos.y() as u32;
    let tz = pos.z() as u32;
    let type_mask = pos.type_id();

    let packed_dir = dirs[axon_id];
    let dx = (packed_dir & 0xFF) as i8;
    let dy = ((packed_dir >> 8) & 0xFF) as i8;
    let dz = ((packed_dir >> 16) & 0xFF) as i8;

    let shift_x = if dx > 64 {
        1
    } else if dx < -64 {
        -1
    } else {
        0
    };
    let shift_y = if dy > 64 {
        1
    } else if dy < -64 {
        -1
    } else {
        0
    };
    let shift_z = if dz > 64 {
        1
    } else if dz < -64 {
        -1
    } else {
        0
    };

    let new_tx = tx as i32 + shift_x;
    let new_ty = ty as i32 + shift_y;
    let new_tz = tz as i32 + shift_z;

    // [DOD FIX] Handover Trigger with dynamic Z-bounds
    if new_tx < 0
        || new_tx >= max_x as i32
        || new_ty < 0
        || new_ty >= max_y as i32
        || new_tz < 0
        || new_tz >= max_z as i32
    {
        if *handovers_count < axicor_core::ipc::MAX_HANDOVERS_PER_NIGHT {
            let len = lengths[axon_id] as u16;
            handovers[*handovers_count] = AxonHandoverEvent {
                origin_zone_hash: zone_hash, // [DOD FIX] Stamp our ID
                local_axon_id: axon_id as u32,
                entry_x: new_tx.clamp(0, 2047) as u16,
                entry_y: new_ty.clamp(0, 2047) as u16,
                vector_x: dx,
                vector_y: dy,
                vector_z: dz,
                type_mask,
                remaining_length: 256u16.saturating_sub(len),
                entry_z: new_tz.clamp(0, 255) as u8,
                _padding: 0,
            };
            *handovers_count += 1;
        }
        // Axon left the shard. Zero out Tip to stop local growth.
        tips[axon_id] = 0;
        return;
    }

    let next_tip = axicor_core::types::PackedPosition::pack_raw(
        new_tx as u32,
        new_ty as u32,
        new_tz as u32,
        type_mask,
    )
    .0;
    tips[axon_id] = next_tip;

    let len = lengths[axon_id] as usize;
    if len < 256 {
        paths[axon_id * 256 + len] = next_tip;
        lengths[axon_id] = (len + 1) as u8;
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[allow(clippy::too_many_arguments)]
pub fn run_sprouting_pass(
    targets: &mut [u32],
    weights: &mut [i32],
    flags: &[u8],
    ghost_origins: &[u32], // [DOD FIX] Origin Tracking
    handovers: &mut [AxonHandoverEvent],
    incoming_handovers_count: usize,
    axon_tips_uvw: &mut [u32],
    axon_dirs_xyz: &mut [u32],
    soma_to_axon: &[u32],
    padded_n: usize,
    total_ghosts: usize,
    virtual_axons: usize, // [DOD FIX]
    max_x: u32,
    max_y: u32,
    max_z: u32,
    blueprints: Option<&BlueprintsConfig>,
    _epoch: u64,
    lengths: &mut [u8],
    paths: &mut [u32],
    soma_positions: &[u32],
    master_seed: u64,
    zone_hash: u32,
    max_sprouts_per_night: u16,
    prune_threshold: i16, // [DOD FIX] For initial weight protection
    shm_ptr: *mut u8,     // [DOD FIX] For prune writing
) -> (usize, usize, Vec<axicor_core::ipc::AxonHandoverAck>) {
    let total_axons = axon_tips_uvw.len();
    let ghost_start = padded_n + virtual_axons; // [DOD FIX] Ghosts follow Virtual
    let ghost_end = ghost_start + total_ghosts;

    // [DOD FIX] Axon Liveness Tracking (Reference Counting)
    let mut active_axons = vec![false; total_axons];

    // 1. Mark existing connections as active
    for t in targets.iter() {
        if *t != 0 {
            let axon_id = axicor_core::layout::unpack_axon_id(*t) as usize;
            if axon_id < total_axons {
                active_axons[axon_id] = true;
            }
        }
    }

    // [DOD FIX] O(1) Reverse Lookup Map (Axon -> Soma)
    // Prevents O(N) soma search when checking axon activity or ownership.
    let mut axon_to_soma = vec![usize::MAX; total_axons];
    for (s_idx, &a_idx) in soma_to_axon.iter().enumerate() {
        if a_idx != u32::MAX && (a_idx as usize) < total_axons {
            axon_to_soma[a_idx as usize] = s_idx;
        }
    }

    // 0. Absorption of incoming Ghost Axons (before SHM rewrite)
    let mut generated_acks = Vec::with_capacity(incoming_handovers_count);

    let mut next_free_ghost = ghost_start;
    #[allow(clippy::needless_range_loop)]
    for i in 0..incoming_handovers_count {
        let ev = &handovers[i];

        while next_free_ghost < ghost_end && axon_tips_uvw[next_free_ghost] != 0 {
            next_free_ghost += 1;
        }

        if next_free_ghost >= ghost_end {
            warn!("Ghost capacity exceeded!");
            break;
        }

        // [DOD FIX] Create ACK for sender
        generated_acks.push(axicor_core::ipc::AxonHandoverAck {
            target_zone_hash: ev.origin_zone_hash,
            receiver_zone_hash: zone_hash, // [DOD FIX] Inject our identity
            src_axon_id: ev.local_axon_id,
            dst_ghost_id: next_free_ghost as u32,
        });

        // [DOD FIX] Correct shifts for 10-10-8-4 layout
        let packed_tip = ((ev.type_mask as u32) << 28)
            | ((ev.entry_z as u32) << 20)
            | ((ev.entry_y as u32) << 10)
            | (ev.entry_x as u32);

        let packed_dir = ((ev.vector_z as u8 as u32) << 16)
            | ((ev.vector_y as u8 as u32) << 8)
            | (ev.vector_x as u8 as u32);

        axon_tips_uvw[next_free_ghost] = packed_tip;
        axon_dirs_xyz[next_free_ghost] = packed_dir;
        lengths[next_free_ghost] = 0; // Reset length, just born here
        paths[next_free_ghost * 256] = packed_tip;

        next_free_ghost += 1;
    }

    let mut handovers_count = 0;

    // 1. Living Axons (Local)
    for soma_idx in 0..padded_n {
        // [DOD FIX] Check spike accumulator for the full batch (bits 3:1), not just the last microsecond
        let f = flags[soma_idx];
        let burst_count = (f >> 1) & 0x07;
        let is_spiking = f & 0x01;

        if burst_count != 0 || is_spiking != 0 {
            let axon_id = soma_to_axon[soma_idx];
            if axon_id != u32::MAX && (axon_id as usize) < total_axons {
                nudge_axon(
                    axon_id as usize,
                    axon_tips_uvw,
                    axon_dirs_xyz,
                    lengths,
                    paths,
                    handovers,
                    &mut handovers_count,
                    max_x,
                    max_y,
                    max_z,
                    zone_hash,
                );
            }
        }
    }

    // 2. Ghost Axons (Unconditional growth by inertia)
    // [DOD FIX] Strict iteration bounds to prevent OOB and virtual axon corruption
    for axon_id in ghost_start..ghost_end {
        nudge_axon(
            axon_id,
            axon_tips_uvw,
            axon_dirs_xyz,
            lengths,
            paths,
            handovers,
            &mut handovers_count,
            max_x,
            max_y,
            max_z,
            zone_hash,
        );
    }

    // 3. Build Spatial Grid from paths
    let segment_grid = AxonSegmentGrid::build_from_paths(lengths, paths, total_axons, 2);

    // 4. Synaptogenesis (Zero-Cost Spatial Search with Type Scoring)
    let mut new_synapses = 0;
    let mut connected_axons = Vec::with_capacity(MAX_DENDRITE_SLOTS);

    for i in 0..padded_n {
        connected_axons.clear();

        let my_pos_raw = soma_positions[i];
        if my_pos_raw == 0 {
            continue;
        }

        // [DOD FIX] Check spike accumulator for the full batch (bits 3:1), not just the last microsecond
        let f = flags[i];
        let burst_count = (f >> 1) & 0x07;
        let is_spiking = f & 0x01;

        if burst_count == 0 && is_spiking == 0 {
            continue; // Neuron was physically silent all day
        }

        let my_pos = PackedPosition(my_pos_raw);
        let my_type_idx = my_pos.type_id() as usize;

        let my_type_cfg = blueprints.and_then(|bp| bp.neuron_types.get(my_type_idx));

        for slot in 0..MAX_DENDRITE_SLOTS {
            let t = targets[slot * padded_n + i];
            if t != 0 {
                connected_axons.push(axicor_core::layout::unpack_axon_id(t));
            }
        }

        let mut sprouts_tonight = 0;
        // [DOD FIX] Iterate FORWARD (0..128).
        // Array is compacted (dense) after GPU Sort & Prune.
        // First encountered 0 is the end of the dense block and ideal place for a new synapse.
        for slot in 0..MAX_DENDRITE_SLOTS {
            let col_idx = slot * padded_n + i;
            if targets[col_idx] != 0 {
                continue; // Skip live synapses, find first empty slot
            }

            let mut best_candidate = None;
            let mut best_score = -1.0;

            // O(K) candidate scan
            segment_grid.for_each_in_radius(&my_pos, 2, |seg_ref| {
                if soma_to_axon[i] == seg_ref.axon_id {
                    return;
                } // Self-connection guard

                // Rule of Uniqueness
                if connected_axons.contains(&seg_ref.axon_id) {
                    return;
                }

                // [DOD FIX] Heuristic: Power Index + Type Affinity + Explore Noise
                let cand_type_idx = seg_ref.type_idx as usize;
                let is_same_type = (my_type_idx == cand_type_idx) as i32 as f32;

                // Deterministic noise based on axon and epoch
                let noise = crate::bake::seed::random_f32(
                    master_seed
                        .wrapping_add(seg_ref.axon_id as u64)
                        .wrapping_add(_epoch),
                );

                // [DOD FIX] O(1) Target Power calculation
                let owner_soma = axon_to_soma[seg_ref.axon_id as usize];
                let target_power = if owner_soma == usize::MAX {
                    1.0 // [DOD FIX] Virtual / Ghost axons have maximum attraction!
                } else {
                    compute_power_index(owner_soma, weights, padded_n)
                };

                let mut score = 1.0;
                if let Some(cfg) = my_type_cfg {
                    // Use sprouting_weight_type from config!
                    score = cfg.sprouting_weight_distance * 1.0 // Consider distance close (r=2)
                          + cfg.sprouting_weight_explore * noise
                          + cfg.sprouting_weight_type * is_same_type
                          + cfg.sprouting_weight_power * target_power;
                }

                if score > best_score {
                    best_score = score;
                    best_candidate = Some(*seg_ref);
                }
            });

            if let Some(seg) = best_candidate {
                let new_target =
                    axicor_core::layout::pack_dendrite_target(seg.axon_id, seg.seg_idx as u32);
                let type_id = seg.type_idx as usize;

                let (is_inhibitory_src, initial_weight) = if let Some(bp) = blueprints {
                    if let Some(nt) = bp.neuron_types.get(type_id) {
                        // [DOD FIX] Shift u16 blueprint weight into i32 Mass Domain
                        let mut start_w = (nt.initial_synapse_weight as i32) << 16;
                        // Shift prune threshold to match Mass Domain comparison
                        let prune_i32 = (prune_threshold.abs() as i32) << 16;

                        // Dead on Arrival protection
                        if start_w <= prune_i32 {
                            start_w = prune_i32 + start_w.max(100 << 16);
                        }
                        (nt.is_inhibitory, start_w)
                    } else {
                        (false, 74i32 << 16)
                    }
                } else {
                    (false, 74i32 << 16)
                };

                targets[col_idx] = new_target;
                connected_axons.push(seg.axon_id); // Update local cache

                weights[col_idx] = if is_inhibitory_src {
                    -initial_weight
                } else {
                    initial_weight
                };

                // [DOD FIX] Axon received a new connection, it is alive
                active_axons[seg.axon_id as usize] = true;

                new_synapses += 1;
                sprouts_tonight += 1;
            } else {
                // [DOD FIX] If there are no more suitable axons around the soma for this slot,
                // there won't be any for the others. Stop senseless memory scan.
                break;
            }

            if sprouts_tonight >= max_sprouts_per_night as i32 {
                break;
            }
        }
    }

    // [DOD FIX] 5. GC Sweep: Scan for orphaned Ghost Axons
    let mut prunes = Vec::new();
    for ghost_id in ghost_start..ghost_end {
        if !active_axons[ghost_id] && axon_tips_uvw[ghost_id] != 0 {
            // Found a ghost without connections!
            let idx = ghost_id - ghost_start;
            let target_zone_hash = ghost_origins[idx];

            if target_zone_hash != 0 {
                // Register death in SHM
                prunes.push(axicor_core::ipc::AxonHandoverPrune {
                    target_zone_hash,
                    receiver_zone_hash: zone_hash, // [DOD FIX] Inject our identity
                    dst_ghost_id: ghost_id as u32,
                });

                // Physical kill: write sentinel into BurstHeads8 (axon_heads)
                // AXON_SENTINEL = 0xFFFFFFFF
                axon_tips_uvw[ghost_id] = 0; // On host
                                             // In VRAM (via SHM won't work, wait for disk write or now?)
                                             // We are in Baker, we write to our local Tips/Dirs structures.
                                             // These structures will later be baked or synchronized.
                                             // In this case we just zero out Tips, and nudge_axon(ghost_id) next night
                                             // will just skip this axon.
            }
        }
    }

    // Write Prunes into SHM
    if !prunes.is_empty() {
        let hdr = unsafe { &mut *(shm_ptr as *mut axicor_core::ipc::ShmHeader) };
        let dest = unsafe {
            shm_ptr.add(hdr.prunes_offset as usize) as *mut axicor_core::ipc::AxonHandoverPrune
        };
        let count = prunes.len().min(1000); // Night limit
        unsafe {
            std::ptr::copy_nonoverlapping(prunes.as_ptr(), dest, count);
            hdr.prunes_count = count as u32;
        }
    }

    (new_synapses, handovers_count, generated_acks)
}

/// Continues growth of axons that crossed the shard boundary (Ghost Axons).
pub fn inject_ghost_axons(
    ghost_packets: &[crate::bake::axon_growth::GhostPacket],
    positions: &[PackedPosition],
    _const_mem: &axicor_core::config::blueprints::AxicorConstantMemory,
    sim: &crate::parser::simulation::SimulationConfig,
    shard_bounds: &crate::bake::axon_growth::ShardBounds,
    master_seed: u64,
) -> (
    Vec<crate::bake::axon_growth::GrownAxon>,
    Vec<crate::bake::axon_growth::GhostPacket>,
) {
    let voxel_um = sim.simulation.voxel_size_um;

    let max_search_radius_vox = sim.simulation.segment_length_voxels as f32 * 3.0;
    let spatial_grid = crate::bake::spatial_grid::SpatialGrid::new(
        positions.to_vec(),
        f32::max(1.0, max_search_radius_vox.ceil()) as u32,
    );
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
        let current_pos_um = current_pos * voxel_um;
        let forward_dir = packet.entry_dir;

        let ghost_seed = master_seed
            .wrapping_add(packet.soma_idx as u64)
            .wrapping_add(packet.origin_shard_id as u64);

        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(ghost_seed);

        use crate::bake::cone_tracing::ConeParams;
        let params = ConeParams {
            radius_um: max_search_radius_vox * voxel_um,
            fov_cos,
            owner_type: packet.type_idx as u8,
            type_affinity: 0.5, // Ghost axons: neutral affinity
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
            target_pos: None, // Ghost axons grow by inertia
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
        let (final_x, final_y, final_z) = if let Some(&last) = segments.last() {
            let pos = axicor_core::types::PackedPosition(last);
            (pos.x() as u32, pos.y() as u32, pos.z() as u32)
        } else {
            (packet.entry_x, packet.entry_y, packet.entry_z)
        };

        grown.push(crate::bake::axon_growth::GrownAxon {
            soma_idx: usize::MAX, // Ghost  no local soma
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
