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

/// Вычисляет "силу" сомы на базе накопленных весов всех её входящих синапсов.
/// Используется для аттракции аксонов к функционально важным нейронам.
pub fn compute_power_index(soma_idx: usize, weights: &[i16], padded_n: usize) -> f32 {
    let mut power = 0u32;
    for slot in 0..MAX_DENDRITE_SLOTS {
        let w = weights[slot * padded_n + soma_idx];
        power += w.unsigned_abs() as u32; // Без float, без бранчей
    }
    // Нормализация в 0.0..1.0 (128 слотов по 32767 макс веса)
    power as f32 / (MAX_DENDRITE_SLOTS as f32 * 32767.0)
}

use genesis_core::config::blueprints::BlueprintsConfig;
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
    zone_hash: u32, // <--- НОВЫЙ ПАРАМЕТР
) {
    let packed_tip = tips[axon_id];
    if packed_tip == 0 { return; } // Мертвый или улетевший аксон

    let pos = genesis_core::types::PackedPosition(tips[axon_id]);
    let tx = pos.x() as u32;
    let ty = pos.y() as u32;
    let tz = pos.z() as u32;
    let type_mask = pos.type_id() as u8;

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
                origin_zone_hash: zone_hash, // [DOD FIX] Штампуем наш ID
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

    let next_tip = genesis_core::types::PackedPosition::pack_raw(new_tx as u32, new_ty as u32, new_tz as u32, type_mask).0;
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
    ghost_origins: &[u32], // [DOD FIX] Origin Tracking
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
    master_seed: u64,
    zone_hash: u32,
    max_sprouts_per_night: u16,
    shm_ptr: *mut u8, // [DOD FIX] For prune writing
) -> (usize, usize, Vec<genesis_core::ipc::AxonHandoverAck>) {
    let total_axons = axon_tips_uvw.len();
    let ghost_start = padded_n;
    let ghost_end = padded_n + total_ghosts;

    // [DOD FIX] Axon Liveness Tracking (Reference Counting)
    let mut active_axons = vec![false; total_axons];
    
    // 1. Mark existing connections as active
    for t in targets.iter() {
        if *t != 0 {
            let axon_id = genesis_core::layout::unpack_axon_id(*t) as usize;
            if axon_id < total_axons {
                active_axons[axon_id] = true;
            }
        }
    }

    // [DOD FIX] O(1) Reverse Lookup Map (Axon -> Soma)
    // Предотвращает O(N) поиск сомы при проверке активности или принадлежности аксона.
    let mut axon_to_soma = vec![usize::MAX; total_axons];
    for (s_idx, &a_idx) in soma_to_axon.iter().enumerate() {
        if a_idx != u32::MAX && (a_idx as usize) < total_axons {
            axon_to_soma[a_idx as usize] = s_idx;
        }
    }

    // 0. Абсорбция входящих Ghost Axons (до перезаписи SHM)
    let mut generated_acks = Vec::with_capacity(incoming_handovers_count);
    
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

        // [DOD FIX] Создаем ACK для отправителя
        generated_acks.push(genesis_core::ipc::AxonHandoverAck {
            target_zone_hash: ev.origin_zone_hash,
            src_axon_id: ev.local_axon_id,
            dst_ghost_id: next_free_ghost as u32,
        });

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
        if (flags[soma_idx] & 0x02) != 0 {
            let axon_id = soma_to_axon[soma_idx];
            if axon_id != u32::MAX && (axon_id as usize) < total_axons {
                nudge_axon(
                    axon_id as usize, axon_tips_uvw, axon_dirs_xyz, lengths, paths,
                    handovers, &mut handovers_count, max_x, max_y, zone_hash
                );
            }
        }
    }

    // 2. Ghost Axons (Безусловный рост по инерции)
    let ghost_end = padded_n + total_ghosts;
    for axon_id in padded_n..ghost_end {
        nudge_axon(
            axon_id, axon_tips_uvw, axon_dirs_xyz, lengths, paths,
            handovers, &mut handovers_count, max_x, max_y, zone_hash
        );
    }

    // 3. Строим Spatial Grid из путей
    let segment_grid = AxonSegmentGrid::build_from_paths(lengths, paths, total_axons, 2);

    // 4. Synaptogenesis (Zero-Cost Spatial Search with Type Scoring)
    let mut new_synapses = 0;

    for i in 0..padded_n {
        let my_pos_raw = soma_positions[i];
        if my_pos_raw == 0 { continue; }
        // [DOD FIX] Читаем все 3 бита burst_count (биты 1, 2, 3). Маска 0x0E (0000_1110)
        if (flags[i] & 0x0E) == 0 { continue; }

        let my_pos = PackedPosition(my_pos_raw);
        let my_type_idx = my_pos.type_id() as usize;
        
        let my_type_cfg = blueprints.and_then(|bp| bp.neuron_types.get(my_type_idx));

        let mut sprouts_tonight = 0;
        for slot in (0..MAX_DENDRITE_SLOTS).rev() {
            let col_idx = slot * padded_n + i;
            if targets[col_idx] != 0 {
                break; // Слоты плотные, конец пустых
            }

            let mut best_candidate = None;
            let mut best_score = -1.0;

            // O(K) сканирование кандидатов
            segment_grid.for_each_in_radius(&my_pos, 2, |seg_ref| {
                if soma_to_axon[i] == seg_ref.axon_id { return; } // Self-connection guard

                // Rule of Uniqueness
                let mut is_dup = false;
                for existing_slot in 0..MAX_DENDRITE_SLOTS {
                    let t = targets[existing_slot * padded_n + i];
                    if t != 0 && genesis_core::layout::unpack_axon_id(t) == seg_ref.axon_id {
                        is_dup = true;
                        break;
                    }
                }
                if is_dup { return; }

                // [DOD FIX] Эвристика: Power Index + Type Affinity + Explore Noise
                let cand_type_idx = seg_ref.type_idx as usize;
                let is_same_type = (my_type_idx == cand_type_idx) as i32 as f32;
                
                // Детерминированный шум на базе аксона и эпохи
                let noise = crate::bake::seed::random_f32(
                    master_seed.wrapping_add(seg_ref.axon_id as u64).wrapping_add(_epoch)
                );

                // [DOD FIX] O(1) Target Power calculation
                let owner_soma = axon_to_soma[seg_ref.axon_id as usize];
                let target_power = if owner_soma == usize::MAX {
                    1.0 // [DOD FIX] Virtual / Ghost аксоны имеют максимальную привлекательность!
                } else {
                    compute_power_index(owner_soma, weights, padded_n)
                };

                let mut score = 1.0;
                if let Some(cfg) = my_type_cfg {
                    // Используем sprouting_weight_type из конфига!
                    score = cfg.sprouting_weight_distance * 1.0 // Считаем дистанцию близкой (r=2)
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
                let new_target = genesis_core::layout::pack_dendrite_target(seg.axon_id, seg.seg_idx as u32);
                let type_id = seg.type_idx as usize;

                let (is_inhibitory_src, initial_weight) = if let Some(bp) = blueprints {
                    if let Some(nt) = bp.neuron_types.get(type_id) {
                        (nt.is_inhibitory, nt.initial_synapse_weight as i16)
                    } else { (false, 74) }
                } else { (false, 74) };

                targets[col_idx] = new_target;
                weights[col_idx] = if is_inhibitory_src { -initial_weight } else { initial_weight };
                
                // [DOD FIX] Аксон получил новую связь, он жив
                active_axons[seg.axon_id as usize] = true;

                new_synapses += 1;
                sprouts_tonight += 1;
            } else {
                // [DOD FIX] Если вокруг сомы больше нет подходящих аксонов для этого слота, 
                // их не будет и для остальных. Прекращаем бессмысленный скан памяти.
                break;
            }
            
            if sprouts_tonight >= max_sprouts_per_night as i32 {
                break;
            }
        }
    }

    // [DOD FIX] 5. GC Sweep: Ищем сирот среди Ghost Axons
    let mut prunes = Vec::new();
    for ghost_id in ghost_start..ghost_end {
        if !active_axons[ghost_id] && axon_tips_uvw[ghost_id] != 0 {
            // Найден призрак без связей!
            let idx = ghost_id - ghost_start;
            let target_zone_hash = ghost_origins[idx];
            
            if target_zone_hash != 0 {
                // Регистрируем смерть в SHM
                prunes.push(genesis_core::ipc::AxonHandoverPrune {
                    target_zone_hash,
                    dst_ghost_id: ghost_id as u32,
                });
                
                // Физическое убийство: записываем сентенель в BurstHeads8 (axon_heads)
                // AXON_SENTINEL = 0xFFFFFFFF
                axon_tips_uvw[ghost_id] = 0; // На хосте
                // В VRAM (через SHM не получится, нужно дождаться записи на диск или прямо сейчас?)
                // Мы находимся в Baker, мы пишем в наши локальные структуры Tips/Dirs.
                // Эти структуры потом будут запечены или синхронизированы.
                // В данном случае мы просто обнуляем Tips, и nudge_axon(ghost_id) в следующую ночь
                // просто пропустит этот аксон. 
            }
        }
    }

    // Записываем Prunes в SHM
    if !prunes.is_empty() {
        let hdr = unsafe { &mut *(shm_ptr as *mut genesis_core::ipc::ShmHeader) };
        let dest = unsafe {
            shm_ptr.add(hdr.prunes_offset as usize) as *mut genesis_core::ipc::AxonHandoverPrune
        };
        let count = prunes.len().min(1000); // Лимит на ночь
        unsafe {
            std::ptr::copy_nonoverlapping(prunes.as_ptr(), dest, count);
            hdr.prunes_count = count as u32;
        }
    }

    (new_synapses, handovers_count, generated_acks)
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
