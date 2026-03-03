// genesis-baker/src/bake/dendrite_connect.rs

use genesis_core::config::blueprints::GenesisConstantMemory;
use genesis_core::layout::{pack_dendrite_target, unpack_axon_id};
use crate::bake::sprouting::compute_sprouting_score;
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use crate::bake::neuron_placement::PlacedNeuron;
use std::collections::{HashMap, HashSet};
use crate::bake::axon_growth::GrownAxon;
use crate::bake::layout::ShardSoA;
use crate::bake::seed::entity_seed;

/// Ключ ячейки пространственной решётки.
type GridCell = (u32, u32, u32);

/// Строит HashMap: grid_cell → список индексов аксонов, хотя бы один сегмент которых проходит через ячейку.
fn build_axon_grid(axons: &[GrownAxon], cell_size: u32) -> HashMap<GridCell, Vec<usize>> {
    let mut grid: HashMap<GridCell, Vec<usize>> = HashMap::new();
    let safe_cell_size = cell_size.max(1);
    for (i, ax) in axons.iter().enumerate() {
        let mut touched_cells = HashSet::new();
        for &seg in &ax.segments {
            let z = (seg >> 20) & 0xFF;
            let y = (seg >> 10) & 0x3FF;
            let x = seg & 0x3FF;
            let cell = (x / safe_cell_size, y / safe_cell_size, z / safe_cell_size);
            touched_cells.insert(cell);
        }
        for cell in touched_cells {
            grid.entry(cell).or_default().push(i);
        }
    }
    grid
}

struct Candidate {
    axon_idx: usize,
    segment_idx: usize,
    score: f32,
}

#[derive(Clone, Copy)]
struct NeuronSlots {
    targets: [u32; MAX_DENDRITE_SLOTS],
    weights: [i16; MAX_DENDRITE_SLOTS],
}

/// Выполняет первичное соединение всех нейронов с аксонами.
pub fn connect_dendrites(
    shard: &mut ShardSoA,
    neurons: &[PlacedNeuron],
    axons: &[GrownAxon],
    const_mem: &GenesisConstantMemory,
    master_seed: u64,
    cell_size: u32,
) {
    let pn = shard.padded_n;
    let search_radius = cell_size as f32;
    let grid_cell_size = (search_radius / 1.5).ceil() as u32; 
    let grid = build_axon_grid(axons, grid_cell_size);

    println!("Baking: Initiating Rayon parallel dendrite search for {} somas...", neurons.len());

    let mut temp_slots = vec![NeuronSlots {
        targets: [0; MAX_DENDRITE_SLOTS],
        weights: [0; MAX_DENDRITE_SLOTS],
    }; pn];

    use rayon::prelude::*;

    temp_slots.par_iter_mut().enumerate().for_each(|(soma_id, slots)| {
        if soma_id >= neurons.len() { return; }

        let neuron = &neurons[soma_id];
        let soma_x = neuron.x();
        let soma_y = neuron.y();
        let soma_z = neuron.z();

        let cell_x = soma_x / grid_cell_size;
        let cell_y = soma_y / grid_cell_size;
        let cell_z = soma_z / grid_cell_size;

        let mut candidates: Vec<Candidate> = Vec::new();
        let mut seen_axons: HashSet<usize> = HashSet::new();

        for dx in 0..=2u32 {
            for dy in 0..=2u32 {
                for dz in 0..=2u32 {
                    let cx = cell_x.saturating_add(dx).saturating_sub(1);
                    let cy = cell_y.saturating_add(dy).saturating_sub(1);
                    let cz = cell_z.saturating_add(dz).saturating_sub(1);

                    if let Some(cell_axons) = grid.get(&(cx, cy, cz)) {
                        for &axon_idx in cell_axons {
                            let ax = &axons[axon_idx];
                            if ax.soma_idx == soma_id { continue; }
                            if seen_axons.contains(&axon_idx) { continue; }

                            let mut min_dist = f32::MAX;
                            let mut best_seg_idx = 0;
                            
                            for (seg_idx, &seg) in ax.segments.iter().enumerate() {
                                let z = (seg >> 20) & 0xFF;
                                let y = (seg >> 10) & 0x3FF;
                                let x = seg & 0x3FF;
                                let dist = crate::bake::sprouting::voxel_dist(soma_x, soma_y, soma_z, x, y, z);
                                if dist < min_dist {
                                    min_dist = dist;
                                    best_seg_idx = seg_idx;
                                }
                            }

                            if min_dist > search_radius { continue; }

                            let noise = {
                                let epoch_seed = entity_seed(master_seed, (soma_id.wrapping_mul(31).wrapping_add(axon_idx)) as u32);
                                (epoch_seed ^ (epoch_seed >> 17)) as f32 / u64::MAX as f32
                            };
                            let score = compute_sprouting_score(const_mem, ax.type_idx.min(15) as u8, min_dist, 0.0, noise);
                            
                            seen_axons.insert(axon_idx);
                            candidates.push(Candidate { axon_idx, segment_idx: best_seg_idx, score });
                        }
                    }
                }
            }
        }

        candidates.sort_unstable_by(|a, b| {
            b.score.total_cmp(&a.score).then_with(|| a.axon_idx.cmp(&b.axon_idx))
        });

        for (slot, cand) in candidates.iter().take(MAX_DENDRITE_SLOTS).enumerate() {
            let axon_idx = cand.axon_idx;
            let variant = &const_mem.variants[axons[axon_idx].type_idx.min(15)];
            let abs_weight = (variant.gsop_potentiation.unsigned_abs() as i16).max(1).min(i16::MAX);
            let weight: i16 = if variant.gsop_depression < 0 { -abs_weight } else { abs_weight };

            slots.targets[slot] = pack_dendrite_target(axon_idx as u32, cand.segment_idx as u32);
            slots.weights[slot] = weight;
        }
    });

    println!("Baking: Transposing to Columnar Layout...");

    for slot in 0..MAX_DENDRITE_SLOTS {
        let col_offset = slot * pn;
        for i in 0..pn {
            shard.dendrite_targets[col_offset + i] = temp_slots[i].targets[slot];
            shard.dendrite_weights[col_offset + i] = temp_slots[i].weights[slot];
        }
    }
}

/// Привязывает один синапс к свободному слоту нейрона.
/// Соблюдает правило уникальности: 1 аксон = 1 дендритная связь.
pub fn bind_synapse(
    soa: &mut ShardSoA,
    soma_dense_idx: usize,
    axon_id: u32,
    segment_offset: u32,
    initial_weight: i16
) -> Result<(), String> {
    let padded_n = soa.padded_n;
    
    // 1. Проверяем правило уникальности и ищем пустой слот
    let mut empty_slot = None;
    for slot in 0..MAX_DENDRITE_SLOTS {
        let col_idx = ShardSoA::columnar_idx(padded_n, soma_dense_idx, slot);
        let target = soa.dendrite_targets[col_idx];
        
        // Target == 0 зарезервирован для пустых слотов
        if target != 0 && unpack_axon_id(target) == axon_id {
            return Ok(()); // Связь уже есть, игнорируем (Spec 04 §1.4)
        }
        if target == 0 && empty_slot.is_none() {
            empty_slot = Some(col_idx);
        }
    }
    
    // 2. Записываем в SoA
    if let Some(col_idx) = empty_slot {
        soa.dendrite_targets[col_idx] = pack_dendrite_target(axon_id, segment_offset);
        soa.dendrite_weights[col_idx] = initial_weight;
        soa.dendrite_timers[col_idx] = 0;
        Ok(())
    } else {
        Err(format!("Soma {} reached 128 dendrites limit", soma_dense_idx))
    }
}

/// Maintenance-подсистема (заглушка для редизайна)
#[allow(dead_code)]
pub fn reconnect_empty_dendrites(
    _targets: &mut [u32],
    _weights: &mut [i16],
    _downloaded_weights: &[i16],
    _padded_n: usize,
    _neurons: &[PlacedNeuron],
    _axons: &[GrownAxon],
    _const_mem: &GenesisConstantMemory,
    _master_seed: u64,
    _cell_size: u32,
) {
    // let mut _fake_soa = ShardSoA::new(_neurons.len(), _axons.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::layout::ShardSoA;
    use genesis_core::layout::{unpack_axon_id, unpack_segment_offset};

    #[test]
    fn test_bind_synapse_uniqueness() {
        let mut soa = ShardSoA::new(32, 100);
        let soma_idx = 5;
        let axon_id = 42;
        
        // Первая привязка
        bind_synapse(&mut soa, soma_idx, axon_id, 3, 1000).unwrap();
        
        // Вторая привязка того же аксона (должна быть проигнорирована)
        bind_synapse(&mut soa, soma_idx, axon_id, 5, 2000).unwrap();
        
        let mut count = 0;
        for slot in 0..MAX_DENDRITE_SLOTS {
            let target = soa.dendrite_targets[ShardSoA::columnar_idx(32, soma_idx, slot)];
            if target != 0 && unpack_axon_id(target) == axon_id {
                count += 1;
                assert_eq!(unpack_segment_offset(target), 3);
            }
        }
        assert_eq!(count, 1, "Axon 42 was bound twice to the same soma!");
    }

    #[test]
    fn test_bind_synapse_columnar() {
        let n = 64;
        let mut soa = ShardSoA::new(n, 100);
        
        // Привязываем в слот 0
        bind_synapse(&mut soa, 10, 1, 0, 500).unwrap();
        // Привязываем другой аксон (попадёт в слот 1)
        bind_synapse(&mut soa, 10, 2, 0, 600).unwrap();
        
        assert_ne!(soa.dendrite_targets[0 * n + 10], 0);
        assert_ne!(soa.dendrite_targets[1 * n + 10], 0);
        assert_eq!(unpack_axon_id(soa.dendrite_targets[1 * n + 10]), 2);
    }

    #[test]
    fn test_bind_synapse_limit() {
        let mut soa = ShardSoA::new(32, 200);
        for i in 0..MAX_DENDRITE_SLOTS {
            bind_synapse(&mut soa, 0, i as u32, 0, 100).unwrap();
        }
        
        // 129-й должен упасть
        let res = bind_synapse(&mut soa, 0, 999, 0, 100);
        assert!(res.is_err());
    }
}
