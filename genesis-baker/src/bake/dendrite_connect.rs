use crate::bake::axon_growth::GrownAxon;
use crate::bake::layout::ShardStateSoA;
use crate::bake::neuron_placement::PlacedNeuron;
use crate::bake::seed::entity_seed;
use crate::bake::sprouting::{sprouting_score, voxel_dist, SproutingWeights};
use crate::parser::blueprints::NeuronType;
use genesis_core::constants::MAX_DENDRITE_SLOTS;
use genesis_core::coords::{pack_target, unpack_target};
use std::collections::HashMap;

pub const INITIAL_EXCITATORY_WEIGHT: i16 = 74;
pub const INITIAL_INHIBITORY_WEIGHT: i16 = -74;

/// Размер ячейки пространственной решётки (в вокселях).
/// Кандидат может находиться не дальше CELL_SIZE от сомы → ищем только в 3×3×3 ячейках.
const CELL_SIZE: u32 = 30;

/// Ключ ячейки пространственной решётки.
type GridCell = (u32, u32, u32);

/// Строит HashMap: grid_cell → список индексов аксонов, хотя бы один сегмент которых проходит через ячейку.
/// Позволяет заменить O(N²) полный перебор на O(N × K) поиск по соседям.
fn build_axon_grid(axons: &[GrownAxon]) -> HashMap<GridCell, Vec<usize>> {
    let mut grid: HashMap<GridCell, Vec<usize>> = HashMap::new();
    for (i, ax) in axons.iter().enumerate() {
        // Мы добавляем аксон во все ячейки, через которые он проходит.
        // Чтобы не дублировать ID аксона в одной и той же ячейке:
        let mut touched_cells = std::collections::HashSet::new();
        
        for &seg in &ax.segments {
            let z = (seg >> 20) & 0xFF;
            let y = (seg >> 10) & 0x3FF;
            let x = seg & 0x3FF;
            
            let cell = (
                x / CELL_SIZE,
                y / CELL_SIZE,
                z / CELL_SIZE,
            );
            touched_cells.insert(cell);
        }
        
        for cell in touched_cells {
            grid.entry(cell).or_default().push(i);
        }
    }
    grid
}

/// Кандидат дендритного слота.
struct Candidate {
    axon_idx: usize,
    segment_idx: usize,
    score: f32,
}

/// Заполняет dendrite_targets и dendrite_weights в ShardStateSoA.
/// Использует пространственную решётку для O(N × K) поиска вместо O(N²).
pub fn connect_dendrites(
    shard: &mut ShardStateSoA,
    neurons: &[PlacedNeuron],
    axons: &[GrownAxon],
    neuron_types: &[NeuronType],
    master_seed: u64,
) {
    let pn = shard.padded_n;

    // Строим пространственную решётку по tip-позициям аксонов
    let grid = build_axon_grid(axons);

    for (soma_id, neuron) in neurons.iter().enumerate() {
        let soma_x = neuron.x();
        let soma_y = neuron.y();
        let soma_z = neuron.z();
        let nt = &neuron_types[neuron.type_idx.min(neuron_types.len() - 1)];
        let cfg = SproutingWeights::from_neuron_type(nt);

        // Диапазон ячеек для поиска
        let cell_x = soma_x / CELL_SIZE;
        let cell_y = soma_y / CELL_SIZE;
        let cell_z = soma_z / CELL_SIZE;

        let mut candidates: Vec<Candidate> = Vec::new();

        // Проверяем 3×3×3 соседних ячейки
        for dx in 0..=2u32 {
            for dy in 0..=2u32 {
                for dz in 0..=2u32 {
                    let cx = cell_x.saturating_add(dx).saturating_sub(1);
                    let cy = cell_y.saturating_add(dy).saturating_sub(1);
                    let cz = cell_z.saturating_add(dz).saturating_sub(1);

                    let Some(cell_axons) = grid.get(&(cx, cy, cz)) else {
                        continue;
                    };

                    for &axon_idx in cell_axons {
                        let ax = &axons[axon_idx];
                        if ax.soma_idx == soma_id {
                            continue;
                        }

                        // Find the closest segment of this axon to the soma
                        let mut min_dist = f32::MAX;
                        let mut best_seg_idx = 0;
                        
                        for (seg_idx, &seg) in ax.segments.iter().enumerate() {
                            let z = (seg >> 20) & 0xFF;
                            let y = (seg >> 10) & 0x3FF;
                            let x = seg & 0x3FF;
                            
                            let dist = voxel_dist(soma_x, soma_y, soma_z, x, y, z);
                            if dist < min_dist {
                                min_dist = dist;
                                best_seg_idx = seg_idx;
                            }
                        }

                        if min_dist > CELL_SIZE as f32 {
                            continue;
                        }

                        let epoch_seed = entity_seed(
                            master_seed,
                            (soma_id.wrapping_mul(31).wrapping_add(axon_idx)) as u32,
                        );
                        let score = sprouting_score(min_dist, 0.0, epoch_seed, &cfg);
                        candidates.push(Candidate { axon_idx, segment_idx: best_seg_idx, score });
                    }
                }
            }
        }

        // Сортируем по убыванию score
        candidates.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Записываем top-N в columnar SoA
        for (slot, cand) in candidates.iter().take(MAX_DENDRITE_SLOTS).enumerate() {
            let axon_idx = cand.axon_idx;
            let target_type = axons[axon_idx].type_idx;
            let is_excitatory = neuron_types
                .get(target_type)
                .map(|n| n.name.contains("Excitatory"))
                .unwrap_or(true);
            let weight = if is_excitatory {
                INITIAL_EXCITATORY_WEIGHT
            } else {
                INITIAL_INHIBITORY_WEIGHT
            };

            let target_packed = pack_target(axon_idx as u32, cand.segment_idx as u32);
            
            let cell = slot * pn + soma_id;
            shard.dendrite_targets[cell] = target_packed;
            shard.dendrite_weights[cell] = weight;
        }
    }
}

/// Вызывается оркестратором во время Night Phase (Maintenance Pipeline).
/// Сканирует существующие `dendrite_targets` на наличие пустых слотов (`0`), 
/// ищет новые аксоны через Cone Tracing (пространственную решётку), 
/// и перезаписывает пустые `targets` и `weights`.
pub fn reconnect_empty_dendrites(
    targets: &mut [u32],
    weights: &mut [i16],
    padded_n: usize,
    neurons: &[PlacedNeuron],
    axons: &[GrownAxon],
    neuron_types: &[NeuronType],
    master_seed: u64,
) {
    let grid = build_axon_grid(axons);

    for (soma_id, neuron) in neurons.iter().enumerate() {
        // Virtual axons have no dendrite slots — skip them
        if soma_id >= padded_n {
            break;
        }
        // Collect indices of empty slots for this neuron
        let mut empty_slots: Vec<usize> = Vec::new();
        for slot in 0..MAX_DENDRITE_SLOTS {
            let cell = slot * padded_n + soma_id;
            if targets[cell] == 0 {
                empty_slots.push(slot);
            }
        }

        if empty_slots.is_empty() {
            continue; // Neuron is fully connected, skip spatial search
        }

        let soma_x = neuron.x();
        let soma_y = neuron.y();
        let soma_z = neuron.z();
        let nt = &neuron_types[neuron.type_idx.min(neuron_types.len() - 1)];
        let cfg = SproutingWeights::from_neuron_type(nt);

        let cell_x = soma_x / CELL_SIZE;
        let cell_y = soma_y / CELL_SIZE;
        let cell_z = soma_z / CELL_SIZE;

        let mut candidates: Vec<Candidate> = Vec::new();

        // 3x3x3 Neighborhood search
        for dx in 0..=2u32 {
            for dy in 0..=2u32 {
                for dz in 0..=2u32 {
                    let cx = cell_x.saturating_add(dx).saturating_sub(1);
                    let cy = cell_y.saturating_add(dy).saturating_sub(1);
                    let cz = cell_z.saturating_add(dz).saturating_sub(1);

                    let Some(cell_axons) = grid.get(&(cx, cy, cz)) else {
                        continue;
                    };

                    for &axon_idx in cell_axons {
                        let ax = &axons[axon_idx];
                        if ax.soma_idx == soma_id {
                            continue; // No self connections
                        }

                        // Also prevent duplicate connections (check existing targets)
                        let mut already_connected = false;
                        for slot in 0..MAX_DENDRITE_SLOTS {
                            let cell = slot * padded_n + soma_id;
                            let target = targets[cell];
                            if unpack_target(target).map_or(false, |(id, _)| id as usize == axon_idx) {
                                already_connected = true;
                                break;
                            }
                        }
                        if already_connected {
                            continue;
                        }

                        let mut min_dist = f32::MAX;
                        let mut best_seg_idx = 0;
                        
                        for (seg_idx, &seg) in ax.segments.iter().enumerate() {
                            let z = (seg >> 20) & 0xFF;
                            let y = (seg >> 10) & 0x3FF;
                            let x = seg & 0x3FF;
                            
                            let dist = voxel_dist(soma_x, soma_y, soma_z, x, y, z);
                            if dist < min_dist {
                                min_dist = dist;
                                best_seg_idx = seg_idx;
                            }
                        }

                        if min_dist > CELL_SIZE as f32 {
                            continue;
                        }

                        let epoch_seed = entity_seed(
                            master_seed,
                            (soma_id.wrapping_mul(31).wrapping_add(axon_idx)) as u32,
                        );
                        // TODO: Pass actual target power from downloaded weights
                        let score = sprouting_score(min_dist, 0.0, epoch_seed, &cfg);
                        candidates.push(Candidate { axon_idx, segment_idx: best_seg_idx, score });
                    }
                }
            }
        }

        candidates.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Fill only the empty slots
        for (cand, &slot) in candidates.iter().zip(empty_slots.iter()) {
            let axon_idx = cand.axon_idx;
            let target_type = axons[axon_idx].type_idx;
            let is_excitatory = neuron_types
                .get(target_type)
                .map(|n| n.name.contains("Excitatory"))
                .unwrap_or(true);
            let weight = if is_excitatory {
                INITIAL_EXCITATORY_WEIGHT
            } else {
                INITIAL_INHIBITORY_WEIGHT
            };

            let target_packed = pack_target(axon_idx as u32, cand.segment_idx as u32);
            
            let cell = slot * padded_n + soma_id;
            targets[cell] = target_packed;
            weights[cell] = weight;
        }
    }
}

