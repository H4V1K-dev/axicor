use crate::bake::layout::ShardSoA;
use crate::bake::axon_growth::GrownAxon;
use genesis_core::types::PackedPosition;
use genesis_core::config::blueprints::NeuronType;
use crate::bake::spatial_grid::{AxonSegmentGrid, SegmentRef};
use rayon::prelude::*;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct TempSlot {
    target: u32,
    weight: i16,
    _pad: u16,
}

pub fn connect_dendrites(
    shard: &mut ShardSoA,
    positions: &[PackedPosition],
    axons: &[GrownAxon],
    types: &[NeuronType],
    voxel_size_um: f32, // [DOD] Вместо хардкод cell_size берем размер вокселя
) -> usize {
    // 1. ИНВЕРСИЯ: Строим SpatialGrid из СЕГМЕНТОВ аксонов.
    // Оптимальный размер чанка для хэш-сетки = 2 вокселя (~50 мкм)
    let cell_size_vox = 2u32;
    let segment_grid = AxonSegmentGrid::build_from_axons(axons, cell_size_vox);

    // 2. Локальные AoS буферы для каждой сомы (Zero Lock Contention)
    let mut aos_dendrites = vec![[TempSlot::default(); genesis_core::constants::MAX_DENDRITE_SLOTS]; shard.padded_n];
    let mut soma_slot_counts = vec![0usize; shard.padded_n];

    // 3. 100% Параллельный сбор связей
    let total_synapses: usize = aos_dendrites.par_iter_mut()
        .zip(soma_slot_counts.par_iter_mut())
        .enumerate()
        .filter(|(soma_id, _)| positions[*soma_id].0 != 0) // Игнорируем пустышки
        .map(|(soma_id, (slots, count))| {
            let my_pos = positions[soma_id];
            let my_type = &types[my_pos.type_id() as usize];

            // [DOD FIX] Динамический радиус дендритов из Бланка (blueprints.toml)
            let radius_vox = (my_type.dendrite_radius_um / voxel_size_um).ceil() as u32;
            let radius_cells = (radius_vox as f32 / cell_size_vox as f32).ceil() as i32;

            // Сома сканирует пространство вокруг себя
            segment_grid.for_each_in_radius(&my_pos, radius_cells, |seg_ref: &SegmentRef| {
                if *count >= genesis_core::constants::MAX_DENDRITE_SLOTS { return; }
                
                let axon_id = seg_ref.axon_id as usize;
                
                // Самоисключение
                if axons[axon_id].soma_idx == soma_id { return; } 
                
                let owner_type_idx = seg_ref.type_idx as usize;
                let owner_name = &types[owner_type_idx].name;
                
                //TODO рефакторинг логики
                if !my_type.dendrite_whitelist.is_empty() && !my_type.dendrite_whitelist.contains(owner_name) {
                    return; 
                }

                // Rule of Uniqueness: O(K) линейный поиск в горячем L1 кэше
                let is_duplicate = slots[0..*count].iter().any(|s| {
                    genesis_core::layout::unpack_axon_id(s.target) == axon_id as u32
                });
                
                if !is_duplicate {
                    let weight = if my_type.is_inhibitory {
                        -(my_type.initial_synapse_weight as i16)
                    } else {
                        my_type.initial_synapse_weight as i16
                    };

                    slots[*count] = TempSlot {
                        target: genesis_core::layout::pack_dendrite_target(axon_id as u32, seg_ref.seg_idx as u32),
                        weight,
                        _pad: 0,
                    };
                    *count += 1;
                }
            });
            *count
        })
        .sum();

    // 4. ТРАНСПОЗИЦИЯ: AoS -> SoA (Один линейный проход)
    for soma_id in 0..shard.padded_n {
        let count = soma_slot_counts[soma_id];
        for slot in 0..count {
            let col_idx = slot * shard.padded_n + soma_id;
            let temp = aos_dendrites[soma_id][slot];
            shard.dendrite_targets[col_idx] = temp.target;
            shard.dendrite_weights[col_idx] = temp.weight;
        }

        // [DOD] Invariant: The rest of the column must remain zero (Empty Slot triggers GPU early exit)
        debug_assert!({
            let mut zeros_valid = true;
            for slot in count..genesis_core::constants::MAX_DENDRITE_SLOTS {
                let col_idx = slot * shard.padded_n + soma_id;
                if shard.dendrite_targets[col_idx] != 0 {
                    zeros_valid = false;
                    break;
                }
            }
            zeros_valid
        }, "Dendrite targets trailing slots must be zero for early exit");
    }

    total_synapses
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::types::PackedPosition;
    use genesis_core::config::blueprints::NeuronType;
    use crate::bake::axon_growth::GrownAxon;
    use crate::bake::layout::ShardSoA;

    #[test]
    fn test_concurrent_somas_connect_to_same_axon() {
        // Prepare mock setup
        let mut shard = ShardSoA::new(32, 32); // Creates padded_n = 32
        
        let mut positions = vec![PackedPosition::pack_raw(0, 0, 0, 0); 32];
        let type_idx = 1;
        // Two somas near each other in the same voxel block
        positions[0] = PackedPosition::pack_raw(10, 10, 10, type_idx); // Soma A
        positions[1] = PackedPosition::pack_raw(10, 10, 11, type_idx); // Soma B
        
        // Setup simple type
        let mut types = vec![NeuronType::default(); 2];
        types[usize::from(type_idx)].initial_synapse_weight = 100;

        // Create one axon from another soma (soma_idx 2) passing through
        let mut segments = vec![];
        for z in 5..=15 {
            segments.push(PackedPosition::pack_raw(10, 10, z, type_idx).0);
        }
        
        let axon = GrownAxon {
            soma_idx: 2,
            type_idx: usize::from(type_idx),
            tip_x: 10, tip_y: 10, tip_z: 15,
            length_segments: segments.len() as u32,
            segments,
            last_dir: glam::Vec3::Z,
        };

        let axons = vec![axon];
        
        let voxel_size_um = 25.0;
        let connections = connect_dendrites(&mut shard, &positions, &axons, &types, voxel_size_um);

        // Since both soma A and soma B scan through space, they both read the same axon.
        // Because data races are eliminated with TempSlot buffers per soma, both must connect safely.
        assert_eq!(connections, 2, "Both somas must connect to the passing axon");

        // Verify uniqueness Rule (no repeated connects to same axon)
        // Soma A is at padded_n index 0. Slot 0 target
        let target_a = shard.dendrite_targets[0 * shard.padded_n + 0];
        assert_eq!(genesis_core::layout::unpack_axon_id(target_a), 0);

        let target_b = shard.dendrite_targets[0 * shard.padded_n + 1];
        assert_eq!(genesis_core::layout::unpack_axon_id(target_b), 0);
        
        // Slot 1 must be empty
        let target_a1 = shard.dendrite_targets[1 * shard.padded_n + 0];
        assert_eq!(target_a1, 0);
    }
}
