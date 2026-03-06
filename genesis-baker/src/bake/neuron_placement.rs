use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng; // Быстрый и детерминированный алгоритм
use genesis_core::types::PackedPosition;
use genesis_core::config::anatomy::AnatomyConfig;
use genesis_core::config::InstanceConfig;

/// [DOD FIX] Размещение нейронов в глобальных координатах из shard.toml.
/// Размеры (w/d/h) берутся напрямую из InstanceConfig — уже в вокселях.
/// Координаты сдвигаются на world_offset, раздвигая зоны по оси X.
pub fn generate_placement_from_config(
    anatomy: &AnatomyConfig,
    shard_cfg: &InstanceConfig,
    global_density: f32,
    master_seed: u64,
    type_names: &[String],
) -> Vec<PackedPosition> {
    let max_x = shard_cfg.dimensions.w;
    let max_y = shard_cfg.dimensions.d;
    let max_z = shard_cfg.dimensions.h;

    let off_x = shard_cfg.world_offset.x;
    let off_y = shard_cfg.world_offset.y;
    let off_z = shard_cfg.world_offset.z;

    // [Spec 03 §1.3] PackedPosition: X — 11 бит (max 2047), Z — 6 бит (max 63)
    assert!(off_x + max_x <= 0x7FF, "Zone X range exceeds 11-bit limit (2047 voxels): off_x={} w={}", off_x, max_x);
    assert!(off_y + max_y <= 0x7FF, "Zone Y range exceeds 11-bit limit (2047 voxels): off_y={} d={}", off_y, max_y);
    assert!(off_z + max_z <= 0x3F,  "Zone Z range exceeds 6-bit limit (63 voxels): off_z={} h={}", off_z, max_z);

    let total_voxels = max_x * max_y * max_z;
    let total_capacity = (total_voxels as f32 * global_density).floor() as usize;

    let mut positions = Vec::with_capacity(total_capacity);
    let mut rng = ChaCha8Rng::seed_from_u64(master_seed);
    let mut current_z_pct = 0.0;

    for layer in &anatomy.layers {
        let z_start_local = (current_z_pct * max_z as f32).floor() as u32;
        let z_end_local   = ((current_z_pct + layer.height_pct) * max_z as f32).floor() as u32;
        current_z_pct += layer.height_pct;

        let layer_budget = (layer.population_pct * total_capacity as f32).floor() as usize;
        if layer_budget == 0 { continue; }

        let layer_volume = max_x * max_y * (z_end_local - z_start_local).max(1);
        let mut pool: Vec<u32> = (0..layer_volume).collect();
        // 100% Детерминированное перемешивание In-Place
        pool.shuffle(&mut rng);

        let most_frequent_type_name = layer.composition.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_default();
        let fallback_type_id = type_names.iter().position(|n| n == &most_frequent_type_name).unwrap_or(0) as u8;

        let mut type_pool = Vec::with_capacity(layer_budget);
        for (type_name, &quota) in &layer.composition {
            let count = (quota * layer_budget as f32).floor() as usize;
            let type_id = type_names.iter().position(|n| n == type_name).unwrap_or(0) as u8;
            for _ in 0..count { type_pool.push(type_id); }
        }
        while type_pool.len() < layer_budget { type_pool.push(fallback_type_id); }

        for type_id in type_pool {
            // O(1) извлечение без коллизий
            let flat_idx = pool.pop().expect("FATAL: Density > 100% or Math Error");
            
            // O(1) Распаковка локальных координат
            let lz = z_start_local + (flat_idx / (max_x * max_y));
            let rem = flat_idx % (max_x * max_y);
            let ly = rem / max_x;
            let lx = rem % max_x;

            // Перевод в глобальные координаты
            let gx = lx + off_x;
            let gy = ly + off_y;
            let gz = lz + off_z;

            // Упаковка
            positions.push(PackedPosition::pack_raw(gx, gy, gz, type_id));
        }
    }

    // Warp Alignment padding (Invariant 2)
    let remainder = positions.len() % 32;
    if remainder != 0 {
        for _ in 0..32 - remainder {
            positions.push(PackedPosition::pack_raw(0, 0, 0, 0));
        }
    }

    // Z-sort (Invariant 7: Determinism + IndexMapping)
    positions.sort_by_key(|p| p.z());
    positions
}


pub struct ZoneDimensions {
    pub width_um: f32,
    pub depth_um: f32,
    pub height_um: f32,
}

/// Выполняет детерминированное размещение нейронов и квантование координат.
/// Возвращает массив PackedPosition (SoA-ready).
pub fn generate_placement(
    anatomy: &AnatomyConfig,
    dims: &ZoneDimensions,
    voxel_size_um: f32,
    global_density: f32,
    master_seed: u64,
    type_names: &[String],
) -> Vec<PackedPosition> {
    // 1. Вычисляем границы воксельной сетки (максимум для 11 бит = 2047, 6 бит = 63)
    let max_x = (dims.width_um / voxel_size_um).floor() as u32;
    let max_y = (dims.depth_um / voxel_size_um).floor() as u32;
    let max_z = (dims.height_um / voxel_size_um).floor() as u32;

    assert!(max_x <= 0x7FF, "Width exceeds 11-bit limit (2047 voxels)");
    assert!(max_y <= 0x7FF, "Depth exceeds 11-bit limit (2047 voxels)");
    assert!(max_z <= 0x3F, "Height exceeds 6-bit limit (63 voxels)");

    let total_voxels = max_x * max_y * max_z;
    let total_capacity = (total_voxels as f32 * global_density).floor() as usize;

    let mut positions = Vec::with_capacity(total_capacity);
    
    // Инициализируем детерминированный генератор
    let mut rng = ChaCha8Rng::seed_from_u64(master_seed);

    let mut current_z_pct = 0.0;

    // 2. Идем по слоям сверху вниз
    for layer in &anatomy.layers {
        // Пространственные рамки слоя в вокселях
        let z_start = (current_z_pct * max_z as f32).floor() as u32;
        let z_end = ((current_z_pct + layer.height_pct) * max_z as f32).floor() as u32;
        current_z_pct += layer.height_pct;

        let layer_budget = (layer.population_pct * total_capacity as f32).floor() as usize;
        if layer_budget == 0 { continue; }

        let layer_volume = max_x * max_y * (z_end - z_start).max(1);
        let mut pool: Vec<u32> = (0..layer_volume).collect();
        // 100% Детерминированное перемешивание In-Place
        pool.shuffle(&mut rng);

        // Поиск самого частого типа для добивания остатка
        let most_frequent_type_name = layer.composition.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_default();
        let fallback_type_id = type_names.iter().position(|n| n == &most_frequent_type_name).unwrap_or(0) as u8;

        // 3. Формируем точный пул типов (Hard Quotas)
        let mut type_pool = Vec::with_capacity(layer_budget);
        for (type_name, &quota) in &layer.composition {
            let count = (quota * layer_budget as f32).floor() as usize;
            let type_id = type_names.iter().position(|n| n == type_name).unwrap_or(0) as u8;
            for _ in 0..count {
                type_pool.push(type_id);
            }
        }

        // Если из-за floor() не хватило нейронов до бюджета, добиваем самым частым типом
        while type_pool.len() < layer_budget {
            type_pool.push(fallback_type_id);
        }

        // 4. Размещаем нейроны со строгим контролем коллизий
        for type_id in type_pool {
            let flat_idx = pool.pop().expect("FATAL: Density > 100% or Math Error");
            
            let z = z_start + (flat_idx / (max_x * max_y));
            let rem = flat_idx % (max_x * max_y);
            let y = rem / max_x;
            let x = rem % max_x;

            positions.push(PackedPosition::pack_raw(x, y, z, type_id));
        }
    }

    // ВАЖНО: Паддинг для Warp Alignment (кратность 32)
    // Это гарантирует 100% Coalesced Access в UpdateNeurons (см. 02_configuration.md §4.3)
    let remainder = positions.len() % 32;
    if remainder != 0 {
        let pad_count = 32 - remainder;
        for _ in 0..pad_count {
            // Добиваем пустышками (0 координаты, 0 тип)
            positions.push(PackedPosition::pack_raw(0, 0, 0, 0));
        }
    }

    // ⚠️ ЗАКОН ДЕТЕРМИНИЗМА 3: Z-Sorting (08_io_matrix.md §3.1)
    // Весь массив сом обязан быть отсортирован по Z-координате (по возрастанию)
    // перед тем как индекс в векторе станет легальным Dense ID.
    positions.sort_by_key(|p| p.z());

    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::config::anatomy::LayerConfig;

    #[test]
    fn test_extreme_density_placement() {
        let mut composition = std::collections::HashMap::new();
        composition.insert("Excitatory".into(), 1.0);

        let anatomy = AnatomyConfig {
            layers: vec![
                LayerConfig {
                    name: "L1".into(),
                    height_pct: 1.0,
                    population_pct: 1.0,
                    composition,
                }
            ],
        };

        let dims = ZoneDimensions {
            width_um: 50.0,
            depth_um: 50.0,
            height_um: 50.0,
        };
        // Ensure max voxel limits don't panic while checking Extreme density.
        let voxel_size = 10.0; 
        // -> max_x = 5, max_y = 5, max_z = 5 -> total 125 

        let type_names = vec!["NullType".into(), "Excitatory".into()];

        let positions = generate_placement(
            &anatomy,
            &dims,
            voxel_size,
            1.0, // 100% density
            0,
            &type_names
        );

        // Before padding: 125 positions.
        // Remainder: 125 % 32 = 29 -> padding needed: 3.
        // Total positions: 125 + 3 = 128
        assert_eq!(positions.len(), 128);

        // Verify that there are exactly 125 non-dummy neurons.
        let real_neurons = positions.iter().filter(|p| p.0 != 0).count();
        assert_eq!(real_neurons, 125);
    }
}
