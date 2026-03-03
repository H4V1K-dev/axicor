use std::collections::HashSet;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng; // Быстрый и детерминированный алгоритм
use genesis_core::types::PackedPosition;
use genesis_core::config::anatomy::AnatomyConfig;

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
    let mut occupancy = HashSet::with_capacity(total_capacity);
    
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

        // 4. Размещаем нейроны со строгим контролем коллизий (Reject-Sampling)
        for type_id in type_pool {
            let mut attempt = 0;
            loop {
                let x = rng.gen_range(0..max_x);
                let y = rng.gen_range(0..max_y);
                let z = rng.gen_range(z_start..z_end.max(z_start + 1)); // Защита от нулевой толщины
                
                // Пространственный хэш вокселя (28 бит)
                let voxel_hash = x | (y << 11) | (z << 22);

                if occupancy.insert(voxel_hash) {
                    positions.push(PackedPosition::pack_raw(x, y, z, type_id));
                    break;
                }

                attempt += 1;
                if attempt > 100 {
                    panic!(
                        "[Baker] FATAL: Occupancy collision limit reached! \
                        Density is too high for layer at Z: {}-{}. \
                        Increase voxel_size or reduce global_density.",
                        z_start, z_end
                    );
                }
            }
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
