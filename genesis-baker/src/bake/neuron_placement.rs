use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng; // Быстрый и детерминированный алгоритм
use genesis_core::types::PackedPosition;
use genesis_core::config::anatomy::AnatomyConfig;
use genesis_core::config::InstanceConfig;

pub fn generate_placement_from_config(
    anatomy: &AnatomyConfig,
    shard_cfg: &InstanceConfig,
    master_seed: u64, // Удалили global_density отсюда
    type_names: &[String],
) -> Vec<PackedPosition> {
    let max_x = shard_cfg.dimensions.w;
    let max_y = shard_cfg.dimensions.d;
    let max_z = shard_cfg.dimensions.h;

    let off_x = shard_cfg.world_offset.x;
    let off_y = shard_cfg.world_offset.y;
    let off_z = shard_cfg.world_offset.z;

    assert!(off_x + max_x <= 0x7FF, "Zone X range exceeds 11-bit limit");
    assert!(off_y + max_y <= 0x7FF, "Zone Y range exceeds 11-bit limit");
    assert!(off_z + max_z <= 0x3F,  "Zone Z range exceeds 6-bit limit");

    let mut positions = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(master_seed);
    let mut current_z_pct = 0.0;

    for layer in &anatomy.layers {
        let z_start_local = (current_z_pct * max_z as f32).floor() as u32;
        let z_end_local   = ((current_z_pct + layer.height_pct) * max_z as f32).floor() as u32;
        current_z_pct += layer.height_pct;

        let layer_volume = max_x * max_y * (z_end_local - z_start_local).max(1);
        
        // [DOD FIX] Бюджет слоя считается от объема слоя и его личной плотности!
        let layer_budget = (layer_volume as f32 * layer.density).floor() as usize;
        if layer_budget == 0 { continue; }

        let mut pool: Vec<u32> = (0..layer_volume).collect();
        // 100% Детерминированное перемешивание In-Place
        pool.shuffle(&mut rng);

        let most_frequent_type_name = layer.composition.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_default();
        let fallback_type_id = type_names.iter().position(|n| n == &most_frequent_type_name)
            .unwrap_or_else(|| panic!("FATAL: Type '{}' specified in anatomy.toml is missing in blueprints.toml!", most_frequent_type_name)) as u8;

        let mut type_pool = Vec::with_capacity(layer_budget);
        for (type_name, &quota) in &layer.composition {
            let count = (quota * layer_budget as f32).floor() as usize;
            let type_id = type_names.iter().position(|n| n == type_name)
                .unwrap_or_else(|| panic!("FATAL: Type '{}' specified in anatomy.toml is missing in blueprints.toml!", type_name)) as u8;
            for _ in 0..count { type_pool.push(type_id); }
        }
        while type_pool.len() < layer_budget { type_pool.push(fallback_type_id); }

        for type_id in type_pool {
            let flat_idx = pool.pop().expect("FATAL: Layer density > 100%");

            let lz = z_start_local + (flat_idx / (max_x * max_y));
            let rem = flat_idx % (max_x * max_y);
            let ly = rem / max_x;
            let lx = rem % max_x;

            let gx = lx + off_x;
            let gy = ly + off_y;
            let gz = lz + off_z;

            positions.push(PackedPosition::pack_raw(gx, gy, gz, type_id));
        }
    }

    // Warp Alignment padding (Invariant 2)
    let remainder = positions.len() % genesis_core::constants::GPU_WARP_SIZE;
    if remainder != 0 {
        for _ in 0..(genesis_core::constants::GPU_WARP_SIZE - remainder) {
            positions.push(PackedPosition::pack_raw(0, 0, 0, 0));
        }
    }

    // Z-sort
    positions.sort_by_key(|p| p.z());
    positions
}
