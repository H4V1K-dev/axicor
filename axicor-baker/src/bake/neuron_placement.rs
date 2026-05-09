use axicor_core::config::anatomy::AnatomyConfig;
use axicor_core::config::InstanceConfig;
use axicor_core::types::PackedPosition;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng; // Fast and deterministic algorithm

pub fn generate_placement_from_config(
    anatomy: &AnatomyConfig,
    shard_cfg: &InstanceConfig,
    master_seed: u64, // Removed global_density from here
    type_names: &[String],
) -> Vec<PackedPosition> {
    let max_x = shard_cfg.dimensions.w;
    let max_y = shard_cfg.dimensions.d;
    let max_z = shard_cfg.dimensions.h;

    let off_x = shard_cfg.world_offset.x;
    let off_y = shard_cfg.world_offset.y;
    let off_z = shard_cfg.world_offset.z;

    assert!(off_x + max_x <= 0x3FF, "Zone X range exceeds 10-bit limit");
    assert!(off_y + max_y <= 0x3FF, "Zone Y range exceeds 10-bit limit");
    assert!(off_z + max_z <= 0xFF, "Zone Z range exceeds 8-bit limit");

    let mut positions = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(master_seed);
    let mut current_z_pct = 0.0;

    for layer in &anatomy.layers {
        let z_start_local = (current_z_pct * max_z as f32).floor() as u32;
        let z_end_local = ((current_z_pct + layer.height_pct) * max_z as f32).floor() as u32;
        current_z_pct += layer.height_pct;

        let layer_volume = max_x * max_y * (z_end_local - z_start_local).max(1);

        // [DOD FIX] Layer budget is calculated from layer volume and its specific density!
        let layer_budget = (layer_volume as f32 * layer.density).floor() as usize;
        if layer_budget == 0 {
            continue;
        }

        let mut pool: Vec<u32> = (0..layer_volume).collect();
        // 100% Deterministic In-Place shuffle
        pool.shuffle(&mut rng);

        let most_frequent_type_name = layer
            .composition
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
            .unwrap_or_default();
        let fallback_type_id = type_names
            .iter()
            .position(|n| n == &most_frequent_type_name)
            .unwrap_or_else(|| {
                panic!(
                    "FATAL: Type '{}' specified in anatomy.toml is missing in blueprints.toml!",
                    most_frequent_type_name
                )
            }) as u8;

        let mut type_pool = Vec::with_capacity(layer_budget);
        for (type_name, &quota) in &layer.composition {
            let count = (quota * layer_budget as f32).floor() as usize;
            let type_id = type_names
                .iter()
                .position(|n| n == type_name)
                .unwrap_or_else(|| {
                    panic!(
                        "FATAL: Type '{}' specified in anatomy.toml is missing in blueprints.toml!",
                        type_name
                    )
                }) as u8;
            for _ in 0..count {
                type_pool.push(type_id);
            }
        }
        while type_pool.len() < layer_budget {
            type_pool.push(fallback_type_id);
        }

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

    // Warp Alignment padding (Invariant 2 - 64B L2 Cache Line)
    let padded_n = axicor_core::layout::align_to_warp(positions.len());
    while positions.len() < padded_n {
        positions.push(PackedPosition::pack_raw(0, 0, 0, 0));
    }

    // Z-sort
    positions.sort_by_key(|p| p.z());
    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use axicor_core::config::anatomy::{AnatomyConfig, LayerConfig};
    use axicor_core::config::InstanceConfig;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_placement_zero_collisions() {
        let mut composition = HashMap::new();
        composition.insert("TypeA".to_string(), 1.0);

        let anatomy = AnatomyConfig {
            layers: vec![LayerConfig {
                name: "L1".to_string(),
                height_pct: 1.0,
                density: 1.0,
                composition,
            }],
        };

        let shard_cfg_str = r#"
            zone_id = "test_zone"
            [world_offset]
            x = 0
            y = 0
            z = 0
            [dimensions]
            w = 10
            d = 10
            h = 10
            [neighbors]
            [settings]
            bsp_sync_timeout_ms = 500
        "#;
        let shard_cfg: InstanceConfig = toml::from_str(shard_cfg_str).unwrap();

        let type_names = vec!["TypeA".to_string()];
        let positions = generate_placement_from_config(&anatomy, &shard_cfg, 42, &type_names);

        let mut unique_positions = HashSet::new();
        let mut dummies = 0;
        for p in &positions {
            if p.0 == 0 {
                dummies += 1;
            } else {
                unique_positions.insert(p.0);
            }
        }

        // We might have one real neuron at 0,0,0 with type 0, which .0 is 0.
        // Total positions = 1000. Padded = 1024.
        // We expect exactly 1000 unique real positions + dummy paddings.
        // If 0,0,0 is a real position, dummies will be 24 (padding) + 1 (real).
        assert_eq!(
            unique_positions.len() + (if dummies > 24 { 1 } else { 0 }),
            1000
        );
    }

    #[test]
    fn test_placement_warp_alignment() {
        let mut composition = HashMap::new();
        composition.insert("TypeA".to_string(), 1.0);

        let anatomy = AnatomyConfig {
            layers: vec![LayerConfig {
                name: "L2".to_string(),
                height_pct: 1.0,
                density: 0.042, // 1000 * 0.042 = 42
                composition,
            }],
        };

        let shard_cfg_str = r#"
            zone_id = "test_zone"
            [world_offset]
            x = 0
            y = 0
            z = 0
            [dimensions]
            w = 10
            d = 10
            h = 10
            [neighbors]
            [settings]
            bsp_sync_timeout_ms = 500
        "#;
        let shard_cfg: InstanceConfig = toml::from_str(shard_cfg_str).unwrap();

        let type_names = vec!["TypeA".to_string()];
        let positions = generate_placement_from_config(&anatomy, &shard_cfg, 123, &type_names);

        assert_eq!(positions.len(), 64);

        let dummies_count = positions.iter().filter(|p| p.0 == 0).count();
        // Expecting exactly 22 paddings. Plus maybe 1 real if at 0,0,0
        assert!(dummies_count == 22 || dummies_count == 23);
    }
}
