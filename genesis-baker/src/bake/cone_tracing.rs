use crate::bake::neuron_placement::PlacedNeuron;
use crate::bake::spatial_grid::SpatialGrid;
use glam::Vec3;

// We need a way to determine how "attractive" a neuron is.
// In specs: "target.attraction_gradient / (dist_sq + 1e-5)"
// For now, let's treat every valid target as having weight = 1.0. 
// A more advanced version could use `soma_power_index` from the NightPhase sprouting score.
const ATTRACTION_GRADIENT: f32 = 1.0;

/// Returns the V_attract vector and the total weight.
pub fn calculate_v_attract(
    head_pos: Vec3,
    forward_dir: Vec3,
    fov_cos: f32, // cos(FOV / 2)
    max_search_radius_vox: f32,
    spatial_grid: &SpatialGrid,
    neurons: &[PlacedNeuron],
    owner_type_mask: u8,
    owner_soma_idx: usize,
    type_affinity: f32,
) -> Vec3 {
    let mut v_attract = Vec3::ZERO;
    let mut total_weight = 0.0;
    let max_radius_sq = max_search_radius_vox * max_search_radius_vox;

    let candidates = spatial_grid.get_in_radius(head_pos, max_search_radius_vox);

    for idx in candidates {
        if idx == owner_soma_idx {
            continue;
        }

        let target = &neurons[idx];
        
        // Type Affinity: модулируем вес притяжения по типу
        let is_same_type = target.type_idx == (owner_type_mask as usize);
        let affinity_multiplier = if is_same_type {
            type_affinity
        } else {
            1.0 - type_affinity
        };

        if affinity_multiplier < 0.01 {
            continue;
        }

        let target_pos = Vec3::new(target.x() as f32, target.y() as f32, target.z() as f32);
        let dir = target_pos - head_pos;
        let dist_sq = dir.length_squared();

        if dist_sq > max_radius_sq || dist_sq < 1e-5 {
            continue;
        }

        let dist = dist_sq.sqrt();
        let dir_norm = dir / dist;

        // Check if target is inside the cone
        if forward_dir.dot(dir_norm) >= fov_cos {
            let weight = affinity_multiplier * ATTRACTION_GRADIENT / (dist_sq + 1e-5);
            v_attract += dir_norm * weight;
            total_weight += weight;
        }
    }

    if total_weight > 0.0 {
        (v_attract / total_weight).normalize_or_zero()
    } else {
        forward_dir
    }
}
