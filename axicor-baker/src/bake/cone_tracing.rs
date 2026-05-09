use crate::bake::spatial_grid::SpatialGrid;
use axicor_core::types::PackedPosition;
use glam::Vec3;

pub struct ConeParams {
    pub radius_um: f32,
    pub fov_cos: f32,       // cos(FOV / 2.0). If FOV = 60, then cos(30)  0.866
    pub owner_type: u8,     // [DOD] Raw 4-bit axon owner type
    pub type_affinity: f32, // [DOD] 0.0=attracted to others, 0.5=neutral, 1.0=attracted to own type
}

/// Zero-Cost unpacking from 32-bit to f32 vector (micrometers)
#[inline(always)]
pub fn unpack_to_vec3(pos: PackedPosition, voxel_size_um: f32) -> Vec3 {
    Vec3::new(
        (pos.x() as f32) * voxel_size_um,
        (pos.y() as f32) * voxel_size_um,
        (pos.z() as f32) * voxel_size_um,
    )
}

/// Scans the space ahead of the axon and calculates the attraction gradient (V_attract)
pub fn calculate_v_attract(
    origin_pos: PackedPosition,
    current_dir: Vec3,
    params: &ConeParams,
    grid: &SpatialGrid,
    voxel_size_um: f32,
) -> Vec3 {
    let origin_vec = unpack_to_vec3(origin_pos, voxel_size_um);

    // Convert search radius from m to chunks for SpatialGrid
    let radius_cells = (params.radius_um / (grid.cell_size as f32 * voxel_size_um)).ceil() as i32;

    let mut v_attract = Vec3::ZERO;

    // O(K) Zero-allocation spatial query
    grid.for_each_in_radius(&origin_pos, radius_cells, |dense_id| {
        let neighbor_pos = grid.get_position(dense_id);

        // [DOD FIX] Warp Padding Guard: Do not compute gravity for dummy neurons
        if neighbor_pos.0 == 0 {
            return;
        }

        // Ignore self (coordinate collision)
        if neighbor_pos.0 == origin_pos.0 {
            return;
        }

        let target_vec = unpack_to_vec3(neighbor_pos, voxel_size_um);
        let diff = target_vec - origin_vec;
        let dist_sq = diff.length_squared();

        // Fast sphere culling (Squared  no sqrt!)
        if dist_sq > params.radius_um * params.radius_um || dist_sq == 0.0 {
            return;
        }

        let dist = dist_sq.sqrt();
        let dir_to_target = diff / dist;

        // Cone Frustum Culling
        let dot = current_dir.dot(dir_to_target);
        if dot > params.fov_cos {
            // [DOD] Branchless Type Affinity Math
            // is_same = 1.0 if types match, 0.0 if different
            let is_same = (neighbor_pos.type_id() == params.owner_type) as i32 as f32;

            // When is_same=1.0  use affinity
            // When is_same=0.0  use (1.0 - affinity)
            // 2.0: with affinity=0.5 the multiplier becomes 1.0 for all
            let affinity_mod = (is_same * params.type_affinity
                + (1.0 - is_same) * (1.0 - params.type_affinity))
                * 2.0;

            // [DOD FIX] Chemical Diffusion Gradient (1/r instead of 1/r^2) prevents Singularity Trap
            let weight = (1.0 / (dist + 1.0)) * affinity_mod;
            v_attract += dir_to_target * weight;
        }
    });

    // If cone is empty, vector is zero. Otherwise return normalized attraction.
    if v_attract.length_squared() > 0.0 {
        v_attract.normalize()
    } else {
        Vec3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::spatial_grid::SpatialGrid;

    #[test]
    fn test_cone_tracing_degenerate_v_attract() {
        let grid = SpatialGrid::new(vec![], 10);
        let origin = PackedPosition::pack_raw(10, 10, 10, 0);
        let params = ConeParams {
            radius_um: 100.0,
            fov_cos: 0.0,
            owner_type: 0,
            type_affinity: 0.5,
        };

        let v = calculate_v_attract(origin, Vec3::X, &params, &grid, 1.0);
        assert_eq!(v, Vec3::ZERO, "Degenerate cone must return Vec3::ZERO, not NaN");
    }

    #[test]
    fn test_cone_tracing_fov_culling() {
        let origin = PackedPosition::pack_raw(10, 10, 10, 0);
        // target is at (10, 10, 5), meaning vector from origin to target is (0, 0, -5)
        let target = PackedPosition::pack_raw(10, 10, 5, 0);
        let grid = SpatialGrid::new(vec![origin, target], 10);
        
        let params = ConeParams {
            radius_um: 100.0,
            fov_cos: 0.707, // 90 degrees FOV (45 half angle)
            owner_type: 0,
            type_affinity: 0.5,
        };

        // Looking straight UP (0, 0, 1)
        let current_dir = Vec3::Z;
        let v = calculate_v_attract(origin, current_dir, &params, &grid, 1.0);
        
        // Target is BEHIND the origin. V_attract should be ZERO, completely ignored.
        assert_eq!(v, Vec3::ZERO, "Target behind the FOV cone must be ignored");
    }
}
