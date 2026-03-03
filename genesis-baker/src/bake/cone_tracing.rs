use glam::Vec3;
use genesis_core::types::PackedPosition;
use crate::bake::spatial_grid::SpatialGrid;

pub struct ConeParams {
    pub radius_um: f32,
    pub fov_cos: f32, // cos(FOV / 2.0). Если FOV = 60°, то cos(30°) ≈ 0.866
    pub target_type: Option<u8>,
}

/// Zero-Cost распаковка из 32 бит в f32 вектор (микрометры)
#[inline(always)]
pub fn unpack_to_vec3(pos: PackedPosition, voxel_size_um: f32) -> Vec3 {
    Vec3::new(
        (pos.x() as f32) * voxel_size_um,
        (pos.y() as f32) * voxel_size_um,
        (pos.z() as f32) * voxel_size_um,
    )
}

/// Сканирует пространство перед аксоном и вычисляет градиент притяжения (V_attract)
pub fn calculate_v_attract(
    origin_pos: PackedPosition,
    current_dir: Vec3,
    params: &ConeParams,
    grid: &SpatialGrid,
    voxel_size_um: f32,
) -> Vec3 {
    let origin_vec = unpack_to_vec3(origin_pos, voxel_size_um);
    
    // Переводим радиус поиска из мкм в чанки для SpatialGrid
    let radius_cells = (params.radius_um / (grid.cell_size as f32 * voxel_size_um)).ceil() as i32;

    // O(1) извлечение соседей из хэш-таблицы
    let neighbors = grid.query_radius(&origin_pos, radius_cells);
    let mut v_attract = Vec3::ZERO;

    for dense_id in neighbors {
        let neighbor_pos = grid.get_position(dense_id);
        
        // 1. Быстрый аппаратный фильтр по маске типа (0 бит float-математики)
        if let Some(t) = params.target_type {
            if neighbor_pos.type_id() != t { continue; }
        }

        // 2. Игнорируем себя (коллизия координат)
        if neighbor_pos.0 == origin_pos.0 { continue; }

        let target_vec = unpack_to_vec3(neighbor_pos, voxel_size_um);
        let diff = target_vec - origin_vec;
        let dist_sq = diff.length_squared();

        // 3. Быстрое отсечение по сфере (через квадрат, никаких sqrt!)
        if dist_sq > params.radius_um * params.radius_um || dist_sq == 0.0 {
            continue;
        }

        let dist = dist_sq.sqrt();
        let dir_to_target = diff / dist;

        // 4. Отсечение по Конусу (Cone Frustum Culling)
        // Dot Product = 1 такт ALU. Если dot > cos(FOV), мы смотрим на цель.
        let dot = current_dir.dot(dir_to_target);
        if dot > params.fov_cos {
            // 5. Взвешивание (Weighting). Inverse Square Law: чем ближе цель, тем сильнее тянет.
            let weight = 1.0 / (dist_sq + 1.0); // +1.0 защита от NaN
            v_attract += dir_to_target * weight;
        }
    }

    // Если в конусе пусто, вектор нулевой. Иначе возвращаем нормализованную тягу.
    if v_attract.length_squared() > 0.0 {
        v_attract.normalize()
    } else {
        Vec3::ZERO
    }
}
