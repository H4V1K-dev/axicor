// genesis-baker/src/bake/cone_tracing.rs

use crate::bake::neuron_placement::PlacedNeuron;
use crate::bake::spatial_grid::SpatialGrid;
use genesis_core::types::PackedPosition;
use genesis_core::coords::pack_position;
use glam::Vec3;

/// Квантование f32 позиции в PackedPosition [Type(4)|Z(8)|Y(10)|X(10)].
/// Защита от дрифта обеспечивается тем, что f32 позиция прокидывается снаружи.
#[inline(always)]
pub fn step_and_pack(
    current_pos: Vec3,
    direction: Vec3,
    step_um: f32,
    owner_type_id: u8
) -> PackedPosition {
    let next = current_pos + direction * step_um;
    
    // Квантование: round() предпочтительнее floor() для минимизации ошибки.
    pack_position(
        next.x.round() as u32,
        next.y.round() as u32,
        next.z.round() as u32,
        (owner_type_id & 0xF) as u32
    )
}

/// Проверка попадания в конус без тригонометрии (Spec 04 §4.1).
/// fov_cos = cos(FOV / 2). Вычисляется 1 раз перед циклом.
#[inline(always)]
pub fn is_in_cone(
    head_pos: Vec3,
    growth_dir: Vec3, // MB: Обязан быть нормализован!
    target_pos: Vec3,
    lookahead_sq: f32,
    fov_cos: f32,
) -> bool {
    let dir = target_pos - head_pos;
    let dist_sq = dir.length_squared();
    
    if dist_sq > lookahead_sq || dist_sq < 1e-5 { return false; }
    
    let dist = dist_sq.sqrt();
    let dot = dir.dot(growth_dir) / dist;
    
    dot >= fov_cos
}

// Постоянная притяжения (Spec 04 §4.2)
const ATTRACTION_GRADIENT: f32 = 1.0;

/// Рассчитывает вектор V_attract как средневзвешенное направление на кандидатов.
pub fn calculate_v_attract(
    head_pos: Vec3,
    forward_dir: Vec3,
    fov_cos: f32,
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

    // SpatialGrid возвращает Dense ID кандидатов за O(1)
    let candidates = spatial_grid.get_in_radius(head_pos, max_search_radius_vox);

    for idx in candidates {
        if idx == owner_soma_idx { continue; }

        let target = &neurons[idx];
        
        // Модуляция веса по типу нейрона
        let is_same_type = target.type_idx == (owner_type_mask as usize);
        let affinity_multiplier = if is_same_type { type_affinity } else { 1.0 - type_affinity };

        if affinity_multiplier < 0.01 { continue; }

        let target_pos = Vec3::new(target.x() as f32, target.y() as f32, target.z() as f32);
        
        if is_in_cone(head_pos, forward_dir, target_pos, max_radius_sq, fov_cos) {
            let dir = target_pos - head_pos;
            let dist_sq = dir.length_squared();
            let weight = affinity_multiplier * ATTRACTION_GRADIENT / (dist_sq + 1e-5);
            
            v_attract += dir.normalize_or_zero() * weight;
            total_weight += weight;
        }
    }

    if total_weight > 0.0 {
        (v_attract / total_weight).normalize_or_zero()
    } else {
        forward_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_is_in_cone_precision() {
        let head = Vec3::ZERO;
        let dir = Vec3::Z; // Looking straight up
        let fov_cos = 45.0f32.to_radians().cos(); // 90 deg FOV
        
        // 1. По центру
        assert!(is_in_cone(head, dir, Vec3::new(0.0, 0.0, 10.0), 400.0, fov_cos));
        
        // 2. На границе (45 градусов)
        // x=10, z=10 -> tan(angle) = 1 -> angle = 45 deg
        assert!(is_in_cone(head, dir, Vec3::new(10.0, 0.0, 10.0), 400.0, fov_cos));
        
        // 3. Снаружи (46 градусов)
        assert!(!is_in_cone(head, dir, Vec3::new(11.0, 0.0, 10.0), 400.0, fov_cos));
        
        // 4. Сзади
        assert!(!is_in_cone(head, dir, Vec3::new(0.0, 0.0, -10.0), 400.0, fov_cos));

        // 5. Слишком далеко (radius check)
        assert!(!is_in_cone(head, dir, Vec3::new(0.0, 0.0, 21.0), 400.0, fov_cos));
    }

    #[test]
    fn test_step_and_pack_quantization() {
        let pos = Vec3::new(10.4, 20.6, 5.5);
        let dir = Vec3::X;
        let step = 1.0;
        let packed = step_and_pack(pos, dir, step, 1);
        
        let (x, y, z, t) = genesis_core::coords::unpack_position(packed);
        // 10.4 + 1.0 = 11.4 -> round(11.4) = 11
        assert_eq!(x, 11);
        assert_eq!(y, 21);
        assert_eq!(z, 6);
        assert_eq!(t, 1);
    }
}
