use genesis_core::config::blueprints::GenesisConstantMemory;

/// Нормализованный «вес» сомы = Σ|dendrite_weights| / (128 × 32767).
/// При первом Baking все веса = 0 → power_index = 0.0 (новые нейроны равны).
/// (04_connectivity.md §1.6.1)
pub fn compute_power_index(soma_id: usize, weights: &[i16], padded_n: usize) -> f32 {
    let mut power = 0u32;
    for slot in 0..128 {
        let w = weights[slot * padded_n + soma_id];
        power += w.unsigned_abs() as u32;
    }
    power as f32 / (128.0 * 32767.0)
}

/// Вычисляет привлекательность сомы-кандидата для растущего аксона.
/// Вся математика здесь легально использует f32, так как это Night Phase.
#[inline]
pub fn compute_sprouting_score(
    const_mem: &GenesisConstantMemory,
    target_type_idx: u8,
    distance: f32,
    power_index: f32,
    noise: f32,
) -> f32 {
    // Прямое чтение параметров нужного варианта за O(1)
    let _variant = &const_mem.variants[target_type_idx as usize];
    
    // Пример скоринга, основанного на параметрах конкретного типа.
    // Если для этого типа прописано мощное влияние дистанции:
    let dist_weight = 1.0; // В будущем можно вынести в параметры VariantParameters
    let power_weight = 0.5;
    
    let score = (1.0 / (distance + 1.0)) * dist_weight 
              + power_index * power_weight 
              + noise;
              
    score
}

/// Евклидово расстояние в вокселях между двумя точками.
pub fn voxel_dist(ax: u32, ay: u32, az: u32, bx: u32, by: u32, bz: u32) -> f32 {
    let dx = ax as f32 - bx as f32;
    let dy = ay as f32 - by as f32;
    let dz = az as f32 - bz as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

