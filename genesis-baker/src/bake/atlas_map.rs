use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// FNV-1a для детерминированного шума
fn hash_jitter(seed: u64, salt: u32) -> f32 {
    let mut hash = 0x811c9dc5_u32;
    for &b in &seed.to_le_bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash ^= salt;
    hash = hash.wrapping_mul(0x01000193);
    
    // Нормализация в диапазон [-1.0 .. 1.0]
    ((hash % 2000) as f32 / 1000.0) - 1.0
}

/// Генерирует .ghosts файл на основе UV-проекции
pub fn bake_atlas_connection(
    out_dir: &Path,
    from_name: &str,
    to_name: &str,
    src_packed_pos: &[u32],
    src_size_um: (f32, f32), // (width, depth)
    conn_grid: (u16, u16),   // Разрешение проекции (width, height)
    dst_ghost_offset: u32,
    master_seed: u64,
) -> u32 {
    let (grid_w, grid_h) = conn_grid;
    let count = (grid_w as u32) * (grid_h as u32);
    
    let mut src_indices = Vec::with_capacity(count as usize);
    let mut dst_indices = Vec::with_capacity(count as usize);

    for py in 0..grid_h {
        for px in 0..grid_w {
            // 1. UV нормализация (0.0 .. 1.0)
            let u = (px as f32) / (grid_w as f32);
            let v = (py as f32) / (grid_h as f32);

            // 2. Детерминированный Jitter (шум до 5% от размера сетки)
            let salt = (py as u32) << 16 | (px as u32);
            let jitter_u = hash_jitter(master_seed, salt) * 0.05;
            let jitter_v = hash_jitter(master_seed, salt.wrapping_mul(2)) * 0.05;

            let u_noisy = (u + jitter_u).clamp(0.0, 1.0);
            let v_noisy = (v + jitter_v).clamp(0.0, 1.0);

            // 3. Целевая физическая точка в зоне-источнике
            let target_x = u_noisy * src_size_um.0;
            let target_y = v_noisy * src_size_um.1;

            // 4. Z-Sort: Ищем ближайшую сому-отправителя
            let mut best_soma_id = u32::MAX;
            let mut min_dist_sq = f32::MAX;

            for (dense_id, &packed) in src_packed_pos.iter().enumerate() {
                // Распаковка 10-бит X, 10-бит Y
                let vx = (packed & 0x3FF) as f32; 
                let vy = ((packed >> 10) & 0x3FF) as f32; 
                
                let dx = vx - target_x;
                let dy = vy - target_y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                    best_soma_id = dense_id as u32;
                }
            }

            assert!(best_soma_id != u32::MAX, "Fatal: Topology is completely empty in {}", from_name);

            src_indices.push(best_soma_id);
            dst_indices.push(dst_ghost_offset + (py as u32 * grid_w as u32) + px as u32);
        }
    }

    // 5. Запись бинарного контракта (Zero-Copy Ready)
    let path = out_dir.join(format!("{}_{}.ghosts", from_name, to_name));
    let mut file = BufWriter::new(File::create(path).expect("Failed to create .ghosts file"));
    
    file.write_all(&(count as u32).to_le_bytes()).unwrap();
    
    let src_bytes = unsafe { std::slice::from_raw_parts(src_indices.as_ptr() as *const u8, src_indices.len() * 4) };
    let dst_bytes = unsafe { std::slice::from_raw_parts(dst_indices.as_ptr() as *const u8, dst_indices.len() * 4) };
    
    file.write_all(src_bytes).unwrap();
    file.write_all(dst_bytes).unwrap();

    count
}
