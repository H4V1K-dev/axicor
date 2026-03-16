// genesis-baker/src/bake/output_map.rs
//
// Фаза B: Readout Interface (GXO)
// Спецификация: 08_io_matrix.md §3.1 / 09_baking_pipeline.md §2.2
//
// Контракты:
//   1. Empty Pixel Trap: пустой столбец → EMPTY_PIXEL (0xFFFF_FFFF).
//   2. Z-Sort: при наличии нескольких сом в пикселе выбирается та, у которой Z минимален.
//   3. output_count в заголовке = количество УСПЕШНО привязанных сом (без сентинелей).

use genesis_core::hash::fnv1a_32;
use genesis_core::ipc::EMPTY_PIXEL;
use genesis_core::constants::GXO_MAGIC;
use std::path::Path;

/// Дескриптор одной матрицы в файле .gxo (16 байт)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GxoMatrixDescriptor {
    pub name_hash: u32,
    pub offset:    u32, // Индекс в Soma Array
    pub width:     u16,
    pub height:    u16,
    pub stride:    u8,
    pub _padding:  [u8; 3],
}

/// Результат запекания одной матрицы выхода.
#[derive(Clone)]
pub struct BakedGxo {
    pub name_hash: u32,
    pub width: u16,
    pub height: u16,
    pub stride: u8,
    /// Flat array размером `total_pixels`.
    /// EMPTY_PIXEL (0xFFFF_FFFF) означает «нет сомы в этом пикселе».
    pub mapped_soma_ids: Vec<u32>,
}

/// Основная функция маппинга выходов на слой нейронов (Z-Sort + Sentinel).
pub fn build_gxo_mapping(
    matrix_name: &str,
    _zone_name: &str,
    matrix_width: u32,
    matrix_height: u32,
    zone_width_vox: u32,
    zone_depth_vox: u32,
    neurons_packed_pos: &[u32],
    stride: u8,
    target_type_id: Option<u8>,
) -> BakedGxo {
    let total_pixels = (matrix_width * matrix_height) as usize;
    let mut mapped_soma_ids = vec![EMPTY_PIXEL; total_pixels];
    let mut min_z_per_pixel = vec![u32::MAX; total_pixels];

    for (dense_id, &packed) in neurons_packed_pos.iter().enumerate() {
        if packed == 0 { continue; }
        
        let p_struct = genesis_core::types::PackedPosition(packed);
        let type_idx = p_struct.type_id();
        
        if let Some(target) = target_type_id {
            if type_idx != target { continue; }
        }

        let vx = p_struct.x();
        let vy = p_struct.y();
        let vz = p_struct.z();

        let px = ((vx as u64 * matrix_width as u64) / zone_width_vox.max(1) as u64)
            .min(matrix_width as u64 - 1) as u32;
        let py = ((vy as u64 * matrix_height as u64) / zone_depth_vox.max(1) as u64)
            .min(matrix_height as u64 - 1) as u32;

        let pixel_idx = (py * matrix_width + px) as usize;

        if (vz as u32) < min_z_per_pixel[pixel_idx] {
            min_z_per_pixel[pixel_idx] = vz as u32;
            mapped_soma_ids[pixel_idx] = dense_id as u32;
        }
    }

    let name_hash = fnv1a_32(matrix_name.as_bytes());

    BakedGxo { 
        name_hash,
        width: matrix_width as u16,
        height: matrix_height as u16,
        stride,
        mapped_soma_ids 
    }
}

/// Сериализует `BakedGxo` в файл `<out_dir>/shard.gxo` (zero-copy).
pub fn write_gxo_file(out_dir: &Path, matrices: &[BakedGxo]) {
    use std::io::Write;
    let path = out_dir.join("shard.gxo");
    let mut file = std::fs::File::create(path).expect("Failed to create .gxo file");

    let total_pixels: u32 = matrices.iter().map(|m| m.mapped_soma_ids.len() as u32).sum();
    let num_matrices = matrices.len() as u16;

    // Header (12 bytes)
    file.write_all(&GXO_MAGIC.to_le_bytes()).unwrap(); // Magic
    file.write_all(&[1u8, 0u8]).unwrap();              // Version 1 + Padding
    file.write_all(&num_matrices.to_le_bytes()).unwrap();
    file.write_all(&total_pixels.to_le_bytes()).unwrap();

    // Matrix Descriptors (16 bytes each)
    let mut current_offset = 0;
    for m in matrices {
        let desc = GxoMatrixDescriptor {
            name_hash: m.name_hash,
            offset: current_offset,
            width: m.width,
            height: m.height,
            stride: m.stride,
            _padding: [0; 3],
        };
        unsafe {
            let bytes = std::slice::from_raw_parts(
                (&desc as *const GxoMatrixDescriptor) as *const u8,
                std::mem::size_of::<GxoMatrixDescriptor>()
            );
            file.write_all(bytes).unwrap();
        }
        current_offset += m.mapped_soma_ids.len() as u32;
    }

    // Soma Array (u32 per pixel) 
    for m in matrices {
        let payload_bytes = unsafe {
            std::slice::from_raw_parts(
                m.mapped_soma_ids.as_ptr() as *const u8,
                m.mapped_soma_ids.len() * std::mem::size_of::<u32>(),
            )
        };
        file.write_all(payload_bytes).expect("Failed to write soma IDs");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::coords::pack_position;
    use genesis_core::ipc::EMPTY_PIXEL;

    fn pack(x: u32, y: u32, z: u32) -> u32 {
        pack_position(x, y, z, 0).0
    }

    #[test]
    fn test_gxo_z_sort() {
        // Three neurons in the SAME pixel (world 10×10 vox, matrix 1×1)
        // Z values: 10, 50, 5 — algorithm MUST pick Dense ID 2 (Z=5)
        let neurons = vec![
            pack(5, 5, 10),  // Dense ID 0
            pack(5, 5, 50),  // Dense ID 1
            pack(5, 5,  5),  // Dense ID 2  ← winner
        ];

        let gxo = build_gxo_mapping("out", "zone", 1, 1, 10, 10, &neurons, 1, None);

        assert_eq!(gxo.mapped_soma_ids[0], 2, "Z-Sort must pick Dense ID 2 (Z=5)");
    }

    #[test]
    fn test_gxo_empty_pixel() {
        // 2×2 matrix, only top-left cell has a neuron
        // world: 20 vox wide × 20 vox deep, matrix 2×2 → each pixel is 10×10 vox
        let neurons = vec![
            pack(2, 2, 0),  // Pixel (0,0) — Dense ID 0
        ];

        let gxo = build_gxo_mapping("out", "zone", 2, 2, 20, 20, &neurons, 1, None);

        assert_eq!(gxo.mapped_soma_ids[0], 0,          "pixel 0 should map to soma 0");
        assert_eq!(gxo.mapped_soma_ids[1], EMPTY_PIXEL, "pixel 1 (top-right) must be sentinel");
        assert_eq!(gxo.mapped_soma_ids[2], EMPTY_PIXEL, "pixel 2 (bottom-left) must be sentinel");
        assert_eq!(gxo.mapped_soma_ids[3], EMPTY_PIXEL, "pixel 3 (bottom-right) must be sentinel");
    }

    #[test]
    fn test_gxo_no_neurons() {
        let gxo = build_gxo_mapping("out", "zone", 2, 2, 20, 20, &[], 1, None);
        assert!(gxo.mapped_soma_ids.iter().all(|&id| id == EMPTY_PIXEL));
    }
}
