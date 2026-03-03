// genesis-baker/src/bake/input_map.rs
//
// Фаза A: Input Matrix / Virtual Axons (GXI)
// Спецификация: 08_io_matrix.md §2.1 / 09_baking_pipeline.md §2.1
//
// Контракты:
//   1. Virtual Axon Offset: каждый пиксель → аксон.
//      Аксоны ДОЛЖНЫ быть добавлены в конец axon_heads шарда.
//   2. axon_id в .gxi = base_axon_id + pixel_index.
//   3. Заголовок: GxiHeader (32 байта).

use genesis_core::hash::fnv1a_32;
use genesis_core::constants::GXI_MAGIC;
use std::path::Path;
use std::io::Write;

/// Дескриптор одной матрицы в файле .gxi (16 байт)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GxiMatrixDescriptor {
    pub name_hash: u32,
    pub offset:    u32, // Индекс в Axon Array
    pub width:     u16,
    pub height:    u16,
    pub stride:    u8,
    pub _padding:  [u8; 3],
}

/// Результат запекания одной матрицы входа.
pub struct BakedGxi {
    pub name_hash: u32,
    pub width: u16,
    pub height: u16,
    pub stride: u8,
    /// flat: pixel_index → axon_id (= base_axon_id + pixel_index)
    pub axon_ids: Vec<u32>,
}

/// Генерирует маппинг входной матрицы → Virtual Axons.
pub fn build_gxi_mapping(
    matrix_name: &str,
    _zone_name: &str,
    matrix_width: u32,
    matrix_height: u32,
    base_axon_id: u32,
    stride: u8,
) -> BakedGxi {
    let total_pixels = matrix_width * matrix_height;
    let axon_ids: Vec<u32> = (0..total_pixels).map(|i| base_axon_id + i).collect();
    let name_hash = fnv1a_32(matrix_name.as_bytes());

    BakedGxi { 
        name_hash,
        width: matrix_width as u16,
        height: matrix_height as u16,
        stride,
        axon_ids 
    }
}

/// Zero-copy сериализация в `<out_dir>/shard.gxi`.
pub fn write_gxi_file(out_dir: &Path, matrices: &[BakedGxi]) {
    let path = out_dir.join("shard.gxi");
    let mut file = std::fs::File::create(path).expect("Failed to create .gxi file");

    let total_pixels: u32 = matrices.iter().map(|m| m.axon_ids.len() as u32).sum();
    let num_matrices = matrices.len() as u16;

    // Header (12 bytes)
    file.write_all(&GXI_MAGIC.to_le_bytes()).unwrap(); // Magic
    file.write_all(&[1u8, 0u8]).unwrap();              // Version 1 + Padding
    file.write_all(&num_matrices.to_le_bytes()).unwrap();
    file.write_all(&total_pixels.to_le_bytes()).unwrap();

    // Matrix Descriptors (16 bytes each)
    let mut current_offset = 0;
    for m in matrices {
        let desc = GxiMatrixDescriptor {
            name_hash: m.name_hash,
            offset: current_offset,
            width: m.width,
            height: m.height,
            stride: m.stride,
            _padding: [0; 3],
        };
        unsafe {
            let bytes = std::slice::from_raw_parts(
                (&desc as *const GxiMatrixDescriptor) as *const u8,
                std::mem::size_of::<GxiMatrixDescriptor>()
            );
            file.write_all(bytes).unwrap();
        }
        current_offset += m.axon_ids.len() as u32;
    }

    // Axon Array (u32 per pixel)
    for m in matrices {
        let payload_bytes = unsafe {
            std::slice::from_raw_parts(
                m.axon_ids.as_ptr() as *const u8,
                m.axon_ids.len() * std::mem::size_of::<u32>(),
            )
        };
        file.write_all(payload_bytes).expect("Failed to write axon IDs");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gxi_pixel_count() {
        let gxi = build_gxi_mapping("sens", "zone_a", 4, 4, 0, 1);
        assert_eq!(gxi.axon_ids.len(), 16);
        assert_eq!(gxi.width, 4);
        assert_eq!(gxi.height, 4);
    }

    #[test]
    fn test_gxi_axon_offset() {
        // base_axon_id = 500 → pixels 0..15 → axons 500..515
        let gxi = build_gxi_mapping("sens", "zone", 4, 4, 500, 1);
        assert_eq!(gxi.axon_ids[0], 500);
        assert_eq!(gxi.axon_ids[15], 515);
    }
}
