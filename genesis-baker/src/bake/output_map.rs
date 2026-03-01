// genesis-baker/src/bake/output_map.rs
use genesis_core::constants::GXO_MAGIC;
use genesis_core::types::PackedPosition;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// DTO для генерации матрицы
pub struct OutputMatrixDef {
    pub name_hash: u32,
    pub width: u16,
    pub height: u16,
    pub stride: u8,
}

/// Запекает .gxo файл используя Z-Sort алгоритм для выбора сом-кандидатов.
pub fn bake_outputs(
    out_dir: &Path,
    matrices: &[OutputMatrixDef],
    zone_width_um: f32,
    zone_depth_um: f32,
    neurons_packed_pos: &[u32], // Массив PackedPosition, где индекс = Dense_ID сомы
) {
    let mut total_pixels = 0;
    for m in matrices {
        total_pixels += (m.width as u32) * (m.height as u32);
    }

    let mut payload_soma_ids = vec![0u32; total_pixels as usize];
    let mut current_offset = 0;

    for matrix in matrices {
        let pixels = (matrix.width as u32) * (matrix.height as u32);
        
        // Z-Sort: Для каждого пикселя ищем сому с минимальным Z
        for py in 0..matrix.height {
            for px in 0..matrix.width {
                // Вычисляем физические границы пикселя (региона) в микрометрах (без умножения)
                let x_min = (px as f32 / matrix.width as f32) * zone_width_um;
                let x_max = ((px + 1) as f32 / matrix.width as f32) * zone_width_um;
                let y_min = (py as f32 / matrix.height as f32) * zone_depth_um;
                let y_max = ((py + 1) as f32 / matrix.height as f32) * zone_depth_um;

                let mut best_soma_id = u32::MAX;
                let mut min_z = u32::MAX;

                for (dense_id, &packed) in neurons_packed_pos.iter().enumerate() {
                    // Распаковка 10-бит X, 10-бит Y, 8-бит Z
                    let vx = (packed & 0x3FF) as f32; 
                    let vy = ((packed >> 10) & 0x3FF) as f32; 
                    let vz = (packed >> 20) & 0xFF;

                    if vx >= x_min && vx < x_max && vy >= y_min && vy < y_max {
                        if vz < min_z {
                            min_z = vz;
                            best_soma_id = dense_id as u32;
                        }
                    }
                }

                if best_soma_id == u32::MAX {
                    eprintln!("Warning: Z-Sort failed. Empty pixel at ({}, {}) in output matrix hash {}. Density too low!", px, py, matrix.name_hash);
                    // Fallback to soma 0 so it doesn't crash if sparsely generated
                    best_soma_id = 0; 
                }

                let pixel_idx = (py as u32 * matrix.width as u32) + px as u32;
                payload_soma_ids[(current_offset + pixel_idx) as usize] = best_soma_id;
            }
        }
        current_offset += pixels;
    }

    write_gxo_binary(out_dir, matrices, total_pixels, &payload_soma_ids);
}

/// Хардкорная запись байт-в-байт без serde
fn write_gxo_binary(
    out_dir: &Path, 
    matrices: &[OutputMatrixDef], 
    total_pixels: u32, 
    payload: &[u32]
) {
    let path = out_dir.join("shard.gxo");
    let mut file = BufWriter::new(File::create(path).expect("Failed to create .gxo file"));

    // Header (12 bytes)
    file.write_all(&GXO_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&[1u8, 0u8]).unwrap(); // Version 1, Padding 1
    file.write_all(&(matrices.len() as u16).to_le_bytes()).unwrap();
    file.write_all(&total_pixels.to_le_bytes()).unwrap();

    // Descriptors (16 bytes each)
    let mut current_offset: u32 = 0;
    for m in matrices {
        file.write_all(&m.name_hash.to_le_bytes()).unwrap();
        file.write_all(&current_offset.to_le_bytes()).unwrap();
        file.write_all(&m.width.to_le_bytes()).unwrap();
        file.write_all(&m.height.to_le_bytes()).unwrap();
        file.write_all(&[m.stride, 0, 0, 0]).unwrap(); // Stride + 3 bytes padding
        
        current_offset += (m.width as u32) * (m.height as u32);
    }

    // Payload
    let payload_bytes = unsafe {
        std::slice::from_raw_parts(
            payload.as_ptr() as *const u8,
            payload.len() * 4
        )
    };
    file.write_all(payload_bytes).unwrap();
}
