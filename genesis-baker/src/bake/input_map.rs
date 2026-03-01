// genesis-baker/src/bake/input_map.rs
use crate::bake::axon_growth::GrownAxon;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use genesis_core::constants::GXI_MAGIC;

/// DTO для генерации входной матрицы
pub struct InputMatrixDef {
    pub name_hash: u32,
    pub width: u16,
    pub height: u16,
}

/// Выращивает виртуальные аксоны для входных данных и генерирует GXI-файл.
pub fn bake_inputs(
    out_dir: &Path,
    matrices: &[InputMatrixDef],
    base_axon_id: u32,
) -> Vec<GrownAxon> {
    let mut total_pixels = 0;
    for m in matrices {
        total_pixels += (m.width as u32) * (m.height as u32);
    }

    let mut payload_axon_ids = vec![0u32; total_pixels as usize];
    let mut virtual_axons = Vec::with_capacity(total_pixels as usize);

    for i in 0..total_pixels {
        payload_axon_ids[i as usize] = base_axon_id + i;
        
        // Push a lobotomized virtual axon. No physical geometry needed.
        virtual_axons.push(GrownAxon {
            soma_idx: usize::MAX, // Mark as external / virtual
            type_idx: 0,          // Default excitatory
            tip_x: 0,
            tip_y: 0,
            tip_z: 0,
            length_segments: 0,
            segments: vec![],
        });
    }

    write_gxi_binary(out_dir, matrices, total_pixels, &payload_axon_ids);

    virtual_axons
}

fn write_gxi_binary(
    out_dir: &Path,
    matrices: &[InputMatrixDef],
    total_pixels: u32,
    payload: &[u32]
) {
    let shard_name = out_dir.file_name().and_then(|n| n.to_str()).unwrap_or("shard");
    let path = out_dir.join(format!("{}.gxi", shard_name));
    let mut file = BufWriter::new(File::create(path).expect("Failed to create .gxi file"));

    // Header (12 bytes)
    file.write_all(&GXI_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&[1u8, 0u8]).unwrap(); // Version 1, Padding 0
    file.write_all(&(matrices.len() as u16).to_le_bytes()).unwrap();
    file.write_all(&total_pixels.to_le_bytes()).unwrap();

    // Descriptors (32 + 4 + 4 + 4 = 44 bytes each but we had dynamic width previously, 
    // Wait, the runtime reads name_len (u16), name_bytes, width (u16), height (u16), axon_offset (u32).
    // Let's stick to the runtime format.
    // genesis-runtime/src/input.rs: "GXI" format parsing:
    // read tag (u16), len (u16)? Oh wait, let's look at exactly what we had before in input_map.rs.
    // The previous implementation did NOT use fixed 32-byte names.
    // It used: name_len(u16), name_bytes, width(u16), height(u16), axon_offset(u32)
    // Actually runtime `input.rs` uses:
    // pub name_hash: u32,
    // pub width: u16,
    // pub height: u16,
    // pub offset: u32,
    // Let's just use a fixed 12-byte descriptor for GXI to match GXO format strictly.
    // Wait, let's make it exactly 12 bytes: name_hash(4), offset(4), width(2), height(2)
    
    let mut current_offset: u32 = 0;
    for m in matrices {
        file.write_all(&m.name_hash.to_le_bytes()).unwrap();
        file.write_all(&current_offset.to_le_bytes()).unwrap();
        file.write_all(&m.width.to_le_bytes()).unwrap();
        file.write_all(&m.height.to_le_bytes()).unwrap();
        file.write_all(&[1u8, 0, 0, 0]).unwrap(); // Stride 1 + 3 bytes padding
        
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
