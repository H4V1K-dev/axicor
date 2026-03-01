use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Запекает бинарный файл связей (Sender-Side Mapping)
/// src_soma_to_axon - массив локальных аксонов зоны-отправителя
/// dst_ghost_offset - смещение, с которого начинаются Ghost-аксоны в целевой зоне
pub fn bake_ghost_connection(
    out_dir: &Path,
    from_name: &str,
    to_name: &str,
    src_soma_to_axon: &[u32],
    dst_ghost_offset: u32,
) -> u32 {
    let count = src_soma_to_axon.len() as u32;
    let mut src_indices = Vec::with_capacity(count as usize);
    let mut dst_indices = Vec::with_capacity(count as usize);

    // Проецируем 1 к 1 (Все сомы SensoryCortex пускают аксоны в MotorCortex)
    for (i, &src_axon) in src_soma_to_axon.iter().enumerate() {
        src_indices.push(src_axon);
        dst_indices.push(dst_ghost_offset + i as u32);
    }

    let path = out_dir.join(format!("{}_{}.ghosts", from_name, to_name));
    let mut file = BufWriter::new(File::create(path).expect("Fatal: Failed to create .ghosts file"));
    
    // Формат: [u32 count] [u32 array SRC] [u32 array DST]
    file.write_all(&(count as u32).to_le_bytes()).unwrap();
    
    let src_bytes = unsafe { std::slice::from_raw_parts(src_indices.as_ptr() as *const u8, src_indices.len() * 4) };
    let dst_bytes = unsafe { std::slice::from_raw_parts(dst_indices.as_ptr() as *const u8, dst_indices.len() * 4) };
    
    file.write_all(src_bytes).unwrap();
    file.write_all(dst_bytes).unwrap();

    count
}
