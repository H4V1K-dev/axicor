use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub fn pack_directory_to_axic(project_dir: &Path, out_file: &Path) -> anyhow::Result<()> {
    let mut files = Vec::new();
    collect_files_recursive(project_dir, project_dir, &mut files);

    let mut file = File::create(out_file)?;

    // 1. Header (12 bytes)
    file.write_all(b"AXIC")?;
    file.write_all(&1u32.to_le_bytes())?; // Version 1
    file.write_all(&(files.len() as u32).to_le_bytes())?; // File Count

    let toc_start = file.stream_position()?;

    // 2. Dummy TOC reservation (272 bytes per file: 256 path + 8 offset + 8 size)
    let dummy_toc = vec![0u8; files.len() * 272];
    file.write_all(&dummy_toc)?;

    let mut toc_data = Vec::with_capacity(files.len() * 272);

    // 3. Write Payloads with Strict 4096-byte Page Alignment
    for (rel_path, abs_path) in files {
        let current_pos = file.stream_position()?;

        // [DOD FIX] Align offset to 4096 boundary (OS Page Size).
        // This is vital for Zero-Copy mmap of a specific file from the archive!
        let padding = (4096 - (current_pos % 4096)) % 4096;
        if padding > 0 {
            static ZEROS: [u8; 4096] = [0; 4096];
            file.write_all(&ZEROS[..padding as usize])?;
        }

        let aligned_offset = file.stream_position()?;
        let mut f_in = File::open(&abs_path)?;
        let size = std::io::copy(&mut f_in, &mut file)?;

        // Build TOC Entry
        let mut path_buf = [0u8; 256];
        let bytes = rel_path.as_bytes();
        let len = bytes.len().min(256);
        path_buf[..len].copy_from_slice(&bytes[..len]);

        toc_data.extend_from_slice(&path_buf);
        toc_data.extend_from_slice(&aligned_offset.to_le_bytes());
        toc_data.extend_from_slice(&size.to_le_bytes());
    }

    // 4. Flush TOC
    file.seek(SeekFrom::Start(toc_start))?;
    file.write_all(&toc_data)?;

    Ok(())
}

fn collect_files_recursive(dir: &Path, base: &Path, files: &mut Vec<(String, PathBuf)>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_files_recursive(&path, base, files);
            } else {
                let rel = path
                    .strip_prefix(base)
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                // Unify slashes for Windows/Linux compatibility
                files.push((rel.replace("\\", "/"), path));
            }
        }
    }
}
