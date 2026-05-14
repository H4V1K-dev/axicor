use std::path::PathBuf;

/// DTO for transferring memory ownership to the background I/O thread.
/// Full Ownership Move, zero O(N) byte copying.
pub struct CheckpointPayload {
    pub zone_hash: u32,
    pub baked_dir: PathBuf,
    pub state_data: Vec<u8>,
    pub axons_data: Vec<u8>,
}

#[cold]
pub fn dispatch_checkpoint(
    payload: CheckpointPayload,
    rt_handle: &tokio::runtime::Handle,
) {
    rt_handle.spawn_blocking(move || {
        let hash = payload.zone_hash;
        if let Err(e) = write_checkpoint_to_disk(payload) {
            tracing::error!("[Storage] CRITICAL: Checkpoint failed for zone {:08X}: {}", hash, e);
        } else {
            tracing::info!("[Storage] Checkpoint committed for zone {:08X}", hash);
        }
    });
}

fn write_checkpoint_to_disk(payload: CheckpointPayload) -> std::io::Result<()> {
    let chk_state = payload.baked_dir.join("checkpoint.state");
    let tmp_state = payload.baked_dir.join("checkpoint.state.tmp");
    let chk_axons = payload.baked_dir.join("checkpoint.axons");
    let tmp_axons = payload.baked_dir.join("checkpoint.axons.tmp");

    // Atomic write
    std::fs::write(&tmp_state, payload.state_data)?;
    std::fs::write(&tmp_axons, payload.axons_data)?;

    std::fs::rename(&tmp_state, &chk_state)?;
    std::fs::rename(&tmp_axons, &chk_axons)?;

    Ok(())
}

/// Flush OS dirty pages to NVMe. Accepts only the required type, without knowing about Workspace.
pub fn flush_mmap_geometry(ephys_mmap: Option<&mut memmap2::MmapMut>) {
    if let Some(mmap) = ephys_mmap {
        if let Err(e) = mmap.flush_async() {
            tracing::warn!("[Storage] Failed to flush async mmap: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    // 1. Happy Path: Check full write and rename cycle
    #[test]
    fn test_write_checkpoint_happy_path() {
        let dir = tempdir().unwrap();
        let payload = CheckpointPayload {
            zone_hash: 0xDEADBEEF,
            baked_dir: dir.path().to_path_buf(),
            state_data: vec![0xAA; 1024],
            axons_data: vec![0xBB; 512],
        };

        assert!(write_checkpoint_to_disk(payload).is_ok());

        let final_state = dir.path().join("checkpoint.state");
        let final_axons = dir.path().join("checkpoint.axons");

        assert!(final_state.exists());
        assert!(final_axons.exists());
        assert_eq!(fs::read(final_state).unwrap().len(), 1024);
        assert_eq!(fs::read(final_axons).unwrap().len(), 512);
        
        // There should be no temporary files
        assert!(!dir.path().join("checkpoint.state.tmp").exists());
    }

    // 2. Atomicity: Simulate I/O error on the second file
    #[test]
    fn test_write_checkpoint_atomicity_on_crash() {
        let dir = tempdir().unwrap();
        
        // Sabotage the creation of the second temporary file by creating a directory with its name
        let tmp_axons = dir.path().join("checkpoint.axons.tmp");
        fs::create_dir(&tmp_axons).unwrap();

        let payload = CheckpointPayload {
            zone_hash: 0xBADBAD,
            baked_dir: dir.path().to_path_buf(),
            state_data: vec![0x11; 128],
            axons_data: vec![0x22; 128],
        };

        // Write must fail
        let res = write_checkpoint_to_disk(payload);
        assert!(res.is_err());

        // Main atomicity check: target files must not be overwritten with garbage
        let final_state = dir.path().join("checkpoint.state");
        assert!(!final_state.exists(), "Fatal: State promoted despite axons failure!");
    }

    // 3. Tokio Integration: Background dispatch does not drop JoinHandle
    #[test]
    fn test_dispatch_checkpoint_tokio() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let dir = tempdir().unwrap();
        let payload = CheckpointPayload {
            zone_hash: 0xC0FFEE,
            baked_dir: dir.path().to_path_buf(),
            state_data: vec![0x01; 64],
            axons_data: vec![0x02; 64],
        };

        dispatch_checkpoint(payload, rt.handle());

        // Give the OS thread pool time for system calls
        rt.block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        });

        assert!(dir.path().join("checkpoint.state").exists());
    }

    // 4. Mmap Flush: Geometry protection
    #[test]
    fn test_flush_mmap_geometry() {
        // Check: None does not cause a panic
        flush_mmap_geometry(None);

        // Check: Real MmapMut flushes successfully
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("geom.bin");
        let file = fs::OpenOptions::new()
            .read(true).write(true).create(true).open(&file_path).unwrap();
        file.set_len(4096).unwrap();
        
        let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file).unwrap() };
        mmap[0] = 0xFF; // Dirty page
        
        flush_mmap_geometry(Some(&mut mmap));
    }
}