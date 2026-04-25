use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// FNV-1a for deterministic jitter
fn hash_jitter(seed: u64, salt: u32) -> f32 {
    let mut hash = 0x811c9dc5_u32;
    for &b in &seed.to_le_bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash ^= salt;
    hash = hash.wrapping_mul(0x01000193);

    // Normalization to [-1.0 .. 1.0] range
    ((hash % 2000) as f32 / 1000.0) - 1.0
}

#[allow(clippy::too_many_arguments)]
pub fn bake_atlas_connection(
    out_dir: &Path,
    from_name: &str,
    to_name: &str,
    src_packed_pos: &[u32],
    src_size_um: (f32, f32), // (width, depth)
    conn_grid: (u16, u16),   // Projection resolution (width, height)
    dst_ghost_offset: u32,
    master_seed: u64,
) -> u32 {
    let (grid_w, grid_h) = conn_grid;
    let count = (grid_w as u32) * (grid_h as u32);

    let mut src_indices = Vec::with_capacity(count as usize);
    let mut dst_indices = Vec::with_capacity(count as usize);

    for py in 0..grid_h {
        for px in 0..grid_w {
            // 1. UV normalization (0.0 .. 1.0)
            let u = (px as f32) / (grid_w as f32);
            let v = (py as f32) / (grid_h as f32);

            // 2. Deterministic Jitter (noise up to 5% of grid size)
            let salt = (py as u32) << 16 | (px as u32);
            let jitter_u = hash_jitter(master_seed, salt) * 0.05;
            let jitter_v = hash_jitter(master_seed, salt.wrapping_mul(2)) * 0.05;

            let u_noisy = (u + jitter_u).clamp(0.0, 1.0);
            let v_noisy = (v + jitter_v).clamp(0.0, 1.0);

            // 3. Target physical point in source zone
            let target_x = u_noisy * src_size_um.0;
            let target_y = v_noisy * src_size_um.1;

            // 4. Z-Sort: find nearest source soma
            let mut best_soma_id = u32::MAX;
            let mut min_dist_sq = f32::MAX;

            for (dense_id, &packed) in src_packed_pos.iter().enumerate() {
                // [DOD FIX] Recalculation: Voxel Coords (0..1024) -> Microns (0..25600)
                let vx_um = (packed & 0x3FF) as f32 * 25.0;
                let vy_um = ((packed >> 10) & 0x3FF) as f32 * 25.0;

                let dx = vx_um - target_x;
                let dy = vy_um - target_y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                    best_soma_id = dense_id as u32;
                }
            }

            assert!(
                best_soma_id != u32::MAX,
                "Fatal: Topology is completely empty in {}",
                from_name
            );

            src_indices.push(best_soma_id);
            dst_indices.push(dst_ghost_offset + (py as u32 * grid_w as u32) + px as u32);
        }
    }

    // 5. Binary contract write (Zero-Copy Ready)
    let path = out_dir.join(format!("{}_{}.ghosts", from_name, to_name));
    let mut file = BufWriter::new(File::create(path).expect("Failed to create .ghosts file"));

    // [DOD FIX] Using new C-ABI structures from ipc.rs
    use axicor_core::hash::fnv1a_32;
    use axicor_core::ipc::{GhostConnection, GhostsHeader};

    let header = GhostsHeader::new(
        fnv1a_32(from_name.as_bytes()),
        fnv1a_32(to_name.as_bytes()),
        count,
    );
    file.write_all(header.as_bytes()).unwrap();

    let mut connections = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        connections.push(GhostConnection {
            src_soma_id: src_indices[i],
            target_ghost_id: dst_indices[i],
        });
    }

    file.write_all(GhostConnection::slice_as_bytes(&connections))
        .unwrap();

    count
}
