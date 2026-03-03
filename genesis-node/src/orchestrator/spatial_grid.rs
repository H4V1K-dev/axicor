// genesis-runtime/src/orchestrator/spatial_grid.rs
use std::collections::HashMap;

/// Size of a hash-grid cell in voxels. 1 voxel = 25 um (typically).
const CELL_SIZE: u32 = 4; // 100 um cells

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GridCell {
    x: i32,
    y: i32,
    z: i32,
}

#[derive(Debug, Clone)]
pub struct SpatialGrid {
    // cell -> list of axon IDs
    cells: HashMap<GridCell, Vec<u32>>,
}

impl SpatialGrid {
    pub fn new() -> Self {
        Self { cells: HashMap::with_capacity(1000) }
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    pub fn insert(&mut self, axon_id: u32, x: u32, y: u32, z: u32) {
        let cx = (x / CELL_SIZE) as i32;
        let cy = (y / CELL_SIZE) as i32;
        let cz = (z / CELL_SIZE) as i32;
        self.cells.entry(GridCell { x: cx, y: cy, z: cz }).or_default().push(axon_id);
    }

    /// Returns a random candidate axon within the neighborhood of the given position.
    pub fn get_random_candidate(&self, x: u32, y: u32, z: u32, seed: u64) -> Option<u32> {
        let cx = (x / CELL_SIZE) as i32;
        let cy = (y / CELL_SIZE) as i32;
        let cz = (z / CELL_SIZE) as i32;

        // Search in a 3x3x3 block
        let mut candidates = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(ids) = self.cells.get(&GridCell { x: cx + dx, y: cy + dy, z: cz + dz }) {
                        candidates.extend_from_slice(ids);
                    }
                }
            }
        }

        if candidates.is_empty() {
            None
        } else {
            let idx = (seed % candidates.len() as u64) as usize;
            Some(candidates[idx])
        }
    }
}
