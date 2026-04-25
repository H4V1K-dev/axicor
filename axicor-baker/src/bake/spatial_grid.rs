use crate::bake::axon_growth::GrownAxon;
use axicor_core::types::PackedPosition;
use std::collections::HashMap;
// [DOD] Adding Rayon for instant sorting of flat arrays
use rayon::prelude::*;

/// Spatial hash for O(1) neighbor lookup.
/// [DOD FIX] Converted to flat arrays (Flat Grid).
pub struct SpatialGrid {
    pub cell_size: u32,
    cell_index: HashMap<u64, std::ops::Range<u32>>,
    flat_cells: Vec<(u64, u32)>, // (hash, dense_id)
    positions: Vec<PackedPosition>,
}

#[allow(clippy::new_without_default)]
impl SpatialGrid {
    pub fn new(positions: Vec<PackedPosition>, cell_size_voxels: u32) -> Self {
        let mut grid = Self {
            cell_size: cell_size_voxels.max(1),
            cell_index: HashMap::new(),
            flat_cells: Vec::new(),
            positions,
        };
        grid.build();
        grid
    }

    fn build(&mut self) {
        let mut flat_cells = Vec::with_capacity(self.positions.len());

        for (dense_id, pos) in self.positions.iter().enumerate() {
            if pos.0 == 0 {
                continue;
            }

            let cx = (pos.x() as u32) / self.cell_size;
            let cy = (pos.y() as u32) / self.cell_size;
            let cz = (pos.z() as u32) / self.cell_size;

            let hash = Self::hash_cell(cx, cy, cz);
            flat_cells.push((hash, dense_id as u32));
        }

        // [DOD FIX] O(N log N) parallel sorting by hash.
        // Elements in the same cell are physically adjacent in memory.
        flat_cells.par_sort_unstable_by_key(|k| k.0);

        let mut cell_index = HashMap::with_capacity(flat_cells.len() / 10);
        let mut start = 0;

        for i in 1..=flat_cells.len() {
            if i == flat_cells.len() || flat_cells[i].0 != flat_cells[i - 1].0 {
                cell_index.insert(flat_cells[start].0, (start as u32)..(i as u32));
                start = i;
            }
        }

        self.flat_cells = flat_cells;
        self.cell_index = cell_index;
    }

    #[inline(always)]
    pub fn hash_cell(cx: u32, cy: u32, cz: u32) -> u64 {
        ((cx as u64) & 0xFFF) | (((cy as u64) & 0xFFF) << 12) | (((cz as u64) & 0xFF) << 24)
    }

    #[inline(always)]
    pub fn for_each_in_radius<F>(&self, pos: &PackedPosition, radius_cells: i32, mut f: F)
    where
        F: FnMut(u32),
    {
        let cx = (pos.x() as u32 / self.cell_size) as i32;
        let cy = (pos.y() as u32 / self.cell_size) as i32;
        let cz = (pos.z() as u32 / self.cell_size) as i32;

        for z in (cz - radius_cells)..=(cz + radius_cells) {
            if z < 0 {
                continue;
            }
            for y in (cy - radius_cells)..=(cy + radius_cells) {
                if y < 0 {
                    continue;
                }
                for x in (cx - radius_cells)..=(cx + radius_cells) {
                    if x < 0 {
                        continue;
                    }

                    let hash = Self::hash_cell(x as u32, y as u32, z as u32);
                    // [DOD FIX] One lookup followed by slice and hardware Prefetching
                    if let Some(range) = self.cell_index.get(&hash) {
                        for i in range.start..range.end {
                            f(self.flat_cells[i as usize].1);
                        }
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub fn get_position(&self, dense_id: u32) -> PackedPosition {
        self.positions[dense_id as usize]
    }

    pub fn positions_len(&self) -> usize {
        self.positions.len()
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SegmentRef {
    pub axon_id: u32,
    pub seg_idx: u16,
    pub type_idx: u8,
}

pub struct AxonSegmentGrid {
    pub cell_size: u32,
    cell_index: HashMap<u64, std::ops::Range<u32>>,
    flat_cells: Vec<(u64, SegmentRef)>,
}

impl AxonSegmentGrid {
    pub fn build_from_paths(
        lengths: &[u8],
        paths: &[u32],
        total_axons: usize,
        cell_size_voxels: u32,
    ) -> Self {
        let cell_size = cell_size_voxels.max(1);
        let mut flat_cells = Vec::with_capacity(total_axons * 10);

        for axon_id in (0..total_axons).rev() {
            let len = lengths[axon_id] as usize;
            let offset = axon_id * 256;

            for seg_idx in 0..len {
                let packed = paths[offset + seg_idx];
                if packed == 0 {
                    continue;
                }

                let pos = PackedPosition(packed);
                let cx = (pos.x() as u32) / cell_size;
                let cy = (pos.y() as u32) / cell_size;
                let cz = (pos.z() as u32) / cell_size;

                let hash = SpatialGrid::hash_cell(cx, cy, cz);
                flat_cells.push((
                    hash,
                    SegmentRef {
                        axon_id: axon_id as u32,
                        seg_idx: seg_idx as u16,
                        type_idx: pos.type_id(),
                    },
                ));
            }
        }

        flat_cells.par_sort_unstable_by_key(|k| k.0);

        let mut cell_index = HashMap::with_capacity(flat_cells.len() / 10);
        let mut start = 0;
        for i in 1..=flat_cells.len() {
            if i == flat_cells.len() || flat_cells[i].0 != flat_cells[i - 1].0 {
                cell_index.insert(flat_cells[start].0, (start as u32)..(i as u32));
                start = i;
            }
        }

        Self {
            cell_size,
            cell_index,
            flat_cells,
        }
    }

    pub fn build_from_axons(axons: &[GrownAxon], cell_size_voxels: u32) -> Self {
        let cell_size = cell_size_voxels.max(1);
        let est_segs: usize = axons.iter().map(|a| a.segments.len()).sum();
        let mut flat_cells = Vec::with_capacity(est_segs);

        for (axon_id, axon) in axons.iter().enumerate().rev() {
            let type_idx = axon.type_idx as u8;
            for (seg_idx, &packed) in axon.segments.iter().enumerate() {
                let pos = PackedPosition(packed);
                let cx = (pos.x() as u32) / cell_size;
                let cy = (pos.y() as u32) / cell_size;
                let cz = (pos.z() as u32) / cell_size;

                let hash = SpatialGrid::hash_cell(cx, cy, cz);
                flat_cells.push((
                    hash,
                    SegmentRef {
                        axon_id: axon_id as u32,
                        seg_idx: seg_idx as u16,
                        type_idx,
                    },
                ));
            }
        }

        flat_cells.par_sort_unstable_by_key(|k| k.0);

        let mut cell_index = HashMap::with_capacity(flat_cells.len() / 10);
        let mut start = 0;
        for i in 1..=flat_cells.len() {
            if i == flat_cells.len() || flat_cells[i].0 != flat_cells[i - 1].0 {
                cell_index.insert(flat_cells[start].0, (start as u32)..(i as u32));
                start = i;
            }
        }

        Self {
            cell_size,
            cell_index,
            flat_cells,
        }
    }

    #[inline(always)]
    pub fn for_each_in_radius<F>(&self, pos: &PackedPosition, radius_cells: i32, mut f: F)
    where
        F: FnMut(&SegmentRef),
    {
        let cx = (pos.x() as u32 / self.cell_size) as i32;
        let cy = (pos.y() as u32 / self.cell_size) as i32;
        let cz = (pos.z() as u32 / self.cell_size) as i32;

        for z in (cz - radius_cells)..=(cz + radius_cells) {
            if z < 0 {
                continue;
            }
            for y in (cy - radius_cells)..=(cy + radius_cells) {
                if y < 0 {
                    continue;
                }
                for x in (cx - radius_cells)..=(cx + radius_cells) {
                    if x < 0 {
                        continue;
                    }

                    let hash = SpatialGrid::hash_cell(x as u32, y as u32, z as u32);
                    if let Some(range) = self.cell_index.get(&hash) {
                        for i in range.start..range.end {
                            f(&self.flat_cells[i as usize].1);
                        }
                    }
                }
            }
        }
    }
}
