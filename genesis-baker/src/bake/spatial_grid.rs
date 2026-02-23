use crate::bake::neuron_placement::PlacedNeuron;
use glam::Vec3;
use std::collections::HashMap;

/// Size of a hash-grid cell in voxels (e.g., 2 = 50 um grid cells).
/// Tune this based on MAX_SEARCH_RADIUS.
const CELL_SIZE: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GridCell {
    x: i32,
    y: i32,
    z: i32,
}

#[derive(Debug, Clone)]
pub struct SpatialGrid {
    cells: HashMap<GridCell, Vec<usize>>,
}

impl SpatialGrid {
    pub fn new(neurons: &[PlacedNeuron]) -> Self {
        let mut cells: HashMap<GridCell, Vec<usize>> = HashMap::new();
        
        for (i, neuron) in neurons.iter().enumerate() {
            let cx = (neuron.x() as f32 / CELL_SIZE).floor() as i32;
            let cy = (neuron.y() as f32 / CELL_SIZE).floor() as i32;
            let cz = (neuron.z() as f32 / CELL_SIZE).floor() as i32;
            
            cells.entry(GridCell { x: cx, y: cy, z: cz }).or_default().push(i);
        }
        
        Self { cells }
    }

    /// Returns a list of neuron indices that could be within `radius` voxels.
    /// The radius is in voxels!
    pub fn get_in_radius(&self, pos: Vec3, radius: f32) -> Vec<usize> {
        let mut result = Vec::new();
        
        let min_cx = ((pos.x - radius) / CELL_SIZE).floor() as i32;
        let max_cx = ((pos.x + radius) / CELL_SIZE).floor() as i32;
        let min_cy = ((pos.y - radius) / CELL_SIZE).floor() as i32;
        let max_cy = ((pos.y + radius) / CELL_SIZE).floor() as i32;
        let min_cz = ((pos.z - radius) / CELL_SIZE).floor() as i32;
        let max_cz = ((pos.z + radius) / CELL_SIZE).floor() as i32;

        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                for cz in min_cz..=max_cz {
                    if let Some(indices) = self.cells.get(&GridCell { x: cx, y: cy, z: cz }) {
                        result.extend_from_slice(indices);
                    }
                }
            }
        }
        
        result
    }
}
