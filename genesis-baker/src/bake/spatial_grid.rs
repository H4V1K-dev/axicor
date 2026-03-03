// genesis-baker/src/bake/spatial_grid.rs

use genesis_core::types::PackedPosition;
use genesis_core::coords::unpack_position;
use crate::bake::neuron_placement::PlacedNeuron;
use glam::Vec3;

/// Плоская структура для O(1) поиска нейронов.
/// Гарантирует Data Locality, так как хранит только Dense ID (u32).
/// Построена через паттерн Counting Sort.
pub struct SpatialGrid {
    /// Префиксные суммы. Индекс = hash_cell, Значение = начало интервала в cell_entries.
    /// Размер = 1 << 20 (для сетки 128x128x64 ячеек).
    pub cell_offsets: Vec<u32>,
    /// Плоский массив Dense ID нейронов (индексы в исходном массиве positions).
    pub cell_entries: Vec<u32>,
    /// Размер ячейки сетки в вокселях.
    pub cell_size: f32,
}

impl SpatialGrid {
    /// Лимит хэш-таблицы (2^20 ячеек для покрытия 1280x1280x256 вокселей при CS=10).
    pub const GRID_CAPACITY: usize = 1 << 20;

    #[inline(always)]
    pub fn hash_cell(x: u32, y: u32, z: u32, cell_size: f32) -> usize {
        let cs = cell_size as u32;
        let cx = (x / cs) & 0x7F; // 7 bits
        let cy = (y / cs) & 0x7F; // 7 bits
        let cz = (z / 4) & 0x3F;  // 6 bits (Z levels in groups of 4 voxels)
        (cz << 14 | cy << 7 | cx) as usize
    }

    /// Legacy constructor for compatibility with PlacedNeuron-based tests.
    pub fn new(neurons: &[PlacedNeuron], cell_size: f32) -> Self {
        let positions: Vec<PackedPosition> = neurons.iter().map(|n| n.position).collect();
        let mut grid = Self::build(&positions, cell_size);
        grid.cell_size = cell_size;
        grid
    }

    /// Строит сетку за 3 прохода (Counting Sort).
    /// 0 аллокаций внутри циклов.
    pub fn build(positions: &[PackedPosition], cell_size: f32) -> Self {
        let mut cell_offsets = vec![0u32; Self::GRID_CAPACITY + 1];
        let mut cell_entries = vec![0u32; positions.len()];

        // Проход 1: Histogram
        for &pos in positions {
            let (x, y, z, _) = unpack_position(pos);
            let h = Self::hash_cell(x, y, z, cell_size);
            cell_offsets[h] += 1;
        }

        // Проход 2: Prefix Sum (Cumulative)
        let mut sum = 0;
        for i in 0..Self::GRID_CAPACITY {
            let count = cell_offsets[i];
            cell_offsets[i] = sum;
            sum += count;
        }
        cell_offsets[Self::GRID_CAPACITY] = sum;

        // Временный массив курсоров для записи
        let mut cursors = cell_offsets.clone();

        // Проход 3: Scatter
        for (dense_id, &pos) in positions.iter().enumerate() {
            let (x, y, z, _) = unpack_position(pos);
            let h = Self::hash_cell(x, y, z, cell_size);
            let target_idx = cursors[h] as usize;
            cell_entries[target_idx] = dense_id as u32;
            cursors[h] += 1;
        }

        Self {
            cell_offsets,
            cell_entries,
            cell_size,
        }
    }

    /// Возвращает срез Dense ID нейронов в конкретной ячейке.
    #[inline(always)]
    pub fn get_cell_slice(&self, hash: usize) -> &[u32] {
        if hash >= Self::GRID_CAPACITY {
            return &[];
        }
        let start = self.cell_offsets[hash] as usize;
        let end = self.cell_offsets[hash + 1] as usize;
        &self.cell_entries[start..end]
    }

    /// Legacy method for radius-based lookup used in existing tests.
    pub fn get_in_radius(&self, pos: Vec3, radius: f32) -> Vec<usize> {
        let mut result = Vec::new();
        
        // Пересекаем ячейки GRID_CAPACITY в радиусе
        let min_x = (pos.x - radius).max(0.0) as u32;
        let max_x = (pos.x + radius) as u32;
        let min_y = (pos.y - radius).max(0.0) as u32;
        let max_y = (pos.y + radius) as u32;
        let min_z = (pos.z - radius).max(0.0) as u32;
        let max_z = (pos.z + radius) as u32;

        let cs = self.cell_size as u32;
        
        // Цикл по ячейкам (не по вокселям!)
        let start_cx = min_x / cs;
        let end_cx = max_x / cs;
        let start_cy = min_y / cs;
        let end_cy = max_y / cs;
        let start_cz = min_z / 4;
        let end_cz = max_z / 4;

        for cx in start_cx..=end_cx {
            for cy in start_cy..=end_cy {
                for cz in start_cz..=end_cz {
                    // Хэшируем ячейку
                    let h = ((cz & 0x3F) << 14 | (cy & 0x7F) << 7 | (cx & 0x7F)) as usize;
                    result.extend(self.get_cell_slice(h).iter().map(|&id| id as usize));
                }
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::coords::pack_position;

    #[test]
    fn test_spatial_grid_building() {
        let mut positions = Vec::new();
        // Два нейрона в одной ячейке (CS=10)
        positions.push(pack_position(5, 5, 1, 0)); // Dense ID 0
        positions.push(pack_position(8, 2, 2, 1)); // Dense ID 1
        // Один в соседней
        positions.push(pack_position(15, 5, 1, 2)); // Dense ID 2

        let grid = SpatialGrid::build(&positions, 10.0);
        
        let h1 = SpatialGrid::hash_cell(5, 5, 1, 10.0);
        let h2 = SpatialGrid::hash_cell(15, 5, 1, 10.0);
        
        let slice1 = grid.get_cell_slice(h1);
        let slice2 = grid.get_cell_slice(h2);
        
        assert_eq!(slice1.len(), 2);
        assert!(slice1.contains(&0));
        assert!(slice1.contains(&1));
        
        assert_eq!(slice2.len(), 1);
        assert_eq!(slice2[0], 2);
    }
}
