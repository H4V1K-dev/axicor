use std::collections::HashMap;
use genesis_core::types::PackedPosition;

/// Пространственный хэш для O(1) поиска соседей.
/// Хранит только dense_id (индексы в массиве PackedPosition).
pub struct SpatialGrid {
    pub cell_size: u32,
    cells: HashMap<u64, Vec<u32>>,
    positions: Vec<PackedPosition>, // Read-only копия или ссылка для быстрого доступа
}

impl SpatialGrid {
    pub fn new(positions: Vec<PackedPosition>, cell_size_voxels: u32) -> Self {
        let mut grid = Self {
            cell_size: cell_size_voxels.max(1),
            cells: HashMap::with_capacity(positions.len() / 10),
            positions,
        };
        grid.build();
        grid
    }

    fn build(&mut self) {
        for (dense_id, pos) in self.positions.iter().enumerate() {
            // Пропускаем пустышки от Warp Alignment (x=0, y=0, z=0, type=0)
            if pos.0 == 0 {
                continue;
            }

            let cx = (pos.x() as u32) / self.cell_size;
            let cy = (pos.y() as u32) / self.cell_size;
            let cz = (pos.z() as u32) / self.cell_size;
            
            let hash = Self::hash_cell(cx, cy, cz);
            self.cells.entry(hash).or_default().push(dense_id as u32);
        }
    }

    #[inline(always)]
    pub fn hash_cell(cx: u32, cy: u32, cz: u32) -> u64 {
        // Упаковываем координаты чанка в 64-битный ключ
        ((cx as u64) & 0xFFF) | (((cy as u64) & 0xFFF) << 12) | (((cz as u64) & 0xFF) << 24)
    }

    /// Возвращает итератор по всем dense_id в заданном радиусе (в чанках)
    pub fn query_radius(&self, pos: &PackedPosition, radius_cells: i32) -> Vec<u32> {
        let cx = (pos.x() as u32 / self.cell_size) as i32;
        let cy = (pos.y() as u32 / self.cell_size) as i32;
        let cz = (pos.z() as u32 / self.cell_size) as i32;

        let mut result = Vec::new();

        for z in (cz - radius_cells)..=(cz + radius_cells) {
            if z < 0 { continue; }
            for y in (cy - radius_cells)..=(cy + radius_cells) {
                if y < 0 { continue; }
                for x in (cx - radius_cells)..=(cx + radius_cells) {
                    if x < 0 { continue; }
                    
                    let hash = Self::hash_cell(x as u32, y as u32, z as u32);
                    if let Some(ids) = self.cells.get(&hash) {
                        result.extend_from_slice(ids);
                    }
                }
            }
        }
        result
    }

    #[inline(always)]
    pub fn get_position(&self, dense_id: u32) -> PackedPosition {
        self.positions[dense_id as usize]
    }
    
    pub fn positions_len(&self) -> usize {
        self.positions.len()
    }
}
