use crate::bake::seed::{entity_seed, random_f32, shuffle_indices};
use crate::parser::{anatomy::Anatomy, simulation::SimulationConfig};
use genesis_core::coords::{pack_position, unpack_position};
use genesis_core::types::PackedPosition;

/// Размещённый нейрон в 3D-пространстве.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PlacedNeuron {
    /// Packed voxel coordinate: [Type(4b)|Z(8b)|Y(10b)|X(10b)]
    pub position: PackedPosition,
    /// Индекс нейронного типа из blueprints.neuron_type[]
    pub type_idx: usize,
    /// Имя слоя в котором размещён нейрон (для отладки)
    pub layer_name: String,
}

impl PlacedNeuron {
    /// X-координата в вокселях.
    #[inline] pub fn x(&self) -> u32 { unpack_position(self.position).0 }
    /// Y-координата в вокселях.
    #[inline] pub fn y(&self) -> u32 { unpack_position(self.position).1 }
    /// Z-координата в вокселях.
    #[inline] pub fn z(&self) -> u32 { unpack_position(self.position).2 }
    /// Тип-маска нейрона (4 бита).
    #[inline] pub fn type_mask(&self) -> u32 { unpack_position(self.position).3 }
}

/// Размещает все нейроны зоны в 3D-пространстве.
///
/// Алгоритм (02_configuration.md §5.5):
/// 1. `Layer_Budget = floor(population_pct × total_budget)`
/// 2. Для каждого типа: `count = floor(quota × Layer_Budget)`
/// 3. Нейроны размещаются равномерно в [Z_start, Z_end], XY — случайно в [0, width)
/// 4. Детерминированный shuffle через wyhash + master_seed
pub fn place_neurons(
    sim: &SimulationConfig,
    anatomy: &Anatomy,
    type_names: &[String], // blueprints.neuron_type[i].name → индекс
    master_seed: u64,
) -> Vec<PlacedNeuron> {
    let total_budget = sim.neuron_budget();
    let voxel_um = sim.simulation.voxel_size_um;

    let world_w_vox = sim.world.width_um / voxel_um;
    let world_d_vox = sim.world.depth_um / voxel_um;
    let world_h_vox = sim.world.height_um / voxel_um;

    let mut all_neurons: Vec<PlacedNeuron> = Vec::with_capacity(total_budget as usize);

    let mut z_cursor_pct = 0.0f32;

    for layer in &anatomy.layers {
        let layer_budget = (total_budget as f32 * layer.population_pct) as u64;
        let z_start = (z_cursor_pct * world_h_vox as f32) as u32;
        let z_end = ((z_cursor_pct + layer.height_pct) * world_h_vox as f32) as u32;
        let z_range = (z_end - z_start).max(1);
        z_cursor_pct += layer.height_pct;

        // Для каждого типа — разместить floor(quota × budget) нейронов
        for (type_name, &quota) in &layer.composition {
            let count = (layer_budget as f32 * quota) as u64;
            let type_idx = type_names.iter().position(|n| n == type_name).unwrap_or(0);
            // type_mask — просто индекс типа (0..3), 4 бита
            let type_mask = (type_idx & 0xF) as u32;

            // Генерируем `count` позиций с детерминированным shuffle
            let shuffle = shuffle_indices(
                count as usize,
                entity_seed(master_seed, all_neurons.len() as u32),
            );

            for (i, &si) in shuffle.iter().enumerate() {
                // Равномерное распределение по Z внутри слоя
                let z = z_start + (si as u32 % z_range);
                // XY — псевдослучайно через seed
                let pos_seed = entity_seed(master_seed, (all_neurons.len() + i) as u32);
                let x = (random_f32(pos_seed) * world_w_vox as f32) as u32;
                let y = (random_f32(pos_seed.wrapping_mul(6364136223846793005))
                    * world_d_vox as f32) as u32;

                let x = x.min(world_w_vox - 1);
                let y = y.min(world_d_vox - 1);
                let z = z.min(255); // Z 8-bit cap

                all_neurons.push(PlacedNeuron {
                    position: pack_position(x, y, z, type_mask),
                    type_idx,
                    layer_name: layer.name.clone(),
                });
            }
        }
    }

    all_neurons
}

