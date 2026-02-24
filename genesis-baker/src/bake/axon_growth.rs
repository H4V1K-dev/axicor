use crate::bake::neuron_placement::PlacedNeuron;
use crate::bake::seed::{entity_seed, random_f32};
use crate::parser::anatomy::Anatomy;
use crate::parser::blueprints::NeuronType;
use crate::parser::simulation::SimulationConfig;

/// Выращенный аксон готовый к записи в ShardStateSoA.
#[derive(Debug, Clone)]
pub struct GrownAxon {
    /// Индекс сомы в массиве нейронов
    pub soma_idx: usize,
    /// Тип нейрона (копируется для дендритной фильтрации без lookup)
    pub type_idx: usize,
    /// Tip position — конечная точка аксона (где ищем дендриты)
    pub tip_x: u32,
    pub tip_y: u32,
    pub tip_z: u32,
    /// Длина аксона в сегментах (для инициализации axon_head)
    pub length_segments: u32,
    /// Геометрия кусочно-линейная (PackedPositions: Type|Z|Y|X)
    pub segments: Vec<u32>,
}

/// Кэш Z-диапазонов слоёв (вычисляется один раз из anatomy + sim)
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LayerZRange {
    pub name: String,
    pub z_start_vox: u32,
    pub z_end_vox: u32,
}

/// Предвычисляет Z-диапазоны всех слоёв в вокселях.
pub fn compute_layer_ranges(anatomy: &Anatomy, sim: &SimulationConfig) -> Vec<LayerZRange> {
    let voxel_um = sim.simulation.voxel_size_um;
    let world_h_vox = sim.world.height_um / voxel_um;
    let mut cursor_pct = 0.0f32;
    let mut ranges = Vec::with_capacity(anatomy.layers.len());
    for layer in &anatomy.layers {
        let h_um = (sim.world.height_um as f32 * layer.height_pct) as u32;
        let z_start = (cursor_pct * world_h_vox as f32) as u32;
        let z_end = ((cursor_pct + layer.height_pct) * world_h_vox as f32) as u32;
        cursor_pct += layer.height_pct;
        ranges.push(LayerZRange {
            name: layer.name.clone(),
            z_start_vox: z_start,
            z_end_vox: z_end.min(255),
        });
    }
    ranges
}

/// Cone Tracing: вычитает конечную позицию аксона для каждого нейрона.
///
/// Алгоритм (04_connectivity.md §1.3):
/// 1. Найти слой сомы по Z-координате.
/// 2. `Soma_Rel_Z = (soma_z - layer_z_start) / layer_height`
/// 3. Целевой слой = слой выше (Z+), если нейрон не в верхнем слое.
///    (Для первого Baking — каждый нейрон тянется в ближайший вышестоящий слой)
/// 4. `Target_Z = target_z_start + Soma_Rel_Z × target_height`
/// 5. XY: небольшой дрейф конуса (FOV) относительно оригинальной XY-позиции.
///    `tip_x = soma_x + Δx`, где `|Δx| ≤ cone_radius`
/// 6. Длина аксона = |target_z - soma_z| + 1 (в сегментах-вокселях)
use crate::bake::cone_tracing::calculate_v_attract;
use crate::bake::spatial_grid::SpatialGrid;
use glam::Vec3;

pub fn grow_axons(
    neurons: &[PlacedNeuron],
    layer_ranges: &[LayerZRange],
    neuron_types: &[NeuronType],
    sim: &SimulationConfig,
    master_seed: u64,
) -> Vec<GrownAxon> {
    let world_w_vox = sim.world.width_um / sim.simulation.voxel_size_um;
    let world_d_vox = sim.world.depth_um / sim.simulation.voxel_size_um;
    let world_h_vox = sim.world.height_um / sim.simulation.voxel_size_um;

    let mut axons = Vec::with_capacity(neurons.len());
    let spatial_grid = SpatialGrid::new(neurons);

    for (soma_idx, neuron) in neurons.iter().enumerate() {
        let soma_z = neuron.z();
        let soma_x = neuron.x();
        let soma_y = neuron.y();
        let type_idx = neuron.type_idx;
        let nt = &neuron_types[type_idx.min(neuron_types.len() - 1)];

        // 1. Найти домашний слой сомы
        let home_layer = find_layer(soma_z, layer_ranges);
        let (home_z_start, home_z_end) = match home_layer {
            Some(l) => (l.z_start_vox, l.z_end_vox),
            None => (soma_z, soma_z + 1),
        };

        // 2. Soma_Rel_Z — относительная позиция в домашнем слое [0.0, 1.0)
        let layer_h = (home_z_end - home_z_start).max(1) as f32;
        let soma_rel_z = (soma_z.saturating_sub(home_z_start) as f32) / layer_h;

        // 3. Целевой слой — следующий вверх по Z (index + 1)
        let target_layer = find_target_layer(soma_z, layer_ranges);
        let (target_z_start, target_z_end) = match target_layer {
            Some(l) => (l.z_start_vox, l.z_end_vox),
            None => {
                // Верхний слой — аксон тянется вниз к предыдущему
                layer_ranges
                    .first()
                    .map_or((0u32, 1u32), |l| (l.z_start_vox, l.z_end_vox))
            }
        };

        // 4. Target_Z = target_z_start + Soma_Rel_Z × target_height
        let target_h = (target_z_end - target_z_start).max(1) as f32;
        let tip_z = (target_z_start as f32 + soma_rel_z * target_h) as u32;
        let tip_z = tip_z.clamp(target_z_start, target_z_end).min(255);

        // Global segment length from config (fixed for all types)
        let segment_length_vox = sim.simulation.segment_length_voxels as f32;
        let cone_seed = entity_seed(master_seed, soma_idx as u32);
        let owner_type_mask = type_idx as u8; // We assume type_idx fits into 4 bits
        
        // Approximate specs fields replaced with actual config
        let fov_cos = (nt.steering_fov_deg / 2.0).to_radians().cos(); 
        let max_search_radius_vox = nt.steering_radius_um / (sim.simulation.voxel_size_um as f32);
        let weight_inertia = nt.steering_weight_inertia;
        let weight_sensor = nt.steering_weight_sensor;
        let weight_jitter = nt.steering_weight_jitter;

        // V_global (Goal) — bias определяет вертикальный vs горизонтальный рост
        let bias = nt.growth_vertical_bias.clamp(0.0, 1.0);

        // Вертикальная цель: целевой слой по Z (существующая логика)
        let v_vertical_target = Vec3::new(soma_x as f32, soma_y as f32, tip_z as f32);

        // Горизонтальная цель: случайная точка в XY плоскости своего слоя
        let horiz_seed = entity_seed(master_seed, soma_idx as u32 + 0x48_4F_52_5A); // "HORZ"
        let target_x = random_f32(horiz_seed) * world_w_vox as f32;
        let target_y = random_f32(horiz_seed.wrapping_mul(6364136223846793005)) * world_d_vox as f32;
        let v_horizontal_target = Vec3::new(target_x, target_y, soma_z as f32);

        // Смешиваем цели по bias
        let target_pos = v_vertical_target * bias + v_horizontal_target * (1.0 - bias);

        let is_growing_up = tip_z >= soma_z;
        let mut forward_dir = (target_pos - Vec3::new(soma_x as f32, soma_y as f32, soma_z as f32)).normalize_or_zero();
        if forward_dir.length_squared() < 0.01 {
            forward_dir = if is_growing_up { Vec3::Z } else { Vec3::NEG_Z };
        }
        
        let mut current_pos = Vec3::new(soma_x as f32, soma_y as f32, soma_z as f32);

        let mut segments = Vec::new();
        let max_steps = sim.simulation.axon_growth_max_steps;
        let mut step = 0;

        while step < max_steps {
            step += 1;
            
            // Check if reached V_global (Z-plane)
            let finished = if is_growing_up {
                current_pos.z >= target_pos.z
            } else {
                current_pos.z <= target_pos.z
            };

            if finished {
                break;
            }

            // V_global steering vector (always points toward the target plane/xy column)
            let v_global = (target_pos - current_pos).normalize_or_zero();
            
            // Sensing → Weighting
            let v_attract = calculate_v_attract(
                current_pos,
                forward_dir,
                fov_cos,
                max_search_radius_vox,
                &spatial_grid,
                neurons,
                owner_type_mask,
                soma_idx,
                nt.type_affinity,
            );

            // Jitter
            let s = cone_seed.wrapping_add(step as u64);
            let v_noise = Vec3::new(
                random_f32(s) * 2.0 - 1.0,
                random_f32(s.wrapping_add(1)) * 2.0 - 1.0,
                random_f32(s.wrapping_add(2)) * 2.0 - 1.0,
            ).normalize_or_zero();

            // Steering
            forward_dir = (v_global * weight_inertia + v_attract * weight_sensor + v_noise * weight_jitter).normalize_or_zero();
            
            // Step & Pack
            current_pos += forward_dir * segment_length_vox;
            
            let x = (current_pos.x.round() as u32).min(world_w_vox.saturating_sub(1)).min(1023); // 10 bits
            let y = (current_pos.y.round() as u32).min(world_d_vox.saturating_sub(1)).min(1023); // 10 bits
            let z = (current_pos.z.round() as u32).min(world_h_vox.saturating_sub(1)).min(255); // 8 bits
            let t = (owner_type_mask & 0x0F) as u32; // 4 bits

            let packed = (t << 28) | (z << 20) | (y << 10) | x;

            if let Some(&last) = segments.last() {
                if last == packed {
                    // Stagnation
                    break;
                }
            }
            
            segments.push(packed);
        }

        let length_segments = segments.len() as u32;
        let (final_x, final_y, final_z) = if let Some(last) = segments.last() {
            let z = (last >> 20) & 0xFF;
            let y = (last >> 10) & 0x3FF;
            let x = last & 0x3FF;
            (x, y, z)
        } else {
            (soma_x, soma_y, soma_z)
        };

        axons.push(GrownAxon {
            soma_idx,
            type_idx,
            tip_x: final_x,
            tip_y: final_y,
            tip_z: final_z,
            length_segments,
            segments,
        });
    }

    axons
}

/// Инициализировать axon_head по spec:
/// `axon_head = AXON_SENTINEL - length_segments * v_seg`
/// Это позволяет PropagateAxons в первый же тик корректно распространить сигнал.
pub fn init_axon_head(length_segments: u32, v_seg: u32) -> u32 {
    use genesis_core::constants::AXON_SENTINEL;
    AXON_SENTINEL.wrapping_sub(length_segments * v_seg)
}

fn find_layer(z: u32, ranges: &[LayerZRange]) -> Option<&LayerZRange> {
    ranges
        .iter()
        .find(|l| z >= l.z_start_vox && z < l.z_end_vox)
}

fn find_target_layer(soma_z: u32, ranges: &[LayerZRange]) -> Option<&LayerZRange> {
    // Следующий слой выше по Z (z_start больше z текущего слоя)
    ranges
        .iter()
        .filter(|l| l.z_start_vox > soma_z)
        .min_by_key(|l| l.z_start_vox)
}



/// Generates fake "External" Ghost Axons that originate from other shards (Atlas Routing).
/// They aren't attached to any local soma, so `soma_idx` is set to `usize::MAX`.
pub fn grow_external_axons(
    io_config: &crate::parser::io::IoConfig,
    layer_ranges: &[LayerZRange],
    sim: &SimulationConfig,
    master_seed: u64,
) -> Vec<GrownAxon> {
    let mut ext_axons = Vec::new();
    let world_w_vox = sim.world.width_um / sim.simulation.voxel_size_um;
    let world_d_vox = sim.world.depth_um / sim.simulation.voxel_size_um;

    for (channel_idx, channel) in io_config.inputs.iter().enumerate() {
        // Find the target layer bounds
        let layer_z = layer_ranges.iter().find(|l| l.name == channel.target_layer);
        let (z_start, z_end) = match layer_z {
            Some(l) => (l.z_start_vox, l.z_end_vox),
            None => continue, // Layer not found, skip projecting here
        };

        let depth = (z_end - z_start).max(1) as f32;

        for i in 0..channel.axon_count {
            // Seed uniquely identifies this external projection
            let s = master_seed
                .wrapping_add(channel_idx as u64)
                .wrapping_add(i as u64);

            // Jitter the entry tip throughout the destination layer
            let tip_x = (random_f32(s) * world_w_vox as f32) as u32;
            let tip_y = (random_f32(s.wrapping_add(1)) * world_d_vox as f32) as u32;
            let tip_z = z_start + (random_f32(s.wrapping_add(2)) * depth) as u32;

            let tip_x = tip_x.min(world_w_vox.saturating_sub(1));
            let tip_y = tip_y.min(world_d_vox.saturating_sub(1));
            let tip_z = tip_z.min(z_end);
            let t = (channel.type_mask as u32) & 0x0F;

            ext_axons.push(GrownAxon {
                soma_idx: usize::MAX, // Signifies external origin
                type_idx: channel.type_mask as usize, // Stuffs the 4-bit config phenotype
                tip_x,
                tip_y,
                tip_z,
                length_segments: 1, // Assume they just "entered" the shard at the border
                segments: vec![(t << 28) | (tip_z << 20) | (tip_y << 10) | tip_x],
            });
        }
    }

    ext_axons
}

/// Создаёт 2D-сетку виртуальных аксонов (Mock Retina).
/// Они располагаются на плоскости Z=0 и смотрят вверх.
pub fn grow_mock_retina(
    num_virtual: u32,
    sim: &SimulationConfig,
) -> Vec<GrownAxon> {
    if num_virtual == 0 {
        return Vec::new();
    }

    let mut retina = Vec::with_capacity(num_virtual as usize);
    let world_w_vox = sim.world.width_um / sim.simulation.voxel_size_um;
    let world_d_vox = sim.world.depth_um / sim.simulation.voxel_size_um;

    // Предположим, что мы распределяем сетчатку квадратом
    let side = (num_virtual as f32).sqrt().ceil() as u32;
    let step_x = (world_w_vox as f32 / side as f32).max(1.0);
    let step_y = (world_d_vox as f32 / side as f32).max(1.0);

    for i in 0..num_virtual {
        let ix = i % side;
        let iy = i / side;

        let tip_x = ((ix as f32 * step_x) as u32).min(world_w_vox.saturating_sub(1));
        let tip_y = ((iy as f32 * step_y) as u32).min(world_d_vox.saturating_sub(1));
        let tip_z = 0; // Сетчатка лежит на дне (z=0)
        let t = 0u32; // type=0 

        retina.push(GrownAxon {
            soma_idx: usize::MAX, // Signifies external origin
            type_idx: 0, // Условно 0 (Excitatory)
            tip_x,
            tip_y,
            tip_z,
            length_segments: 1,
            // (Type=0 | Z=0 | Y=tip_y | X=tip_x)
            segments: vec![(t << 28) | (tip_z << 20) | (tip_y << 10) | tip_x],
        });
    }

    retina
}

