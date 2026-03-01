use crate::bake::neuron_placement::PlacedNeuron;
use crate::bake::seed::{entity_seed, random_f32};
use crate::parser::anatomy::Anatomy;
use crate::parser::simulation::SimulationConfig;
use genesis_core::config::blueprints::GenesisConstantMemory;

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

/// Границы шарда в мировых координатах (вокселях).
/// Аксоны, пересекающие эту границу, становятся Ghost Axons в соседнем шарде.
/// В монорежиме (один шард) передавать `ShardBounds::full_world(&sim)` —
/// тогда crossing никогда не произойдёт и поведение идентично текущему.
#[derive(Debug, Clone)]
pub struct ShardBounds {
    pub x_start: u32, pub x_end: u32,
    pub y_start: u32, pub y_end: u32,
    pub z_start: u32, pub z_end: u32,
}

impl ShardBounds {
    /// Единый шард = весь мировой объём. Ghost Packets генерироваться не будут.
    pub fn full_world(sim: &SimulationConfig) -> Self {
        Self {
            x_start: 0,
            x_end: sim.world.width_um / sim.simulation.voxel_size_um,
            y_start: 0,
            y_end: sim.world.depth_um / sim.simulation.voxel_size_um,
            z_start: 0,
            z_end: sim.world.height_um / sim.simulation.voxel_size_um,
        }
    }

    /// Проверяет, вышла ли точка за пределы шарда.
    #[inline]
    pub fn is_outside(&self, x: u32, y: u32, z: u32) -> bool {
        x < self.x_start || x >= self.x_end
        || y < self.y_start || y >= self.y_end
        || z < self.z_start || z >= self.z_end
    }
}

/// Пакет межшардовой передачи.
/// Генерируется, когда аксон в шарде A пересекает свою границу.
/// Шард B получает этот пакет и создаёт Ghost Axon на своей стороне границы.
#[derive(Debug, Clone)]
pub struct GhostPacket {
    /// ID шарда-источника (для маршрутизации в полишардовой топологии)
    pub origin_shard_id: u32,
    /// аксон без локальной сомы → usize::MAX
    pub soma_idx: usize,
    /// Тип нейрона (для whitelist, affinity и т.д.)
    pub type_idx: usize,
    /// Точка входа в новый шард (уже в его координатах)
    pub entry_x: u32,
    pub entry_y: u32,
    pub entry_z: u32,
    /// Направление движения в момент пересечения (для сохранения инерции)
    pub entry_dir: Vec3,
    /// Сколько шагов осталось до окончания роста
    pub remaining_steps: u32,
}

/// Предвычисляет Z-диапазоны всех слоёв в вокселях.
pub fn compute_layer_ranges(anatomy: &Anatomy, sim: &SimulationConfig) -> Vec<LayerZRange> {
    let voxel_um = sim.simulation.voxel_size_um;
    let world_h_vox = sim.world.height_um / voxel_um;
    let mut cursor_pct = 0.0f32;
    let mut ranges = Vec::with_capacity(anatomy.layers.len());
    for layer in &anatomy.layers {
        let _h_um = (sim.world.height_um as f32 * layer.height_pct) as u32;
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
    const_mem: &GenesisConstantMemory,
    sim: &SimulationConfig,
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (Vec<GrownAxon>, Vec<GhostPacket>) {
    let world_w_vox = sim.world.width_um / sim.simulation.voxel_size_um;
    let world_d_vox = sim.world.depth_um / sim.simulation.voxel_size_um;
    let world_h_vox = sim.world.height_um / sim.simulation.voxel_size_um;

    let mut axons = Vec::with_capacity(neurons.len());
    let mut ghost_packets: Vec<GhostPacket> = Vec::new();
    let spatial_grid = SpatialGrid::new(neurons);

    for (soma_idx, neuron) in neurons.iter().enumerate() {
        let soma_z = neuron.z();
        let soma_x = neuron.x();
        let soma_y = neuron.y();
        let type_idx = neuron.type_idx;

        let (axon, ghost) = grow_single_axon(
            soma_x, soma_y, soma_z,
            soma_idx,
            type_idx as u8,
            const_mem,
            sim,
            world_w_vox, world_d_vox, world_h_vox,
            layer_ranges,
            &spatial_grid,
            neurons,
            shard_bounds,
            master_seed,
        );

        axons.push(axon);
        if let Some(gp) = ghost {
            ghost_packets.push(gp);
        }
    }

    (axons, ghost_packets)
}

#[allow(clippy::too_many_arguments)]
pub fn grow_single_axon(
    soma_x: u32, soma_y: u32, soma_z: u32,
    soma_idx: usize,
    type_idx: u8,
    const_mem: &GenesisConstantMemory,
    sim: &SimulationConfig,
    world_w_vox: u32, world_d_vox: u32, world_h_vox: u32,
    layer_ranges: &[LayerZRange],
    spatial_grid: &SpatialGrid,
    neurons: &[PlacedNeuron],
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (GrownAxon, Option<GhostPacket>) {
    use crate::bake::cone_tracing::calculate_v_attract;

    // Чтение параметров роста за O(1) из плоского массива:
    let variant = &const_mem.variants[type_idx as usize];
    let _max_length = variant.signal_propagation_length;
    let _velocity = variant.conduction_velocity;

    // 1. Найдём слой сомы (Index_home)
    let home_layer = layer_ranges.iter().find(|l| soma_z >= l.z_start_vox && soma_z < l.z_end_vox);
    let (home_z_start, home_z_end) = match home_layer {
        Some(l) => (l.z_start_vox, l.z_end_vox),
        None => (soma_z, soma_z + 1), // fallback, if soma is outside defined layers
    };

    // 2. Soma_Rel_Z — относительная позиция в домашнем слое [0.0, 1.0)
    let layer_h = (home_z_end - home_z_start).max(1) as f32;
    let soma_rel_z = (soma_z.saturating_sub(home_z_start) as f32) / layer_h;

    // 3. Целевой слой — следующий вверх по Z (index + 1)
    let target_layer = layer_ranges.iter().find(|l| l.z_start_vox > soma_z);
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
    let owner_type_mask = type_idx; // 4-bit mask
    
    // TODO: Steering params — вынести в VariantParameters или отдельный CPU-конфиг
    let fov_cos = (45.0_f32 / 2.0).to_radians().cos();
    let max_search_radius_vox = 100.0 / (sim.simulation.voxel_size_um as f32);
    let weight_inertia = 0.6_f32;
    let weight_sensor = 0.3_f32;
    let weight_jitter = 0.1_f32;

    // V_global (Goal) — bias определяет вертикальный vs горизонтальный рост
    let bias = 0.8_f32; // TODO: вынести в CPU-конфиг
    let type_idx_usize = type_idx as usize;
    let is_horizontal = bias < 0.5;

    let mut current_pos = Vec3::new(soma_x as f32, soma_y as f32, soma_z as f32);
    let is_growing_up = tip_z >= soma_z;

    let (mut forward_dir, target_pos) = if is_horizontal {
        // Случайный радиальный вектор в XY
        let horiz_seed = entity_seed(master_seed, soma_idx as u32 + 0x48_4F_52_5A); // "HORZ"
        let angle = random_f32(horiz_seed) * std::f32::consts::TAU; // 0..2π
        let dir = Vec3::new(angle.cos(), angle.sin(), 0.0);
        
        // target_pos не используется для остановки H-нейронов, но используется
        // для генерации константного v_global на каждом шаге. 
        // Cоздаем целевую точку далеко в выбранном направлении.
        let far_target = current_pos + dir * 5000.0;
        (dir, far_target)
    } else {
        // Вертикальная цель: целевой слой по Z 
        let v_vertical_target = Vec3::new(soma_x as f32, soma_y as f32, tip_z as f32);
        
        // Горизонтальная компонента (ограниченно)
        let horiz_seed = entity_seed(master_seed, soma_idx as u32 + 0x48_4F_52_5A);
        let target_x = random_f32(horiz_seed) * world_w_vox as f32;
        let target_y = random_f32(horiz_seed.wrapping_mul(6364136223846793005)) * world_d_vox as f32;
        let v_horizontal_target = Vec3::new(target_x, target_y, soma_z as f32);

        let t_pos = v_vertical_target * bias + v_horizontal_target * (1.0 - bias);
        let dir = (t_pos - current_pos).normalize_or_zero();
        let final_dir = if dir.length_squared() < 0.01 {
            if is_growing_up { Vec3::Z } else { Vec3::NEG_Z }
        } else {
            dir
        };
        (final_dir, t_pos)
    };

    let mut segments = Vec::new();
    let max_steps = sim.simulation.axon_growth_max_steps;
    let mut step = 0;
    let mut ghost_packet = None;

    while step < max_steps {
        step += 1;
        
        // Check if reached Stop Condition
        let finished = if is_horizontal {
            // H: вышел из своего слоя по Z? (заблудился по вертикали)
            let z = current_pos.z as u32;
            z < home_z_start || z > home_z_end
        } else {
            // V: достиг целевой Z-plane?
            if is_growing_up {
                current_pos.z >= target_pos.z
            } else {
                current_pos.z <= target_pos.z
            }
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
            spatial_grid,
            neurons,
            owner_type_mask,
            soma_idx,
            0.5, // TODO: type_affinity → вынести в CPU-конфиг
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

        // Crossing detection — аксон вышел за границы шарда?
        if shard_bounds.is_outside(x, y, z) {
            ghost_packet = Some(GhostPacket {
                origin_shard_id: 0, // текущий шард ID (0 в монорежиме)
                soma_idx,
                type_idx: type_idx_usize,
                entry_x: x.min(world_w_vox.saturating_sub(1)).min(1023),
                entry_y: y.min(world_d_vox.saturating_sub(1)).min(1023),
                entry_z: z.min(world_h_vox.saturating_sub(1)).min(255),
                entry_dir: forward_dir,
                remaining_steps: max_steps - step,
            });
            break; // Аксон в этом шарде заканчивается
        }

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

    let axon = GrownAxon {
        soma_idx,
        type_idx: type_idx_usize,
        tip_x: final_x,
        tip_y: final_y,
        tip_z: final_z,
        length_segments,
        segments,
    };

    (axon, ghost_packet)
}

/// Инициализировать axon_head по spec:
/// `axon_head = AXON_SENTINEL - length_segments * v_seg`
/// Это позволяет PropagateAxons в первый же тик корректно распространить сигнал.
pub fn init_axon_head(length_segments: u32, v_seg: u32) -> u32 {
    use genesis_core::constants::AXON_SENTINEL;
    AXON_SENTINEL.wrapping_sub(length_segments * v_seg)
}

/// Продолжает рост аксонов, пересёкших границу шарда (Ghost Axons).
///
/// Каждый GhostPacket описывает аксон, вошедший в этот шард через границу.
/// Ghost Axon продолжает рост с сохранением инерции (`entry_dir`) и
/// притяжением к нейронам текущего шарда.
///
/// - `soma_idx = usize::MAX` — нет локальной сомы, GSOP не применяется
/// - Если Ghost Axon снова пересекает границу → генерируется новый GhostPacket
///
/// Возвращает: `(grown_ghosts, outgoing_packets)`
pub fn inject_ghost_axons(
    ghost_packets: &[GhostPacket],
    neurons: &[PlacedNeuron],
    const_mem: &GenesisConstantMemory,
    sim: &SimulationConfig,
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (Vec<GrownAxon>, Vec<GhostPacket>) {
    let world_w_vox = sim.world.width_um / sim.simulation.voxel_size_um;
    let world_d_vox = sim.world.depth_um / sim.simulation.voxel_size_um;
    let world_h_vox = sim.world.height_um / sim.simulation.voxel_size_um;
    let segment_length_vox = sim.simulation.segment_length_voxels as f32;

    let spatial_grid = SpatialGrid::new(neurons);
    let mut grown = Vec::with_capacity(ghost_packets.len());
    let mut outgoing: Vec<GhostPacket> = Vec::new();

    for packet in ghost_packets {
        let _variant = &const_mem.variants[packet.type_idx.min(15)];
        // TODO: Steering params — вынести в VariantParameters или CPU-конфиг
        let fov_cos = (45.0_f32 / 2.0).to_radians().cos();
        let max_search_radius_vox = 100.0 / (sim.simulation.voxel_size_um as f32);
        let owner_type_mask = packet.type_idx as u8;

        let mut current_pos = Vec3::new(
            packet.entry_x as f32,
            packet.entry_y as f32,
            packet.entry_z as f32,
        );
        let mut forward_dir = packet.entry_dir;
        let mut segments = Vec::new();

        let ghost_seed = master_seed
            .wrapping_add(packet.soma_idx as u64)
            .wrapping_add(packet.origin_shard_id as u64);

        let mut exited_again = false;
        for step in 0..packet.remaining_steps {
            let v_attract = calculate_v_attract(
                current_pos,
                forward_dir,
                fov_cos,
                max_search_radius_vox,
                &spatial_grid,
                neurons,
                owner_type_mask,
                usize::MAX, // Ghost не имеет сомы — self-check никогда не сработает
                0.5, // TODO: type_affinity → вынести в CPU-конфиг
            );

            let s = ghost_seed.wrapping_add(step as u64);
            let v_noise = Vec3::new(
                random_f32(s) * 2.0 - 1.0,
                random_f32(s.wrapping_add(1)) * 2.0 - 1.0,
                random_f32(s.wrapping_add(2)) * 2.0 - 1.0,
            ).normalize_or_zero();

            // Нет v_global — аксон летит свободно: только инерция + притяжение + шум
            forward_dir = (
                forward_dir * 0.6
                + v_attract * 0.3
                + v_noise * 0.1
            ).normalize_or_zero();

            current_pos += forward_dir * segment_length_vox;

            let x = (current_pos.x.round() as u32).min(world_w_vox.saturating_sub(1)).min(1023);
            let y = (current_pos.y.round() as u32).min(world_d_vox.saturating_sub(1)).min(1023);
            let z = (current_pos.z.round() as u32).min(world_h_vox.saturating_sub(1)).min(255);
            let t = (owner_type_mask & 0x0F) as u32;

            // Повторный выход → пакет для следующего шарда
            if shard_bounds.is_outside(x, y, z) {
                outgoing.push(GhostPacket {
                    origin_shard_id: packet.origin_shard_id,
                    soma_idx: packet.soma_idx,
                    type_idx: packet.type_idx,
                    entry_x: x.min(world_w_vox.saturating_sub(1)).min(1023),
                    entry_y: y.min(world_d_vox.saturating_sub(1)).min(1023),
                    entry_z: z.min(world_h_vox.saturating_sub(1)).min(255),
                    entry_dir: forward_dir,
                    remaining_steps: packet.remaining_steps - step,
                });
                exited_again = true;
                break;
            }

            let packed = (t << 28) | (z << 20) | (y << 10) | x;
            if segments.last().copied() == Some(packed) {
                break; // Стагнация
            }
            segments.push(packed);
        }

        if segments.is_empty() && !exited_again {
            continue; // Ghost ничего не вырастил — пропускаем
        }

        let length_segments = segments.len() as u32;
        let (final_x, final_y, final_z) = if let Some(last) = segments.last() {
            ((last & 0x3FF), ((last >> 10) & 0x3FF), ((last >> 20) & 0xFF))
        } else {
            (packet.entry_x, packet.entry_y, packet.entry_z)
        };

        grown.push(GrownAxon {
            soma_idx: usize::MAX, // Ghost — нет локальной сомы
            type_idx: packet.type_idx,
            tip_x: final_x,
            tip_y: final_y,
            tip_z: final_z,
            length_segments,
            segments,
        });
    }

    (grown, outgoing)
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
// NOTE: grow_external_axons and grow_mock_retina have been removed.
// They will be replaced by the new Input Map Builder in `input_map.rs`
// as per Spec 05 §2.1.


#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn mock_bounds() -> ShardBounds {
        ShardBounds { x_start: 0, x_end: 100, y_start: 0, y_end: 100, z_start: 0, z_end: 100 }
    }

    #[test]
    fn test_shard_bounds_inside() {
        let b = mock_bounds();
        assert!(!b.is_outside(0, 0, 0));
        assert!(!b.is_outside(50, 50, 50));
        assert!(!b.is_outside(99, 99, 99));
    }

    #[test]
    fn test_shard_bounds_outside() {
        let b = mock_bounds();
        assert!(b.is_outside(100, 50, 50)); // x >= x_end
        assert!(b.is_outside(50, 100, 50)); // y >= y_end
        assert!(b.is_outside(50, 50, 100)); // z >= z_end
        assert!(b.is_outside(200, 200, 200));
    }

    #[test]
    fn test_ghost_packet_fields() {
        let pkt = GhostPacket {
            origin_shard_id: 1,
            soma_idx: usize::MAX,
            type_idx: 2,
            entry_x: 10, entry_y: 20, entry_z: 30,
            entry_dir: Vec3::Z,
            remaining_steps: 15,
        };
        assert_eq!(pkt.soma_idx, usize::MAX);
        assert_eq!(pkt.remaining_steps, 15);
        assert_eq!(pkt.type_idx, 2);
    }

    #[test]
    fn test_inject_empty_packets() {
        let neurons: Vec<PlacedNeuron> = vec![];
        use genesis_core::config::blueprints::{GenesisConstantMemory, VariantParameters};
        let empty_const_mem = GenesisConstantMemory {
            variants: [VariantParameters {
                threshold: 0, rest_potential: 0, leak_rate: 0, homeostasis_penalty: 0,
                gsop_potentiation: 0, gsop_depression: 0, homeostasis_decay: 0,
                signal_propagation_length: 0, conduction_velocity: 0,
                slot_decay_ltm: 0, slot_decay_wm: 0,
                refractory_period: 0, synapse_refractory_period: 0,
                inertia_curve: [0; 16], _reserved: [0; 16],
            }; 16],
        };
        let toml = r#"
            [world]
            width_um = 100
            depth_um = 100
            height_um = 100

            [simulation]
            tick_duration_us = 100
            total_ticks = 1000
            master_seed = "TEST"
            global_density = 0.05
            voxel_size_um = 25
            signal_speed_um_tick = 50
            sync_batch_ticks = 10
        "#;
        let sim = SimulationConfig::parse(toml).unwrap();
        
        let (grown, outgoing) = inject_ghost_axons(
            &[],
            &neurons,
            &empty_const_mem,
            &sim,
            &mock_bounds(),
            0,
        );
        assert!(grown.is_empty());
        assert!(outgoing.is_empty());
    }
}
