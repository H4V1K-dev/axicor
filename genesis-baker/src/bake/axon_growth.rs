use genesis_core::types::PackedPosition;
use crate::bake::seed::{entity_seed, random_f32};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use genesis_core::config::blueprints::{GenesisConstantMemory, NeuronType};
use crate::parser::simulation::SimulationConfig;
use crate::parser::anatomy::Anatomy;

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
    /// Last segment vector (for handover)
    pub last_dir: glam::Vec3,
}

/// Плоский контекст для единого цикла роста аксона
pub struct GrowthContext {
    pub current_pos_um: Vec3,
    pub current_pos_vox: Vec3,
    pub forward_dir: Vec3,
    pub target_pos: Option<Vec3>, // None для Ghost-аксонов (летят по инерции)
    
    pub remaining_steps: u32,
    pub owner_type_idx: u8,
    pub soma_idx: usize,          // usize::MAX для Ghost-аксонов
    pub origin_shard_id: u32,
}

pub struct SteeringWeights {
    pub global: f32,
    pub attract: f32,
    pub noise: f32,
}

/// Вычисляет следующий шаг аксона, смешивая градиенты, и возвращает (Непрерывную позицию, Квантованную позицию)
#[inline(always)]
pub fn step_and_pack(
    current_pos_um: Vec3,
    v_global: Vec3,
    v_attract: Vec3,
    weights: &SteeringWeights,
    step_size_um: f32,
    type_id: u8,
    rng: &mut ChaCha8Rng,
    voxel_size_um: f32,
) -> (Vec3, PackedPosition) {
    // 1. V_noise: Выход из локальных минимумов (WyHash/ChaCha джиттер)
    let theta = rng.gen_range(0.0..std::f32::consts::TAU);
    let phi = rng.gen_range(0.0..std::f32::consts::PI);
    let v_noise = Vec3::new(
        phi.sin() * theta.cos(),
        phi.sin() * theta.sin(),
        phi.cos()
    );

    // 2. Взвешенное смешивание (Steering)
    let mut v_final = (v_global * weights.global) 
                    + (v_attract * weights.attract) 
                    + (v_noise * weights.noise);

    // Защита от идеального математического нуля
    if v_final.length_squared() < 0.0001 {
        v_final = v_global;
    } else {
        v_final = v_final.normalize();
    }

    // 3. Шаг в непрерывном пространстве (сохраняем f32 для защиты от дрейфа сетки)
    let next_pos_um = current_pos_um + (v_final * step_size_um);

    // 4. Квантование (Zero-Cost конвертация в индексы вокселей)
    // Ограничиваем снизу нулем, чтобы избежать underflow при приведении к u32
    let x_idx = (next_pos_um.x / voxel_size_um).max(0.0) as u32;
    let y_idx = (next_pos_um.y / voxel_size_um).max(0.0) as u32;
    let z_idx = (next_pos_um.z / voxel_size_um).max(0.0) as u32;

    let packed = PackedPosition::pack_raw(x_idx, y_idx, z_idx, type_id);

    (next_pos_um, packed)
}

/// Единый физический конвейер роста. Используется и для локальных, и для Ghost аксонов.
pub fn execute_growth_loop(
    ctx: &mut GrowthContext,
    params: &crate::bake::cone_tracing::ConeParams,
    weights: &SteeringWeights,
    spatial_grid: &SpatialGrid,
    sim: &SimulationConfig,
    shard_bounds: &ShardBounds,
    mut rng: rand_chacha::ChaCha8Rng,
) -> (Vec<u32>, Option<GhostPacket>) {
    let mut segments = Vec::new();
    let mut ghost_packet = None;
    let voxel_size_um = sim.simulation.voxel_size_um as f32;
    let segment_length_um = sim.simulation.segment_length_voxels as f32 * voxel_size_um;

    let world_w_vox = (sim.world.width_um as f32 / voxel_size_um) as u32;
    let world_d_vox = (sim.world.depth_um as f32 / voxel_size_um) as u32;
    let world_h_vox = (sim.world.height_um as f32 / voxel_size_um) as u32;

    while ctx.remaining_steps > 0 {
        ctx.remaining_steps -= 1;

        // 1. Определение V_global
        let v_global = if let Some(target) = ctx.target_pos {
            (target - ctx.current_pos_vox).normalize_or_zero()
        } else {
            ctx.forward_dir // Ghost-аксоны используют инерцию как цель
        };

        // 2. Остановка по достижению Z-цели (только если есть target)
        if let Some(target) = ctx.target_pos {
            // H-аксоны или V-аксоны: проверяем достижение цели
            // (В оригинале была разная логика для H и V, тут пробуем унифицировать)
            let is_growing_up = target.z >= ctx.current_pos_vox.z;
            if (is_growing_up && ctx.current_pos_vox.z >= target.z) || 
               (!is_growing_up && ctx.current_pos_vox.z <= target.z) {
                break;
            }
        }

        // 3. Sensing
        let current_packed = PackedPosition::new(
            ctx.current_pos_vox.x.round() as u32,
            ctx.current_pos_vox.y.round() as u32,
            ctx.current_pos_vox.z.round() as u32,
            ctx.owner_type_idx
        );
        
        let v_attract = calculate_v_attract(
            current_packed, ctx.forward_dir, params, spatial_grid, voxel_size_um
        );

        // 4. Steering & Step
        let (next_pos_um, next_packed) = step_and_pack(
            ctx.current_pos_um, v_global, v_attract, weights, 
            segment_length_um, ctx.owner_type_idx, &mut rng, voxel_size_um
        );

        ctx.current_pos_um = next_pos_um;
        ctx.forward_dir = (next_pos_um - (ctx.current_pos_vox * voxel_size_um)).normalize_or_zero();
        ctx.current_pos_vox = next_pos_um / voxel_size_um;

        let x = next_packed.x() as u32;
        let y = next_packed.y() as u32;
        let z = next_packed.z() as u32;

        // 5. Границы шарда
        if shard_bounds.is_outside(x, y, z) {
            ghost_packet = Some(GhostPacket {
                origin_shard_id: ctx.origin_shard_id,
                soma_idx: ctx.soma_idx,
                type_idx: ctx.owner_type_idx as usize,
                entry_x: x.min(world_w_vox.saturating_sub(1)),
                entry_y: y.min(world_d_vox.saturating_sub(1)),
                entry_z: z.min(world_h_vox.saturating_sub(1)),
                entry_dir: ctx.forward_dir,
                remaining_steps: ctx.remaining_steps,
            });
            break;
        }

        // 6. Защита от стагнации
        if segments.last().copied() == Some(next_packed.0) {
            break; 
        }

        segments.push(next_packed.0);
    }

    (segments, ghost_packet)
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
        let voxel_um = sim.simulation.voxel_size_um;
        Self {
            x_start: 0,
            x_end: (sim.world.width_um as f32 / voxel_um) as u32,
            y_start: 0,
            y_end: (sim.world.depth_um as f32 / voxel_um) as u32,
            z_start: 0,
            z_end: (sim.world.height_um as f32 / voxel_um) as u32,
        }
    }

    /// [DOD FIX] Реальные границы конкретной зоны из shard.toml (в вокселях).
    /// Аксон, пересёкший x_end, корректно создаст GhostPacket вместо
    /// бесконечного блуждания до края глобального мира.
    pub fn from_config(cfg: &genesis_core::config::InstanceConfig) -> Self {
        Self {
            x_start: cfg.world_offset.x,
            x_end:   cfg.world_offset.x + cfg.dimensions.w,
            y_start: cfg.world_offset.y,
            y_end:   cfg.world_offset.y + cfg.dimensions.d,
            z_start: cfg.world_offset.z,
            z_end:   cfg.world_offset.z + cfg.dimensions.h,
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
    let world_h_vox = (sim.world.height_um as f32 / voxel_um) as u32;
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
    positions: &[PackedPosition],
    layer_ranges: &[LayerZRange],
    types: &[NeuronType],
    sim: &SimulationConfig,
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (Vec<GrownAxon>, Vec<GhostPacket>) {
    let voxel_um = sim.simulation.voxel_size_um;
    let world_w_vox = (sim.world.width_um as f32 / voxel_um) as u32;
    let world_d_vox = (sim.world.depth_um as f32 / voxel_um) as u32;
    let world_h_vox = (sim.world.height_um as f32 / voxel_um) as u32;

    use rayon::prelude::*;

    let max_search_radius_vox = sim.simulation.segment_length_voxels as f32 * 3.0;
    let spatial_grid = SpatialGrid::new(positions.to_vec(), f32::max(1.0, max_search_radius_vox.ceil()) as u32);

    let results: Vec<(GrownAxon, Option<GhostPacket>)> = positions
        .par_iter()
        .enumerate()
        .filter(|(_, pos)| pos.0 != 0) // Skip dummy warp-aligned neurons
        .map(|(soma_idx, pos)| {
            let soma_z = pos.z() as u32;
            let soma_x = pos.x() as u32;
            let soma_y = pos.y() as u32;
            let type_idx = pos.type_id();

            grow_single_axon(
                soma_x, soma_y, soma_z,
                soma_idx,
                type_idx as u8,
                types,
                sim,
                world_w_vox, world_d_vox, world_h_vox,
                layer_ranges,
                &spatial_grid,
                shard_bounds,
                master_seed,
            )
        })
        .collect();

    let (axons, ghost_packets_opts): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    let ghost_packets = ghost_packets_opts.into_iter().flatten().collect();

    (axons, ghost_packets)
}

#[allow(clippy::too_many_arguments)]
pub fn grow_single_axon(
    soma_x: u32, soma_y: u32, soma_z: u32,
    soma_idx: usize,
    type_idx: u8,
    types: &[NeuronType],
    sim: &SimulationConfig,
    world_w_vox: u32, world_d_vox: u32, world_h_vox: u32,
    layer_ranges: &[LayerZRange],
    spatial_grid: &SpatialGrid,
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (GrownAxon, Option<GhostPacket>) {
    use crate::bake::cone_tracing::ConeParams;

    // Чтение параметров роста за O(1) из плоского массива:
    let type_params = &types[type_idx as usize];

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
            if layer_ranges.len() == 1 {
                // Одиночный слой: растём к противоположной границе мира
                if soma_rel_z < 0.5 {
                    (world_h_vox, world_h_vox.saturating_add(1))
                } else {
                    (0, 1)
                }
            } else {
                // Если слоёв много, а мы в верхнем — тянемся вниз к самому первому
                layer_ranges
                    .first()
                    .map_or((0u32, 1u32), |l| (l.z_start_vox, l.z_end_vox))
            }
        }
    };

    // 4. Target_Z = target_z_start + Soma_Rel_Z × target_height
    // [DOD Stage 45.2] If target_layer is None, we use Isotropic (Inertial) growth.
    let (tip_z, has_target_z) = match target_layer {
        Some(_) => {
            let target_h = (target_z_end - target_z_start).max(1) as f32;
            let tz = (target_z_start as f32 + (soma_rel_z * target_h).round()) as u32;
            (tz.clamp(target_z_start, target_z_end).min(255), true)
        }
        None => (soma_z, false), // No Z-target in isotropic mode
    };

    // Global segment length from config (fixed for all types)
    let segment_length_vox = sim.simulation.segment_length_voxels as f32;
    let cone_seed = entity_seed(master_seed, soma_idx as u32);
    let _owner_type_mask = type_idx; // 4-bit mask
    
    let fov_cos = (type_params.steering_fov_deg / 2.0).to_radians().cos();
    let _max_search_radius_vox = type_params.steering_radius_um / sim.simulation.voxel_size_um as f32;
    let weights = SteeringWeights {
        global: type_params.steering_weight_inertia,
        attract: type_params.steering_weight_sensor,
        noise: type_params.steering_weight_jitter,
    };

    let bias = type_params.growth_vertical_bias;
    let type_idx_usize = type_idx as usize;
    let is_horizontal = bias < 0.1; // Only pure horizontal if bias is very low (e.g. 0.0)


    let current_pos = Vec3::new(soma_x as f32, soma_y as f32, soma_z as f32);
    let voxel_size_um = sim.simulation.voxel_size_um as f32;
    let _segment_length_um = segment_length_vox * voxel_size_um;
    let current_pos_um = current_pos * voxel_size_um;
    let is_growing_up = tip_z >= soma_z;

    let (forward_dir, target_pos) = if is_horizontal {
        // Случайный радиальный вектор в XY
        let horiz_seed = entity_seed(master_seed, soma_idx as u32 + 0x48_4F_52_5A); // "HORZ"
        let angle = random_f32(horiz_seed) * std::f32::consts::TAU; // 0..2π
        let dir = Vec3::new(angle.cos(), angle.sin(), 0.0);
        
        // target_pos не используется для остановки H-нейронов, но используется
        // для генерации константного v_global на каждом шаге. 
        // Cоздаем целевую точку далеко в выбранном направлении.
        let far_target = current_pos + dir * 5000.0;
        (dir, Some(far_target))
    } else if has_target_z {
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
        (final_dir, Some(t_pos))
    } else {
        // [DOD Stage 45.2] Isotropic mode: Use inherent forward dir (inertia)
        let horiz_seed = entity_seed(master_seed, soma_idx as u32 + 0x48_4F_52_5A);
        let angle = random_f32(horiz_seed) * std::f32::consts::TAU;
        let pitch = (random_f32(horiz_seed.wrapping_mul(123456789)) - 0.5) * std::f32::consts::PI;
        let dir = Vec3::new(
            pitch.cos() * angle.cos(),
            pitch.cos() * angle.sin(),
            pitch.sin()
        ).normalize_or_zero();
        (dir, None) // No target_pos -> execute_growth_loop uses forward_dir
    };

    let params = ConeParams {
        radius_um: type_params.steering_radius_um,
        fov_cos,
        owner_type: type_idx,                       // [DOD] Сырой 4-битный тип
        type_affinity: type_params.type_affinity,   // Читаем из Blueprint
    };

    let mut ctx = GrowthContext {
        current_pos_um,
        current_pos_vox: current_pos,
        forward_dir,
        target_pos,
        remaining_steps: sim.simulation.axon_growth_max_steps,
        owner_type_idx: type_idx,
        soma_idx,
        origin_shard_id: 0,
    };

    let (segments, ghost_packet) = execute_growth_loop(
        &mut ctx,
        &params,
        &weights,
        spatial_grid,
        sim,
        shard_bounds,
        ChaCha8Rng::seed_from_u64(cone_seed),
    );

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
        last_dir: ctx.forward_dir,
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
    positions: &[PackedPosition],
    _const_mem: &GenesisConstantMemory,
    sim: &SimulationConfig,
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (Vec<GrownAxon>, Vec<GhostPacket>) {
    let voxel_um = sim.simulation.voxel_size_um;

    let max_search_radius_vox = sim.simulation.segment_length_voxels as f32 * 3.0;
    let spatial_grid = SpatialGrid::new(positions.to_vec(), f32::max(1.0, max_search_radius_vox.ceil()) as u32);
    let mut grown = Vec::with_capacity(ghost_packets.len());
    let mut outgoing: Vec<GhostPacket> = Vec::new();

    for packet in ghost_packets {
        // TODO: Steering params — вынести в VariantParameters или CPU-конфиг
        let fov_cos = (45.0_f32 / 2.0).to_radians().cos();
        let max_search_radius_vox = sim.simulation.segment_length_voxels as f32 * 4.0;

        let current_pos = Vec3::new(
            packet.entry_x as f32,
            packet.entry_y as f32,
            packet.entry_z as f32,
        );
        let current_pos_um = current_pos * voxel_um as f32;
        let forward_dir = packet.entry_dir;

        let ghost_seed = master_seed
            .wrapping_add(packet.soma_idx as u64)
            .wrapping_add(packet.origin_shard_id as u64);
            
        let rng = ChaCha8Rng::seed_from_u64(ghost_seed);

        use crate::bake::cone_tracing::ConeParams;
        let params = ConeParams {
            radius_um: max_search_radius_vox * voxel_um as f32,
            fov_cos,
            owner_type: packet.type_idx as u8,
            type_affinity: 0.5, // Ghost-аксоны: нейтральное сродство
        };
        let weights = SteeringWeights {
            global: 0.6,
            attract: 0.3,
            noise: 0.1,
        };

        let mut ctx = GrowthContext {
            current_pos_um,
            current_pos_vox: current_pos,
            forward_dir,
            target_pos: None, // Ghost-аксоны летят по инерции
            remaining_steps: packet.remaining_steps,
            owner_type_idx: packet.type_idx as u8,
            soma_idx: packet.soma_idx,
            origin_shard_id: packet.origin_shard_id,
        };

        let (segments, maybe_outgoing) = execute_growth_loop(
            &mut ctx,
            &params,
            &weights,
            &spatial_grid,
            sim,
            shard_bounds,
            rng,
        );

        let has_outgoing = maybe_outgoing.is_some();
        if let Some(pkt) = maybe_outgoing {
            outgoing.push(pkt);
        }

        if segments.is_empty() && !has_outgoing {
            continue; 
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
            last_dir: ctx.forward_dir,
        });
    }

    (grown, outgoing)
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
        let positions: Vec<PackedPosition> = vec![];
        use genesis_core::layout::{VariantParameters};
        use genesis_core::config::blueprints::{GenesisConstantMemory};
        let empty_const_mem = GenesisConstantMemory {
            variants: [VariantParameters {
                threshold: 0, rest_potential: 0, leak_rate: 0, homeostasis_penalty: 0,
                gsop_potentiation: 0, gsop_depression: 0, homeostasis_decay: 0,
                signal_propagation_length: 0,
                slot_decay_ltm: 0, slot_decay_wm: 0,
                refractory_period: 0, synapse_refractory_period: 0,
                ..VariantParameters::default()
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
            voxel_size_um = 25.0
            signal_speed_m_s = 0.5
            sync_batch_ticks = 10
        "#;
        let sim = SimulationConfig::parse(toml).unwrap();
        
        let (grown, outgoing) = inject_ghost_axons(
            &[],
            &positions,
            &empty_const_mem,
            &sim,
            &mock_bounds(),
            0,
        );
        assert!(grown.is_empty());
        assert!(outgoing.is_empty());
    }
}

// =============================================================================
// § Half-Duplex SHM Bridge: Входящие межзональные аксоны
// =============================================================================

/// Конвертирует сетевые AxonHandoverEvent (из SHM) в физические GhostPacket
/// и продолжает их рост внутри нового шарда.
pub fn inject_handover_events(
    handovers: &[genesis_core::ipc::AxonHandoverEvent],
    positions: &[PackedPosition],
    _layer_ranges: &[LayerZRange],
    types: &[genesis_core::config::blueprints::NeuronType],
    sim: &crate::parser::simulation::SimulationConfig,
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (Vec<GrownAxon>, Vec<GhostPacket>) {
    if handovers.is_empty() {
        return (vec![], vec![]);
    }

    let voxel_um = sim.simulation.voxel_size_um;
    // SpatialGrid для притяжения к существующим нейронам
    let search_r = (sim.simulation.segment_length_voxels as f32 * 3.0).max(1.0).ceil() as u32;
    let spatial_grid = crate::bake::spatial_grid::SpatialGrid::new(positions.to_vec(), search_r);

    let mut grown_axons = Vec::with_capacity(handovers.len());
    let mut outgoing_ghosts = Vec::new();

    for (idx, ev) in handovers.iter().enumerate() {
        let raw_dir = glam::Vec3::new(
            ev.vector_x as f32 / 127.0,
            ev.vector_y as f32 / 127.0,
            ev.vector_z as f32 / 127.0,
        );

        let type_idx = (ev.type_mask as usize).min(types.len().saturating_sub(1));

        // Точка входа в глобальных координатах
        let entry_x = ev.entry_x as u32 + shard_bounds.x_start;
        let entry_y = ev.entry_y as u32 + shard_bounds.y_start;
        let entry_z = if ev.vector_z > 0 {
            shard_bounds.z_start 
        } else {
            shard_bounds.z_end.saturating_sub(1)
        };

        // Seed детерминированный
        let ghost_seed = master_seed.wrapping_add(idx as u64).wrapping_add(0x4748_5354_0000_0000);

        let entry_dir = if raw_dir.length_squared() > 0.001 {
            raw_dir.normalize()
        } else {
            glam::Vec3::Z
        };

        let mut ctx = GrowthContext {
            current_pos_um: Vec3::new(entry_x as f32, entry_y as f32, entry_z as f32) * voxel_um as f32,
            current_pos_vox: Vec3::new(entry_x as f32, entry_y as f32, entry_z as f32),
            forward_dir: entry_dir,
            target_pos: None, // Handover: летим по инерции пока не поймаем гравитацию (или можно задать цель)
            remaining_steps: sim.simulation.axon_growth_max_steps, // TODO: брать из события
            owner_type_idx: type_idx as u8,
            soma_idx: usize::MAX, 
            origin_shard_id: 0, // TODO: брать из события
        };

        // Т.к. params зависят от типа, достанем их:
        let t_params = &types[type_idx];
        let fov_cos = (t_params.steering_fov_deg / 2.0).to_radians().cos();
        let weights = SteeringWeights {
            global: t_params.steering_weight_inertia,
            attract: t_params.steering_weight_sensor,
            noise: t_params.steering_weight_jitter,
        };
        let params = crate::bake::cone_tracing::ConeParams {
            radius_um: t_params.steering_radius_um,
            fov_cos,
            owner_type: type_idx as u8,
            type_affinity: t_params.type_affinity,
        };

        let (segments, maybe_outgoing) = execute_growth_loop(
            &mut ctx,
            &params,
            &weights,
            &spatial_grid,
            sim,
            shard_bounds,
            rand_chacha::ChaCha8Rng::seed_from_u64(ghost_seed),
        );

        let length_segments = segments.len() as u32;
        let (final_x, final_y, final_z) = if let Some(last) = segments.last() {
            ((last & 0x3FF), ((last >> 10) & 0x3FF), ((last >> 20) & 0xFF))
        } else {
            (entry_x, entry_y, entry_z)
        };

        grown_axons.push(GrownAxon {
            soma_idx: usize::MAX,
            type_idx,
            tip_x: final_x,
            tip_y: final_y,
            tip_z: final_z,
            length_segments,
            segments,
            last_dir: ctx.forward_dir,
        });

        if let Some(outgoing) = maybe_outgoing {
            outgoing_ghosts.push(outgoing);
        }
    }

    (grown_axons, outgoing_ghosts)
}
