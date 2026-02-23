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
    let mut ranges = Vec::with_capacity(anatomy.layer.len());
    for layer in &anatomy.layer {
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

        // Cone Tracing setup
        let cone_seed = entity_seed(master_seed, soma_idx as u32);
        let segment_length_vox = nt.axon_growth_step as f32;
        let owner_type_mask = type_idx as u8; // We assume type_idx fits into 4 bits
        
        // Approximate specs fields
        let fov_cos = 0.866; // cos(30 deg) = 60 deg cone
        let max_search_radius_vox = 100.0 / (sim.simulation.voxel_size_um as f32); // e.g. 100 microns
        let weight_inertia = 0.6;
        let weight_sensor = 0.3;
        let weight_jitter = 0.1;

        // V_global (Goal)
        let mut target_pos = Vec3::new(soma_x as f32, soma_y as f32, tip_z as f32);
        
        let is_growing_up = tip_z >= soma_z;
        let mut forward_dir = if is_growing_up { Vec3::Z } else { Vec3::NEG_Z };
        let mut current_pos = Vec3::new(soma_x as f32, soma_y as f32, soma_z as f32);

        let mut segments = Vec::new();
        let max_steps = 1000;
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
            let z = (current_pos.z.round() as u32).min(255); // 8 bits
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::neuron_placement::place_neurons;
    use crate::bake::seed::{seed_from_str, DEFAULT_MASTER_SEED};
    use crate::parser::{anatomy, blueprints, simulation};
    use genesis_core::constants::AXON_SENTINEL;

    const SIM: &str = r#"
[world]
width_um = 3500
depth_um = 3500
height_um = 10250
[simulation]
tick_duration_us = 100
total_ticks = 10000
master_seed = "GENESIS"
global_density = 0.01
voxel_size_um = 25
signal_speed_um_tick = 50
sync_batch_ticks = 1000
"#;
    const ANATOMY: &str = r#"
[[layer]]
name = "L4"
height_pct = 0.40
population_pct = 0.55
[layer.composition]
"Vertical_Excitatory"   = 0.80
"Horizontal_Inhibitory" = 0.20

[[layer]]
name = "L2/3"
height_pct = 0.60
population_pct = 0.45
[layer.composition]
"Vertical_Excitatory"   = 0.85
"Horizontal_Inhibitory" = 0.15
"#;
    const BP: &str = r#"
[[neuron_type]]
name = "Vertical_Excitatory"
threshold = 42000
rest_potential = 10000
leak_rate = 1200
refractory_period = 15
synapse_refractory_period = 15
conduction_velocity = 200
signal_propagation_length = 10
axon_growth_step = 12
homeostasis_penalty = 5000
homeostasis_decay = 10
slot_decay_ltm = 160
slot_decay_wm = 96
sprouting_weight_distance = 0.5
sprouting_weight_power = 0.4
sprouting_weight_explore = 0.1

[[neuron_type]]
name = "Horizontal_Inhibitory"
threshold = 40000
rest_potential = 10000
leak_rate = 1500
refractory_period = 10
synapse_refractory_period = 5
conduction_velocity = 100
signal_propagation_length = 5
axon_growth_step = 10
homeostasis_penalty = 3000
homeostasis_decay = 15
slot_decay_ltm = 140
slot_decay_wm = 80
sprouting_weight_distance = 0.6
sprouting_weight_power = 0.3
sprouting_weight_explore = 0.1
"#;

    fn setup() -> (
        Vec<PlacedNeuron>,
        Vec<GrownAxon>,
        Vec<LayerZRange>,
        Vec<NeuronType>,
        SimulationConfig,
    ) {
        let sim = simulation::parse(SIM).unwrap();
        let an = anatomy::parse(ANATOMY).unwrap();
        let bp = blueprints::parse(BP).unwrap();
        let type_names: Vec<String> = bp.neuron_type.iter().map(|n| n.name.clone()).collect();
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        let neurons = place_neurons(&sim, &an, &type_names, master);
        let ranges = compute_layer_ranges(&an, &sim);
        let axons = grow_axons(&neurons, &ranges, &bp.neuron_type, &sim, master);
        (neurons, axons, ranges, bp.neuron_type, sim)
    }

    #[test]
    fn axon_count_matches_neuron_count() {
        let (neurons, axons, ..) = setup();
        assert_eq!(axons.len(), neurons.len());
    }

    #[test]
    fn tip_z_in_world_bounds() {
        let (_, axons, _, _, sim) = setup();
        let max_z = (sim.world.height_um / sim.simulation.voxel_size_um).min(255) as u32;
        for ax in axons.iter().take(500) {
            assert!(ax.tip_z <= max_z, "tip_z={} > max_z={}", ax.tip_z, max_z);
        }
    }

    #[test]
    fn init_axon_head_not_zero() {
        // Нулевой head вызвал бы эпилептический спайк в тик 0
        let v_seg = 2u32;
        let head = init_axon_head(10, v_seg);
        assert_ne!(head, 0, "axon_head must never be 0 at init");
        assert_ne!(head, AXON_SENTINEL, "active axon must not be SENTINEL");
    }

    #[test]
    fn growth_is_deterministic() {
        let (neurons, axons_a, ranges, nt, sim) = setup();
        let master = seed_from_str(DEFAULT_MASTER_SEED);
        let axons_b = grow_axons(&neurons, &ranges, &nt, &sim, master);
        assert!(
            axons_a
                .iter()
                .zip(axons_b.iter())
                .all(|(a, b)| a.tip_z == b.tip_z && a.tip_x == b.tip_x && a.tip_y == b.tip_y),
            "axon growth must be deterministic"
        );
    }
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

    for (channel_idx, channel) in io_config.input.iter().enumerate() {
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

            ext_axons.push(GrownAxon {
                soma_idx: usize::MAX, // Signifies external origin
                type_idx: channel.type_mask as usize, // Stuffs the 4-bit config phenotype
                tip_x,
                tip_y,
                tip_z,
                length_segments: 1, // Assume they just "entered" the shard at the border
                segments: vec![(channel.type_mask as u32) << 28 | (tip_z << 20) | (tip_y << 10) | tip_x],
            });
        }
    }

    ext_axons
}
