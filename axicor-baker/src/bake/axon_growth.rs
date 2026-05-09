use crate::bake::seed::{entity_seed, random_f32};
use crate::parser::anatomy::Anatomy;
use crate::parser::simulation::SimulationConfig;
use axicor_core::config::blueprints::NeuronType;
use axicor_core::types::PackedPosition;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use glam::Vec3;
use crate::bake::cone_tracing::{calculate_v_attract, ConeParams};
use crate::bake::spatial_grid::SpatialGrid;

/// Grown axon ready for writing to ShardStateSoA.
#[derive(Debug, Clone)]
pub struct GrownAxon {
    /// Soma index in the neuron array
    pub soma_idx: usize,
    /// Neuron type (copied for dendritic filtering without lookup)
    pub type_idx: usize,
    /// Tip position  axon endpoint (where dendrites are sought)
    pub tip_x: u32,
    pub tip_y: u32,
    pub tip_z: u32,
    /// Axon length in segments (for axon_head initialization)
    pub length_segments: u32,
    /// Piecewise-linear geometry (PackedPositions: Type|Z|Y|X)
    pub segments: Vec<u32>,
    /// Last segment vector (for handover)
    pub last_dir: glam::Vec3,
}

/// Flat context for a single axon growth cycle
pub struct GrowthContext {
    pub current_pos_um: Vec3,
    pub current_pos_vox: Vec3,
    pub forward_dir: Vec3,
    pub target_pos: Option<Vec3>, // None for Ghost axons (fly by inertia)

    pub remaining_steps: u32,
    pub owner_type_idx: u8,
    pub soma_idx: usize, // usize::MAX for Ghost axons
    pub origin_shard_id: u32,
}

pub struct SteeringWeights {
    pub global: f32,
    pub attract: f32,
    pub noise: f32,
}

/// Calculates the next axon step by mixing gradients and returns (Continuous position, Quantized position)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
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
    // 1. V_noise: Escape from local minima (WyHash/ChaCha jitter)
    // [DOD FIX] Uniform sphere sampling (Malley's Method)
    // Sample cos(phi) uniformly in [-1, 1] to avoid Z-Pole density clustering
    let theta = rng.gen_range(0.0..std::f32::consts::TAU);
    let u: f32 = rng.gen_range(-1.0_f32..1.0_f32);
    let r = (1.0 - u * u).sqrt();
    let v_noise = Vec3::new(r * theta.cos(), r * theta.sin(), u);

    // 2. Weighted steering mix
    let mut v_final =
        (v_global * weights.global) + (v_attract * weights.attract) + (v_noise * weights.noise);

    // Protection against mathematical zero
    if v_final.length_squared() < 0.0001 {
        v_final = v_global;
    } else {
        v_final = v_final.normalize();
    }

    // 3. Step in continuous space (retain f32 to protect against grid drift)
    let next_pos_um = current_pos_um + (v_final * step_size_um);

    // 4. Quantization (Zero-Cost conversion to voxel indices)
    // Clamp bottom to zero to avoid underflow when casting to u32
    // [DOD FIX] Round-to-nearest prevents truncation bias and false stagnation
    let x_idx = (next_pos_um.x / voxel_size_um).round().max(0.0) as u32;
    let y_idx = (next_pos_um.y / voxel_size_um).round().max(0.0) as u32;
    let z_idx = (next_pos_um.z / voxel_size_um).round().max(0.0) as u32;

    // [DOD FIX] Strict clamping to prevent bit overflow into adjacent coordinate fields
    let packed = PackedPosition::pack_raw(
        x_idx.min(1023),
        y_idx.min(1023),
        z_idx.min(255),
        type_id,
    );

    (next_pos_um, packed)
}

// [DOD FIX] Strict State Machine for Axon Growth
pub enum GrowthEvent {
    Advanced(u32),                  // Успешный шаг (возвращает raw u32 PackedPosition)
    TargetReached,                  // Достиг цели по Z
    Stagnated,                      // Застрял (дистанция < epsilon или уперся в границу)
    OutOfBounds(GhostPacket),       // Покинул шард (уходит в сеть)
}

/// Unified physical growth pipeline. Used for both local and Ghost axons.
pub fn execute_growth_loop(
    ctx: &mut GrowthContext,
    params: &ConeParams,
    weights: &SteeringWeights,
    spatial_grid: &SpatialGrid,
    sim: &SimulationConfig,
    shard_bounds: &ShardBounds,
    mut rng: rand_chacha::ChaCha8Rng,
) -> (Vec<u32>, Option<GhostPacket>) {
    let mut segments = Vec::new();
    let mut ghost_packet = None;

    while ctx.remaining_steps > 0 {
        ctx.remaining_steps -= 1;
        
        let event = compute_growth_step(ctx, params, weights, spatial_grid, sim, shard_bounds, &mut rng, segments.last().copied());
        
        match event {
            GrowthEvent::Advanced(packed_pos) => segments.push(packed_pos),
            GrowthEvent::TargetReached | GrowthEvent::Stagnated => break,
            GrowthEvent::OutOfBounds(packet) => {
                ghost_packet = Some(packet);
                break;
            }
        }
    }
    (segments, ghost_packet)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_growth_step(
    ctx: &mut GrowthContext,
    params: &ConeParams,
    weights: &SteeringWeights,
    spatial_grid: &SpatialGrid,
    sim: &SimulationConfig,
    shard_bounds: &ShardBounds,
    rng: &mut ChaCha8Rng,
    last_packed: Option<u32>,
) -> GrowthEvent {
    let voxel_size_um = sim.simulation.voxel_size_um;
    let segment_length_um = sim.simulation.segment_length_voxels as f32 * voxel_size_um;

    let world_w_vox = (sim.world.width_um as f32 / voxel_size_um) as u32;
    let world_d_vox = (sim.world.depth_um as f32 / voxel_size_um) as u32;
    let world_h_vox = (sim.world.height_um as f32 / voxel_size_um) as u32;

    // 1. Determine V_global
    let v_global = if let Some(target) = ctx.target_pos {
        (target - ctx.current_pos_vox).normalize_or_zero()
    } else {
        ctx.forward_dir // Ghost axons use inertia as target
    };

    // 2. Stop upon reaching Z-target (only if target exists)
    if let Some(target) = ctx.target_pos {
        let is_growing_up = target.z >= ctx.current_pos_vox.z;
        if (is_growing_up && ctx.current_pos_vox.z >= target.z)
            || (!is_growing_up && ctx.current_pos_vox.z <= target.z)
        {
            return GrowthEvent::TargetReached;
        }
    }

    // 3. Sensing
    // [DOD FIX] Round-to-nearest for spatial grid lookup consistency
    let current_packed = PackedPosition::new(
        ctx.current_pos_vox.x.round() as u32,
        ctx.current_pos_vox.y.round() as u32,
        ctx.current_pos_vox.z.round() as u32,
        ctx.owner_type_idx,
    );

    let v_attract = calculate_v_attract(
        current_packed,
        ctx.forward_dir,
        params,
        spatial_grid,
        voxel_size_um,
    );

    // 4. Steering & Step
    let (next_pos_um, next_packed) = step_and_pack(
        ctx.current_pos_um,
        v_global,
        v_attract,
        weights,
        segment_length_um,
        ctx.owner_type_idx,
        rng,
        voxel_size_um,
    );

    // [DOD FIX] Calculate true direction vector before mutating current state
    ctx.forward_dir = (next_pos_um - ctx.current_pos_um).normalize_or_zero();
    ctx.current_pos_um = next_pos_um;
    ctx.current_pos_vox = next_pos_um / voxel_size_um;

    // [DOD FIX] Round-to-nearest for correct OutOfBounds boundary detection
    let raw_x = ctx.current_pos_vox.x.round().max(0.0) as u32;
    let raw_y = ctx.current_pos_vox.y.round().max(0.0) as u32;
    let raw_z = ctx.current_pos_vox.z.round().max(0.0) as u32;

    // 5. Shard boundaries
    if shard_bounds.is_outside(raw_x, raw_y, raw_z) {
        return GrowthEvent::OutOfBounds(GhostPacket {
            origin_shard_id: ctx.origin_shard_id,
            soma_idx: ctx.soma_idx,
            type_idx: ctx.owner_type_idx as usize,
            entry_x: raw_x.min(world_w_vox.saturating_sub(1)),
            entry_y: raw_y.min(world_d_vox.saturating_sub(1)),
            entry_z: raw_z.min(world_h_vox.saturating_sub(1)),
            entry_dir: ctx.forward_dir,
            remaining_steps: ctx.remaining_steps,
        });
    }

    // 6. Stagnation protection
    if last_packed == Some(next_packed.0) {
        return GrowthEvent::Stagnated;
    }

    GrowthEvent::Advanced(next_packed.0)
}

/// Cache of layer Z-ranges (calculated once from anatomy + sim)
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LayerZRange {
    pub name: String,
    pub z_start_vox: u32,
    pub z_end_vox: u32,
}

/// Shard boundaries in world coordinates (voxels).
/// Axons crossing this boundary become Ghost Axons in a neighboring shard.
/// In mono-mode (single shard), pass `ShardBounds::full_world(&sim)`
/// crossing will never occur and behavior remains identical.
#[derive(Debug, Clone)]
pub struct ShardBounds {
    pub x_start: u32,
    pub x_end: u32,
    pub y_start: u32,
    pub y_end: u32,
    pub z_start: u32,
    pub z_end: u32,
}

impl ShardBounds {
    /// Single shard = entire world volume. Ghost Packets will not be generated.
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

    /// [DOD FIX] Real boundaries for a specific zone from shard.toml (in voxels).
    /// Axons crossing x_end will correctly create a GhostPacket instead of
    /// wandering indefinitely to the edge of the global world.
    pub fn from_config(cfg: &axicor_core::config::InstanceConfig) -> Self {
        Self {
            x_start: cfg.world_offset.x,
            x_end: cfg.world_offset.x + cfg.dimensions.w,
            y_start: cfg.world_offset.y,
            y_end: cfg.world_offset.y + cfg.dimensions.d,
            z_start: cfg.world_offset.z,
            z_end: cfg.world_offset.z + cfg.dimensions.h,
        }
    }

    /// Checks if a point has exited shard boundaries.
    #[inline]
    pub fn is_outside(&self, x: u32, y: u32, z: u32) -> bool {
        x < self.x_start
            || x >= self.x_end
            || y < self.y_start
            || y >= self.y_end
            || z < self.z_start
            || z >= self.z_end
    }
}

/// Inter-shard transmission packet.
/// Generated when an axon in shard A crosses its boundary.
/// Shard B receives this packet and creates a Ghost Axon on its side of the boundary.
#[derive(Debug, Clone)]
pub struct GhostPacket {
    /// Source shard ID (for routing in multi-shard topologies)
    pub origin_shard_id: u32,
    /// Axon without local soma  usize::MAX
    pub soma_idx: usize,
    /// Neuron type (for whitelist, affinity, etc.)
    pub type_idx: usize,
    /// Entry point into the new shard (already in its coordinates)
    pub entry_x: u32,
    pub entry_y: u32,
    pub entry_z: u32,
    /// Direction at the moment of crossing (to maintain inertia)
    pub entry_dir: Vec3,
    /// Number of steps remaining until growth completion
    pub remaining_steps: u32,
}

/// Precalculates Z-ranges of all layers in voxels.
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

/// Cone Tracing: calculates the axon endpoint for each neuron.
///
/// Algorithm (04_connectivity.md 1.3):
/// 1. Find home layer by soma Z-coordinate.
/// 2. `Soma_Rel_Z = (soma_z - layer_z_start) / layer_height`
/// 3. Target layer = next layer up (Z+), unless in the topmost layer.
///    (For initial Baking  each neuron seeks the nearest layer above)
/// 4. `Target_Z = target_z_start + Soma_Rel_Z  target_height`
/// 5. XY: small cone drift (FOV) relative to original XY position.
///    `tip_x = soma_x + x`, where `|x|  cone_radius`
/// 6. Axon length = |target_z - soma_z| + 1 (in segments/voxels)
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
    let spatial_grid = SpatialGrid::new(
        positions.to_vec(),
        f32::max(1.0, max_search_radius_vox.ceil()) as u32,
    );

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
                soma_x,
                soma_y,
                soma_z,
                soma_idx,
                type_idx,
                types,
                sim,
                world_w_vox,
                world_d_vox,
                world_h_vox,
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
    soma_x: u32,
    soma_y: u32,
    soma_z: u32,
    soma_idx: usize,
    type_idx: u8,
    types: &[NeuronType],
    sim: &SimulationConfig,
    world_w_vox: u32,
    world_d_vox: u32,
    world_h_vox: u32,
    layer_ranges: &[LayerZRange],
    spatial_grid: &SpatialGrid,
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (GrownAxon, Option<GhostPacket>) {
    // Removed local import, moved to top.

    // O(1) growth parameter lookup from flat array:
    let type_params = &types[type_idx as usize];

    // 1. Find home layer (Index_home)
    let home_layer = layer_ranges
        .iter()
        .find(|l| soma_z >= l.z_start_vox && soma_z < l.z_end_vox);
    let (home_z_start, home_z_end) = match home_layer {
        Some(l) => (l.z_start_vox, l.z_end_vox),
        None => (soma_z, soma_z + 1), // fallback, if soma is outside defined layers
    };

    // 2. Soma_Rel_Z  relative position in home layer [0.0, 1.0)
    let layer_h = (home_z_end - home_z_start).max(1) as f32;
    let soma_rel_z = (soma_z.saturating_sub(home_z_start) as f32) / layer_h;

    // 3. Target layer  next one up along Z (index + 1)
    let target_layer = layer_ranges.iter().find(|l| l.z_start_vox > soma_z);
    let (target_z_start, target_z_end) = match target_layer {
        Some(l) => (l.z_start_vox, l.z_end_vox),
        None => {
            if layer_ranges.len() == 1 {
                // Single layer: grow toward opposite world boundary
                if soma_rel_z < 0.5 {
                    (world_h_vox, world_h_vox.saturating_add(1))
                } else {
                    (0, 1)
                }
            } else {
                // If multiple layers and we're in the topmost  pull down to the very first
                layer_ranges
                    .first()
                    .map_or((0u32, 1u32), |l| (l.z_start_vox, l.z_end_vox))
            }
        }
    };

    // 4. Target_Z = target_z_start + Soma_Rel_Z  target_height
    let target_h = (target_z_end - target_z_start).max(1) as f32;
    let tip_z = (target_z_start as f32 + (soma_rel_z * target_h).round()) as u32; // Use round to avoid truncation
    let tip_z = tip_z.clamp(target_z_start, target_z_end).min(255);

    // Global segment length from config (fixed for all types)
    let segment_length_vox = sim.simulation.segment_length_voxels as f32;
    let cone_seed = entity_seed(master_seed, soma_idx as u32);
    let _owner_type_mask = type_idx; // 4-bit mask

    let fov_cos = (type_params.steering_fov_deg / 2.0).to_radians().cos();
    let _max_search_radius_vox =
        type_params.steering_radius_um / sim.simulation.voxel_size_um;
    let weights = SteeringWeights {
        global: type_params.steering_weight_inertia,
        attract: type_params.steering_weight_sensor,
        noise: type_params.steering_weight_jitter,
    };

    let bias = type_params.growth_vertical_bias;
    let type_idx_usize = type_idx as usize;
    let is_horizontal = bias < 0.1; // Only pure horizontal if bias is very low (e.g. 0.0)

    let current_pos = Vec3::new(soma_x as f32, soma_y as f32, soma_z as f32);
    let voxel_size_um = sim.simulation.voxel_size_um;
    let _segment_length_um = segment_length_vox * voxel_size_um;
    let current_pos_um = current_pos * voxel_size_um;
    let is_growing_up = tip_z >= soma_z;

    let (forward_dir, target_pos) = if is_horizontal {
        // Random radial vector in XY
        let horiz_seed = entity_seed(master_seed, soma_idx as u32 + 0x48_4F_52_5A); // "HORZ"
        let angle = random_f32(horiz_seed) * std::f32::consts::TAU; // 0..2
        let dir = Vec3::new(angle.cos(), angle.sin(), 0.0);

        // target_pos is not used to stop H-neurons, but is used
        // to generate constant v_global at each step.
        // Create target point far in the chosen direction.
        let far_target = current_pos + dir * 5000.0;
        (dir, far_target)
    } else {
        // Vertical target: target layer along Z
        let v_vertical_target = Vec3::new(soma_x as f32, soma_y as f32, tip_z as f32);

        // Horizontal component (limited)
        let horiz_seed = entity_seed(master_seed, soma_idx as u32 + 0x48_4F_52_5A);
        let target_x = random_f32(horiz_seed) * world_w_vox as f32;
        let target_y =
            random_f32(horiz_seed.wrapping_mul(6364136223846793005)) * world_d_vox as f32;
        let v_horizontal_target = Vec3::new(target_x, target_y, soma_z as f32);

        let t_pos = v_vertical_target * bias + v_horizontal_target * (1.0 - bias);
        let dir = (t_pos - current_pos).normalize_or_zero();
        let final_dir = if dir.length_squared() < 0.01 {
            if is_growing_up {
                Vec3::Z
            } else {
                Vec3::NEG_Z
            }
        } else {
            dir
        };
        (final_dir, t_pos)
    };

    let params = ConeParams {
        radius_um: type_params.steering_radius_um,
        fov_cos,
        owner_type: type_idx,                     // [DOD] Raw 4-bit type
        type_affinity: type_params.type_affinity, // Read from Blueprint
    };

    let mut ctx = GrowthContext {
        current_pos_um,
        current_pos_vox: current_pos,
        forward_dir,
        target_pos: Some(target_pos),
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
    let (final_x, final_y, final_z) = if let Some(&last) = segments.last() {
        let pos = axicor_core::types::PackedPosition(last);
        (pos.x() as u32, pos.y() as u32, pos.z() as u32)
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

/// Initialize axon_head according to spec:
/// `axon_head = AXON_SENTINEL - length_segments * v_seg`
/// This allows PropagateAxons to correctly distribute signals in the very first tick.
pub fn init_axon_head(length_segments: u32, v_seg: u32) -> u32 {
    use axicor_core::constants::AXON_SENTINEL;
    AXON_SENTINEL.wrapping_sub(length_segments * v_seg)
}

/// Continues growth of axons that crossed the shard boundary (Ghost Axons).
///
/// Each GhostPacket describes an axon that entered this shard through a boundary.
/// Ghost Axon continues growth while maintaining inertia (`entry_dir`) and
/// attraction to neurons of the current shard.
///
/// - `soma_idx = usize::MAX`  no local soma, GSOP is not applied
/// - If Ghost Axon crosses a boundary again  a new GhostPacket is generated
///
/// Returns: `(grown_ghosts, outgoing_packets)`
pub fn inject_ghost_axons(
    ghost_packets: &[GhostPacket],
    positions: &[PackedPosition],
    types: &[axicor_core::config::blueprints::NeuronType], // [DOD FIX] Type propagation
    sim: &SimulationConfig,
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (Vec<GrownAxon>, Vec<GhostPacket>) {
    let voxel_um = sim.simulation.voxel_size_um;

    let max_search_radius_vox = sim.simulation.segment_length_voxels as f32 * 3.0;
    let spatial_grid = SpatialGrid::new(
        positions.to_vec(),
        f32::max(1.0, max_search_radius_vox.ceil()) as u32,
    );
    let mut grown = Vec::with_capacity(ghost_packets.len());
    let mut outgoing: Vec<GhostPacket> = Vec::new();

    for packet in ghost_packets {
        // [DOD FIX] No more hardcode. Read the axon owner's physics!
        let t_params = &types[packet.type_idx];
        let fov_cos = (t_params.steering_fov_deg / 2.0).to_radians().cos();

        let current_pos = Vec3::new(
            packet.entry_x as f32,
            packet.entry_y as f32,
            packet.entry_z as f32,
        );
        let current_pos_um = current_pos * voxel_um;
        let forward_dir = packet.entry_dir;

        let ghost_seed = master_seed
            .wrapping_add(packet.soma_idx as u64)
            .wrapping_add(packet.origin_shard_id as u64);

        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(ghost_seed);

        use crate::bake::cone_tracing::ConeParams;
        let params = ConeParams {
            radius_um: t_params.steering_radius_um,
            fov_cos,
            owner_type: packet.type_idx as u8,
            type_affinity: t_params.type_affinity, // Apply affinity from type
        };
        let weights = SteeringWeights {
            global: t_params.steering_weight_inertia,
            attract: t_params.steering_weight_sensor,
            noise: t_params.steering_weight_jitter,
        };

        let mut ctx = GrowthContext {
            current_pos_um,
            current_pos_vox: current_pos,
            forward_dir,
            target_pos: None,
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
        let (final_x, final_y, final_z) = if let Some(&last) = segments.last() {
            let pos = axicor_core::types::PackedPosition(last);
            (pos.x() as u32, pos.y() as u32, pos.z() as u32)
        } else {
            (packet.entry_x, packet.entry_y, packet.entry_z)
        };

        grown.push(GrownAxon {
            soma_idx: usize::MAX, // Ghost  no local soma
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
// as per Spec 05 2.1.

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn mock_bounds() -> ShardBounds {
        ShardBounds {
            x_start: 0,
            x_end: 100,
            y_start: 0,
            y_end: 100,
            z_start: 0,
            z_end: 100,
        }
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
            entry_x: 10,
            entry_y: 20,
            entry_z: 30,
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
        use axicor_core::config::blueprints::AxicorConstantMemory;
        use axicor_core::layout::VariantParameters;
        let _empty_const_mem = AxicorConstantMemory {
            variants: [VariantParameters {
                threshold: 0,
                rest_potential: 0,
                leak_shift: 0,
                homeostasis_penalty: 0,
                gsop_potentiation: 0,
                gsop_depression: 0,
                homeostasis_decay: 0,
                signal_propagation_length: 0,
                refractory_period: 0,
                synapse_refractory_period: 0,
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
            voxel_size_um = 25.0
            signal_speed_m_s = 0.5
            segment_length_voxels = 2
            sync_batch_ticks = 10
            axon_growth_max_steps = 100
        "#;
        let sim = SimulationConfig::parse(toml).unwrap();

        let dummy_types = vec![];

        let (grown, outgoing) =
            inject_ghost_axons(&[], &positions, &dummy_types, &sim, &mock_bounds(), 0);
        assert!(grown.is_empty());
        assert!(outgoing.is_empty());
    }

    #[test]
    fn test_uniform_sphere_sampling_isotropy() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut sectors = [0; 6]; // +X, -X, +Y, -Y, +Z, -Z
        let iters = 100_000;
        
        for _ in 0..iters {
            let theta = rng.gen_range(0.0..std::f32::consts::TAU);
            let u: f32 = rng.gen_range(-1.0_f32..1.0_f32);
            let r = (1.0 - u * u).sqrt();
            let v_noise = Vec3::new(r * theta.cos(), r * theta.sin(), u);
            
            let mut max_axis = 0;
            let mut max_val = v_noise.x.abs();
            if v_noise.y.abs() > max_val { max_val = v_noise.y.abs(); max_axis = 1; }
            if v_noise.z.abs() > max_val { max_axis = 2; }
            
            let is_positive = match max_axis {
                0 => v_noise.x > 0.0,
                1 => v_noise.y > 0.0,
                _ => v_noise.z > 0.0,
            };
            
            sectors[max_axis * 2 + (if is_positive { 0 } else { 1 })] += 1;
        }
        
        let expected = iters / 6;
        let tolerance = (iters as f32 * 0.03) as i32; // 3% tolerance
        for &count in &sectors {
            assert!((count as i32 - expected as i32).abs() < tolerance, "Count {} out of bounds for expected {}", count, expected);
        }
    }

    #[test]
    fn test_quantization_round_vs_truncate() {
        // (49.9 / 25.0) -> round(1.996) = 2. Truncate would be 1.
        let pos_um = Vec3::new(49.9, 50.1, 74.8);
        let voxel_size_um = 25.0;
        
        let x_idx = (pos_um.x / voxel_size_um).round().max(0.0) as u32;
        let y_idx = (pos_um.y / voxel_size_um).round().max(0.0) as u32;
        let z_idx = (pos_um.z / voxel_size_um).round().max(0.0) as u32;
        
        assert_eq!(x_idx, 2);
        assert_eq!(y_idx, 2);
        assert_eq!(z_idx, 3);
    }

    #[test]
    fn test_boundary_detection_with_rounding() {
        let b = ShardBounds {
            x_start: 0, x_end: 100,
            y_start: 0, y_end: 100,
            z_start: 0, z_end: 100,
        };
        
        // Simulating pos = 99.7 voxels
        let pos_vox_x = 99.7_f32;
        let raw_x = pos_vox_x.round().max(0.0) as u32;
        
        assert_eq!(raw_x, 100);
        assert!(b.is_outside(raw_x, 50, 50));
    }

    #[test]
    fn test_stagnation_false_positive_reduction() {
        let pos1 = 1.1_f32; // Step 1
        let pos2 = 1.9_f32; // Step 2
        
        // Truncation (old behavior)
        let p1_trunc = pos1.max(0.0) as u32;
        let p2_trunc = pos2.max(0.0) as u32;
        assert_eq!(p1_trunc, p2_trunc, "Truncation would cause false stagnation here (both 1)");
        
        // Rounding (new behavior)
        let p1_round = pos1.round().max(0.0) as u32;
        let p2_round = pos2.round().max(0.0) as u32;
        assert_ne!(p1_round, p2_round, "Rounding resolves false stagnation (1 vs 2)");
    }
}

// =============================================================================
//  Half-Duplex SHM Bridge: Incoming inter-zone axons
// =============================================================================

/// Converts network `AxonHandoverEvent`s (from SHM) into physical `GhostPacket`s
/// and continues their growth within the new shard.
pub fn inject_handover_events(
    handovers: &[axicor_core::ipc::AxonHandoverEvent],
    positions: &[PackedPosition],
    _layer_ranges: &[LayerZRange],
    types: &[axicor_core::config::blueprints::NeuronType],
    sim: &crate::parser::simulation::SimulationConfig,
    shard_bounds: &ShardBounds,
    master_seed: u64,
) -> (Vec<GrownAxon>, Vec<GhostPacket>) {
    if handovers.is_empty() {
        return (vec![], vec![]);
    }

    let voxel_um = sim.simulation.voxel_size_um;
    // SpatialGrid for attraction to existing neurons
    let search_r = (sim.simulation.segment_length_voxels as f32 * 3.0)
        .max(1.0)
        .ceil() as u32;
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

        // Entry point in global coordinates
        let entry_x = ev.entry_x as u32 + shard_bounds.x_start;
        let entry_y = ev.entry_y as u32 + shard_bounds.y_start;
        let entry_z = ev.entry_z as u32 + shard_bounds.z_start;

        // Deterministic seed
        let ghost_seed = master_seed
            .wrapping_add(idx as u64)
            .wrapping_add(0x4748_5354_0000_0000);

        let entry_dir = if raw_dir.length_squared() > 0.001 {
            raw_dir.normalize()
        } else {
            glam::Vec3::Z
        };

        let mut ctx = GrowthContext {
            current_pos_um: Vec3::new(entry_x as f32, entry_y as f32, entry_z as f32)
                * voxel_um,
            current_pos_vox: Vec3::new(entry_x as f32, entry_y as f32, entry_z as f32),
            forward_dir: entry_dir,
            target_pos: None,
            // [DOD FIX] Extract remaining growth resource from the network event!
            remaining_steps: ev.remaining_length as u32,
            owner_type_idx: type_idx as u8,
            soma_idx: usize::MAX,
            origin_shard_id: 0, // Routing hash is used at orchestrator level
        };

        // Since params depend on type, fetch them:
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
        let (final_x, final_y, final_z) = if let Some(&last) = segments.last() {
            let pos = axicor_core::types::PackedPosition(last);
            (pos.x() as u32, pos.y() as u32, pos.z() as u32)
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
