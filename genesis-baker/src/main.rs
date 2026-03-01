use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

mod bake;
mod parser;
mod validator;


#[derive(Parser)]
#[command(
    name = "baker",
    about = "Genesis Baking Tool — TOML configs → binary .state/.axons blobs + ghost maps"
)]
#[command(version)]
struct Cli {
    #[arg(long, default_value = "config/brain.toml")]
    brain: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    let brain_config = genesis_core::config::brain::parse_brain_config(&cli.brain)
        .map_err(|e| anyhow::anyhow!(e))?;
    println!("[baker] Processing Brain Architecture: {} zones", brain_config.zones.len());
    
    // Store compilation results for ghost connections linking
    // Tuple: (soma_to_axon_map, base_axons_count, packed_pos, size_um)
    let mut compiled_zones: std::collections::HashMap<String, (Vec<u32>, usize, Vec<u32>, (f32, f32))> = std::collections::HashMap::new();

    // 1. Compile all zones
    for zone in &brain_config.zones {
        println!("\n[baker] === Compiling Zone: {} ===", zone.name);
        let result_tuple = compile(
            &brain_config.simulation.config,
            &zone.blueprints,
            &zone.anatomy,
            &zone.io,
            &zone.baked_dir,
            &zone.name,
        )?;
        compiled_zones.insert(zone.name.clone(), result_tuple);
    }
    
    // 2. Generate Ghost Maps
    if !brain_config.connections.is_empty() {
        println!("\n[baker] === Baking Ghost Axon Mappings ===");
    }
    
    for conn in &brain_config.connections {
        let src_tuple = compiled_zones.get(&conn.from).expect("Source zone missing");
        let dst_tuple = compiled_zones.get(&conn.to).expect("Dest zone missing");
        
        // Target offset is the end of the dest zone's local+virtual axons
        let dst_ghost_offset = dst_tuple.1 as u32;
        
        let out_dir = &brain_config.zones.iter().find(|z| z.name == conn.from).unwrap().baked_dir;
        
        let sent_ghosts = if let (Some(w), Some(h)) = (conn.width, conn.height) {
            println!("[baker] Generating UV Atlas Projection {} -> {} ({}x{})", conn.from, conn.to, w, h);
            bake::atlas_map::bake_atlas_connection(
                out_dir,
                &conn.from,
                &conn.to,
                &src_tuple.2, // src_packed_pos
                src_tuple.3,  // src_size_um
                (w, h),       // conn_grid
                dst_ghost_offset,
                42, // master_seed hardcoded for MVP
            )
        } else {
            bake::ghost_map::bake_ghost_connection(
                out_dir,
                &conn.from,
                &conn.to,
                &src_tuple.0,
                dst_ghost_offset
            )
        };
        
        println!("[baker] ✓ Ghost link {} -> {}: {} axons established.", conn.from, conn.to, sent_ghosts);
        
        // Update total_axons in destination zone's `.state` file
        let dest_zone = brain_config.zones.iter().find(|z| z.name == conn.to).unwrap();
        let state_path = dest_zone.baked_dir.join("shard.state");
        
        let mut state_bytes = std::fs::read(&state_path)?;
        let mut header = *genesis_core::layout::StateFileHeader::from_bytes(&state_bytes).unwrap();
        header.total_base_axons += sent_ghosts;
        
        let header_bytes = header.as_bytes();
        state_bytes[..header_bytes.len()].copy_from_slice(header_bytes);
        std::fs::write(&state_path, state_bytes)?;
        
        println!("[baker] ✓ Patched {} VRAM allocation: +{} ghost slots", conn.to, sent_ghosts);
    }

    Ok(())
}

fn compile(sim_path: &Path, bp_path: &Path, an_path: &Path, io_path: &Path, out_dir: &Path, zone_name: &str) -> Result<(Vec<u32>, usize, Vec<u32>, (f32, f32))> {
    // --- 1. Parse ---
    println!("[baker] Parsing configs...");
    let sim = parser::simulation::parse(
        &std::fs::read_to_string(sim_path)
            .with_context(|| format!("Cannot read {}", sim_path.display()))?,
    )
    .context("Failed to parse simulation.toml")?;
    let (const_mem, name_map) = parser::blueprints::parse_blueprints(
        &std::fs::read_to_string(bp_path)
            .with_context(|| format!("Cannot read {}", bp_path.display()))?,
    );
    let anatomy = parser::anatomy::parse(
        &std::fs::read_to_string(an_path)
            .with_context(|| format!("Cannot read {}", an_path.display()))?,
    )
    .context("Failed to parse anatomy.toml")?;

    println!(
        "[baker] ✓ Parsed: {} neuron types, {} layers",
        name_map.len(),
        anatomy.layers.len()
    );

    // --- 2. Validate ---
    println!("[baker] Validating invariants...");
    validator::validate_all(&sim, &const_mem, &anatomy).context("Config validation failed")?;
    println!("[baker] ✓ All invariants passed");

    // --- 3. Master seed ---
    let master_seed_str = &sim.simulation.master_seed;
    let master_seed = bake::seed::seed_from_str(master_seed_str);
    println!("[baker] Master seed: \"{}\"", master_seed_str);

    // Ensure output directory exists before any writing
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("Cannot create output dir: {}", out_dir.display()))?;

    // --- 4. Place neurons ---
    println!("[baker] Placing neurons...");
    // Build sorted type_names from name_map for placement ordering
    let mut type_name_pairs: Vec<(&String, &u8)> = name_map.iter().collect();
    type_name_pairs.sort_by_key(|(_, &idx)| idx);
    let type_names: Vec<String> = type_name_pairs.into_iter().map(|(n, _)| n.clone()).collect();
    let neurons = bake::neuron_placement::place_neurons(&sim, &anatomy, &type_names, master_seed);
    println!("[baker] ✓ Placed {} neurons", neurons.len());

    // --- 5. Cone Tracing: grow axons ---
    println!("[baker] Growing axons (Cone Tracing)...");
    let layer_ranges = bake::axon_growth::compute_layer_ranges(&anatomy, &sim);
    let shard_bounds = bake::axon_growth::ShardBounds::full_world(&sim);
    let (mut axons, mut ghost_packets) = bake::axon_growth::grow_axons(
        &neurons,
        &layer_ranges,
        &const_mem,
        &sim,
        &shard_bounds,
        master_seed,
    );
    let local_axons_count = axons.len();

    // --- 5.2 External Inputs (Virtual Axons CartPole Mock) ---
    let mut num_virtual = 0;
    if zone_name == "SensoryCortex" {
        println!("[baker] Processing CartPole Input Maps...");
        
        let input_matrices = vec![
            bake::input_map::InputMatrixDef {
                name_hash: 0x87654321,
                width: 8,
                height: 8,
            }
        ];

        let mut virtual_axons = bake::input_map::bake_inputs(
            out_dir,
            &input_matrices,
            axons.len() as u32,
        );
        num_virtual = virtual_axons.len();
        axons.append(&mut virtual_axons);

        println!(
            "[baker] ✓ Written CartPole .gxi with {} virtual axons",
            num_virtual
        );
    }

    // --- 5.3 External Outputs (Readout Maps CartPole Mock) ---
    if zone_name == "MotorCortex" {
        println!("[baker] Processing CartPole Output Maps...");
        
        let packed_positions: Vec<u32> = neurons.iter().map(|n| n.position).collect();
        
        let output_matrices = vec![
            bake::output_map::OutputMatrixDef {
                name_hash: 0x12345678, // Matching runtime payload hash expectation
                width: 2, // Left/Right
                height: 1,
                stride: 1,
            }
        ];

        bake::output_map::bake_outputs(
            out_dir,
            &output_matrices,
            sim.world.width_um as f32,
            sim.world.depth_um as f32,
            &packed_positions,
        );

        println!(
            "[baker] ✓ Written CartPole .gxo for {} somas",
            2
        );
    }
    if !ghost_packets.is_empty() {
        println!("[baker] ✓ {} ghost packet(s) detected — injecting into shard B...", ghost_packets.len());
        let (mut ghost_axons, leftover) = bake::axon_growth::inject_ghost_axons(
            &ghost_packets,
            &neurons,
            &const_mem,
            &sim,
            &shard_bounds,
            master_seed,
        );
        if !leftover.is_empty() {
            println!("[baker] ⚠ {} ghost(s) exited shard B boundary (further shards not yet wired)", leftover.len());
        }
        println!("[baker] ✓ Injected {} ghost axons", ghost_axons.len());
        axons.append(&mut ghost_axons);
    }

    // --- 5.5 Atlas Routing (White Matter) ---
    // Временно отключено, будет переработано под новый IoConfig

    println!("[baker] ✓ Total Grown: {} axons ({} local, {} virtual)", 
        axons.len(), local_axons_count, num_virtual);

    // --- 6. Build SoA state ---
    // rest_potential: берём из первого варианта const_mem
    let rest_potential = const_mem.variants[0].rest_potential;
    let mut shard = bake::layout::ShardSoA::new(neurons.len(), axons.len());
    // Инициализация потенциалов из rest_potential типа 0
    for v in shard.voltage.iter_mut() {
        *v = rest_potential;
    }

    // --- 7. Init axon heads ---
    let physics = genesis_core::physics::compute_derived_physics(
        sim.simulation.signal_speed_um_tick as u32,
        sim.simulation.voxel_size_um,
        sim.simulation.segment_length_voxels,
    ).expect("v_seg validation failed. Run baker validation first.");
    let v_seg = physics.v_seg;
    for (i, ax) in axons.iter().enumerate() {
        if i < shard.axon_heads.len() {
            shard.axon_heads[i] = bake::axon_growth::init_axon_head(ax.length_segments, v_seg);
        }
        if ax.soma_idx != usize::MAX && ax.soma_idx < shard.soma_to_axon.len() {
            shard.soma_to_axon[ax.soma_idx] = i as u32;
        }
    }
    println!("[baker] ✓ Axon heads initialized (v_seg={})", v_seg);

    // --- 8. Connect dendrites (spatial search + sprouting score) ---
    println!("[baker] Connecting dendrites...");
    bake::dendrite_connect::connect_dendrites(
        &mut shard,
        &neurons,
        &axons,
        &const_mem,
        master_seed,
    );
    let connected = shard.dendrite_targets.iter().filter(|&&t| t != 0).count();
    println!(
        "[baker] ✓ Connected {} synapses ({:.1} avg per neuron)",
        connected,
        connected as f64 / neurons.len() as f64
    );

    // --- 9. Dump to disk (Zero-Copy binary: raw bytes, no serde) ---


    shard.dump_to_disk(out_dir);
    println!("[baker] ✓ Written: shard.state + shard.axons");

    println!("[baker] Zone {} Done.", zone_name);
    
    let packed_pos: Vec<u32> = neurons.iter().map(|n| n.position).collect();
    let size_um = (sim.world.width_um as f32, sim.world.depth_um as f32);
    
    Ok((shard.soma_to_axon.clone(), axons.len(), packed_pos, size_um))
}

/// Формат файла `.axons`:
/// - Header: `AxonsFileHeader` (8 bytes)
/// - Array of `total_axons`:
///   - tip_x: u16, tip_y: u16, tip_z: u16
///   - length: u32 (N)
///   - segments: [u32; N] (PackedPositions)
fn serialize_axons(axons: &[bake::axon_growth::GrownAxon]) -> Vec<u8> {
    // Оцениваем размер: 8 байт заголовок + для каждого (10 байт заголовок + 4 байта на сегмент)
    let estimated_size = 8 + axons.iter().map(|ax| 10 + ax.segments.len() * 4).sum::<usize>();
    let mut out = Vec::with_capacity(estimated_size);
    let header = genesis_core::layout::AxonsFileHeader::new(axons.len() as u32);
    out.extend_from_slice(header.as_bytes());
    for ax in axons {
        out.extend_from_slice(&(ax.tip_x as u16).to_le_bytes());
        out.extend_from_slice(&(ax.tip_y as u16).to_le_bytes());
        out.extend_from_slice(&(ax.tip_z as u16).to_le_bytes());
        
        out.extend_from_slice(&(ax.segments.len() as u32).to_le_bytes());
        for &seg in &ax.segments {
            out.extend_from_slice(&seg.to_le_bytes());
        }
    }
    out
}

/// Атомарная запись: пишем в .tmp, потом rename. Защита от краша.
fn atomic_write(path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
    let path = path.as_ref();
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data).with_context(|| format!("Cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("Cannot rename .tmp → {}", path.display()))?;
    Ok(())
}

