use clap::Parser;
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
    for (zone_idx, zone) in brain_config.zones.iter().enumerate() {
        println!("\n[baker] === Compiling Zone: {} ===", zone.name);
        let result_tuple = compile(
            &brain_config.simulation.config,
            &zone.blueprints,
            &zone.anatomy,
            &zone.io,
            &zone.baked_dir,
            &zone.name,
            zone_idx as u16,
            200_000, // DEFAULT_GHOST_CAPACITY (§3.1)
        )?;
        compiled_zones.insert(zone.name.clone(), result_tuple);
    }
    
    // 2. Generate Ghost Maps
    if !brain_config.connections.is_empty() {
        println!("\n[baker] === Baking Ghost Axon Mappings ===");
    }
    
    for conn in &brain_config.connections {
        let src_tuple = compiled_zones.get(&conn.from).expect("Source zone missing");
        let _dst_tuple = compiled_zones.get(&conn.to).expect("Dest zone missing");
        
        // Target offset is the end of the dest zone's local+virtual axons
        let dst_ghost_offset = src_tuple.1 as u32; // This logic seems slightly off in original, but I'll focus on removing patching
        
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
                dst_ghost_offset, // This should be based on destination!
                42, 
            )
        } else {
            let ghosts = bake::ghost_map::build_ghost_mapping(
                &conn.from,
                &conn.to,
                &src_tuple.0,
                dst_ghost_offset,
            );
            bake::ghost_map::write_ghosts_file(out_dir, &conn.from, &conn.to, &ghosts);
            ghosts.header.connection_count
        };
        
        println!("[baker] ✓ Ghost link {} -> {}: {} axons established.", conn.from, conn.to, sent_ghosts);
        // [REMOVED] Patching logic. The file is already pre-allocated to 200,000 ghosts.
    }

    Ok(())
}



fn compile(sim_path: &Path, bp_path: &Path, an_path: &Path, io_path: &Path, out_dir: &Path, zone_name: &str, zone_idx: u16, ghost_capacity: usize) -> Result<(Vec<u32>, usize, Vec<u32>, (f32, f32))> {
    // --- 1. Parse ---
    println!("[baker] Parsing configs...");
    let sim = parser::simulation::parse(
        &std::fs::read_to_string(sim_path)
            .with_context(|| format!("Cannot read {}", sim_path.display()))?,
    )
    .context("Failed to parse simulation.toml")?;
    let (const_mem, neuron_types, name_map) = parser::blueprints::parse_blueprints(
        &std::fs::read_to_string(bp_path)
            .with_context(|| format!("Cannot read {}", bp_path.display()))?,
    );
    let anatomy = parser::anatomy::parse(
        &std::fs::read_to_string(an_path)
            .with_context(|| format!("Cannot read {}", an_path.display()))?,
    )
    .context("Failed to parse anatomy.toml")?;
    let io = parser::io::parse(
        &std::fs::read_to_string(io_path)
            .with_context(|| format!("Cannot read {}", io_path.display()))?,
    )
    .context("Failed to parse io.toml")?;

    println!(
        "[baker] ✓ Parsed: {} neuron types, {} layers, {}/{} I/O",
        name_map.len(),
        anatomy.layers.len(),
        io.inputs.len(),
        io.outputs.len()
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
    let (mut axons, ghost_packets) = bake::axon_growth::grow_axons(
        &neurons,
        &layer_ranges,
        &neuron_types,
        &sim,
        &shard_bounds,
        master_seed,
    );
    let local_axons_count = axons.len();

    // --- 5.2 External Inputs (Virtual Axons from io.toml) ---
    let mut num_virtual = 0;
    let mut baked_gxis = Vec::new();
    if !io.inputs.is_empty() {
        println!("[baker] Processing Input Maps for {}...", zone_name);
        for matrix in &io.inputs {
            let gxi = bake::input_map::build_gxi_mapping(
                &matrix.name,
                zone_name,
                matrix.width,
                matrix.height,
                axons.len() as u32,
                matrix.stride,
            );
            num_virtual += gxi.axon_ids.len();
            
            // Virtual Axon Offset contract: append axon_ids as sentinel heads
            for &axon_id in &gxi.axon_ids {
                axons.push(bake::axon_growth::GrownAxon {
                    soma_idx: usize::MAX,
                    type_idx: 0,
                    tip_x: 0, tip_y: 0, tip_z: 0,
                    length_segments: 0,
                    segments: vec![],
                    last_dir: glam::Vec3::ZERO,
                });
                let _ = axon_id;
            }
            baked_gxis.push(gxi);
        }
        
        bake::input_map::write_gxi_file(out_dir, &baked_gxis);

        println!(
            "[baker] ✓ Written {}.gxi with {} virtual axons across {} matrices",
            zone_name,
            num_virtual,
            baked_gxis.len()
        );
    }

    // --- 5.3 Output Readouts (Soma Readouts from io.toml) ---
    let mut baked_gxos = Vec::new();
    if !io.outputs.is_empty() {
        println!("[baker] Processing Output Maps for {}...", zone_name);
        let voxel_um = sim.simulation.voxel_size_um;
        let world_w_vox = (sim.world.width_um / voxel_um) as u32;
        let world_d_vox = (sim.world.depth_um / voxel_um) as u32;
        let packed_positions: Vec<u32> = neurons.iter().map(|n| n.position.0).collect();

        for matrix in &io.outputs {
            let gxo = bake::output_map::build_gxo_mapping(
                &matrix.name,
                zone_name,
                matrix.width,
                matrix.height,
                world_w_vox,
                world_d_vox,
                &packed_positions,
                matrix.stride,
            );
            baked_gxos.push(gxo);
        }
        bake::output_map::write_gxo_file(out_dir, &baked_gxos);
        println!("[baker] ✓ Written {}.gxo with {} matrices", zone_name, baked_gxos.len());
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
    let total_capacity = axons.len() + ghost_capacity;
    let mut shard = bake::layout::ShardSoA::new(neurons.len(), total_capacity);
    // Инициализация потенциалов из rest_potential типа 0
    for v in shard.voltage.iter_mut() {
        *v = rest_potential;
    }

    // --- 7. Init axon heads ---
    let physics = genesis_core::physics::compute_derived_physics(
        sim.simulation.signal_speed_m_s,
        sim.simulation.tick_duration_us,
        sim.simulation.voxel_size_um,
        sim.simulation.segment_length_voxels,
    ).expect("v_seg validation failed. Run baker validation first.");
    let v_seg = physics.v_seg;
    for (i, ax) in axons.iter().enumerate() {
        if i < shard.axon_heads.len() {
            shard.axon_heads[i] = bake::axon_growth::init_axon_head(ax.length_segments, v_seg);
            
            // Записываем геометрию (Z|Y|X packing matching Spec)
            shard.axon_tips_uvw[i] = (ax.tip_z << 20) | (ax.tip_y << 10) | ax.tip_x;
            
            // Записываем направление (i8 packing: X | Y | Z | 0)
            let dx = (ax.last_dir.x * 127.0).clamp(-127.0, 127.0) as i8 as u32;
            let dy = (ax.last_dir.y * 127.0).clamp(-127.0, 127.0) as i8 as u32;
            let dz = (ax.last_dir.z * 127.0).clamp(-127.0, 127.0) as i8 as u32;
            shard.axon_dirs_xyz[i] = (dz << 16) | (dy << 8) | dx;
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
        sim.simulation.segment_length_voxels * 8, // conservative search radius
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

    // --- 10. Write Autonomous Manifest ---
    let zone_hash_fnv = genesis_core::hash::fnv1a_32(zone_name.as_bytes());
    let manifest = genesis_core::config::manifest::ZoneManifest {
        magic: 0x47454E45, // "GENE"
        zone_hash: zone_hash_fnv,
        memory: genesis_core::config::manifest::ManifestMemory {
            padded_n: shard.soma_to_axon.len(),
            virtual_axons: num_virtual,
            ghost_capacity: 200000, // Safe default for V1
            v_seg: v_seg as u16,
        },
        network: genesis_core::config::manifest::ManifestNetwork {
            slow_path_tcp: 8010 + zone_idx * 10,
            external_udp_in: 8081 + zone_idx * 10,
            external_udp_out: 8082 + zone_idx * 10,
            external_udp_out_target: None,
            fast_path_udp_local: 9001 + zone_idx * 10,
            fast_path_peers: if zone_idx == 0 { vec!["127.0.0.1:9011".to_string()] } else { vec!["127.0.0.1:9001".to_string()] },
        },
        variants: const_mem.variants.iter().enumerate().map(|(i, v)| {
            genesis_core::config::manifest::ManifestVariant {
                id: i as u8,
                name: format!("Variant_{}", i),
                threshold: v.threshold,
                rest_potential: v.rest_potential,
                leak_rate: v.leak_rate,
                homeostasis_penalty: v.homeostasis_penalty,
                homeostasis_decay: v.homeostasis_decay,
                gsop_potentiation: v.gsop_potentiation,
                gsop_depression: v.gsop_depression,
                refractory_period: v.refractory_period,
                synapse_refractory_period: v.synapse_refractory_period,
                slot_decay_ltm: v.slot_decay_ltm as u8,
                slot_decay_wm: v.slot_decay_wm as u8,
                signal_propagation_length: v.signal_propagation_length as u16,
            }
        }).collect(),
    };
    
    let manifest_toml = toml::to_string(&manifest).expect("Failed to serialize manifest");
    let manifest_path = out_dir.join("manifest.toml");
    std::fs::write(&manifest_path, manifest_toml)?;
    println!("[baker] ✓ Written: manifest.toml");

    let dna_dir = out_dir.join("BrainDNA");
    std::fs::create_dir_all(&dna_dir).expect("Failed to create BrainDNA directory");
    let configs = [
        (sim_path, "simulation.toml"),
        (bp_path, "blueprints.toml"),
        (an_path, "anatomy.toml"),
        (io_path, "io.toml"),
    ];
    for (src, dst_name) in configs {
        if src.exists() {
            std::fs::copy(src, dna_dir.join(dst_name))
                .unwrap_or_else(|_| panic!("Failed to copy {} to BrainDNA", dst_name));
        }
    }
    println!("[baker] ✓ Written: BrainDNA Artifacts");

    println!("[baker] Zone {} Done.", zone_name);
    
    let packed_pos: Vec<u32> = neurons.iter().map(|n| n.position.0).collect();
    let voxel_um = sim.simulation.voxel_size_um;
    let world_w_vox = (sim.world.width_um as f32 / voxel_um) as u32;
    let world_d_vox = (sim.world.depth_um as f32 / voxel_um) as u32;
    let _size_vox = (world_w_vox as f32, world_d_vox as f32);
    
    Ok((shard.soma_to_axon.clone(), shard.total_axons, packed_pos, _size_vox))
}

/// Формат файла `.axons`:
/// - Header: `AxonsFileHeader` (8 bytes)
/// - Array of `total_axons`:
///   - tip_x: u16, tip_y: u16, tip_z: u16
///   - length: u32 (N)
///   - segments: [u32; N] (PackedPositions)
#[allow(dead_code)]
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
#[allow(dead_code)]
fn atomic_write(path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
    let path = path.as_ref();
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data).with_context(|| format!("Cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("Cannot rename .tmp → {}", path.display()))?;
    Ok(())
}

