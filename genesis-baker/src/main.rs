#![deny(warnings)]
#![deny(unused_variables)]
#![deny(dead_code)]
use clap::Parser;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};

use genesis_baker::{bake, parser, validator};


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
    let mut compiled_zones: std::collections::HashMap<String, bake::layout::CompiledShard> = std::collections::HashMap::new();

    // 1. Compile all zones
    for (zone_idx, zone) in brain_config.zones.iter().enumerate() {
        println!("\n[baker] === Compiling Zone: {} ===", zone.name);
        // [DOD FIX] Путь к shard.toml хранится рядом с anatomy.toml
        let shard_cfg_path = zone.anatomy.parent().unwrap().join("shard.toml");
        
        let workspace = parse_and_validate(
            &brain_config,
            &brain_config.simulation.config,
            &zone.blueprints,
            &zone.anatomy,
            &zone.io,
            &shard_cfg_path,
            &zone.baked_dir,
            &zone.name,
            zone_idx as u16,
            200_000, // DEFAULT_GHOST_CAPACITY (§3.1)
        )?;

        let (shard_soa, compiled_shard, v_seg, num_virtual, gxis, gxos) = build_local_topology(&workspace);
        serialize_artifacts(&shard_soa, &workspace, num_virtual, v_seg, &gxis, &gxos)?;

        compiled_zones.insert(zone.name.clone(), compiled_shard);
    }
    
    // 2. Generate Ghost Maps
    if !brain_config.connections.is_empty() {
        println!("\n[baker] === Baking Ghost Axon Mappings ===");
    }
    
    for conn in &brain_config.connections {
        let src_shard = compiled_zones.get(&conn.from).expect("Source zone missing");
        let dst_shard = compiled_zones.get(&conn.to).expect("Dest zone missing");
        
        // Target offset is the end of the dest zone's local axons
        let dst_ghost_offset = dst_shard.local_axons_count as u32; 
        
        let out_dir = &brain_config.zones.iter().find(|z| z.name == conn.from).unwrap().baked_dir;
        
        let sent_ghosts = if let (Some(w), Some(h)) = (conn.width, conn.height) {
            println!("[baker] Generating UV Atlas Projection {} -> {} ({}x{})", conn.from, conn.to, w, h);
            bake::atlas_map::bake_atlas_connection(
                out_dir,
                &conn.from,
                &conn.to,
                &src_shard.packed_positions, // src_packed_pos
                src_shard.bounds_um,  // src_size_um
                (w, h),       // conn_grid
                dst_ghost_offset, // This should be based on destination!
                42, 
            )
        } else {
            let ghosts = bake::ghost_map::build_ghost_mapping(
                &conn.from,
                &conn.to,
                &src_shard.soma_to_axon_map,
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



pub struct BakeWorkspace {
    pub sim: genesis_core::config::SimulationConfig,
    pub const_mem: genesis_core::config::blueprints::GenesisConstantMemory,
    pub neuron_types: Vec<genesis_core::config::blueprints::NeuronType>,
    pub name_map: std::collections::HashMap<String, u8>,
    pub anatomy: genesis_core::config::anatomy::AnatomyConfig,
    pub io: genesis_core::config::io::IoConfig,
    pub shard_cfg: genesis_core::config::InstanceConfig,
    pub out_dir: PathBuf,
    pub zone_name: String,
    pub zone_idx: u16,
    pub ghost_capacity: usize,
    pub master_seed: u64,
    // Original paths for DNA copying
    pub sim_path: PathBuf,
    pub bp_path: PathBuf,
    pub an_path: PathBuf,
    pub io_path: PathBuf,
    pub shard_cfg_path: PathBuf,
    pub brain_config: genesis_core::config::brain::BrainConfig,
}

fn parse_and_validate(brain_config: &genesis_core::config::brain::BrainConfig, sim_path: &Path, bp_path: &Path, an_path: &Path, io_path: &Path, shard_cfg_path: &Path, out_dir: &Path, zone_name: &str, zone_idx: u16, ghost_capacity: usize) -> Result<BakeWorkspace> {
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

    println!("[baker] Validating invariants...");
    validator::validate_all(&sim, &const_mem, &anatomy).context("Config validation failed")?;
    println!("[baker] ✓ All invariants passed");

    let master_seed_str = &sim.simulation.master_seed;
    let master_seed = bake::seed::seed_from_str(master_seed_str);
    println!("[baker] Master seed: \"{}\"", master_seed_str);

    let shard_cfg = genesis_core::config::InstanceConfig::load(shard_cfg_path)
        .map_err(anyhow::Error::msg)?;

    Ok(BakeWorkspace {
        sim,
        const_mem,
        neuron_types,
        name_map,
        anatomy,
        io,
        shard_cfg,
        out_dir: out_dir.to_path_buf(),
        zone_name: zone_name.to_string(),
        zone_idx,
        ghost_capacity,
        master_seed,
        sim_path: sim_path.to_path_buf(),
        bp_path: bp_path.to_path_buf(),
        an_path: an_path.to_path_buf(),
        io_path: io_path.to_path_buf(),
        shard_cfg_path: shard_cfg_path.to_path_buf(),
        brain_config: brain_config.clone(),
    })
}

fn build_local_topology(workspace: &BakeWorkspace) -> (bake::layout::ShardSoA, bake::layout::CompiledShard, u32, usize, Vec<crate::bake::input_map::BakedGxi>, Vec<crate::bake::output_map::BakedGxo>) {
    bake::topology::build_local_topology_internal(
        &workspace.sim,
        &workspace.anatomy,
        &workspace.shard_cfg,
        &workspace.neuron_types,
        &workspace.const_mem,
        &workspace.name_map,
        &workspace.io,
        &workspace.zone_name,
        workspace.master_seed,
        workspace.ghost_capacity,
    )
}

fn serialize_artifacts(
    shard: &bake::layout::ShardSoA, 
    workspace: &BakeWorkspace, 
    num_virtual: usize, 
    v_seg: u32,
    gxis: &[crate::bake::input_map::BakedGxi],
    gxos: &[crate::bake::output_map::BakedGxo],
) -> Result<()> {
    std::fs::create_dir_all(&workspace.out_dir)
        .with_context(|| format!("Cannot create output dir: {}", workspace.out_dir.display()))?;

    shard.dump_to_disk(&workspace.out_dir);
    println!("[baker] ✓ Written: shard.state + shard.axons");

    if !gxis.is_empty() {
        crate::bake::input_map::write_gxi_file(&workspace.out_dir, gxis);
        println!("[baker] ✓ Written: shard.gxi");
    }
    if !gxos.is_empty() {
        crate::bake::output_map::write_gxo_file(&workspace.out_dir, gxos);
        println!("[baker] ✓ Written: shard.gxo");
    }

    let zone_hash_fnv = genesis_core::hash::fnv1a_32(workspace.zone_name.as_bytes());
    let manifest = genesis_core::config::manifest::ZoneManifest {
        magic: 0x47454E45, // "GENE"
        zone_hash: zone_hash_fnv,
        simulation: Some(genesis_core::config::brain::SimulationConfigRef {
            config: std::path::PathBuf::from("BrainDNA/simulation.toml"),
        }),
        memory: genesis_core::config::manifest::ManifestMemory {
            padded_n: shard.soma_to_axon.len(),
            virtual_axons: num_virtual,
            ghost_capacity: workspace.ghost_capacity, // Expected usize if manifest changed, or cast if u32 required. Error said expected usize, found u32.
            v_seg: v_seg as u16,
        },
        network: genesis_core::config::manifest::ManifestNetwork {
            slow_path_tcp: 8010 + workspace.zone_idx * 10,
            external_udp_in: 8081 + workspace.zone_idx * 10,
            external_udp_out: 8082 + workspace.zone_idx * 10,
            external_udp_out_target: None,
            fast_path_udp_local: 9001 + workspace.zone_idx * 10,
            fast_path_peers: if workspace.zone_idx == 0 { vec!["127.0.0.1:9011".to_string()] } else { vec!["127.0.0.1:9001".to_string()] },
        },
        connections: workspace.brain_config.connections.iter()
            .filter(|c| c.from == workspace.zone_name || c.to == workspace.zone_name)
            .map(|c| genesis_core::config::manifest::ManifestConnection {
                from: c.from.clone(),
                to: c.to.clone(),
                width: c.width,
                height: c.height,
            })
            .collect(),
        variants: workspace.const_mem.variants.iter().enumerate().map(|(i, v)| {
            genesis_core::config::manifest::ManifestVariant {
                id: i as u8,
                name: format!("Variant_{}", i),
                threshold: v.threshold,
                rest_potential: v.rest_potential,
                leak_rate: v.leak_rate,
                homeostasis_penalty: v.homeostasis_penalty,
                homeostasis_decay: v.homeostasis_decay as i32,
                gsop_potentiation: v.gsop_potentiation as i32,
                gsop_depression: v.gsop_depression as i32,
                refractory_period: v.refractory_period,
                synapse_refractory_period: v.synapse_refractory_period,
                slot_decay_ltm: v.slot_decay_ltm as u8,
                slot_decay_wm: v.slot_decay_wm as u8,
                signal_propagation_length: v.signal_propagation_length as u8,
                ltm_slot_count: 80, // default from blueprints
                inertia_curve: v.inertia_curve,
                prune_threshold: 15, // [DOD FIX] TODO: брать из blueprints.toml
            }
        }).collect(),
    };
    
    let manifest_toml = toml::to_string(&manifest).expect("Failed to serialize manifest");
    let manifest_path = workspace.out_dir.join("manifest.toml");
    std::fs::write(&manifest_path, manifest_toml)?;
    println!("[baker] ✓ Written: manifest.toml");

    let dna_dir = workspace.out_dir.join("BrainDNA");
    std::fs::create_dir_all(&dna_dir).expect("Failed to create BrainDNA directory");
    let configs = [
        (&workspace.sim_path, "simulation.toml"),
        (&workspace.bp_path, "blueprints.toml"),
        (&workspace.an_path, "anatomy.toml"),
        (&workspace.io_path, "io.toml"),
        (&workspace.shard_cfg_path, "shard.toml"),
    ];
    for (src, dst_name) in configs {
        if src.exists() {
            std::fs::copy(src, dna_dir.join(dst_name))
                .unwrap_or_else(|_| panic!("Failed to copy {} to BrainDNA", dst_name));
        }
    }
    
    Ok(())
}



