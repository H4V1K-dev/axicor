//! genesis-baker-daemon — Night Phase Sprouting daemon.
//!
//! Runs as a long-lived process alongside genesis-runtime.
//! At startup: reads configs, loads shard geometry, creates POSIX SHM.
//! Main loop: waits for `night_start` on Unix socket, runs Sprouting,
//! writes updated targets to SHM, replies `night_done`.

use genesis_baker::bake::dendrite_connect::reconnect_empty_dendrites;
use genesis_core::ipc::{shm_name, shm_size, ShmHeader, ShmState};
use genesis_baker::bake::neuron_placement::PlacedNeuron;
use genesis_baker::bake::axon_growth::GrownAxon;
use std::io::{BufRead, BufReader, Write};
use genesis_baker::parser::blueprints;
use genesis_baker::parser::simulation;
use std::os::unix::net::UnixListener;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::ffi::CString;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "genesis-baker-daemon", about = "Night Phase Sprouting daemon")]
struct Cli {
    #[arg(short = 'z', long)]
    zone: u16,

    #[arg(short = 's', long, default_value = "config/simulation.toml")]
    pub sim: PathBuf,

    /// Path to blueprints.toml
    #[arg(short = 'b', long, default_value = "config/zones/V1/blueprints.toml")]
    pub blueprints: PathBuf,

    /// Directory containing baked shard files (.state, .axons)
    #[arg(long, default_value = "baked/")]
    shard_dir: PathBuf,

    /// Unix socket path for control channel
    #[arg(long)]
    socket: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let zone_id = cli.zone;

    println!("[baker-daemon] Starting for zone {zone_id}");

    // ── 1. Load configs ──
    let sim_src = std::fs::read_to_string(&cli.sim)
        .with_context(|| format!("Failed to read {:?}", cli.sim))?;
    let bp_src = std::fs::read_to_string(&cli.blueprints)
        .with_context(|| format!("Failed to read {:?}", cli.blueprints))?;

    let sim = simulation::parse(&sim_src)
        .with_context(|| format!("Failed to parse {:?}", cli.sim))?;
    let (const_mem, _name_map) = blueprints::parse_blueprints(&bp_src);
    
    let master_seed = genesis_core::seed::MasterSeed::from_str(&sim.simulation.master_seed);

    let _segment_length = sim.simulation.segment_length_voxels;

    // ── 2. Load shard geometry (neurons + axons) ──
    let state_path = cli.shard_dir.join("shard.state");
    let axons_path = cli.shard_dir.join("shard.axons");

    let state_bytes = std::fs::read(&state_path)
        .with_context(|| format!("Missing {:?}", state_path))?;
    let axons_bytes = std::fs::read(&axons_path)
        .with_context(|| format!("Missing {:?}", axons_path))?;

    let neurons = load_neurons(&state_bytes)?;
    let axons   = load_axons(&axons_bytes)?;

    println!(
        "[baker-daemon] Loaded {} neurons, {} axons.",
        neurons.len(), axons.len(),
    );

    // ── 3. Bind Unix socket ──
    let socket_path = cli
        .socket
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/genesis_baker_{zone_id}.sock")));

    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("Cannot bind Unix socket {:?}", socket_path))?;

    println!("[baker-daemon] Listening on {:?}. Ready.", socket_path);

    // ── 4. Main daemon loop ──
    let mut shm_ptr: *mut u8 = std::ptr::null_mut();
    let mut shm_len: usize = 0;

    for stream in listener.incoming() {
        match stream {
            Err(e) => { eprintln!("[baker-daemon] Accept error: {e}"); continue; }
            Ok(stream) => {
                let mut reader = BufReader::new(&stream);
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() { continue; }
                let line = line.trim().to_string();

                if line.contains("shutdown") {
                    println!("[baker-daemon] Shutdown requested.");
                    break;
                }

                if !line.contains("night_start") { continue; }

                let padded_n = extract_u64(&line, "padded_n").unwrap_or(0) as usize;
                let epoch    = extract_u64(&line, "epoch").unwrap_or(0);

                println!("[baker-daemon] night_start epoch={epoch} padded_n={padded_n}");

                if padded_n == 0 {
                    let _ = write_response(&stream, "error", zone_id, epoch, Some("padded_n=0 in night_start"));
                    continue;
                }

                // Open / resize SHM if needed
                let needed = shm_size(padded_n);
                if shm_ptr.is_null() || shm_len != needed {
                    if !shm_ptr.is_null() {
                        unsafe { libc::munmap(shm_ptr as *mut _, shm_len) };
                    }
                    match open_shm(zone_id, needed) {
                        Ok((ptr, len)) => { shm_ptr = ptr; shm_len = len; }
                        Err(e) => {
                            eprintln!("[baker-daemon] SHM open failed: {e}");
                            let _ = write_response(&stream, "error", zone_id, epoch, Some(&e.to_string()));
                            continue;
                        }
                    }
                    // Write header
                    let hdr = ShmHeader::new(zone_id, padded_n as u32);
                    unsafe { std::ptr::write(shm_ptr as *mut ShmHeader, hdr) };
                }

                let hdr = unsafe { std::ptr::read(shm_ptr as *const ShmHeader) };

                // Mark SPROUTING
                unsafe { shm_ptr.add(5).write_volatile(ShmState::Sprouting as u8) };

                // Read weights + targets from SHM
                let slot_n = padded_n * 128;
                let w_off = hdr.weights_offset as usize;
                let t_off = hdr.targets_offset as usize;

                let mut weights = vec![0i16; slot_n];
                let mut targets = vec![0u32; slot_n];

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        shm_ptr.add(w_off) as *const i16,
                        weights.as_mut_ptr(),
                        slot_n,
                    );
                    std::ptr::copy_nonoverlapping(
                        shm_ptr.add(t_off) as *const u32,
                        targets.as_mut_ptr(),
                        slot_n,
                    );
                }

                let downloaded_weights = weights.clone();

                // Run Sprouting
                reconnect_empty_dendrites(
                    &mut targets,
                    &mut weights,
                    &downloaded_weights,
                    padded_n,
                    &neurons,
                    &axons,
                    &const_mem,
                    master_seed.raw() ^ epoch,
                );

                // Write updated targets back to SHM
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        targets.as_ptr() as *const u8,
                        shm_ptr.add(t_off),
                        slot_n * std::mem::size_of::<u32>(),
                    );
                }

                // Mark NIGHT_DONE
                unsafe { shm_ptr.add(5).write_volatile(ShmState::NightDone as u8) };

                println!("[baker-daemon] night_done epoch={epoch}");
                let _ = write_response(&stream, "night_done", zone_id, epoch, None);
            }
        }
    }

    if !shm_ptr.is_null() {
        unsafe { libc::munmap(shm_ptr as *mut _, shm_len) };
    }

    Ok(())
}

// ── Shard deserialization (mirrors memory.rs originals) ──

fn load_neurons(bytes: &[u8]) -> Result<Vec<PlacedNeuron>> {
    if bytes.len() % 4 != 0 {
        anyhow::bail!("shard.state size not divisible by 4");
    }
    // Positions are in the voltage/flags/etc SoA. We only need positions for
    // sprouting spatial queries. The first pn*4 bytes are voltage (i32).
    // Use a simpler approach: treat first N u32s as packed positions.
    // TODO: proper state deserialization when format is stable.
    let n = (bytes.len() / 910).max(1); // pn estimated from 910 bytes/neuron
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        if off + 4 > bytes.len() { break; }
        let packed = u32::from_le_bytes(bytes[off..off+4].try_into().unwrap());
        let type_idx = ((packed >> 28) & 0xF) as usize;
        out.push(PlacedNeuron { position: packed, type_idx, layer_name: String::new() });
    }
    Ok(out)
}

fn load_axons(bytes: &[u8]) -> Result<Vec<GrownAxon>> {
    if bytes.len() < 4 { anyhow::bail!("shard.axons too small"); }
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut axons = Vec::with_capacity(count);
    let mut offset = 4usize;

    for soma_idx in 0..count {
        if offset + 10 > bytes.len() { break; }
        let tip_x = u16::from_le_bytes(bytes[offset..offset+2].try_into().unwrap()) as u32;
        let tip_y = u16::from_le_bytes(bytes[offset+2..offset+4].try_into().unwrap()) as u32;
        let tip_z = u16::from_le_bytes(bytes[offset+4..offset+6].try_into().unwrap()) as u32;
        let seg_count = u32::from_le_bytes(bytes[offset+6..offset+10].try_into().unwrap()) as usize;
        offset += 10;
        if offset + seg_count * 4 > bytes.len() { break; }
        let mut segments = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            segments.push(u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap()));
            offset += 4;
        }
        let type_idx = segments.first().map(|&s| ((s >> 28) & 0xF) as usize).unwrap_or(0);
        axons.push(GrownAxon { soma_idx, type_idx, tip_x, tip_y, tip_z,
            length_segments: seg_count as u32, segments });
    }
    Ok(axons)
}

// ── Helpers ──

fn open_shm(zone_id: u16, size: usize) -> Result<(*mut u8, usize)> {
    let name = shm_name(zone_id);
    let c_name = CString::new(name.as_str()).unwrap();
    let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
    anyhow::ensure!(fd >= 0, "shm_open failed: {}", std::io::Error::last_os_error());
    anyhow::ensure!(unsafe { libc::ftruncate(fd, size as libc::off_t) } == 0,
        "ftruncate failed: {}", std::io::Error::last_os_error());
    let ptr = unsafe {
        libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED, fd, 0)
    };
    unsafe { libc::close(fd) };
    anyhow::ensure!(ptr != libc::MAP_FAILED, "mmap failed: {}", std::io::Error::last_os_error());
    Ok((ptr as *mut u8, size))
}

fn write_response(mut stream: &std::os::unix::net::UnixStream, cmd: &str,
    zone_id: u16, epoch: u64, msg: Option<&str>) -> Result<()> {
    if let Some(m) = msg {
        writeln!(stream, r#"{{"cmd":"{cmd}","zone_id":{zone_id},"epoch":{epoch},"msg":"{m}"}}"#)?;
    } else {
        writeln!(stream, r#"{{"cmd":"{cmd}","zone_id":{zone_id},"epoch":{epoch}}}"#)?;
    }
    stream.flush().map_err(Into::into)
}

fn extract_u64(json: &str, key: &str) -> Option<u64> {
    let pattern = format!(r#""{key}":"#);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start_matches(' ');
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}
