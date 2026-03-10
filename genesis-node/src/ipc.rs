/// genesis-runtime IPC client — communicates with genesis-baker-daemon.
///
/// Transport:
///   - Data:    File-backed mmap (cross-platform)
///   - Control: Unix socket (Linux) / TCP (Windows)
///
/// Usage:
///   1. Call `BakerClient::connect(zone_hash, socket_path)` at startup.
///   2. Call `run_night(weights, targets, padded_n, timeout)` during Night Phase.
///   3. Returns updated targets (with sprouted connections filled in).
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use genesis_core::ipc::{shm_file_path, shm_size, ShmHeader, ShmState};

/// Runtime-side IPC client for the baker daemon.
pub struct BakerClient {
    zone_hash: u32,
    socket_addr: String,
    pub shm_ptr: *mut u8,
    pub shm_len: usize,
    _mmap: memmap2::MmapMut,
}

// SAFETY: BakerClient is not Send/Sync by default due to raw pointer.
// We implement them manually — the mmap region is only accessed from the
// Night Phase (single-threaded path in runtime main loop).
unsafe impl Send for BakerClient {}
unsafe impl Sync for BakerClient {}

impl BakerClient {
    /// Open and mmap the SHM file, then validate the header written by daemon.
    /// The daemon must already be running and have created the SHM before this is called.
    pub fn connect(zone_hash: u32, socket_path: &Path) -> Result<Self> {
        let shm_path = shm_file_path(zone_hash);

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&shm_path)
            .with_context(|| format!("Cannot open SHM file {:?}", shm_path))?;

        let header_size = std::mem::size_of::<ShmHeader>();
        let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };

        if mmap.len() < header_size {
            bail!("SHM file too small");
        }

        let hdr = unsafe { std::ptr::read(mmap.as_ptr() as *const ShmHeader) };
        hdr.validate()
            .map_err(|e| anyhow::anyhow!("SHM header invalid: {e}"))?;

        let shm_len = shm_size(hdr.padded_n as usize);
        if mmap.len() < shm_len {
            bail!("SHM file size {} < expected {}", mmap.len(), shm_len);
        }

        let socket_addr = socket_path.to_string_lossy().into_owned();

        Ok(Self {
            zone_hash,
            socket_addr,
            shm_ptr: mmap.as_mut_ptr(),
            shm_len,
            _mmap: mmap,
        })
    }

    /// Run one Night Phase Sprouting cycle:
    /// 1. Zero-Copy Write Handovers into Shared Memory
    /// 2. Binary Trigger (16 bytes) - Fast Path
    /// 3. Wait for Binary Ack (4 bytes)
    pub fn run_night(
        &mut self,
        handovers: &[genesis_core::ipc::AxonHandoverEvent],
        _padded_n: usize,
        timeout: Duration,
        prune_threshold: i16,
    ) -> Result<Vec<genesis_core::ipc::AxonHandoverAck>> {
        if handovers.len() > genesis_core::ipc::MAX_HANDOVERS_PER_NIGHT {
            bail!("Too many handovers: {} > {}", handovers.len(), genesis_core::ipc::MAX_HANDOVERS_PER_NIGHT);
        }

        // ── 1. Zero-Copy Write Handovers into Shared Memory ──
        unsafe {
            let hdr_ptr = self.shm_ptr as *mut ShmHeader;
            let dest = self.shm_ptr.add((*hdr_ptr).handovers_offset as usize) as *mut genesis_core::ipc::AxonHandoverEvent;

            std::ptr::copy_nonoverlapping(handovers.as_ptr(), dest, handovers.len());
            (*hdr_ptr).handovers_count = handovers.len() as u32;
        }

        // ── 2. Binary Trigger (16 bytes) - Fast Path ──
        let mut stream = self.connect_stream()?;
        stream.set_read_timeout(Some(timeout))?;

        let req = genesis_core::ipc::BakeRequest {
            magic: genesis_core::ipc::BAKE_MAGIC,
            zone_hash: self.zone_hash,
            current_tick: 0,
            prune_threshold,
            _padding: 0,
        };

        unsafe {
            let req_bytes = std::slice::from_raw_parts(
                &req as *const _ as *const u8,
                std::mem::size_of::<genesis_core::ipc::BakeRequest>(),
            );
            stream.write_all(req_bytes)?;
        }
        stream.flush()?;

        // ── 3. Wait for Binary Ack (4 bytes) ──
        let mut ack = [0u8; 4];
        stream.read_exact(&mut ack).context("Waiting for baker BKOK")?;
        let magic_resp = u32::from_le_bytes(ack);

        if magic_resp != genesis_core::ipc::BAKE_READY_MAGIC {
            self.set_state(ShmState::Idle);
            bail!("Baker daemon returned error magic: {:08X}", magic_resp);
        }

        // Читаем ACKs
        let mut count_buf = [0u8; 4];
        stream.read_exact(&mut count_buf).context("Reading ACK count")?;
        let ack_count = u32::from_le_bytes(count_buf) as usize;
        
        let mut acks = vec![genesis_core::ipc::AxonHandoverAck { target_zone_hash: 0, src_axon_id: 0, dst_ghost_id: 0 }; ack_count];
        if ack_count > 0 {
            let bytes = unsafe { 
                std::slice::from_raw_parts_mut(acks.as_mut_ptr() as *mut u8, ack_count * std::mem::size_of::<genesis_core::ipc::AxonHandoverAck>()) 
            };
            stream.read_exact(bytes).context("Reading ACK payloads")?;
        }

        self.set_state(ShmState::Idle);
        Ok(acks)
    }

    #[cfg(unix)]
    fn connect_stream(&self) -> Result<std::os::unix::net::UnixStream> {
        use std::os::unix::net::UnixStream;
        UnixStream::connect(&self.socket_addr)
            .with_context(|| format!("Cannot connect to baker socket {}", self.socket_addr))
    }

    #[cfg(windows)]
    fn connect_stream(&self) -> Result<std::net::TcpStream> {
        std::net::TcpStream::connect(&self.socket_addr)
            .with_context(|| format!("Cannot connect to baker TCP {}", self.socket_addr))
    }

    fn set_state(&mut self, state: ShmState) {
        unsafe {
            self.shm_ptr.add(5).write_volatile(state as u8);
        }
    }
}

// Drop: _mmap handles unmapping automatically
