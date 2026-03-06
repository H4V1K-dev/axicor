/// genesis-runtime IPC client — communicates with genesis-baker-daemon.
///
/// Transport:
///   - Data:    POSIX SHM `/genesis_shard_{zone_hash:08X}` (mmap, no copies)
///   - Control: Unix domain socket (JSON-line, single command per Night Phase)
///
/// Usage:
///   1. Call `BakerClient::connect(zone_hash, socket_path)` at startup.
///   2. Call `run_night(weights, targets, padded_n, timeout)` during Night Phase.
///   3. Returns updated targets (with sprouted connections filled in).
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use genesis_core::ipc::{shm_name, shm_size, ShmHeader, ShmState};

// POSIX SHM wrappers (libc calls)
use std::ffi::CString;

/// Runtime-side IPC client for the baker daemon.
pub struct BakerClient {
    zone_hash: u32,
    socket_path: std::path::PathBuf,
    pub shm_ptr: *mut u8,
    pub shm_len: usize,
}

// SAFETY: BakerClient is not Send/Sync by default due to raw pointer.
// We implement them manually — the mmap region is only accessed from the
// Night Phase (single-threaded path in runtime main loop).
unsafe impl Send for BakerClient {}
unsafe impl Sync for BakerClient {}

impl BakerClient {
    /// Open and mmap the SHM segment, then validate the header written by daemon.
    /// The daemon must already be running and have created the SHM before this is called.
    pub fn connect(zone_hash: u32, socket_path: &Path) -> Result<Self> {
        let name = shm_name(zone_hash);
        let c_name = CString::new(name.as_str()).unwrap();

        // Open existing SHM segment (daemon creates it at startup)
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0o600) };
        if fd < 0 {
            bail!(
                "shm_open({}) failed: {}",
                name,
                std::io::Error::last_os_error()
            );
        }

        // Read header to learn the real size
        let header_size = std::mem::size_of::<ShmHeader>();
        let mut hdr = std::mem::MaybeUninit::<ShmHeader>::uninit();
        let n = unsafe { libc::read(fd, hdr.as_mut_ptr() as *mut _, header_size) };
        if n < header_size as isize {
            unsafe { libc::close(fd) };
            bail!("SHM too small to read header");
        }
        let hdr = unsafe { hdr.assume_init() };
        hdr.validate()
            .map_err(|e| anyhow::anyhow!("SHM header invalid: {e}"))?;

        let shm_len = shm_size(hdr.padded_n as usize);

        // Map the full segment
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                shm_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        unsafe { libc::close(fd) };

        if ptr == libc::MAP_FAILED {
            bail!("mmap failed: {}", std::io::Error::last_os_error());
        }

        Ok(Self {
            zone_hash,
            socket_path: socket_path.to_path_buf(),
            shm_ptr: ptr as *mut u8,
            shm_len,
        })
    }

    /// Run one Night Phase Sprouting cycle:
    /// 1. Write weights+targets into SHM
    /// 2. Signal daemon (`night_start`)
    /// 3. Wait for `night_done` (or `error`) with timeout
    /// 4. Return updated targets from SHM
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
    ) -> Result<()> {
        if handovers.len() > genesis_core::ipc::MAX_HANDOVERS_PER_NIGHT {
            bail!("Too many handovers: {} > {}", handovers.len(), genesis_core::ipc::MAX_HANDOVERS_PER_NIGHT);
        }

        // ── 1. Zero-Copy Write Handovers into Shared Memory ──
        unsafe {
            let hdr_ptr = self.shm_ptr as *mut ShmHeader;
            let dest = self.shm_ptr.add((*hdr_ptr).handovers_offset as usize) as *mut genesis_core::ipc::AxonHandoverEvent;
            
            // Прямое копирование из RAM в SHM, видимую демоном (DMA-style)
            std::ptr::copy_nonoverlapping(handovers.as_ptr(), dest, handovers.len());
            (*hdr_ptr).handovers_count = handovers.len() as u32;
        }

        // ── 2. Binary Trigger (16 bytes) - Fast Path ──
        let mut stream = UnixStream::connect(&self.socket_path)
            .with_context(|| format!("Cannot connect to baker socket {:?}", self.socket_path))?;
        stream.set_read_timeout(Some(timeout))?;

        let req = genesis_core::ipc::BakeRequest {
            magic: genesis_core::ipc::BAKE_MAGIC,
            zone_hash: self.zone_hash,
            current_tick: 0, 
            prune_threshold, // [DOD FIX] Передано аргументом
            _padding: 0,
        };

        unsafe {
            let req_bytes = std::slice::from_raw_parts(
                &req as *const _ as *const u8, 
                std::mem::size_of::<genesis_core::ipc::BakeRequest>()
            );
            stream.write_all(req_bytes)?;
        }
        stream.flush()?;

        // ── 3. Wait for Binary Ack (4 bytes) ──
        let mut ack = [0u8; 4];
        stream.read_exact(&mut ack).context("Waiting for baker BKOK")?;
        let magic_resp = u32::from_le_bytes(ack);

        if magic_resp == genesis_core::ipc::BAKE_READY_MAGIC {
            self.set_state(ShmState::Idle);
            Ok(())
        } else {
            self.set_state(ShmState::Idle);
            bail!("Baker daemon returned error magic: {:08X}", magic_resp);
        }
    }

    fn _header(&self) -> ShmHeader {
        unsafe { std::ptr::read(self.shm_ptr as *const ShmHeader) }
    }

    fn set_state(&mut self, state: ShmState) {
        // state is at byte offset 5 in ShmHeader
        unsafe {
            self.shm_ptr.add(5).write_volatile(state as u8);
        }
    }
}

impl Drop for BakerClient {
    fn drop(&mut self) {
        if !self.shm_ptr.is_null() {
            unsafe { libc::munmap(self.shm_ptr as *mut _, self.shm_len) };
        }
    }
}
