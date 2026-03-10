/// Night Phase IPC — Shared Memory layout between genesis-runtime and
/// genesis-baker-daemon.
///
/// SHM name: `/genesis_shard_{zone_hash:08X}`
/// Layout:
///   [0..64)   ShmHeader  (fixed, repr C, 64 bytes)
///   [64..)    weights: i16 × 128 × padded_n  (little-endian)
///             targets: u32 × 128 × padded_n  (little-endian)
///
/// State machine (single-writer invariant):
///   IDLE       → runtime writes              → NIGHT_START
///   NIGHT_START → daemon reads & begins work  → SPROUTING
///   SPROUTING  → daemon writes result         → NIGHT_DONE
///   NIGHT_DONE → runtime reads & resets       → IDLE
///   Any state  → daemon panics               → ERROR

/// Magic number at offset 0 of every SHM segment.
pub const SHM_MAGIC: u32 = 0x47454E53; // "GENS"

/// IPC protocol version. Bump on incompatible ShmHeader changes.
pub const SHM_VERSION: u8 = 2;

pub const MAX_HANDOVERS_PER_NIGHT: usize = 10_000;

use serde::{Serialize, Deserialize};

/// Сетевой пакет межзональной передачи аксона (Half-Duplex SHM Data Plane).
/// MUST remain exactly 20 bytes — SHM layout depends on this.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxonHandoverEvent {
    pub origin_zone_hash: u32, // <--- КРИТИЧНО: Обратный адрес для ACK!
    pub local_axon_id: u32,
    pub entry_x: u16,
    pub entry_y: u16,
    pub vector_x: i8,
    pub vector_y: i8,
    pub vector_z: i8,
    pub type_mask: u8,
    pub remaining_length: u16,
    pub entry_z: u8, // [DOD FIX] Z-координата входа
    pub _padding: u8,
}
const _: () = assert!(
    std::mem::size_of::<AxonHandoverEvent>() == 20,
    "AxonHandoverEvent must be 20 bytes for SHM layout"
);

// [DOD FIX] События структурной пластичности для Dynamic Capacity Routing

/// Подтверждение от соседа, что Ghost-аксон создан. Содержит выделенный слот.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AxonHandoverAck {
    pub target_zone_hash: u32,
    pub src_axon_id: u32,
    pub dst_ghost_id: u32, // Индекс в VRAM соседа
}

/// Уведомление соседу (или себе), что связь разорвана и слот нужно очистить.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AxonHandoverPrune {
    pub target_zone_hash: u32,
    pub dst_ghost_id: u32,
}

pub fn shm_name(zone_hash: u32) -> String {
    format!("genesis_shard_{:08X}", zone_hash)
}

/// Path to the shared memory file. On Unix: /dev/shm/name (or POSIX shm name).
/// On Windows: temp_dir/name (file-backed mmap).
pub fn shm_file_path(zone_hash: u32) -> std::path::PathBuf {
    let name = shm_name(zone_hash);
    #[cfg(unix)]
    {
        std::path::PathBuf::from("/dev/shm").join(&name)
    }
    #[cfg(windows)]
    {
        std::env::temp_dir().join(&name)
    }
}

/// POSIX shm_open name (Unix only). Includes leading slash for shm namespace.
#[cfg(unix)]
pub fn shm_posix_name(zone_hash: u32) -> String {
    format!("/{}", shm_name(zone_hash))
}

/// Default control channel address. Unix: socket path. Windows: TCP host:port.
pub fn default_socket_path(zone_hash: u32) -> String {
    #[cfg(unix)]
    {
        format!("/tmp/genesis_baker_{:08X}.sock", zone_hash)
    }
    #[cfg(windows)]
    {
        let port = default_socket_port(zone_hash);
        format!("127.0.0.1:{}", port)
    }
}

/// TCP port for baker IPC (Windows). Base 19000 + zone_hash % 1000.
#[cfg(windows)]
pub fn default_socket_port(zone_hash: u32) -> u16 {
    (19000 + (zone_hash % 1000)) as u16
}

pub const fn shm_size(padded_n: usize) -> usize {
    let weights_bytes = padded_n * 128 * 2;
    let targets_bytes = padded_n * 128 * 4;
    let handovers_bytes = MAX_HANDOVERS_PER_NIGHT * std::mem::size_of::<AxonHandoverEvent>();
    64 + weights_bytes + targets_bytes + handovers_bytes
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShmHeader {
    pub magic: u32,             // 0..4
    pub version: u8,            // 4..5
    pub state: u8,              // 5..6
    pub _pad: u16,              // 6..8 (бывший zone_id)
    pub padded_n: u32,          // 8..12
    pub dendrite_slots: u32,    // 12..16
    pub weights_offset: u32,    // 16..20
    pub targets_offset: u32,    // 20..24
    pub epoch: u64,             // 24..32
    pub total_axons: u32,       // 32..36
    pub handovers_offset: u32,  // 36..40
    pub handovers_count: u32,   // 40..44
    pub zone_hash: u32,         // 44..48 [DOD FIX: Уникальный ID]
    pub prunes_offset: u32,         // 48..52
    pub prunes_count: u32,          // 52..56
    pub incoming_prunes_count: u32, // 56..60
    pub flags_offset: u32,          // 60..64
}


const _: () = assert!(std::mem::size_of::<ShmHeader>() == 64, "ShmHeader MUST be 64 bytes");

impl ShmHeader {
    pub fn new(zone_hash: u32, padded_n: u32, total_axons: u32) -> Self {
        let weights_offset = std::mem::size_of::<ShmHeader>() as u32;
        let weights_bytes = padded_n * 128 * 2;
        let targets_offset = weights_offset + weights_bytes;
        let targets_bytes = padded_n * 128 * 4;
        let handovers_offset = targets_offset + targets_bytes;

        Self {
            magic: SHM_MAGIC,
            version: SHM_VERSION,
            state: ShmState::Idle as u8,
            _pad: 0,
            padded_n,
            dendrite_slots: 128,
            weights_offset,
            targets_offset,
            epoch: 0,
            total_axons,
            handovers_offset,
            handovers_count: 0,
            zone_hash,
            prunes_offset: 0,
            prunes_count: 0,
            incoming_prunes_count: 0,
            flags_offset: 0,
        }
    }

    /// Validate a header read from shared memory.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.magic != SHM_MAGIC {
            return Err("SHM magic mismatch");
        }
        if self.version != SHM_VERSION {
            return Err("SHM version mismatch");
        }
        if self.dendrite_slots != 128 {
            return Err("SHM dendrite_slots != 128");
        }
        Ok(())
    }
}

/// State machine for the SHM Night Phase protocol.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShmState {
    /// SHM ready. Daemon waiting. Runtime may start Night Phase.
    Idle = 0,
    /// Runtime wrote weights+targets. Daemon should start Sprouting.
    NightStart = 1,
    /// Daemon is running Sprouting. Do not touch SHM data.
    Sprouting = 2,
    /// Daemon finished. Updated targets ready for runtime to read.
    NightDone = 3,
    /// Daemon encountered an error. Runtime should skip this night.
    Error = 4,
}

impl ShmState {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Idle),
            1 => Some(Self::NightStart),
            2 => Some(Self::Sprouting),
            3 => Some(Self::NightDone),
            4 => Some(Self::Error),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Trigger-Only UDS IPC (Strike 3)
// ---------------------------------------------------------------------------

pub const BAKE_MAGIC: u32 = 0x42414B45;       // "BAKE"
pub const BAKE_READY_MAGIC: u32 = 0x424B4F4B; // "BKOK"

/// Lightweight trigger for the Baker Daemon.
/// Exactly 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BakeRequest {
    pub magic: u32,           // BAKE_MAGIC
    pub zone_hash: u32,       // FNV-1a of zone name
    pub current_tick: u32,
    pub prune_threshold: i16,
    pub _padding: u16,
}

const _: () = assert!(std::mem::size_of::<BakeRequest>() == 16, "BakeRequest must be 16 bytes");

// =============================================================================
// §2  File-format IPC: baked binary blobs (.gxi / .gxo / .ghosts)
// =============================================================================

/// Sentinel written to `mapped_soma_ids` for a GXO pixel that has no somas.
/// GPU RecordReadout kernel must Early-Exit when it sees this value.
pub const EMPTY_PIXEL: u32 = 0xFFFF_FFFF;

/// Header of a baked input-matrix blob (.gxi).
/// Exactly 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GxiHeader {
    pub magic:        u32,       // GXI_MAGIC (0x47584900)
    pub zone_hash:    u32,       // FNV-1a of zone name
    pub matrix_hash:  u32,       // FNV-1a of matrix name
    pub input_count:  u32,       // Number of virtual axons
    pub total_pixels: u32,       // W × H
    pub _padding:     [u32; 3],  // Reserved; always zero
}
const _: () = assert!(std::mem::size_of::<GxiHeader>() == 32, "GxiHeader must be 32 bytes");

impl GxiHeader {
    pub fn new(zone_hash: u32, matrix_hash: u32, total_pixels: u32) -> Self {
        Self {
            magic:        crate::constants::GXI_MAGIC,
            zone_hash,
            matrix_hash,
            input_count:  total_pixels,
            total_pixels,
            _padding:     [0; 3],
        }
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// Header of a baked output-matrix blob (.gxo).
/// Exactly 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GxoHeader {
    pub magic:        u32,       // GXO_MAGIC (0x47584F00)
    pub zone_hash:    u32,       // FNV-1a of zone name
    pub matrix_hash:  u32,       // FNV-1a of matrix name
    pub output_count: u32,       // Number of mapped somas (NOT sentinel-filled pixels)
    pub _padding:     [u32; 4],  // Reserved; always zero
}
const _: () = assert!(std::mem::size_of::<GxoHeader>() == 32, "GxoHeader must be 32 bytes");

impl GxoHeader {
    pub fn new(zone_hash: u32, matrix_hash: u32, output_count: u32) -> Self {
        Self {
            magic: crate::constants::GXO_MAGIC,
            zone_hash,
            matrix_hash,
            output_count,
            _padding: [0; 4],
        }
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// Header of an inter-zone ghost-routing blob (.ghosts).
/// Exactly 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GhostsHeader {
    pub magic:            u32,   // GHST_MAGIC (0x47485354)
    pub from_zone_hash:   u32,   // FNV-1a of source zone name
    pub to_zone_hash:     u32,   // FNV-1a of destination zone name
    pub connection_count: u32,   // Length of the GhostConnection array that follows
}
const _: () = assert!(std::mem::size_of::<GhostsHeader>() == 16, "GhostsHeader must be 16 bytes");

impl GhostsHeader {
    pub fn new(from_zone_hash: u32, to_zone_hash: u32, connection_count: u32) -> Self {
        Self {
            magic: GHST_MAGIC,
            from_zone_hash,
            to_zone_hash,
            connection_count,
        }
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// A single inter-zone connection record written into a .ghosts blob.
/// Exactly 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GhostConnection {
    pub src_soma_id:    u32,     // Dense ID of the soma in Zone A (from .gxo)
    pub target_ghost_id: u32,   // Index of the ghost axon in Zone B
}
const _: () = assert!(std::mem::size_of::<GhostConnection>() == 8, "GhostConnection must be 8 bytes");

impl GhostConnection {
    #[inline(always)]
    pub fn slice_as_bytes(slice: &[Self]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                slice.len() * std::mem::size_of::<Self>(),
            )
        }
    }
}

use bytemuck::{Pod, Zeroable};

/// Header for inter-node spike batches.
/// Exactly 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct SpikeBatchHeader {
    pub magic:    u32, // 0x5350494B ("SPIK")
    pub batch_id: u32, // Batch counter for ordering
}
const _: () = assert!(std::mem::size_of::<SpikeBatchHeader>() == 8, "SpikeBatchHeader must be 8 bytes");

/// A single spike event recorded for inter-zone transfer or readout.
/// Exactly 8 bytes, repr(C) for Coalesced Access on GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct SpikeEvent {
    pub ghost_axon_id: u32, // Direct index in axon_heads[] of receiver
    pub tick_offset:   u32, // Offset within the batch
}
const _: () = assert!(std::mem::size_of::<SpikeEvent>() == 8, "SpikeEvent must be 8 bytes");

// ===========================================================================
// IDE Telemetry (Step 14)
// ===========================================================================

pub const TELE_MAGIC: u32 = 0x454C4554; // "TELE" in Little-Endian

/// Header for binary telemetry frames sent over WebSocket.
/// Exactly 16 bytes, repr(C).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct TelemetryFrameHeader {
    pub magic:        u32, // TELE_MAGIC
    pub tick:         u32, // Simulation tick
    pub spikes_count: u32, // Number of fired neurons in this frame
    pub _padding:     u32, // 16-byte alignment
}
const _: () = assert!(std::mem::size_of::<TelemetryFrameHeader>() == 16, "TelemetryFrameHeader must be 16 bytes");

/// Magic bytes for inter-zone ghost routing file.
pub const GHST_MAGIC: u32 = 0x47485354; // "GHST"

#[cfg(test)]
mod file_ipc_tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<GxiHeader>(),     32);
        assert_eq!(std::mem::size_of::<GxoHeader>(),     32);
        assert_eq!(std::mem::size_of::<GhostsHeader>(),  16);
        assert_eq!(std::mem::size_of::<GhostConnection>(), 8);
    }

    #[test]
    fn test_gxi_header_magic() {
        let h = GxiHeader::new(0xDEAD, 0xBEEF, 64);
        assert_eq!(h.magic, crate::constants::GXI_MAGIC);
        assert_eq!(h.total_pixels, 64);
        assert_eq!(h.input_count, 64);
        assert_eq!(h._padding, [0; 3]);
    }

    #[test]
    fn test_gxo_header_magic() {
        let h = GxoHeader::new(0x1234, 0x5678, 30);
        assert_eq!(h.magic, crate::constants::GXO_MAGIC);
        assert_eq!(h.output_count, 30);
        assert_eq!(h._padding, [0; 4]);
    }

    #[test]
    fn test_ghosts_header() {
        let h = GhostsHeader::new(0xAABB, 0xCCDD, 5);
        assert_eq!(h.magic, GHST_MAGIC);
        assert_eq!(h.connection_count, 5);
    }

    #[test]
    fn test_gxi_as_bytes_length() {
        let h = GxiHeader::new(1, 2, 16);
        assert_eq!(h.as_bytes().len(), 32);
    }

    #[test]
    fn test_gxo_as_bytes_roundtrip() {
        let h = GxoHeader::new(0, 0, 99);
        let b = h.as_bytes();
        // First 4 bytes are GXO_MAGIC in little-endian
        let magic = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        assert_eq!(magic, crate::constants::GXO_MAGIC);
    }

    #[test]
    fn test_ghost_connection_slice_bytes() {
        let conns = [
            GhostConnection { src_soma_id: 10, target_ghost_id: 20 },
            GhostConnection { src_soma_id: 11, target_ghost_id: 21 },
        ];
        assert_eq!(GhostConnection::slice_as_bytes(&conns).len(), 16);
    }

    #[test]
    fn test_empty_pixel_sentinel() {
        assert_eq!(EMPTY_PIXEL, 0xFFFF_FFFF);
    }
}

/// Header for external asynchronous I/O (Sensors/Motors) over UDP.
/// Exactly 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct ExternalIoHeader {
    pub magic:        u32, // GSIO_MAGIC or GSOO_MAGIC
    pub zone_hash:    u32, // FNV-1a of zone name
    pub matrix_hash:  u32, // FNV-1a of matrix name
    pub payload_size: u32, // Length of data following this header
    pub global_reward: i16, // [DOD] R-STDP Dopamine Modulator
    pub _padding:     u16,
}
const _: () = assert!(std::mem::size_of::<ExternalIoHeader>() == 20, "ExternalIoHeader must be 20 bytes");

impl ExternalIoHeader {
    pub fn new(magic: u32, zone_hash: u32, matrix_hash: u32, payload_size: u32) -> Self {
        Self {
            magic,
            zone_hash,
            matrix_hash,
            payload_size,
            global_reward: 0,
            _padding: 0,
        }
    }
}
// ===========================================================================
// §3 Self-Healing & Replication (Step 19)
// ===========================================================================

pub const SNAP_MAGIC: u32 = 0x50414E53; // "SNAP"

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct ShardStateHeader {
    pub magic: u32,
    pub zone_hash: u32,
    pub tick: u32,
    pub _padding1: u32,      // Добиваем до 16 байт для выравнивания u64
    pub payload_size: u64,   // Смещено на 16 байт
    pub _padding2: [u64; 1], // Добиваем еще 8 байт до 32
}

const _: () = assert!(std::mem::size_of::<ShardStateHeader>() == 32);

pub const ROUT_MAGIC: u32 = 0x54554F52; // "ROUT"

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct RouteUpdate {
    pub magic: u32,
    pub zone_hash: u32,
    pub new_ipv4: u32, // u32 representation of IPv4Addr
    pub new_port: u16,
    pub _padding: u16,
}

const _: () = assert!(std::mem::size_of::<RouteUpdate>() == 16);
