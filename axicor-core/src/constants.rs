/// Maximum spikes per tick in the spike schedule ring buffer.
pub const MAX_SPIKES_PER_TICK: usize = 1024;

/// LTM / WM boundary slot index.
pub const WM_SLOT_START: usize = 80;

/// Hard Constraint: Exactly 128 dendrites per soma.
/// Guarantees memory alignment (Columnar Layout) and 100% L1 cache line utilization.
pub const MAX_DENDRITE_SLOTS: usize = 128;

/// Sentinel value for inactive axons.
/// When shifted (0x80000000 - seg_idx) yields a negative number, disabling the Active Tail.
pub const AXON_SENTINEL: u32 = 0x80000000;

/// Trigger for Maintenance Pipeline: reset overflowed axons (every ~50 simulation hours)
pub const SENTINEL_REFRESH_TICKS: u64 = 1_800_000_000;
pub const SENTINEL_DANGER_THRESHOLD: u32 = 0x7000_0000;

/// target_packed bit layout: [31..24] Segment_Offset (8 bits) | [23..0] Axon_ID + 1 (24 bits)
pub const TARGET_AXON_MASK: u32 = 0x00FF_FFFF;
pub const TARGET_SEG_SHIFT: u32 = 24;

/// Warp size for GPU alignment (padded_n must be multiple of this).
#[cfg(feature = "amd")]
pub const GPU_WARP_SIZE: usize = 64;
#[cfg(not(feature = "amd"))]
pub const GPU_WARP_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Physical constants (Spec 01 1.6)  Fixed configuration
// Any changes to these require V_SEG recalculation and compiler check.
// ---------------------------------------------------------------------------

/// Time step: 100 s = 0.1 ms.
pub const TICK_DURATION_US: u32 = 100;

/// Voxel size in m.
pub const VOXEL_SIZE_UM: u32 = 25;

/// Length of one axon segment in voxels.
pub const SEGMENT_LENGTH_VOXELS: u32 = 2;

/// Segment length in m (= VOXEL_SIZE_UM  SEGMENT_LENGTH_VOXELS).
pub const SEGMENT_LENGTH_UM: u32 = VOXEL_SIZE_UM * SEGMENT_LENGTH_VOXELS; // 50

/// Signal speed in m/tick (0.5 m/s = 50 m/tick).
pub const SIGNAL_SPEED_UM_TICK: u32 = 50;

/// Discrete speed: segments per tick. MUST be an integer.
pub const V_SEG: u32 = SIGNAL_SPEED_UM_TICK / SEGMENT_LENGTH_UM; // 1

/// Invariant 1.6: signal_speed_um_tick MUST be divisible by segment_length_um without remainder.
/// If v_seg is fractional, the GPU cannot operate without floats, violating Integer Physics.
#[allow(clippy::eq_op)]
const _: () = assert!(
    SIGNAL_SPEED_UM_TICK.is_multiple_of(SEGMENT_LENGTH_UM),
    "Spec 01 1.6 violation: signal_speed_um_tick must be divisible by segment_length_um (v_seg must be integer)"
);

// ==========================================
// Binary format magic numbers (Little-Endian)
// ==========================================

/// Input Mapping file header (.gxi)
pub const GXI_MAGIC: u32 = 0x47584900; // "GXI\0"

/// Output Mapping file header (.gxo)
pub const GXO_MAGIC: u32 = 0x47584F00; // "GXO\0"

/// UDP telemetry header (IDE WebSocket & Fast Path)
pub const TELEMETRY_MAGIC: u32 = 0x474E5353; // "GNSS"

/// External I/O: Magic numbers for UDP packets
pub const GSIO_MAGIC: u32 = 0x4F495347; // "GSIO" (Input)
pub const GSOO_MAGIC: u32 = 0x4F4F5347; // "GSOO" (Output)

/// Maximum UDP payload size (MTU limit)
pub const MAX_UDP_PAYLOAD: usize = 65507;
