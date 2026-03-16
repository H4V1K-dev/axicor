/// Maximum spikes per tick in the spike schedule ring buffer.
pub const MAX_SPIKES_PER_TICK: usize = 1024;

/// LTM / WM boundary slot index.
pub const WM_SLOT_START: usize = 80;

/// Hard Constraint: Ровно 128 дендритов на сому. 
/// Гарантирует выравнивание памяти (Columnar Layout) и 100% утилизацию кэш-линий L1.
pub const MAX_DENDRITE_SLOTS: usize = 128;

/// Sentinel-значение для неактивных аксонов. 
/// При сдвиге (0x80000000 - seg_idx) даёт отрицательное число, отключая Active Tail.
pub const AXON_SENTINEL: u32 = 0x80000000;

/// Триггер для Maintenance Pipeline: сброс переполненных аксонов (каждые ~50 часов симуляции)
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
// Физические константы (Spec 01 §1.6) — Фиксированная конфигурация
// Изменение любого из них требует пересчёта V_SEG и проверки компилятора.
// ---------------------------------------------------------------------------

/// Шаг времени: 100 мкс = 0.1 мс.
pub const TICK_DURATION_US: u32 = 100;

/// Размер вокселя в мкм.
pub const VOXEL_SIZE_UM: u32 = 25;

/// Длина одного сегмента аксона в вокселях.
pub const SEGMENT_LENGTH_VOXELS: u32 = 2;

/// Длина сегмента в мкм (= VOXEL_SIZE_UM × SEGMENT_LENGTH_VOXELS).
pub const SEGMENT_LENGTH_UM: u32 = VOXEL_SIZE_UM * SEGMENT_LENGTH_VOXELS; // 50

/// Скорость сигнала в мкм/тик (0.5 м/с = 50 мкм/тик).
pub const SIGNAL_SPEED_UM_TICK: u32 = 50;

/// Дискретная скорость: сегментов за тик. Обязана быть целым числом.
pub const V_SEG: u32 = SIGNAL_SPEED_UM_TICK / SEGMENT_LENGTH_UM; // 1

/// Инвариант §1.6: signal_speed_um_tick ОБЯЗАНА делиться на segment_length_um без остатка.
/// Если v_seg дробное — GPU не может работать без флоатов — нарушение Integer Physics.
#[allow(clippy::eq_op)]
const _: () = assert!(
    SIGNAL_SPEED_UM_TICK % SEGMENT_LENGTH_UM == 0,
    "Spec 01 §1.6 violation: signal_speed_um_tick must be divisible by segment_length_um (v_seg must be integer)"
);

// ==========================================
// Магические числа бинарных форматов (Little-Endian)
// ==========================================

/// Заголовок файла Input Mapping (.gxi)
pub const GXI_MAGIC: u32 = 0x47584900; // "GXI\0"

/// Заголовок файла Output Mapping (.gxo)
pub const GXO_MAGIC: u32 = 0x47584F00; // "GXO\0"

/// Заголовок UDP-телеметрии (IDE WebSocket & Fast Path)
pub const TELEMETRY_MAGIC: u32 = 0x474E5353; // "GNSS"

/// Внешний I/O: Магические числа для UDP пакетов
pub const GSIO_MAGIC: u32 = 0x4F495347; // "GSIO" (Input)
pub const GSOO_MAGIC: u32 = 0x4F4F5347; // "GSOO" (Output)

/// Максимальный размер полезной нагрузки UDP (MTU limit)
pub const MAX_UDP_PAYLOAD: usize = 65507;