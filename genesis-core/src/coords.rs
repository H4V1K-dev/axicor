/// Пространственные координаты, конвертеры и упаковка (Spec 01 §1.1–1.3).
///
/// Три системы координат (§1.1):
///   Microns    — абсолютная (1.0 = 1 мкм), для физики и геометрии
///   Fraction   — нормализованная [0.0, 1.0], для границ слоёв
///   VoxelCoord — дискретная, для GPU и пространственного хэширования
///
/// PackedPosition layout: [Type(4b) | Z(8b) | Y(10b) | X(10b)]
/// Бит-раскладка: type << 28 | z << 20 | y << 10 | x
///
/// Диапазоны:
///   X: 0..=1023  (10 бит)
///   Y: 0..=1023  (10 бит)
///   Z: 0..=255   (8 бит)
///   type_mask: 0..=15 (4 бита)
use crate::types::{Fraction, Microns, PackedPosition, VoxelCoord};

// ---------------------------------------------------------------------------
// §1.1 Конвертеры между системами координат
// ---------------------------------------------------------------------------

/// Абсолютные мкм → воксели (дискретизация пространства).
/// `voxel_size_um` берётся из `constants::VOXEL_SIZE_UM` или из конфига.
#[inline]
pub fn um_to_voxel(um: Microns, voxel_size_um: u32) -> VoxelCoord {
    (um / voxel_size_um as Microns) as VoxelCoord
}

/// Нормализованная доля [0.0, 1.0] → воксели.
/// Используется для перевода `height_pct` / `population_pct` слоёв в воксельные границы.
/// `world_dim_vox` — размер измерения мира в вокселях (например, world_h_vox).
#[inline]
pub fn pct_to_voxel(pct: Fraction, world_dim_vox: u32) -> VoxelCoord {
    (pct * world_dim_vox as Fraction) as VoxelCoord
}

/// Воксели → абсолютные мкм.
#[inline]
pub fn voxel_to_um(vox: VoxelCoord, voxel_size_um: u32) -> Microns {
    vox as Microns * voxel_size_um as Microns
}

/// Упаковывает воксельные координаты и тип нейрона в PackedPosition.
/// Все аргументы проверяются debug_assert в дебаг-билдах.
#[inline]
pub fn pack_position(x: u32, y: u32, z: u32, type_mask: u32) -> PackedPosition {
    debug_assert!(x < 1024, "X={x} exceeds 10-bit range (0..=1023)");
    debug_assert!(y < 1024, "Y={y} exceeds 10-bit range (0..=1023)");
    debug_assert!(z < 256,  "Z={z} exceeds 8-bit range (0..=255)");
    debug_assert!(type_mask < 16, "type_mask={type_mask} exceeds 4-bit range (0..=15)");
    (type_mask << 28) | (z << 20) | (y << 10) | x
}

/// Распаковывает PackedPosition в `(x, y, z, type_mask)`.
#[inline]
pub fn unpack_position(p: PackedPosition) -> (u32, u32, u32, u32) {
    let x         = p & 0x3FF;
    let y         = (p >> 10) & 0x3FF;
    let z         = (p >> 20) & 0xFF;
    let type_mask = p >> 28;
    (x, y, z, type_mask)
}

// ---------------------------------------------------------------------------
// §1.2 PackedTarget — идентификатор (Axon_ID, Segment_Index)
// ---------------------------------------------------------------------------

use crate::constants::{TARGET_AXON_SHIFT, TARGET_SEG_MASK};
use crate::types::PackedTarget;

/// Упаковывает `(axon_id, segment_idx)` в `PackedTarget`.
/// Layout: [31..10] axon_id (22 бита) | [9..0] segment_idx (10 бит).
///
/// Ёмкость шарда: до 4 194 303 аксонов × 1023 сегментов.
#[inline]
pub fn pack_target(axon_id: u32, segment_idx: u32) -> PackedTarget {
    debug_assert!(axon_id < (1 << 22), "axon_id={axon_id} exceeds 22-bit range");
    debug_assert!(segment_idx <= TARGET_SEG_MASK, "segment_idx={segment_idx} exceeds 10-bit range");
    (axon_id << TARGET_AXON_SHIFT) | (segment_idx & TARGET_SEG_MASK)
}

/// Распаковывает `PackedTarget` в `(axon_id, segment_idx)`.
/// Возвращает `None` если `t == 0` (пустой дендритный слот).
#[inline]
pub fn unpack_target(t: PackedTarget) -> Option<(u32, u32)> {
    if t == 0 { return None; }
    Some((t >> TARGET_AXON_SHIFT, t & TARGET_SEG_MASK))
}
