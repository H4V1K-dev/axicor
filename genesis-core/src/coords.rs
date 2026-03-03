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

#[inline]
pub fn pack_position(x: u32, y: u32, z: u32, type_mask: u32) -> PackedPosition {
    PackedPosition::new(x, y, z, type_mask as u8)
}

#[inline]
pub fn unpack_position(p: PackedPosition) -> (u32, u32, u32, u32) {
    (p.x() as u32, p.y() as u32, p.z() as u32, p.type_id() as u32)
}

// ---------------------------------------------------------------------------
// §1.2 PackedTarget — идентификатор (Axon_ID, Segment_Index)
// ---------------------------------------------------------------------------

use crate::layout::{pack_dendrite_target, unpack_axon_id, unpack_segment_offset};
use crate::types::PackedTarget;

/// Упаковывает `(axon_id, segment_idx)` в `PackedTarget`.
/// Layout: [31..24] segment_offset (8 бита) | [23..0] axon_id + 1 (24 бита).
#[inline]
pub fn pack_target(axon_id: u32, segment_idx: u32) -> PackedTarget {
    pack_dendrite_target(axon_id, segment_idx)
}

/// Распаковывает `PackedTarget` в `(axon_id, segment_idx)`.
/// Возвращает `None` если `t == 0` (пустой дендритный слот).
#[inline]
pub fn unpack_target(t: PackedTarget) -> Option<(u32, u32)> {
    if t == 0 { return None; }
    Some((unpack_axon_id(t), unpack_segment_offset(t)))
}
