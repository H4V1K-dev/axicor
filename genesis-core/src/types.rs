/// Абсолютная пространственная единица: 1.0 = 1 мкм.
pub type Microns = f32;

/// Нормализованная координата [0.0, 1.0].
pub type Fraction = f32;

/// Дискретная координата в вокселях.
pub type VoxelCoord = u32;

/// Packed 3D position and neuron type for CPU/Night Phase.
/// Bit layout: [Type(4b) | Z(8b) | Y(10b) | X(10b)]
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedPosition(pub u32);

impl PackedPosition {
    pub const X_MASK: u32    = 0x0000_03FF; // 10 bits (0..1023)
    pub const Y_MASK: u32    = 0x000F_FC00; // 10 bits (0..1023)
    pub const Z_MASK: u32    = 0x0FF0_0000; // 8 bits  (0..255)
    pub const TYPE_MASK: u32 = 0xF000_0000; // 4 bits  (0..15)

    #[inline(always)]
    pub const fn new(x: u32, y: u32, z: u32, type_id: u8) -> Self {
        debug_assert!(x <= 1023 && y <= 1023 && z <= 255 && type_id <= 15, "PackedPosition overflow");
        
        Self(
            (x & 0x3FF) |
            ((y & 0x3FF) << 10) |
            ((z & 0xFF) << 20) |
            (((type_id as u32) & 0xF) << 28)
        )
    }

    #[inline(always)]
    pub const fn x(&self) -> u16 {
        (self.0 & Self::X_MASK) as u16
    }

    #[inline(always)]
    pub const fn y(&self) -> u16 {
        ((self.0 & Self::Y_MASK) >> 10) as u16
    }

    #[inline(always)]
    pub const fn z(&self) -> u8 {
        ((self.0 & Self::Z_MASK) >> 20) as u8
    }

    #[inline(always)]
    pub const fn type_id(&self) -> u8 {
        ((self.0 & Self::TYPE_MASK) >> 28) as u8
    }
}

// --- GPU Runtime Flags ---

pub const FLAG_IS_SPIKING: u8 = 0b0000_0001; // Bit 0
pub const FLAG_TYPE_MASK: u8  = 0b1111_0000; // Bits 4-7

/// Extracts Variant ID (Type ID) from memory flags.
#[inline(always)]
pub const fn extract_variant_id(flags: u8) -> usize {
    ((flags & FLAG_TYPE_MASK) >> 4) as usize
}

// --- Other shared types ---

pub type Tick = u64;
pub type Weight = i16;
pub type Voltage = i32;

/// Axon head position (segment index). AXON_SENTINEL when inactive.
pub type AxonHead = u32;

/// Dendrite target: [31..10] axon_id (22 bits) | [9..0] segment_index (10 bits).
pub type PackedTarget = u32;

/// Индекс сегмента внутри аксона. 10 бит → 0..=1023.
pub type SegmentIndex = u32;

/// Variant ID (0..15)
pub type VariantId = u8;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_position_boundaries() {
        // Max values
        let p = PackedPosition::new(1023, 1023, 255, 15);
        assert_eq!(p.x(), 1023);
        assert_eq!(p.y(), 1023);
        assert_eq!(p.z(), 255);
        assert_eq!(p.type_id(), 15);
        assert_eq!(p.0, 0xFFFFFFFF); // All bits set

        // Zero values
        let p0 = PackedPosition::new(0, 0, 0, 0);
        assert_eq!(p0.x(), 0);
        assert_eq!(p0.y(), 0);
        assert_eq!(p0.z(), 0);
        assert_eq!(p0.type_id(), 0);
        assert_eq!(p0.0, 0);

        // Mixed values
        let pm = PackedPosition::new(123, 456, 78, 9);
        assert_eq!(pm.x(), 123);
        assert_eq!(pm.y(), 456);
        assert_eq!(pm.z(), 78);
        assert_eq!(pm.type_id(), 9);
    }

    #[test]
    fn test_flag_extraction() {
        assert_eq!(extract_variant_id(0b1010_0000), 10);
        assert_eq!(extract_variant_id(0b1111_0001), 15);
        assert_eq!(extract_variant_id(0b0000_0000), 0);
        assert_eq!(extract_variant_id(0b0001_1111), 1);
    }

    #[test]
    fn test_variant_parameters_layout() {
        use crate::layout::VariantParameters;
        // 64B per spec: 16 variants × 64B = 1024B = exactly one CUDA __constant__ block
        assert_eq!(std::mem::size_of::<VariantParameters>(),  64);
        assert_eq!(std::mem::align_of::<VariantParameters>(), 64);
    }

    #[test]
    fn test_columnar_idx() {
        use crate::layout::ShardStateSoA;
        let padded_n = 1024;
        let neuron_idx = 32;
        let slot = 1;
        assert_eq!(ShardStateSoA::columnar_idx(padded_n, neuron_idx, slot), 1056);
    }
}
