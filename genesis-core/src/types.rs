/// Абсолютная пространственная единица: 1.0 = 1 мкм.
pub type Microns = f32;

/// Нормализованная координата [0.0, 1.0].
pub type Fraction = f32;

/// Дискретная координата в вокселях.
pub type VoxelCoord = u32;

use bytemuck::{Pod, Zeroable};

/// Packed 3D position and neuron type for CPU/Night Phase.
/// Bit layout: [Type(4b) | Z(6b) | Y(11b) | X(11b)]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq, Eq, Hash)]
pub struct PackedPosition(pub u32);

impl PackedPosition {
    /// Упаковывает сырые индексы вокселей и тип в один u32 регистр.
    /// Layout: X (11 bit) | Y (11 bit) | Z (6 bit) | Type (4 bit)
    #[inline(always)]
    pub fn pack_raw(x_idx: u32, y_idx: u32, z_idx: u32, type_idx: u8) -> Self {
        let x_q = x_idx & 0x7FF; 
        let y_q = y_idx & 0x7FF;
        let z_q = z_idx & 0x3F;  
        let t_q = (type_idx as u32) & 0xF;   
        
        Self(x_q | (y_q << 11) | (z_q << 22) | (t_q << 28))
    }

    #[inline(always)]
    pub const fn new(x: u32, y: u32, z: u32, type_id: u8) -> Self {
        let x_q = x & 0x7FF; 
        let y_q = y & 0x7FF;
        let z_q = z & 0x3F;  
        let t_q = (type_id as u32) & 0xF;   
        
        Self(x_q | (y_q << 11) | (z_q << 22) | (t_q << 28))
    }

    // Методы для GPU-вычислений (если потребуются на CPU)
    #[inline(always)] pub const fn type_id(&self) -> u8 { (self.0 >> 28) as u8 }
    #[inline(always)] pub const fn x(&self) -> u16 { (self.0 & 0x7FF) as u16 }
    #[inline(always)] pub const fn y(&self) -> u16 { ((self.0 >> 11) & 0x7FF) as u16 }
    #[inline(always)] pub const fn z(&self) -> u8 { ((self.0 >> 22) & 0x3F) as u8 }
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

/// Dendrite target: [31..24] segment_offset (8 bits) | [23..0] axon_id + 1 (24 bits).
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
        // Max values for 11/11/6/4 layout
        let p = PackedPosition::new(2047, 2047, 63, 15);
        assert_eq!(p.x(), 2047);
        assert_eq!(p.y(), 2047);
        assert_eq!(p.z(), 63);
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
        let pm = PackedPosition::new(123, 1456, 48, 9);
        assert_eq!(pm.x(), 123);
        assert_eq!(pm.y(), 1456);
        assert_eq!(pm.z(), 48);
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
