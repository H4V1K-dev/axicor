#[inline(always)]
pub const fn fnv1a_32(name: &[u8]) -> u32 {
    hash_name_fnv1a(name)
}

/// Детерминированный FNV-1a (32-bit) для хэширования имен зон и матриц в UDP-протоколе.
#[inline(always)]
pub const fn hash_name_fnv1a(name: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9dc5;
    let mut i = 0;
    while i < name.len() {
        hash ^= name[i] as u32;
        hash = hash.wrapping_mul(0x01000193);
        i += 1;
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_name_fnv1a_reference() {
        // Фиксированный контракт для 08_io_matrix.md
        // "SensoryCortex" must hash to 0x273fd103
        assert_eq!(hash_name_fnv1a(b"SensoryCortex"), 0x273fd103);
    }
}
