/// Deterministic entity hashing based on `master_seed`.
/// Algorithm: wyhash (01_foundations.md 2.2)
///
/// Rule: the only entropy entry point is `master_seed`.
/// No `time(NULL)`, `std::random_device`, or `SystemTime::now()`.
/// Unified entropy entry point for simulation (2.1).
/// Built from a config string via `seed_from_str`, stored as u64.
/// All derivative seeds are calculated via `entity_seed(self.0, entity_id)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasterSeed(pub u64);

impl MasterSeed {
    /// Creates MasterSeed from any string (ASCII, spaces, Unicode, Emoji).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        Self(seed_from_str(s))
    }

    /// Get seed for a specific entity (neuron, axon).
    pub fn entity(&self, entity_id: u32) -> u64 {
        entity_seed(self.0, entity_id)
    }

    /// Return raw u64 (for passing to GPU Constant Memory).
    pub fn raw(&self) -> u64 {
        self.0
    }
}

pub const DEFAULT_MASTER_SEED: &str = "AXICOR";

/// Hashes the seed string into u64 (FNV-1a 64-bit).
/// Enables the use of readable seeds: "AXICOR", "DEBUG_RUN_42".
pub const fn seed_from_str(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x00000100000001B3);
        i += 1;
    }
    hash
}

/// `Local_Seed = Hash(Master_Seed_u64 + Entity_ID)`  2.2
///
/// Deterministic stateless hash for Entity (WyHash-like 64-bit).
/// Guarantees O(1) computation of soma properties regardless of generation order.
#[inline(always)]
pub const fn entity_seed(master_seed: u64, entity_id: u32) -> u64 {
    let seed = master_seed
        .wrapping_add(entity_id as u64)
        .wrapping_add(0x60bee2bee120fc15);
    // Perform avalanche bit mixing
    let mut tmp = (seed as u128).wrapping_mul(0xa3b195354a39b70d);
    let m1 = (tmp >> 64) ^ tmp;
    tmp = m1.wrapping_mul(0x1b03738712fad5c9);
    ((tmp >> 64) ^ tmp) as u64
}

/// Fast pseudo-random float in [0.0, 1.0) range from seed.
/// Uses the upper 23 bits for the IEEE 754 mantissa.
pub fn random_f32(seed: u64) -> f32 {
    let bits = (seed >> 41) as u32 | 0x3F800000;
    f32::from_bits(bits) - 1.0
}

/// Deterministic shuffle of indices [0..len) via Fisher-Yates + entity_seed.
/// Result is bit-exact identical for the same seed.
pub fn shuffle_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut s = seed;
    for i in (1..len).rev() {
        // Cascade hashing to obtain the next number
        s = entity_seed(s, i as u32);
        let j = (s as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

#[cfg(test)]
#[path = "test_seed.rs"]
mod test_seed;
