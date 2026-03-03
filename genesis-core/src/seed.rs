/// Детерминированное хэширование entity по `master_seed`.
/// Алгоритм: wyhash (01_foundations.md §2.2)
///
/// Правило: единственная точка входа энтропии — `master_seed`.
/// Никаких `time(NULL)`, `std::random_device`, `SystemTime::now()`.
/// Единая точка входа энтропии для симуляции (§2.1).
/// Строится из строки конфига через `seed_from_str`, хранится как u64.
/// Все производные seed вычисляются через `entity_seed(self.0, entity_id)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasterSeed(pub u64);

impl MasterSeed {
    /// Создаёт MasterSeed из любой строки (ASCII, пробелы, Unicode, Emoji).
    pub fn from_str(s: &str) -> Self {
        Self(seed_from_str(s))
    }

    /// Получить seed для конкретного entity (нейрон, аксон).
    pub fn entity(&self, entity_id: u32) -> u64 {
        entity_seed(self.0, entity_id)
    }

    /// Вернуть сырой u64 (для передачи в GPU Constant Memory).
    pub fn raw(&self) -> u64 {
        self.0
    }
}

pub const DEFAULT_MASTER_SEED: &str = "GENESIS";

/// Хэшируем строку-сид в u64 (FNV-1a 64-bit).
/// Позволяет использовать читаемые сиды: "GENESIS", "DEBUG_RUN_42".
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

/// `Local_Seed = Hash(Master_Seed_u64 + Entity_ID)` — §2.2
///
/// Детерминированный stateless-хэш для Entity (WyHash-like 64-bit).
/// Гарантирует O(1) вычисление свойств сомы независимо от порядка генерации.
#[inline(always)]
pub const fn entity_seed(master_seed: u64, entity_id: u32) -> u64 {
    let seed = master_seed.wrapping_add(entity_id as u64).wrapping_add(0x60bee2bee120fc15);
    // Выполняем лавинообразное перемешивание битов (Avalanche effect)
    let mut tmp = (seed as u128).wrapping_mul(0xa3b195354a39b70d);
    let m1 = (tmp >> 64) ^ tmp;
    tmp = m1.wrapping_mul(0x1b03738712fad5c9);
    ((tmp >> 64) ^ tmp) as u64
}

/// Быстрый псевдослучайный float в диапазоне [0.0, 1.0) из seed.
/// Использует старшие 23 бита для мантиссы IEEE 754.
pub fn random_f32(seed: u64) -> f32 {
    let bits = (seed >> 41) as u32 | 0x3F800000;
    f32::from_bits(bits) - 1.0
}

/// Детерминированный shuffle индексов [0..len) через Fisher-Yates + entity_seed.
/// Результат бит-в-бит идентичен для одного и того же seed.
pub fn shuffle_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut s = seed;
    for i in (1..len).rev() {
        // Каскадное хэширование для получения следующего числа
        s = entity_seed(s, i as u32);
        let j = (s as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

#[cfg(test)]
#[path = "test_seed.rs"]
mod test_seed;
