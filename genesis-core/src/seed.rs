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

/// Хэшируем строку-сид в u64.
/// Позволяет использовать читаемые сиды: "GENESIS", "DEBUG_RUN_42".
pub fn seed_from_str(s: &str) -> u64 {
    wyhash::wyhash(s.as_bytes(), 0)
}

/// `Local_Seed = Hash(Master_Seed_u64 + Entity_ID)` — §2.2
///
/// Результат не зависит от порядка вызовов: нейрон №5001 всегда одинаков
/// независимо от того, создали его первым или миллионным.
pub fn entity_seed(master_seed: u64, entity_id: u32) -> u64 {
    wyhash::wyhash(&entity_id.to_le_bytes(), master_seed)
}

/// Быстрый псевдослучайный float в диапазоне [0.0, 1.0) из seed.
/// Использует старшие 23 бита для мантиссы IEEE 754.
pub fn random_f32(seed: u64) -> f32 {
    let bits = (seed >> 41) as u32 | 0x3F800000;
    f32::from_bits(bits) - 1.0
}

/// Детерминированный shuffle индексов [0..len) через Fisher-Yates + wyhash.
/// Результат бит-в-бит идентичен для одного и того же seed.
pub fn shuffle_indices(len: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut s = seed;
    for i in (1..len).rev() {
        s = wyhash::wyhash(&s.to_le_bytes(), s);
        let j = (s as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

#[cfg(test)]
#[path = "test_seed.rs"]
mod test_seed;
