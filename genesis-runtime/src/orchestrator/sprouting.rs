// genesis-runtime/src/orchestrator/sprouting.rs
use genesis_core::constants::MAX_DENDRITE_SLOTS;

// FNV-1a для детерминированного псевдорандома (Stateless Hashing)
fn hash_seed(seed: u64, salt: u32) -> u32 {
    let mut hash = 0x811c9dc5_u32;
    for &b in &seed.to_le_bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    for &b in &salt.to_le_bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

/// Выполняет поиск новых связей для пустых слотов.
/// Работает напрямую с Pinned RAM массивами (Zero-Copy).
pub fn run_cpu_sprouting(
    targets: &mut [u32],
    weights: &mut [i16],
    padded_n: usize,
    total_axons: u32,
    master_seed: u64,
) -> usize {
    let mut new_synapses = 0;
    let base_weight: i16 = 74; // Дефолтный стартовый вес для новой связи

    // Итерируемся по каждому нейрону
    for i in 0..padded_n {
        // GPU отсортировала слоты по убыванию веса. 
        // Пустые слоты (target == 0) гарантированно лежат в конце (slot 127, 126...).
        // Сканируем с конца (Working Memory zone) к началу (LTM zone).
        for slot in (0..MAX_DENDRITE_SLOTS).rev() {
            let col_idx = slot * padded_n + i;
            
            if targets[col_idx] != 0 {
                // Как только встретили живую связь — прерываем цикл. 
                // Дальше пустых слотов нет! Это Early Exit на CPU.
                break;
            }

            // --- SPROUTING LOGIC ---
            // Генерируем псевдослучайного кандидата на основе ID сомы и номера слота.
            // В будущем здесь будет Spatial Hash Grid и Cone Tracing (04_connectivity.md §4.1).
            let salt = (i as u32).wrapping_add(slot as u32);
            let candidate_axon = hash_seed(master_seed, salt) % total_axons;
            
            // Фиктивный сегмент подключения (хвост аксона)
            let segment_idx = (hash_seed(master_seed, salt.wrapping_mul(2)) % 10) as u32;
            
            // Тип кандидата берем условно (в реальности читаем из геометрии аксона)
            let type_id = (hash_seed(master_seed, salt.wrapping_mul(3)) % 4) as u32;

            // Упаковываем target_packed: [31..28] Type | [27..8] AxonID | [7..0] SegIdx
            let new_target = (type_id << 28) | (candidate_axon << 8) | segment_idx;

            targets[col_idx] = new_target;
            weights[col_idx] = base_weight;
            new_synapses += 1;
            
            // Only sprout 1 connection per neuron per Night Phase
            break;
        }
    }

    new_synapses
}
