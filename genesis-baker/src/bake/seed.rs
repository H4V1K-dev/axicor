/// Реэкспорт seed примитивов из genesis-core.
/// Baker использует этот модуль напрямую, runtime и другие крейты берут из genesis-core.

pub use genesis_core::seed::{
    entity_seed,
    random_f32,
    seed_from_str,
    shuffle_indices,
};
