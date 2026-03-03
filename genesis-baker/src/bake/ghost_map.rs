// genesis-baker/src/bake/ghost_map.rs
//
// Фаза C: Inter-zone Ghost Routing (.ghosts)
// Спецификация: 09_baking_pipeline.md §1.3
//
// Контракт:
//   Фаза C строго выполняется ПОСЛЕ Фазы B (GXO).
//   src_soma_ids ОБЯЗАНЫ браться из `BakedGxo.mapped_soma_ids` зоны-источника.
//   Пиксели-сентинели (EMPTY_PIXEL) НЕ порождают связей.

use genesis_core::hash::fnv1a_32;
use genesis_core::ipc::{GhostsHeader, GhostConnection, EMPTY_PIXEL};
use std::path::Path;
use std::io::Write;

/// Результат запекания межзональных связей.
pub struct BakedGhosts {
    pub connections: Vec<GhostConnection>,
    pub header: GhostsHeader,
}

/// Строит межзональные связи по принципу:
/// «Выходная матрица Зоны A (src_mapped_soma_ids) проецируется во входную матрицу Зоны B».
///
/// `src_mapped_soma_ids` — плоский массив из `BakedGxo.mapped_soma_ids` зоны-источника.
///                         ПУСТЫЕ пиксели (EMPTY_PIXEL) порождают соответствующий GHOST аксон,
///                         но не привязывают реальную сому.
/// `dst_base_ghost_id`   — индекс первого ghost-аксона в Зоне B (= base_axon_id из Phase A).
///
/// Контракт: порядок обхода пикселей детерминирован (row-major pixel_index).
/// Все `connection_count` связей записываются, даже для EMPTY_PIXEL, чтобы сохранить
/// 1:1 соответствие пикселей между зонами.
/// Но src_soma_id для пустых пикселей устанавливается в EMPTY_PIXEL,
/// что позволяет рантайму сделать Early Exit и не инжектировать сигнал.
pub fn build_ghost_mapping(
    from_zone_name: &str,
    to_zone_name: &str,
    src_mapped_soma_ids: &[u32],
    dst_base_ghost_id: u32,
) -> BakedGhosts {
    let connections: Vec<GhostConnection> = src_mapped_soma_ids
        .iter()
        .enumerate()
        .map(|(pixel_idx, &src_soma_id)| GhostConnection {
            src_soma_id,                                    // EMPTY_PIXEL if no soma
            target_ghost_id: dst_base_ghost_id + pixel_idx as u32,
        })
        .collect();

    let from_hash = fnv1a_32(from_zone_name.as_bytes());
    let to_hash   = fnv1a_32(to_zone_name.as_bytes());
    let header    = GhostsHeader::new(from_hash, to_hash, connections.len() as u32);

    BakedGhosts { connections, header }
}

/// Zero-copy сериализация в `<out_dir>/<from>_<to>.ghosts`.
pub fn write_ghosts_file(out_dir: &Path, from_name: &str, to_name: &str, ghosts: &BakedGhosts) {
    let filename = format!("{}_{}.ghosts", from_name, to_name);
    let path = out_dir.join(filename);
    let mut file = std::fs::File::create(path).expect("Failed to create .ghosts file");

    file.write_all(ghosts.header.as_bytes()).expect("Failed to write GhostsHeader");
    file.write_all(GhostConnection::slice_as_bytes(&ghosts.connections))
        .expect("Failed to write ghost connections");
}

/// Возвращает количество «реальных» (не-сентинельных) связей в .ghosts блобе.
#[inline]
pub fn count_live_connections(ghosts: &BakedGhosts) -> u32 {
    ghosts.connections.iter()
        .filter(|c| c.src_soma_id != EMPTY_PIXEL)
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::ipc::EMPTY_PIXEL;

    fn make_ids(ids: &[u32]) -> Vec<u32> { ids.to_vec() }

    #[test]
    fn test_ghost_count_matches_pixels() {
        let src = make_ids(&[0, 1, EMPTY_PIXEL, 3]);
        let g = build_ghost_mapping("zone_a", "zone_b", &src, 100);
        assert_eq!(g.connections.len(), 4);
        assert_eq!(g.header.connection_count, 4);
    }

    #[test]
    fn test_ghost_target_ids_sequential() {
        let src = make_ids(&[10, 20, 30]);
        let g = build_ghost_mapping("a", "b", &src, 50);
        assert_eq!(g.connections[0].target_ghost_id, 50);
        assert_eq!(g.connections[1].target_ghost_id, 51);
        assert_eq!(g.connections[2].target_ghost_id, 52);
    }

    #[test]
    fn test_ghost_src_soma_passthrough() {
        let src = make_ids(&[7, EMPTY_PIXEL, 42]);
        let g = build_ghost_mapping("a", "b", &src, 0);
        assert_eq!(g.connections[0].src_soma_id, 7);
        assert_eq!(g.connections[1].src_soma_id, EMPTY_PIXEL);
        assert_eq!(g.connections[2].src_soma_id, 42);
    }

    #[test]
    fn test_ghost_magic_and_hashes() {
        let g = build_ghost_mapping("zone_a", "zone_b", &[0], 0);
        assert_eq!(g.header.magic, genesis_core::ipc::GHST_MAGIC);
        assert_eq!(g.header.from_zone_hash, fnv1a_32(b"zone_a"));
        assert_eq!(g.header.to_zone_hash,   fnv1a_32(b"zone_b"));
    }

    #[test]
    fn test_ghost_live_connection_count() {
        let src = make_ids(&[5, EMPTY_PIXEL, EMPTY_PIXEL, 10]);
        let g = build_ghost_mapping("a", "b", &src, 0);
        assert_eq!(count_live_connections(&g), 2);
    }

    #[test]
    fn test_ghost_phase_c_requires_phase_b() {
        // Simulated pipeline: Phase B produces gxo.mapped_soma_ids,
        // Phase C takes that exact slice.
        let mapped_soma_ids = vec![3u32, EMPTY_PIXEL, 7, 2];
        let g = build_ghost_mapping("src_zone", "dst_zone", &mapped_soma_ids, 200);
        // Pixel 0 → soma 3 → ghost 200
        assert_eq!(g.connections[0].src_soma_id, 3);
        // Pixel 1 → empty  → ghost 201 (but src=EMPTY_PIXEL)
        assert_eq!(g.connections[1].src_soma_id, EMPTY_PIXEL);
        assert_eq!(g.connections[1].target_ghost_id, 201);
    }
}
