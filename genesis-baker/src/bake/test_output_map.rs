// genesis-baker/src/bake/test_output_map.rs
//
// Legacy compatibility tests rewritten to use the new build_gxo_mapping API.

use genesis_core::constants::GXO_MAGIC;
use genesis_core::ipc::EMPTY_PIXEL;
use crate::bake::neuron_placement::PlacedNeuron;
use crate::bake::output_map::build_gxo_mapping;
use genesis_core::coords::pack_position;

fn packed(x: u32, y: u32, z: u32, t: u32) -> u32 {
    pack_position(x, y, z, t).0
}

#[test]
fn test_bake_output_maps_empty() {
    // Zero neurons → output_count = 0, all pixels are EMPTY_PIXEL
    let gxo = build_gxo_mapping("out", "zone", 2, 2, 1000, 1000, &[]);
    assert_eq!(gxo.output_count, 0);
    assert!(gxo.mapped_soma_ids.iter().all(|&id| id == EMPTY_PIXEL));
}

#[test]
fn test_bake_output_maps_basic_assignment() {
    // 2×2 matrix over a 1000×1000 voxel world → each pixel covers 500 voxels.
    // Place one neuron in each quadrant.
    let neurons = vec![
        packed(250, 250, 0, 0), // pixel (0,0)
        packed(750, 250, 0, 0), // pixel (1,0)
        packed(250, 750, 0, 0), // pixel (0,1)
        packed(750, 750, 0, 0), // pixel (1,1)
    ];

    let gxo = build_gxo_mapping("test_map", "V1", 2, 2, 1000, 1000, &neurons);

    assert_eq!(gxo.output_count, 4);
    assert!(gxo.mapped_soma_ids.iter().all(|&id| id != EMPTY_PIXEL));

    // Header magic check
    assert_eq!(&gxo.header.as_bytes()[0..4], &GXO_MAGIC.to_le_bytes());
}

#[test]
fn test_bake_output_maps_type_filtering() {
    // The new build_gxo_mapping does not filter by type (type filter lives in a higher layer).
    // This test verifies that ALL neurons land in the same 1×1 pixel and Z-sort selects
    // the one with smallest Z.
    //
    // Neuron (type 0, Z=0): dense_id=0 → winner
    // Neuron (type 1, Z=0): dense_id=1 → tie, but dense_id 0 was seen first
    let neurons = vec![
        packed(500, 500, 0, 0), // Dense ID 0, type 0, Z=0
        packed(500, 500, 1, 1), // Dense ID 1, type 1, Z=1 → loses
        packed(500, 500, 0, 0), // Dense ID 2, type 0, Z=0 → tie, ID 0 still wins (first seen)
    ];

    let gxo = build_gxo_mapping("only_type_b", "V1", 1, 1, 1000, 1000, &neurons);
    assert_eq!(gxo.output_count, 1);
    // Dense ID 0 has Z=0 and appears first → wins
    assert_eq!(gxo.mapped_soma_ids[0], 0);
}
