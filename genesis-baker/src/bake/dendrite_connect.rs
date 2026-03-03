// genesis-baker/src/bake/dendrite_connect.rs
// Temporarily commented out for cone tracing refactor

use crate::bake::layout::ShardSoA;
use crate::bake::axon_growth::GrownAxon;
use genesis_core::types::PackedPosition;
use genesis_core::config::blueprints::GenesisConstantMemory;

pub fn connect_dendrites(
    _shard: &mut ShardSoA,
    _positions: &[PackedPosition],
    _axons: &[GrownAxon],
    _const_mem: &GenesisConstantMemory,
    _master_seed: u64,
    _cell_size: u32,
) {}

pub fn bind_synapse(
    _soa: &mut ShardSoA,
    _soma_dense_idx: usize,
    _axon_id: u32,
    _segment_offset: u32,
    _initial_weight: i16
) -> Result<(), String> {
    Ok(())
}

pub fn reconnect_empty_dendrites(
    _targets: &mut [u32],
    _weights: &mut [i16],
    _downloaded_weights: &[i16],
    _padded_n: usize,
    _positions: &[PackedPosition],
    _axons: &[GrownAxon],
    _const_mem: &GenesisConstantMemory,
    _master_seed: u64,
    _cell_size: u32,
) {}
