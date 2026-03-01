pub mod geometry_client;
pub mod ring_buffer;
pub mod intra_gpu;
pub mod slow_path;
pub mod telemetry;
pub mod external;
pub mod channel;
pub mod router;
pub mod socket;
pub mod ghosts;
pub mod bsp;
pub mod inter_node;

use bytemuck::{Pod, Zeroable};

#[cfg(test)]
mod test_intra_gpu;

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeBatchHeader {
    pub zone_hash: u32,
    pub count: u32,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeEvent {
    pub ghost_id: u32,
    pub tick_offset: u32,
}
