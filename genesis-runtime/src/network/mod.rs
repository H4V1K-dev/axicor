pub mod ring_buffer;
pub mod bsp;
pub mod router;
pub mod channel;
pub mod intra_gpu;
pub mod slow_path;
pub mod geometry_client;
pub mod socket;
pub mod telemetry;

use bytemuck::{Pod, Zeroable};

#[cfg(test)]
mod test_intra_gpu;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeBatchHeader {
    pub batch_id: u32,
    pub spikes_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeEvent {
    pub receiver_ghost_id: u32,
    pub tick_offset: u8,
    pub _pad: [u8; 3], // align to 64 bits (8 bytes)
}
