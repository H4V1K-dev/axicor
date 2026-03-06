pub mod geometry_client;
pub mod ring_buffer;
pub mod intra_gpu;
pub mod slow_path;
pub mod telemetry;
// pub mod external;
pub mod channel;
pub mod router;
pub mod socket;
pub mod ghosts;
pub mod replication;
pub mod bsp;
pub mod inter_node;
pub mod io_server;
pub mod egress;

// Removed bytemuck imports as they are unused here

// Spike contracts now moved to genesis-core::ipc
pub use genesis_core::ipc::{SpikeEvent, SpikeBatchHeader};

#[cfg(test)]
mod test_intra_gpu;
