pub mod compute;
pub mod ffi;
pub mod memory;
pub mod mock_ffi;

#[cfg(not(feature = "mock-gpu"))]
pub use ffi::*;

#[cfg(feature = "mock-gpu")]
pub use mock_ffi::*;

pub use compute::shard::ShardEngine;
pub use ffi::ShardVramPtrs;
pub use memory::{VramShard, StateOffsets, calculate_state_blob_size, compute_state_offsets};
