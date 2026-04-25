use crossbeam::queue::SegQueue;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// [DOD] AxonHandoverEvent is defined in axicor-core for access from baker-daemon
pub use axicor_core::ipc::AxonHandoverAck;
pub use axicor_core::ipc::AxonHandoverEvent;
pub use axicor_core::ipc::AxonHandoverPrune;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GeometryRequest {
    BulkHandover(Vec<AxonHandoverEvent>),
    BulkAck(Vec<AxonHandoverAck>),
    Prune(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GeometryResponse {
    Ack(AxonHandoverAck),
    Ok,
}

pub struct SlowPathQueues {
    pub incoming_grow: Arc<SegQueue<AxonHandoverEvent>>,
    pub outgoing_ack: Arc<SegQueue<AxonHandoverAck>>,
    pub outgoing_grow: Arc<SegQueue<AxonHandoverEvent>>,
    pub incoming_ack: Arc<SegQueue<AxonHandoverAck>>,
    pub incoming_prune: Arc<SegQueue<AxonHandoverPrune>>,
    pub outgoing_prune: Arc<SegQueue<AxonHandoverPrune>>,
}

impl Default for SlowPathQueues {
    fn default() -> Self {
        Self::new()
    }
}

impl SlowPathQueues {
    pub fn new() -> Self {
        Self {
            incoming_grow: Arc::new(SegQueue::new()),
            outgoing_ack: Arc::new(SegQueue::new()),
            outgoing_grow: Arc::new(SegQueue::new()),
            incoming_ack: Arc::new(SegQueue::new()),
            incoming_prune: Arc::new(SegQueue::new()),
            outgoing_prune: Arc::new(SegQueue::new()),
        }
    }
}
