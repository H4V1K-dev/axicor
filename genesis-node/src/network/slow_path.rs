use crossbeam::queue::SegQueue;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GeometryRequest {
    Handover(AxonHandoverEvent),
    Prune(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GeometryResponse {
    Ack(AxonHandoverAck),
    Ok,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AxonHandoverPrune {
    pub magic: u32,         // 0x44454144 ("DEAD")
    pub ghost_id: u32,      
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AxonHandoverEvent {
    pub local_axon_id: u32,
    pub entry_x: u16,
    pub entry_y: u16,
    pub vector_x: i8,
    pub vector_y: i8,
    pub vector_z: i8,
    pub type_mask: u8,
    pub remaining_length: u16,
    pub _padding: u16,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AxonHandoverAck {
    pub magic: u32,         // 0x41434B48 ("ACKH")
    pub local_axon_id: u32, 
    pub ghost_id: u32,      
}

pub struct SlowPathQueues {
    pub incoming_grow: Arc<SegQueue<AxonHandoverEvent>>,
    pub outgoing_ack: Arc<SegQueue<AxonHandoverAck>>,
    pub outgoing_grow: Arc<SegQueue<AxonHandoverEvent>>,
    pub incoming_ack: Arc<SegQueue<AxonHandoverAck>>,
    pub incoming_prune: Arc<SegQueue<AxonHandoverPrune>>,
    pub outgoing_prune: Arc<SegQueue<AxonHandoverPrune>>,
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
