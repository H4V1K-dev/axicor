use crossbeam::queue::ArrayQueue;
use std::net::SocketAddr;

#[derive(Debug)]
pub struct EgressMessage {
    pub buffer: Vec<u8>, // Strictly capacity = len = MAX_UDP_PAYLOAD
    pub size: usize,
    pub target: SocketAddr,
}

pub struct EgressPool {
    pub free_queue: ArrayQueue<EgressMessage>,
    pub ready_queue: ArrayQueue<EgressMessage>,
}

impl EgressPool {
    pub fn new(capacity: usize) -> Self {
        let free_queue = ArrayQueue::new(capacity);
        let ready_queue = ArrayQueue::new(capacity);

        for _ in 0..capacity {
            let buffer = vec![0u8; axicor_core::constants::MAX_UDP_PAYLOAD];

            free_queue
                .push(EgressMessage {
                    buffer,
                    size: 0,
                    target: "0.0.0.0:0".parse().unwrap(),
                })
                .unwrap();
        }

        Self {
            free_queue,
            ready_queue,
        }
    }
}
