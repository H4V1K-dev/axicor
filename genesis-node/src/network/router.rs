use crate::network::{SpikeEvent, SpikeBatchHeader};
use std::collections::HashMap;
use tokio::net::UdpSocket;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};

pub fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash_value: u32 = 0x811c9dc5;
    for &byte in data {
        hash_value ^= byte as u32;
        hash_value = hash_value.wrapping_mul(0x01000193);
    }
    hash_value
}

pub struct RoutingTable {
    map_ptr: AtomicPtr<HashMap<u32, SocketAddr>>,
}

impl RoutingTable {
    pub fn new(initial_peers: HashMap<u32, SocketAddr>) -> Self {
        let boxed = Box::new(initial_peers);
        Self {
            map_ptr: AtomicPtr::new(Box::into_raw(boxed)),
        }
    }

    pub fn get_address(&self, zone_hash: u32) -> Option<SocketAddr> {
        let ptr = self.map_ptr.load(Ordering::Acquire);
        if ptr.is_null() { return None; }
        unsafe { (*ptr).get(&zone_hash).copied() }
    }

    pub fn get_map_ptr(&self) -> *mut HashMap<u32, SocketAddr> {
        self.map_ptr.load(Ordering::Acquire)
    }

    /// RCU update: atomically swaps the map pointer and schedules old map for cleanup.
    pub unsafe fn update_routes(&self, new_map: HashMap<u32, SocketAddr>) {
        let boxed = Box::new(new_map);
        let new_ptr = Box::into_raw(boxed);
        
        let old_ptr = self.map_ptr.swap(new_ptr, Ordering::Release);
        
        if !old_ptr.is_null() {
            let old_ptr_usize = old_ptr as usize;
            // Deferred cleanup to ensure no readers are still using the old map.
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    let _ = Box::from_raw(old_ptr_usize as *mut HashMap<u32, SocketAddr>);
                });
            } else {
                let _ = Box::from_raw(old_ptr);
            }
        }
    }
}

impl Drop for RoutingTable {
    fn drop(&mut self) {
        let ptr = self.map_ptr.load(Ordering::Acquire);
        if !ptr.is_null() {
            unsafe {
                let _ = Box::from_raw(ptr);
            }
        }
    }
}

pub struct InterNodeRouter {
    pub routing_table: Arc<RoutingTable>,
    pub socket: Arc<UdpSocket>,
}

impl InterNodeRouter {
    pub async fn new(addr: &str, routing_table: Arc<RoutingTable>) -> Self {
        let socket = UdpSocket::bind(addr).await.expect("Failed to bind InterNodeRouter UDP socket");
        Self {
            routing_table,
            socket: Arc::new(socket),
        }
    }

    pub async fn flush_outgoing_batch(&self, target_node_id: u32, events: &[SpikeEvent]) {
        if events.is_empty() { return; }

        let Some(addr) = self.routing_table.get_address(target_node_id) else {
            return; // Node not found or isolated
        };

        let header = SpikeBatchHeader {
            magic: 0x5350494B, // "SPIK"
            batch_id: 0,
        };

        let header_bytes = bytemuck::bytes_of(&header);
        let payload_bytes = bytemuck::cast_slice(events);

        let mut packet = Vec::with_capacity(header_bytes.len() + payload_bytes.len());
        packet.extend_from_slice(header_bytes);
        packet.extend_from_slice(payload_bytes);

        let _ = self.socket.send_to(&packet, addr).await;
    }

    pub fn spawn_receiver_loop(
        socket: Arc<UdpSocket>, 
        bsp_barrier: Arc<crate::network::bsp::BspBarrier>,
    ) {
        tokio::spawn(async move {
            let mut buf = [0u8; 65535];
            
            loop {
                if let Ok((size, _)) = socket.recv_from(&mut buf).await {
                    if size < std::mem::size_of::<SpikeBatchHeader>() { continue; }
                    
                    let header_ptr = buf.as_ptr() as *const SpikeBatchHeader;
                    let header = unsafe { &*header_ptr };
                    
                    if header.magic != 0x5350494B { continue; }
                    
                    let events_count = (size - std::mem::size_of::<SpikeBatchHeader>()) / std::mem::size_of::<SpikeEvent>();
                    let events_ptr = unsafe { buf.as_ptr().add(std::mem::size_of::<SpikeBatchHeader>()) as *const SpikeEvent };
                    let events = unsafe { std::slice::from_raw_parts(events_ptr, events_count) };
                    
                    let schedule = bsp_barrier.get_write_schedule();
                    for event in events {
                        // Using the new tick-based scheduler (Contract §12)
                        schedule.push_spike(event.tick_offset as usize, event.ghost_axon_id);
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lock_free_routing() {
        let mut initial_map = HashMap::new();
        initial_map.insert(0x1111, "127.0.0.1:8080".parse().unwrap());
        
        let table = Arc::new(RoutingTable::new(initial_map));
        
        let mut readers = Vec::new();
        for _ in 0..100 {
            let table_clone = table.clone();
            readers.push(tokio::spawn(async move {
                for _ in 0..1000 {
                    let addr = table_clone.get_address(0x1111);
                    assert!(addr.is_some());
                }
            }));
        }

        let table_clone = table.clone();
        let writer = tokio::spawn(async move {
            let mut new_map = HashMap::new();
            new_map.insert(0x1111, "127.0.0.1:8081".parse().unwrap());
            new_map.insert(0x2222, "127.0.0.1:8082".parse().unwrap());
            
            unsafe { table_clone.update_routes(new_map); }
        });

        writer.await.unwrap();
        for r in readers {
            r.await.unwrap();
        }

        assert_eq!(table.get_address(0x2222).unwrap(), "127.0.0.1:8082".parse().unwrap());
    }
}
