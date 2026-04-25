use crate::network::bsp::BspBarrier;
use crate::network::{SpikeBatchHeader, SpikeEvent};
use axicor_core::ipc::SpikeBatchHeaderV2;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;
use tokio::net::UdpSocket;
use tracing::{info, warn};

pub fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash_value: u32 = 0x811c9dc5;
    for &byte in data {
        hash_value ^= byte as u32;
        hash_value = hash_value.wrapping_mul(0x01000193);
    }
    hash_value
}

pub struct RoutingTable {
    map_ptr: AtomicPtr<HashMap<u32, (SocketAddr, u16)>>,
}

impl RoutingTable {
    pub fn new(initial_peers: HashMap<u32, (SocketAddr, u16)>) -> Self {
        let boxed = Box::new(initial_peers);
        Self {
            map_ptr: AtomicPtr::new(Box::into_raw(boxed)),
        }
    }

    pub fn get_address(&self, zone_hash: u32) -> Option<(SocketAddr, u16)> {
        let ptr = self.map_ptr.load(Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }
        unsafe { (*ptr).get(&zone_hash).copied() }
    }

    pub fn get_map_ptr(&self) -> *mut HashMap<u32, (SocketAddr, u16)> {
        self.map_ptr.load(Ordering::Acquire)
    }

    /// # Safety
    /// Caller must ensure RCU synchronization and thread safety.
    pub unsafe fn update_routes(&self, new_map: HashMap<u32, (SocketAddr, u16)>) {
        let boxed = Box::new(new_map);
        let new_ptr = Box::into_raw(boxed);

        // Atomic pointer swap (Release order guarantees data visibility for readers)
        let old_ptr = self
            .map_ptr
            .swap(new_ptr, std::sync::atomic::Ordering::Release);

        if !old_ptr.is_null() {
            let old_ptr_usize = old_ptr as usize;
            // Deferred deletion: give 100ms for all Egress threads to finish reading old table
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                let _ = Box::from_raw(old_ptr_usize as *mut HashMap<u32, (SocketAddr, u16)>);
                info!("[RCU] Old routing table deallocated safely.");
            });
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
    pub fn new(socket: Arc<UdpSocket>, routing_table: Arc<RoutingTable>) -> Self {
        Self {
            routing_table,
            socket,
        }
    }

    pub async fn flush_outgoing_batch(&self, target_node_id: u32, events: &[SpikeEvent]) {
        if events.is_empty() {
            return;
        }

        let Some((addr, _mtu)) = self.routing_table.get_address(target_node_id) else {
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

    pub fn flush_outgoing_batch_pool(
        &self,
        pool: &crate::network::egress::EgressPool,
        src_zone_hash: u32,
        target_zone_hash: u32,
        events: &[crate::network::SpikeEvent],
        epoch: u32,
    ) {
        let Some((target_addr, peer_mtu)) = self.routing_table.get_address(target_zone_hash) else {
            return;
        };

        let safe_mtu = std::cmp::max(peer_mtu as usize, 1400); // 1400 is the minimum safe MTU for ESP32
        let max_events_per_packet: usize = (safe_mtu - 16) / 8;

        // Send empty Heartbeat if no spikes
        if events.is_empty() {
            let mut msg = loop {
                if let Some(m) = pool.free_queue.pop() {
                    break m;
                }
                std::hint::spin_loop();
            };
            unsafe {
                let header = msg.buffer.as_mut_ptr() as *mut SpikeBatchHeaderV2;
                (*header).src_zone_hash = src_zone_hash;
                (*header).dst_zone_hash = target_zone_hash;
                (*header).epoch = epoch;
                (*header).chunk_idx = 0;
                (*header).total_chunks = 0; // Heartbeat
                msg.size = 16;
            }
            msg.target = target_addr;
            pool.ready_queue.push(msg).unwrap();
            return;
        }

        // L7 Fragmentation
        let chunks = events.chunks(max_events_per_packet);
        let total_chunks = chunks.len();

        for (i, chunk) in chunks.enumerate() {
            let mut msg = loop {
                if let Some(m) = pool.free_queue.pop() {
                    break m;
                }
                std::hint::spin_loop();
            };

            unsafe {
                let header = msg.buffer.as_mut_ptr() as *mut SpikeBatchHeaderV2;
                (*header).src_zone_hash = src_zone_hash;
                (*header).dst_zone_hash = target_zone_hash;
                (*header).epoch = epoch;
                (*header).chunk_idx = i as u16;
                (*header).total_chunks = total_chunks as u16;

                let events_bytes = bytemuck::cast_slice(chunk);
                let total_size = 16 + events_bytes.len();
                
                if total_size > axicor_core::constants::MAX_UDP_PAYLOAD {
                    tracing::warn!("[Egress] Dropping ghost batch: {} bytes exceeds UDP MTU", total_size);
                    return;
                }

                std::ptr::copy_nonoverlapping(
                    events_bytes.as_ptr(),
                    msg.buffer.as_mut_ptr().add(16),
                    events_bytes.len(),
                );
                msg.size = 16 + events_bytes.len();
            }
            msg.target = target_addr;
            pool.ready_queue.push(msg).unwrap();
        }
    }

    // MONOLITH: HIGH — spawn_ghost_listener loop mixes IO, amnesia checks, self-healing, and deduplication logic.
    // REFACTOR: Separate the UDP Listener from the protocol decoder (FastPathDecoder) and state manager.
    #[allow(clippy::needless_range_loop)]
    /// Starts inter-zone spike listener (Sender-Side Mapping)
    pub async fn spawn_ghost_listener(
        port: u16,
        bsp_barrier: Arc<BspBarrier>,
        routing_table: Arc<RoutingTable>,
        cluster_secret: u64, // [DOD FIX] Forward secret for RCU authentication
    ) {
        // [DOD FIX] Listen on all interfaces (0.0.0.0) to receive spikes from other physical nodes
        let sock = tokio::net::UdpSocket::bind(("0.0.0.0", port))
            .await
            .expect("FATAL: Ghost Bind failed");

        #[derive(Clone, Copy)]
        struct PeerState {
            hash: u32,
            epoch: u32,
            received_chunks: u16,
            chunk_mask: [u64; 16], // Bitmask for 1024 chunks (~65MB max batch)
        }

        tokio::spawn(async move {
            let mut peers = [PeerState {
                hash: 0,
                epoch: 0,
                received_chunks: 0,
                chunk_mask: [0; 16],
            }; 32];
            let mut buf = vec![0u8; 65507];
            loop {
                if let Ok((size, _)) = sock.recv_from(&mut buf).await {
                    if size < 16 {
                        continue;
                    }

                    // [DOD FIX] Intercept Control Plane packets (ROUT_MAGIC) before Data Plane parsing
                    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
                    if magic == axicor_core::ipc::ROUT_MAGIC {
                        if size >= std::mem::size_of::<axicor_core::ipc::RouteUpdate>() {
                            let update = unsafe {
                                std::ptr::read_unaligned(
                                    buf.as_ptr() as *const axicor_core::ipc::RouteUpdate
                                )
                            };
                            if update.cluster_secret == cluster_secret {
                                let mut new_map = unsafe { (*routing_table.get_map_ptr()).clone() };
                                let ipv4 = std::net::Ipv4Addr::from(update.new_ipv4);
                                let new_addr = std::net::SocketAddr::from((ipv4, update.new_port));
                                new_map.insert(update.zone_hash, (new_addr, update.mtu));
                                unsafe {
                                    routing_table.update_routes(new_map);
                                }
                                info!(
                                    " [RCU Fast-Path] Dynamic Route Update: 0x{:08X} moved to {}",
                                    update.zone_hash, new_addr
                                );
                            } else {
                                warn!("[WARN] [Security] Unauthorized ROUT_MAGIC on Fast-Path");
                            }
                        }
                        continue;
                    }

                    let header = unsafe {
                        std::ptr::read_unaligned(buf.as_ptr() as *const SpikeBatchHeaderV2)
                    };
                    let current_epoch = bsp_barrier
                        .current_epoch
                        .load(std::sync::atomic::Ordering::Acquire);

                    // 1. Biological Amnesia: Ignore packets from the past
                    if header.epoch < current_epoch {
                        continue;
                    }

                    // 2. Self-Healing: Jump to the future (2.8.1 distributed.md)
                    if header.epoch > current_epoch {
                        let n = bsp_barrier
                            .self_heal_log_counter
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if n.is_multiple_of(100) {
                            warn!("[BSP] Self-Healing: Fast-forwarding epoch {} -> {} (dropped lag data)", current_epoch, header.epoch);
                        }
                        bsp_barrier
                            .current_epoch
                            .store(header.epoch, std::sync::atomic::Ordering::Release);
                        bsp_barrier
                            .completed_peers
                            .store(0, std::sync::atomic::Ordering::Release);
                        bsp_barrier.get_write_schedule().clear(); // Reset garbage from the past
                    }

                    // 3. Process ACK packet
                    if header.chunk_idx == 0xFFFF {
                        bsp_barrier
                            .completed_peers
                            .fetch_add(1, std::sync::atomic::Ordering::Release);
                        continue;
                    }

                    // 4. Zero-Cost O(1) Deduplication State
                    let mut peer_idx = 32;
                    for i in 0..32 {
                        if peers[i].hash == header.src_zone_hash {
                            peer_idx = i;
                            break;
                        }
                        if peers[i].hash == 0 {
                            peers[i].hash = header.src_zone_hash;
                            peer_idx = i;
                            break;
                        }
                    }
                    if peer_idx == 32 {
                        continue;
                    } // Protection against peer overflow

                    let peer = &mut peers[peer_idx];

                    // Reset on new epoch
                    if header.epoch > peer.epoch {
                        peer.epoch = header.epoch;
                        peer.received_chunks = 0;
                        peer.chunk_mask.fill(0);
                    } else if header.epoch == peer.epoch {
                        // Drop duplicates within epoch
                        if header.total_chunks == 0 {
                            continue; // Duplicate Heartbeat
                        }
                        let mask_idx = (header.chunk_idx / 64) as usize;
                        let bit_idx = header.chunk_idx % 64;
                        if mask_idx < 16 && (peer.chunk_mask[mask_idx] & (1 << bit_idx)) != 0 {
                            continue; // Duplicate chunk with spikes (Drop phantom spikes!)
                        }
                    }

                    // If reached here  it's a unique Data-Driven packet.
                    if header.total_chunks > 0 {
                        let mask_idx = (header.chunk_idx / 64) as usize;
                        let bit_idx = header.chunk_idx % 64;
                        if mask_idx < 16 {
                            peer.chunk_mask[mask_idx] |= 1 << bit_idx;
                            peer.received_chunks += 1;
                        }

                        // Read spikes
                        let payload_bytes = &buf[16..size];
                        if payload_bytes.len().is_multiple_of(8) && !payload_bytes.is_empty() {
                            let schedule = bsp_barrier.get_write_schedule();
                            for chunk in payload_bytes.chunks_exact(8) {
                                let ghost_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                                let tick_offset =
                                    u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                                schedule.push_spike(tick_offset as usize, ghost_id);
                            }
                        }
                    }

                    // 5. Trigger barrier and send ACK
                    let is_complete =
                        header.total_chunks == 0 || peer.received_chunks == header.total_chunks;
                    if is_complete {
                        bsp_barrier
                            .completed_peers
                            .fetch_add(1, std::sync::atomic::Ordering::Release);

                        if let Some((src_addr, _mtu)) =
                            routing_table.get_address(header.src_zone_hash)
                        {
                            let ack = SpikeBatchHeaderV2 {
                                src_zone_hash: header.dst_zone_hash,
                                dst_zone_hash: header.src_zone_hash,
                                epoch: header.epoch,
                                chunk_idx: 0xFFFF, // ACK
                                total_chunks: 0,
                            };
                            let _ = sock.send_to(bytemuck::bytes_of(&ack), src_addr).await;
                        }
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
        initial_map.insert(0x1111, ("127.0.0.1:8080".parse().unwrap(), 65507));

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
            new_map.insert(0x1111, ("127.0.0.1:8081".parse().unwrap(), 65507));
            new_map.insert(0x2222, ("127.0.0.1:8082".parse().unwrap(), 65507));

            unsafe {
                table_clone.update_routes(new_map);
            }
        });

        writer.await.unwrap();
        for r in readers {
            r.await.unwrap();
        }

        assert_eq!(
            table.get_address(0x2222).unwrap().0,
            "127.0.0.1:8082".parse::<SocketAddr>().unwrap()
        );
    }
}
