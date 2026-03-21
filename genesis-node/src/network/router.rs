use crate::network::{SpikeEvent, SpikeBatchHeader};
use std::collections::HashMap;
use tokio::net::UdpSocket;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};
use bytemuck::{Pod, Zeroable};

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeBatchHeaderV2 {
    pub src_zone_hash: u32,
    pub dst_zone_hash: u32,
    pub epoch: u32,
    pub is_last: u32, 
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpikeEventV2 {
    pub ghost_id: u32,
    pub tick_offset: u32, 
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
        if ptr.is_null() { return None; }
        unsafe { (*ptr).get(&zone_hash).copied() }
    }

    pub fn get_map_ptr(&self) -> *mut HashMap<u32, (SocketAddr, u16)> {
        self.map_ptr.load(Ordering::Acquire)
    }

    // [DOD FIX] Zero-Lock RCU Routing
    pub unsafe fn update_routes(&self, new_map: HashMap<u32, (SocketAddr, u16)>) {
        let boxed = Box::new(new_map);
        let new_ptr = Box::into_raw(boxed);

        // Атомарный своп указателя (Release order гарантирует видимость данных для читателей)
        let old_ptr = self.map_ptr.swap(new_ptr, std::sync::atomic::Ordering::Release);

        if !old_ptr.is_null() {
            let old_ptr_usize = old_ptr as usize;
            // Отложенное удаление: даем 100мс всем Egress-потокам завершить чтение старой таблицы
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                let _ = Box::from_raw(old_ptr_usize as *mut HashMap<u32, (SocketAddr, u16)>); 
                println!("[RCU] Old routing table deallocated safely.");
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
        if events.is_empty() { return; }

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
        let Some((target_addr, peer_mtu)) = self.routing_table.get_address(target_zone_hash) else { return; };

        let safe_mtu = std::cmp::max(peer_mtu as usize, 1400); // 1400 is the minimum safe MTU for ESP32
        let max_events_per_packet: usize = (safe_mtu - 16) / 8;

        // Отправка пустого Heartbeat, если спайков нет
        if events.is_empty() {
            let mut msg = loop {
                if let Some(m) = pool.free_queue.pop() { break m; }
                std::hint::spin_loop();
            };
            unsafe {
                let header = msg.buffer.as_mut_ptr() as *mut SpikeBatchHeaderV2;
                (*header).src_zone_hash = src_zone_hash;
                (*header).dst_zone_hash = target_zone_hash;
                (*header).epoch = epoch;
                (*header).is_last = 1; // Единственный и последний
                msg.size = 16;
            }
            msg.target = target_addr;
            pool.ready_queue.push(msg).unwrap();
            return;
        }

        // L7 Фрагментация
        let chunks = events.chunks(max_events_per_packet);
        let total_chunks = chunks.len();

        for (i, chunk) in chunks.enumerate() {
            let mut msg = loop {
                if let Some(m) = pool.free_queue.pop() { break m; }
                std::hint::spin_loop();
            };

            unsafe {
                let header = msg.buffer.as_mut_ptr() as *mut SpikeBatchHeaderV2;
                (*header).src_zone_hash = src_zone_hash;
                (*header).dst_zone_hash = target_zone_hash;
                (*header).epoch = epoch;
                // Только последний чанк пробивает барьер получателя
                (*header).is_last = if i == total_chunks - 1 { 1 } else { 0 };

                let events_bytes = bytemuck::cast_slice(chunk);
                std::ptr::copy_nonoverlapping(
                    events_bytes.as_ptr(),
                    msg.buffer.as_mut_ptr().add(16),
                    events_bytes.len()
                );
                msg.size = 16 + events_bytes.len();
            }
            msg.target = target_addr;
            pool.ready_queue.push(msg).unwrap();
        }
    }

    /// Запускает слушатель межзональных спайков (Sender-Side Mapping)
    pub async fn spawn_ghost_listener(
        port: u16,
        bsp_barrier: Arc<crate::network::bsp::BspBarrier>,
        routing_table: Arc<RoutingTable>,
    ) {
        let sock = tokio::net::UdpSocket::bind(("127.0.0.1", port)).await.expect("FATAL: Ghost Bind failed");
        
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65507];
            loop {
                if let Ok((size, _)) = sock.recv_from(&mut buf).await {
                    if size < 16 { continue; }

                    // Safe: network buffer may be unaligned; copy to aligned storage
                    let mut hdr_buf = [0u8; 16];
                    hdr_buf.copy_from_slice(&buf[0..16]);
                    let header: SpikeBatchHeaderV2 = *bytemuck::from_bytes(&hdr_buf);
                    let current_epoch = bsp_barrier.current_epoch.load(std::sync::atomic::Ordering::Acquire);

                    // 1. Biological Amnesia: Игнорируем пакеты из прошлого
                    if header.epoch < current_epoch {
                        continue;
                    }

                    // 2. Self-Healing: Прыжок в будущее (§2.8.1 distributed.md)
                    if header.epoch > current_epoch {
                        let n = bsp_barrier.self_heal_log_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if n % 100 == 0 {
                            println!("⚠️ [BSP] Self-Healing: Fast-forwarding epoch {} -> {} (dropped lag data)", current_epoch, header.epoch);
                        }
                        bsp_barrier.current_epoch.store(header.epoch, std::sync::atomic::Ordering::Release);
                        bsp_barrier.completed_peers.store(0, std::sync::atomic::Ordering::Release);
                        bsp_barrier.get_write_schedule().clear(); // Сброс мусора из прошлого
                    }

                    // 3. Обработка ACK-пакета
                    if header.is_last == 2 {
                        bsp_barrier.completed_peers.fetch_add(1, std::sync::atomic::Ordering::Release);
                        continue;
                    }

                    // 4. Обработка спайков (safe parse: network buffer may be unaligned)
                    let payload_bytes = &buf[16..size];
                    if payload_bytes.len() % 8 == 0 && !payload_bytes.is_empty() {
                        let schedule = bsp_barrier.get_write_schedule();
                        for chunk in payload_bytes.chunks_exact(8) {
                            let ghost_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                            let tick_offset = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                            schedule.push_spike(tick_offset as usize, ghost_id);
                        }
                    }

                    // 5. Триггер барьера и отправка ACK
                    if header.is_last == 1 {
                        bsp_barrier.completed_peers.fetch_add(1, std::sync::atomic::Ordering::Release);

                        // Отправляем ACK отправителю
                        if let Some((src_addr, _mtu)) = routing_table.get_address(header.src_zone_hash) {
                            let ack = SpikeBatchHeaderV2 {
                                src_zone_hash: header.dst_zone_hash, // Меняем местами для обратного роутинга
                                dst_zone_hash: header.src_zone_hash,
                                epoch: header.epoch,
                                is_last: 2, // 2 = ACK
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
            
            unsafe { table_clone.update_routes(new_map); }
        });

        writer.await.unwrap();
        for r in readers {
            r.await.unwrap();
        }

        assert_eq!(table.get_address(0x2222).unwrap().0, "127.0.0.1:8082".parse::<SocketAddr>().unwrap());
    }
}
