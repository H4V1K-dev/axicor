use genesis_compute::ffi;
use crate::network::{SpikeEvent, SpikeBatchHeaderV3};
use std::ptr;
use bytemuck::Zeroable;

pub struct InterNodeChannel {
    pub target_zone_hash: u32,
    pub src_zone_hash: u32,
    pub src_indices_host: Vec<u32>,
    pub src_indices_d: *mut u32,
    pub dst_ghost_ids_d: *mut u32,
    pub count: u32,
    
    // Zero-Copy Pinned RAM (доступен и GPU, и CPU)
    pub out_events_pinned: *mut SpikeEvent,
    pub out_count_pinned: *mut u32,
}

unsafe impl Send for InterNodeChannel {}
unsafe impl Sync for InterNodeChannel {}

impl Drop for InterNodeChannel {
    fn drop(&mut self) {
        unsafe {
            genesis_compute::ffi::gpu_free(self.src_indices_d as *mut _);
            genesis_compute::ffi::gpu_free(self.dst_ghost_ids_d as *mut _);
            genesis_compute::ffi::gpu_host_free(self.out_events_pinned as *mut _);
            genesis_compute::ffi::gpu_host_free(self.out_count_pinned as *mut _);
        }
    }
}

impl InterNodeChannel {
    pub unsafe fn new(src_zone_hash: u32, target_zone_hash: u32, src_indices: &[u32], dst_ghost_ids: &[u32]) -> Self {
        let count = src_indices.len() as u32;
        
        let src_d = ffi::gpu_malloc((count as usize) * 4) as *mut u32;
        let dst_d = ffi::gpu_malloc((count as usize) * 4) as *mut u32;
        ffi::gpu_memcpy_host_to_device_async(src_d as *mut _, src_indices.as_ptr() as *const _, (count as usize) * 4, ptr::null_mut());
        ffi::gpu_memcpy_host_to_device_async(dst_d as *mut _, dst_ghost_ids.as_ptr() as *const _, (count as usize) * 4, ptr::null_mut());

        // Максимум 8 спайков на аксон за батч (8-way Burst model)
        // Используем 8 байт (SpikeEvent pack layout)
        let events_size = (count as usize) * 8 * std::mem::size_of::<SpikeEvent>();
        
        Self {
            target_zone_hash,
            src_zone_hash,
            src_indices_host: src_indices.to_vec(),
            src_indices_d: src_d,
            dst_ghost_ids_d: dst_d,
            count,
            out_events_pinned: ffi::gpu_host_alloc(events_size) as *mut SpikeEvent,
            out_count_pinned: ffi::gpu_host_alloc(4) as *mut u32,
        }
    }

    pub unsafe fn extract_spikes(&self, axon_heads: *const genesis_core::layout::BurstHeads8, sync_batch_ticks: u32, v_seg: u32, stream: ffi::CudaStream) {
        if self.count == 0 { return; }
        genesis_compute::ffi::launch_extract_outgoing_spikes(
            axon_heads,
            self.src_indices_d,
            self.dst_ghost_ids_d,
            self.count,
            sync_batch_ticks,
            v_seg,
            self.out_events_pinned as *mut std::ffi::c_void,
            self.out_count_pinned,
            stream
        );
    }
}

// SpikeBatchHeaderV2 and SpikeEventV2 removed as they are replaced by V3 in ipc.rs

pub struct InterNodeRouter {
    pub socket: std::sync::Arc<tokio::net::UdpSocket>,
    pub routing_table: std::sync::Arc<crate::network::router::RoutingTable>,
}

impl InterNodeRouter {
    pub fn new(socket: std::sync::Arc<tokio::net::UdpSocket>, routing_table: std::sync::Arc<crate::network::router::RoutingTable>) -> Self {
        Self { socket, routing_table }
    }



    /// Zero-Cost отправка батча спайков через Lock-Free Egress Pool с L7-фрагментацией.
    pub fn flush_outgoing_batch_pool(
        &self,
        pool: &crate::network::egress::EgressPool,
        src_zone_hash: u32,
        target_zone_hash: u32,
        events: &[crate::network::SpikeEvent],
        epoch: u32,
        current_tick: u64, // [DOD] Heartbeat Pulse
    ) {
        let Some(target_addr) = self.routing_table.get_address(target_zone_hash) else { return; };
        const MAX_EVENTS_PER_PACKET: usize = 8186;

        // Отправка пустого Heartbeat, если спайков нет
        if events.is_empty() {
            let mut msg = loop {
                if let Some(m) = pool.free_queue.pop() { break m; }
                std::hint::spin_loop();
            };
            unsafe {
                let header = msg.buffer.as_mut_ptr() as *mut SpikeBatchHeaderV3;
                (*header).src_zone_hash = src_zone_hash;
                (*header).dst_zone_hash = target_zone_hash;
                (*header).epoch = epoch;
                (*header).is_last = 1; // Единственный и последний
                (*header).tick = current_tick;
                msg.size = 32;
            }
            msg.target = target_addr;
            pool.ready_queue.push(msg).unwrap();
            return;
        }

        // L7 Фрагментация
        let chunks = events.chunks(MAX_EVENTS_PER_PACKET);
        let total_chunks = chunks.len();

        for (i, chunk) in chunks.enumerate() {
            let mut msg = loop {
                if let Some(m) = pool.free_queue.pop() { break m; }
                std::hint::spin_loop();
            };

            unsafe {
                let header = msg.buffer.as_mut_ptr() as *mut SpikeBatchHeaderV3;
                (*header).src_zone_hash = src_zone_hash;
                (*header).dst_zone_hash = target_zone_hash;
                (*header).epoch = epoch;
                // Только последний чанк пробивает барьер получателя
                (*header).is_last = if i == total_chunks - 1 { 1 } else { 0 };
                (*header).tick = current_tick;
 
                let events_bytes = bytemuck::cast_slice(chunk);
                std::ptr::copy_nonoverlapping(
                    events_bytes.as_ptr(),
                    msg.buffer.as_mut_ptr().add(32),
                    events_bytes.len()
                );
                msg.size = 32 + events_bytes.len();
            }
            msg.target = target_addr;
            pool.ready_queue.push(msg).unwrap();
        }
    }
    /// Запускает слушатель межзональных спайков (Sender-Side Mapping)
    pub async fn spawn_ghost_listener(
        port: u16,
        bsp_barrier: std::sync::Arc<crate::network::bsp::BspBarrier>,
        routing_table: std::sync::Arc<crate::network::router::RoutingTable>,
    ) {
        let sock = tokio::net::UdpSocket::bind(("0.0.0.0", port)).await.expect("FATAL: Ghost Bind failed");
        
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65507];
            loop {
                if let Ok((size, _)) = sock.recv_from(&mut buf).await {
                    if size < 32 { continue; }
 
                    let header: SpikeBatchHeaderV3 = *bytemuck::from_bytes(&buf[0..32]);
                    let current_epoch = bsp_barrier.current_epoch.load(std::sync::atomic::Ordering::Acquire);
 
                    // [DOD Pulse] Refresh health registry
                    bsp_barrier.peer_last_seen.insert(header.src_zone_hash, std::time::Instant::now());
 
                    // 1. Biological Amnesia: Игнорируем пакеты из прошлого
                    if header.epoch < current_epoch {
                        continue;
                    }
 
                    // 2. Self-Healing: Прыжок в будущее, если мы отстали (или пропустили пакет)
                    if header.epoch > current_epoch {
                        println!("⚠️ [BSP] Self-Healing: Fast-forwarding epoch {} -> {} to catch up", current_epoch, header.epoch);
                        bsp_barrier.current_epoch.store(header.epoch, std::sync::atomic::Ordering::Release);
                        bsp_barrier.completed_peers.store(0, std::sync::atomic::Ordering::Release);
                        bsp_barrier.get_write_schedule().clear();
                    }
 
                    // 3. Обработка ACK-пакета
                    if header.is_last == 2 {
                        bsp_barrier.completed_peers.fetch_add(1, std::sync::atomic::Ordering::Release);
                        continue;
                    }
 
                    // 4. Обработка спайков
                    let payload_bytes = &buf[32..size];
                    if payload_bytes.len() % 8 == 0 && !payload_bytes.is_empty() {
                        let events: &[crate::network::SpikeEvent] = bytemuck::cast_slice(payload_bytes);
                        let schedule = bsp_barrier.get_write_schedule();
                        for ev in events {
                            schedule.push_spike(ev.tick_offset as usize, ev.ghost_axon_id);
                        }
                    }

                    // 5. Триггер барьера и отправка ACK
                    if header.is_last == 1 {
                        bsp_barrier.completed_peers.fetch_add(1, std::sync::atomic::Ordering::Release);
 
                        // Отправляем ACK отправителю
                        if let Some(src_addr) = routing_table.get_address(header.src_zone_hash) {
                            let mut ack = SpikeBatchHeaderV3::zeroed();
                            ack.src_zone_hash = header.dst_zone_hash; // Меняем местами для обратного роутинга
                            ack.dst_zone_hash = header.src_zone_hash;
                            ack.epoch = header.epoch;
                            ack.is_last = 2; // 2 = ACK
                            ack.tick = header.tick;
 
                            let _ = sock.send_to(bytemuck::bytes_of(&ack), src_addr).await;
                        }
                    }
                }
            }
        });
    }
}
