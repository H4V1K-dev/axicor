use genesis_compute::ffi;
use crate::network::SpikeEvent;
use std::ptr;

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

        // Максимум 1 спайк на аксон за батч
        // Используем 8 байт (SpikeEvent pack layout)
        let events_size = (count as usize) * std::mem::size_of::<SpikeEvent>();
        
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

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpikeBatchHeaderV2 {
    pub src_zone_hash: u64,
    pub dst_zone_hash: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpikeEventV2 {
    pub ghost_id: u32,
    pub tick_offset: u32, // Выровнено до 8 байт для Coalesced Access
}

pub struct InterNodeRouter {
    pub socket: std::sync::Arc<tokio::net::UdpSocket>,
    pub routing_table: std::sync::Arc<crate::network::router::RoutingTable>,
}

impl InterNodeRouter {
    pub fn new(socket: std::sync::Arc<tokio::net::UdpSocket>, routing_table: std::sync::Arc<crate::network::router::RoutingTable>) -> Self {
        Self { socket, routing_table }
    }

    /// Отправляет батч спайков через UDP (Zero-Copy)
    pub async fn flush_outgoing_batch(
        &self, 
        target_zone_hash: u32, 
        events: &[crate::network::SpikeEvent]
    ) {
        if let Some(target_addr) = self.routing_table.get_address(target_zone_hash) {
            let header = SpikeBatchHeaderV2 {
                src_zone_hash: 0, // Не используется для Ingress, заполняем нулем
                dst_zone_hash: target_zone_hash as u64,
            };
            
            // В сыром виде мы отправляем Header (16 байт) + Slice events (8 байт на ивент)
            // Избегаем аллокаций - используем std::io::IoSlice или копируем в thread-local буфер
            // Для упрощения (т.к. UDP отправляет одним пакетом), формируем буфер здесь:
            let mut packet = Vec::with_capacity(16 + events.len() * std::mem::size_of::<crate::network::SpikeEvent>());
            packet.extend_from_slice(bytemuck::bytes_of(&header));
            packet.extend_from_slice(bytemuck::cast_slice(events));
            
            let _ = self.socket.send_to(&packet, target_addr).await;
        }
    }

    /// Zero-Cost отправка батча спайков через Lock-Free Egress Pool
    pub fn flush_outgoing_batch_pool(
        &self, 
        pool: &crate::network::egress::EgressPool,
        src_zone_hash: u32,
        target_zone_hash: u32, 
        events: &[crate::network::SpikeEvent]
    ) {
        let Some(target_addr) = self.routing_table.get_address(target_zone_hash) else { return; };
        
        let mut msg = loop {
            if let Some(m) = pool.free_queue.pop() {
                break m;
            }
            std::hint::spin_loop();
        };

        unsafe {
            let header = msg.buffer.as_mut_ptr() as *mut SpikeBatchHeaderV2;
            (*header).src_zone_hash = src_zone_hash as u64;
            (*header).dst_zone_hash = target_zone_hash as u64;

            let events_bytes = bytemuck::cast_slice(events);
            if !events_bytes.is_empty() {
                std::ptr::copy_nonoverlapping(
                    events_bytes.as_ptr(),
                    msg.buffer.as_mut_ptr().add(16),
                    events_bytes.len()
                );
            }
            msg.size = 16 + events_bytes.len();
        }
        
        msg.target = target_addr;
        pool.ready_queue.push(msg).unwrap();
    }
    /// Запускает слушатель межзональных спайков (Sender-Side Mapping)
    pub async fn spawn_ghost_listener(
        port: u16,
        bsp_barrier: std::sync::Arc<crate::network::bsp::BspBarrier>,
    ) {
        let sock = tokio::net::UdpSocket::bind(("0.0.0.0", port)).await.expect("FATAL: Ghost Bind failed");
        
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65507];
            loop {
                if let Ok((size, _addr)) = sock.recv_from(&mut buf).await {
                    if size < 16 { continue; }
                    
                    let _header: SpikeBatchHeaderV2 = *bytemuck::from_bytes(&buf[0..16]);
                    let payload_bytes = &buf[16..size];
                    
                    if payload_bytes.len() % 8 != 0 { continue; }
                    
                    let events: &[SpikeEventV2] = bytemuck::cast_slice(payload_bytes);
                    
                    let schedule = bsp_barrier.get_write_schedule();
                    for ev in events {
                        schedule.push_spike(ev.tick_offset as usize, ev.ghost_id);
                    }
                    
                    // Увеличиваем счетчик полученных пакетов для BSP барьера
                    bsp_barrier.packets_received.fetch_add(1, std::sync::atomic::Ordering::Release);
                }
            }
        });
    }
}
