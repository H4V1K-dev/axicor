use crate::network::SpikeEvent;
use std::ptr;

pub struct InterNodeChannel {
    pub target_zone_hash: u32,
    pub src_zone_hash: u32,
    pub capacity: u32, // [DOD FIX] Физический лимит VRAM
    pub count: u32,    // [DOD FIX] Текущее число активных связей

    pub src_indices_host: Vec<u32>,
    pub dst_ghost_ids_host: Vec<u32>, // [DOD FIX] Зеркало для Swap-and-Pop

    pub src_indices_d: *mut u32,
    pub dst_ghost_ids_d: *mut u32,

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
    pub unsafe fn new(src_zone_hash: u32, target_zone_hash: u32, src_indices: &[u32], dst_ghost_ids: &[u32], capacity: u32) -> Self {
        let count = src_indices.len() as u32;
        assert!(count <= capacity, "FATAL: Initial connections exceed capacity");

        // Аллокация под MAX CAPACITY, а не под текущий count
        let bytes_capacity = (capacity as usize) * 4;
        let src_d = genesis_compute::ffi::gpu_malloc(bytes_capacity) as *mut u32;
        let dst_d = genesis_compute::ffi::gpu_malloc(bytes_capacity) as *mut u32;

        // Копируем только РЕАЛЬНЫЕ данные
        if count > 0 {
            let bytes_active = (count as usize) * 4;
            genesis_compute::ffi::gpu_memcpy_host_to_device_async(src_d as *mut _, src_indices.as_ptr() as *const _, bytes_active, ptr::null_mut());
            genesis_compute::ffi::gpu_memcpy_host_to_device_async(dst_d as *mut _, dst_ghost_ids.as_ptr() as *const _, bytes_active, ptr::null_mut());
        }

        // Максимум 8 спайков на аксон за батч. Лимит берется от Capacity.
        let events_size = (capacity as usize) * 8 * std::mem::size_of::<SpikeEvent>();

        // Host-mirroring с преаллокацией
        let mut src_host = Vec::with_capacity(capacity as usize);
        src_host.extend_from_slice(src_indices);
        let mut dst_host = Vec::with_capacity(capacity as usize);
        dst_host.extend_from_slice(dst_ghost_ids);

        Self {
            target_zone_hash,
            src_zone_hash,
            capacity,
            count,
            src_indices_host: src_host,
            dst_ghost_ids_host: dst_host,
            src_indices_d: src_d,
            dst_ghost_ids_d: dst_d,
            out_events_pinned: genesis_compute::ffi::gpu_host_alloc(events_size) as *mut SpikeEvent,
            out_count_pinned: genesis_compute::ffi::gpu_host_alloc(4) as *mut u32,
        }
    }

    pub unsafe fn extract_spikes(&self, axon_heads: *const genesis_core::layout::BurstHeads8, sync_batch_ticks: u32, v_seg: u32, stream: genesis_compute::ffi::CudaStream) {
        if self.count == 0 { return; }
        genesis_compute::ffi::launch_extract_outgoing_spikes(
            axon_heads,
            self.src_indices_d,
            self.dst_ghost_ids_d,
            self.count, // Ядро читает только count, мусор в хвосте VRAM игнорируется
            sync_batch_ticks,
            v_seg,
            self.out_events_pinned as *mut std::ffi::c_void,
            self.out_count_pinned,
            stream
        );
    }

    /// O(1) добавление новой межзональной связи (Hot-Patching)
    pub unsafe fn push_route(&mut self, src_axon: u32, dst_ghost: u32, stream: genesis_compute::ffi::CudaStream) {
        assert!(self.count < self.capacity, "FATAL: Routing capacity exceeded. Increase ghost_capacity.");
        let idx = self.count as usize;

        self.src_indices_host.push(src_axon);
        self.dst_ghost_ids_host.push(dst_ghost);

        let src_ptr = self.src_indices_host.as_ptr().add(idx);
        let dst_ptr = self.dst_ghost_ids_host.as_ptr().add(idx);

        // Микро-DMA (4 байта) прямо в свободный хвост массива VRAM
        genesis_compute::ffi::gpu_memcpy_host_to_device_async(
            self.src_indices_d.add(idx) as *mut _,
            src_ptr as *const _,
            4,
            stream
        );
        genesis_compute::ffi::gpu_memcpy_host_to_device_async(
            self.dst_ghost_ids_d.add(idx) as *mut _,
            dst_ptr as *const _,
            4,
            stream
        );

        self.count += 1;
    }

    /// O(1) удаление связи через Swap-and-Pop
    pub unsafe fn prune_route(&mut self, target_ghost_id: u32, stream: genesis_compute::ffi::CudaStream) {
        // O(N) поиск на хосте допустим, т.к. выполняется только в Night Phase
        let Some(idx) = self.dst_ghost_ids_host.iter().position(|&g| g == target_ghost_id) else {
            return; // Связь уже удалена
        };

        let last_idx = (self.count - 1) as usize;

        // 1. Swap на хосте
        self.src_indices_host.swap(idx, last_idx);
        self.dst_ghost_ids_host.swap(idx, last_idx);

        // 2. Pop хвоста
        self.src_indices_host.pop();
        self.dst_ghost_ids_host.pop();

        // 3. Если удаленный элемент не был последним, нужно пропатчить образовавшуюся дыру в VRAM
        if idx != last_idx {
            let src_ptr = self.src_indices_host.as_ptr().add(idx);
            let dst_ptr = self.dst_ghost_ids_host.as_ptr().add(idx);

            genesis_compute::ffi::gpu_memcpy_host_to_device_async(
                self.src_indices_d.add(idx) as *mut _,
                src_ptr as *const _,
                4,
                stream
            );
            genesis_compute::ffi::gpu_memcpy_host_to_device_async(
                self.dst_ghost_ids_d.add(idx) as *mut _,
                dst_ptr as *const _,
                4,
                stream
            );
        }

        self.count -= 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpikeBatchHeaderV2 {
    pub src_zone_hash: u32,
    pub dst_zone_hash: u32,
    pub epoch: u32,
    pub is_last: u32, // 1 = последний чанк в эпохе (Heartbeat)
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



    /// Zero-Cost отправка батча спайков через Lock-Free Egress Pool с L7-фрагментацией.
    pub fn flush_outgoing_batch_pool(
        &self,
        pool: &crate::network::egress::EgressPool,
        src_zone_hash: u32,
        target_zone_hash: u32,
        events: &[crate::network::SpikeEvent],
        epoch: u32, // [DOD] Инъекция эпохи
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
        let chunks = events.chunks(MAX_EVENTS_PER_PACKET);
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
        bsp_barrier: std::sync::Arc<crate::network::bsp::BspBarrier>,
        routing_table: std::sync::Arc<crate::network::router::RoutingTable>,
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
                            println!("⚠️ [BSP] Self-Healing: Fast-forwarding epoch {} -> {} to catch up ({} jumps)", current_epoch, header.epoch, n + 1);
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
                        if let Some(src_addr) = routing_table.get_address(header.src_zone_hash) {
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
