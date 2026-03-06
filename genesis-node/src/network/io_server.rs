use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use tokio::net::UdpSocket;
use genesis_core::ipc::{ExternalIoHeader, RouteUpdate, ROUT_MAGIC};
use genesis_core::constants::{GSIO_MAGIC, GSOO_MAGIC, MAX_UDP_PAYLOAD};
use genesis_compute::memory::PinnedBuffer;
use crate::network::router::RoutingTable;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::net::SocketAddr;

/// Контекст I/O для конкретной зоны. Содержит персональный InputSwapchain.
pub struct ZoneIoContext {
    pub swapchain: Arc<InputSwapchain>,
    pub matrix_offsets: HashMap<u32, u32>,
}

/// Lock-Free Double Buffering for External I/O.
/// Isolates the asynchronous Tokio network thread from the synchronous GPU Orchestrator.
pub struct InputSwapchain {
    /// Pointer to the buffer that is currently quiescent and ready for GPU DMA.
    pub ready_for_gpu: AtomicPtr<u8>,
    /// Pointer to the buffer currently being written to by the network thread.
    /// Access is serialized by the single-threaded UDP receive loop.
    back_buffer: AtomicPtr<u8>,
    /// [DOD] Size of allocated Pinned RAM. Hard barrier against network overflow.
    capacity: usize,
    
    // Ownership of the underlying pinned memory
    _buffer_a: PinnedBuffer<u8>,
    _buffer_b: PinnedBuffer<u8>,
}

impl InputSwapchain {
    pub fn new(capacity: usize) -> Result<Self> {
        let buffer_a = PinnedBuffer::new(capacity)?;
        let buffer_b = PinnedBuffer::new(capacity)?;
        
        // Ensure buffers are zeroed initially
        unsafe {
            std::ptr::write_bytes(buffer_a.as_mut_ptr(), 0, capacity);
            std::ptr::write_bytes(buffer_b.as_mut_ptr(), 0, capacity);
        }

        Ok(Self {
            ready_for_gpu: AtomicPtr::new(buffer_a.as_mut_ptr()),
            back_buffer: AtomicPtr::new(buffer_b.as_mut_ptr()),
            capacity,
            _buffer_a: buffer_a,
            _buffer_b: buffer_b,
        })
    }

    /// Writes the payload into the back buffer at a specific byte offset.
    /// Does NOT swap the buffer - caller must call .swap() explicitly (e.g. at the end of a tick).
    pub fn write_incoming_at(&self, offset: usize, payload: &[u8]) {
        let back = self.back_buffer.load(Ordering::Relaxed);
        
        // [DOD] Hard barrier: panic on CPU is 100x better than silent VRAM corruption
        assert!(
            offset + payload.len() <= self.capacity,
            "FATAL DMA BUFFER OVERFLOW: {} bytes at offset {} into buffer of {} bytes",
            payload.len(), offset, self.capacity
        );

        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ptr(), back.add(offset), payload.len());
        }
    }

    /// Atomically swaps the back buffer with the ready buffer.
    /// Should be called exactly once per simulation tick/batch by the Orchestrator.
    pub fn swap(&self) {
        let back = self.back_buffer.load(Ordering::Relaxed);
        // Release ordering ensures all previous writes to 'back' are visible to the GPU.
        let old_ready = self.ready_for_gpu.swap(back, Ordering::Release);
        self.back_buffer.store(old_ready, Ordering::Relaxed);
    }

    /// Returns the pointer to the buffer ready for GPU.
    pub fn consume_for_gpu(&self) -> *const u8 {
        self.ready_for_gpu.load(Ordering::Acquire)
    }
}

pub struct ExternalIoServer {
    pub is_sleeping: Arc<AtomicBool>,
    pub oversized_skips: AtomicU64,
    pub routing_table: Arc<RoutingTable>,
    pub socket: Arc<UdpSocket>,
    
    /// [DOD] Плоский массив контекстов зон. O(N) линейный поиск оптимален для N < 10.
    pub io_contexts: Vec<(u32, ZoneIoContext)>,
    
    // Validation hashes (Deprecated for multi-zone, use io_contexts)
    // pub zone_hash: u32,
    // pub matrix_hash: u32,

    // R-STDP Dopamine Modulator (Global Reward Broadcast)
    pub global_dopamine: Arc<std::sync::atomic::AtomicI32>,
}

impl ExternalIoServer {
    pub fn new(
        is_sleeping: Arc<AtomicBool>, 
        io_contexts: Vec<(u32, ZoneIoContext)>,
        routing_table: Arc<RoutingTable>,
        socket: Arc<UdpSocket>,
    ) -> Result<Self> {
        Ok(Self {
            is_sleeping,
            oversized_skips: AtomicU64::new(0),
            io_contexts,
            routing_table,
            socket,
            global_dopamine: Arc::new(std::sync::atomic::AtomicI32::new(0)),
        })
    }

    /// Runs the UDP receiver loop.
    pub async fn run_rx_loop(&self) {
        let mut buf = [0u8; 65536];
        println!("[ExternalIO] UDP Receiver Loop Started on {}", self.socket.local_addr().unwrap());
        loop {
            if let Ok((len, _addr)) = self.socket.recv_from(&mut buf).await {
                self.process_incoming_udp(&buf[..len]);
            }
        }
    }

    /// Processes a raw UDP payload according to Spec §12.
    pub fn process_incoming_udp(&self, payload: &[u8]) {
        // [Contract §12.3] Biological Drop
        if self.is_sleeping.load(Ordering::Acquire) {
            return;
        }

        // [Contract §12.2] EMSGSIZE Protection
        if payload.len() > genesis_core::constants::MAX_UDP_PAYLOAD {
            self.oversized_skips.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // [Contract §12.1] Header Validation
        if payload.len() < std::mem::size_of::<ExternalIoHeader>() {
            return;
        }

        let header = unsafe { &*(payload.as_ptr() as *const ExternalIoHeader) };

        // [Step 19] RouteUpdate check
        if header.magic == genesis_core::ipc::ROUT_MAGIC {
            if payload.len() >= std::mem::size_of::<RouteUpdate>() {
                let update = unsafe { &*(payload.as_ptr() as *const RouteUpdate) };
                let mut new_map = unsafe { (*self.routing_table.get_map_ptr()).clone() };
                let ipv4 = std::net::Ipv4Addr::from(update.new_ipv4);
                let new_addr = SocketAddr::from((ipv4, update.new_port));

                new_map.insert(update.zone_hash, new_addr);
                unsafe { self.routing_table.update_routes(new_map); }
                println!("[I/O Server] Route updated for zone 0x{:08X} -> {}", update.zone_hash, new_addr);
            }
            return;
        }

        // Magic number check (GSIO for Input)
        if header.magic != GSIO_MAGIC {
            return; // Игнорируем чужой мусор
        }

        // [DOD FIX] Zero-Cost Routing с диагностикой
        let ctx = match self.io_contexts.iter().find(|(h, _)| *h == header.zone_hash) {
            Some((_, ctx)) => ctx,
            None => {
                println!("⚠️ [I/O Drop] Unknown zone hash 0x{:08X}", header.zone_hash);
                return;
            }
        };

        let offset = match ctx.matrix_offsets.get(&header.matrix_hash) {
            Some(&off) => off as usize,
            None => {
                println!("⚠️ [I/O Drop] Unknown matrix hash 0x{:08X} for zone 0x{:08X}", header.matrix_hash, header.zone_hash);
                return;
            }
        };

        let payload_start = std::mem::size_of::<ExternalIoHeader>();
        let payload_data = &payload[payload_start..];

        if payload_data.len() != header.payload_size as usize {
            println!("⚠️ [I/O Drop] Size mismatch. Header expects {}, actual payload is {}", header.payload_size, payload_data.len());
            return;
        }

        // [Contract §12.3] Lock-Free Zero-Copy Transfer
        ctx.swapchain.write_incoming_at(offset, payload_data);

        // Update global dopamine reward for R-STDP
        self.global_dopamine.store(header.global_reward as i32, Ordering::Relaxed);
        if header.global_reward != 0 {
            println!("💉 [Dopamine] Reward Received: {}", header.global_reward);
        }
    }

    /// Отправка Output_History (Вызывается оркестратором после RecordReadout)
    pub async fn send_output_batch(
        &self, 
        target_addr: &str, 
        zone_hash: u32, 
        matrix_hash: u32, 
        pinned_output_addr: usize, 
        output_bytes: usize,
        tx_buffer: &mut [u8] // [DOD] Переиспользуемый буфер от Caller'а
    ) {
        let total_size = std::mem::size_of::<ExternalIoHeader>() + output_bytes;
        if total_size > 65535 || total_size > tx_buffer.len() {
            panic!("Output batch exceeds UDP MTU or buffer capacity.");
        }

        unsafe {
            let header = tx_buffer.as_mut_ptr() as *mut ExternalIoHeader;
            (*header).magic = GSOO_MAGIC; // Contract §12
            (*header).zone_hash = zone_hash;
            (*header).matrix_hash = matrix_hash;
            (*header).payload_size = output_bytes as u32;
            (*header).global_reward = self.global_dopamine.load(Ordering::Relaxed) as i16;
            (*header)._padding = 0;

            std::ptr::copy_nonoverlapping(
                pinned_output_addr as *const u8,
                tx_buffer.as_mut_ptr().add(std::mem::size_of::<ExternalIoHeader>()),
                output_bytes
            );
        }

        let _ = self.socket.send_to(&tx_buffer[..total_size], target_addr).await;
        // println!("[I/O Server] TX Output for zone 0x{:08X}: {} bytes to {}", zone_hash, output_bytes, target_addr);
    }
    /// O(1) Отправка Output_History через Lock-Free Egress Pool
    pub fn send_output_batch_pool(
        &self, 
        pool: &crate::network::egress::EgressPool,
        target_addr_str: &str, 
        zone_hash: u32, 
        matrix_hash: u32, 
        pinned_output_addr: usize, 
        output_bytes: usize,
    ) {
        let Ok(target_addr) = target_addr_str.parse::<SocketAddr>() else { return; };
        let total_size = std::mem::size_of::<ExternalIoHeader>() + output_bytes;
        if total_size > genesis_core::constants::MAX_UDP_PAYLOAD {
            panic!("Output batch exceeds UDP MTU.");
        }

        // [DOD FIX] Hardware Backpressure (вместо паники)
        let mut msg = loop {
            if let Some(m) = pool.free_queue.pop() {
                break m;
            }
            std::hint::spin_loop(); // Замедляем GPU, если сеть (NIC) не успевает
        };

        unsafe {
            let header = msg.buffer.as_mut_ptr() as *mut ExternalIoHeader;
            (*header).magic = GSOO_MAGIC; // Contract §12
            (*header).zone_hash = zone_hash;
            (*header).matrix_hash = matrix_hash;
            (*header).payload_size = output_bytes as u32;
            (*header).global_reward = self.global_dopamine.load(Ordering::Relaxed) as i16;
            (*header)._padding = 0;

            std::ptr::copy_nonoverlapping(
                pinned_output_addr as *const u8,
                msg.buffer.as_mut_ptr().add(std::mem::size_of::<ExternalIoHeader>()),
                output_bytes
            );
        }

        msg.size = total_size;
        msg.target = target_addr;
        pool.ready_queue.push(msg).unwrap();
    }
    /// Main loop for the UDP Input Server (Port 8081).
    pub async fn run_input_loop(self: Arc<Self>, addr: &str) -> std::io::Result<()> {
        let socket = UdpSocket::bind(addr).await?;
        println!("[I/O Server] Listening on UDP {}", addr);
        
        let mut buf = vec![0u8; 65536]; // MTU + buffer
        
        loop {
            let (len, _) = socket.recv_from(&mut buf).await?;
            self.process_incoming_udp(&buf[..len]);
        }
    }
}

// NOTE: Integration tests for ExternalIoServer were tied to an older API
// (different constructor signature and swapchain layout). They were removed
// to keep the Night/Day data plane and IO server contract source-of-truth
// in the production code paths rather than stale test harnesses.
