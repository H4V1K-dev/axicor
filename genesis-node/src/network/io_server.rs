use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use tokio::net::UdpSocket;
use genesis_core::ipc::{ExternalIoHeader, RouteUpdate, ROUT_MAGIC};
use genesis_core::constants::{GSIO_MAGIC, MAX_UDP_PAYLOAD};
use genesis_compute::memory::PinnedBuffer;
use crate::network::router::RoutingTable;
use std::collections::HashMap;
use std::net::SocketAddr;

/// Lock-Free Double Buffering for External I/O.
/// Isolates the asynchronous Tokio network thread from the synchronous GPU Orchestrator.
pub struct InputSwapchain {
    /// Pointer to the buffer that is currently quiescent and ready for GPU DMA.
    pub ready_for_gpu: AtomicPtr<u8>,
    /// Pointer to the buffer currently being written to by the network thread.
    /// Access is serialized by the single-threaded UDP receive loop.
    back_buffer: AtomicPtr<u8>,
    
    // Ownership of the underlying pinned memory
    _buffer_a: PinnedBuffer<u8>,
    _buffer_b: PinnedBuffer<u8>,
}

impl InputSwapchain {
    pub fn new(capacity: usize) -> Result<Self, String> {
        let mut buffer_a = PinnedBuffer::new(capacity)?;
        let mut buffer_b = PinnedBuffer::new(capacity)?;
        
        // Ensure buffers are zeroed initially
        unsafe {
            std::ptr::write_bytes(buffer_a.as_mut_ptr(), 0, capacity);
            std::ptr::write_bytes(buffer_b.as_mut_ptr(), 0, capacity);
        }

        Ok(Self {
            ready_for_gpu: AtomicPtr::new(buffer_a.as_mut_ptr()),
            back_buffer: AtomicPtr::new(buffer_b.as_mut_ptr()),
            _buffer_a: buffer_a,
            _buffer_b: buffer_b,
        })
    }

    /// Writes the payload into the back buffer and swaps it with the ready buffer.
    /// This is called by the Tokio thread.
    pub fn write_incoming(&self, payload: &[u8]) {
        // 1. Get the current back buffer. 
        // Relaxation: we are the only writer, so Relaxed is fine for fetching our own ptr.
        let back = self.back_buffer.load(Ordering::Relaxed);
        
        unsafe {
            // 2. Zero-copy (direct memcpy) into pinned memory.
            std::ptr::copy_nonoverlapping(payload.as_ptr(), back, payload.len());
        }

        // 3. Atomically swap the back buffer with the ready buffer.
        // Release ordering ensures the memcpy is visible to the Acquire load in the Orchestrator.
        let old_ready = self.ready_for_gpu.swap(back, Ordering::Release);
        
        // 4. Update our back buffer pointer for the next packet.
        self.back_buffer.store(old_ready, Ordering::Relaxed);
    }

    /// Returns the pointer to the buffer ready for GPU.
    /// Called by the Day Phase Orchestrator.
    pub fn consume_for_gpu(&self) -> *const u8 {
        self.ready_for_gpu.load(Ordering::Acquire)
    }
}

pub struct ExternalIoServer {
    pub is_sleeping: Arc<AtomicBool>,
    pub oversized_skips: AtomicU64,
    pub swapchain: Arc<InputSwapchain>,
    pub dashboard: Arc<crate::tui::DashboardState>,
    pub routing_table: Arc<RoutingTable>,
    pub socket: Arc<UdpSocket>,
    pub matrix_offsets: std::collections::HashMap<u32, u32>,
    
    // Validation hashes
    pub zone_hash: u32,
    pub matrix_hash: u32,
}

impl ExternalIoServer {
    pub fn new(
        is_sleeping: Arc<AtomicBool>, 
        capacity: usize,
        zone_hash: u32,
        matrix_hash: u32,
        dashboard: Arc<crate::tui::DashboardState>,
        routing_table: Arc<RoutingTable>,
        socket: Arc<UdpSocket>,
    ) -> Result<Self, String> {
        Ok(Self {
            is_sleeping,
            oversized_skips: AtomicU64::new(0),
            swapchain: Arc::new(InputSwapchain::new(capacity)?),
            zone_hash,
            matrix_hash,
            dashboard,
            routing_table,
            socket,
            matrix_offsets: std::collections::HashMap::new(),
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
        self.dashboard.udp_in_packets.fetch_add(1, Ordering::Relaxed);

        // [Contract §12.3] Biological Drop
        if self.is_sleeping.load(Ordering::Acquire) {
            return;
        }

        // [Contract §12.2] EMSGSIZE Protection
        if payload.len() > MAX_UDP_PAYLOAD {
            self.dashboard.oversized_skips.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // [Contract §12.1] Header Validation
        if payload.len() < 4 {
            return;
        }

        let magic = u32::from_le_bytes(payload[0..4].try_into().unwrap());

        // [Step 19] RouteUpdate check
        if magic == ROUT_MAGIC {
            if payload.len() >= std::mem::size_of::<RouteUpdate>() {
                let update = unsafe { &*(payload.as_ptr() as *const RouteUpdate) };
                
                // Nuclear RCU update: Atomic swap of the entire table
                let mut new_map = unsafe { (*self.routing_table.get_map_ptr()).clone() };
                let ipv4 = std::net::Ipv4Addr::from(update.new_ipv4);
                let new_addr = SocketAddr::from((ipv4, update.new_port));
                
                new_map.insert(update.zone_hash, new_addr);
                unsafe { self.routing_table.update_routes(new_map); }
                
                println!("[I/O Server] Route updated for zone 0x{:08X} -> {}", update.zone_hash, new_addr);
            }
            return;
        }

        if payload.len() < std::mem::size_of::<ExternalIoHeader>() {
            return;
        }

        let header = unsafe { &*(payload.as_ptr() as *const ExternalIoHeader) };

        // Magic number check (GSIO for Input)
        if header.magic != GSIO_MAGIC {
            return;
        }

        // Hash validation
        if header.zone_hash != self.zone_hash || header.matrix_hash != self.matrix_hash {
            return;
        }

        let payload_start = std::mem::size_of::<ExternalIoHeader>();
        let payload_data = &payload[payload_start..];
        
        if payload_data.len() != header.payload_size as usize {
            return; // Corrupt size field
        }

        // [Contract §12.3] Lock-Free Zero-Copy Transfer
        self.swapchain.write_incoming(payload_data);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[tokio::test]
    async fn test_udp_oversized_drop() {
        let is_sleeping = Arc::new(AtomicBool::new(false));
        let dashboard = Arc::new(crate::tui::DashboardState::new(false));
        let routing_table = Arc::new(RoutingTable::new(HashMap::new()));
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let server = ExternalIoServer::new(is_sleeping, 1024, 0, 0, dashboard, routing_table, socket).unwrap();
        
        let huge_payload = vec![0u8; MAX_UDP_PAYLOAD + 1];
        server.process_incoming_udp(&huge_payload);
        
        assert_eq!(server.oversized_skips.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_io_multiplexing() {
        let is_sleeping = Arc::new(AtomicBool::new(false));
        let zone_hash = 0x1234;
        let matrix_hash = 0xDEADBEEF;
        let dashboard = Arc::new(crate::tui::DashboardState::new(false));
        let routing_table = Arc::new(RoutingTable::new(HashMap::new()));
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let server = ExternalIoServer::new(is_sleeping, 1024, zone_hash, matrix_hash, dashboard, routing_table, socket).unwrap();
        
        let mut packet = Vec::new();
        let header = ExternalIoHeader::new(GSIO_MAGIC, zone_hash, matrix_hash, 4);
        
        unsafe {
            let header_bytes = std::slice::from_raw_parts(
                (&header as *const ExternalIoHeader) as *const u8,
                16
            );
            packet.extend_from_slice(header_bytes);
        }
        packet.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        
        server.process_incoming_udp(&packet);
        
        let ready_ptr = server.swapchain.consume_for_gpu();
        let data = unsafe { std::slice::from_raw_parts(ready_ptr, 4) };
        assert_eq!(data, &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[tokio::test]
    async fn test_route_broadcast_rcu() {
        let is_sleeping = Arc::new(AtomicBool::new(false));
        let dashboard = Arc::new(crate::tui::DashboardState::new(false));
        let routing_table = Arc::new(RoutingTable::new(HashMap::new()));
        
        let zone_a = 0xAAAA_AAAA;
        let old_addr: SocketAddr = "1.1.1.1:1111".parse().unwrap();
        unsafe {
            let mut map = HashMap::new();
            map.insert(zone_a, old_addr);
            routing_table.update_routes(map);
        }
        
        assert_eq!(routing_table.get_address(zone_a), Some(old_addr));

        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let server = ExternalIoServer::new(
            is_sleeping, 1024, 0, 0, dashboard, routing_table.clone(), socket
        ).unwrap();

        let new_ip = std::net::Ipv4Addr::new(2, 2, 2, 2);
        let new_port = 2222;
        let update = RouteUpdate {
            magic: ROUT_MAGIC,
            zone_hash: zone_a,
            new_ipv4: u32::from_be_bytes(new_ip.octets()),
            new_port,
            _padding: 0,
        };
        
        let packet = bytemuck::bytes_of(&update);
        server.process_incoming_udp(packet);
        
        let new_addr: SocketAddr = "2.2.2.2:2222".parse().unwrap();
        assert_eq!(routing_table.get_address(zone_a), Some(new_addr));
    }
}
