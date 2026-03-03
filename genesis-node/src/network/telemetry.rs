use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use genesis_core::ipc::{TelemetryFrameHeader, TELE_MAGIC};
use genesis_compute::memory::PinnedBuffer;
use futures_util::StreamExt;

/// Lock-Free Swapchain for Telemetry.
/// Allows the Day Phase Orchestrator to dump spikes into a back-buffer
/// while the Telemetry Server broadcasts the front-buffer to clients.
pub struct TelemetrySwapchain {
    /// Pointer to the buffer currently being broadcast (read-only for Tokio).
    pub ready_for_export: AtomicPtr<u32>,
    /// Pointer to the buffer currently being filled by GPU DMA (write-only for Orchestrator).
    pub back_buffer: AtomicPtr<u32>,
    
    /// Host-mapped pointer for the count of fired IDs (size = 1 u32).
    pub count_buffer: PinnedBuffer<u32>,

    /// Number of active WebSocket clients. 
    /// If 0, the Orchestrator should skip the DMA transfer to save PCIe bandwidth.
    pub active_clients: AtomicUsize,
    
    /// Number of spikes in the ready buffer.
    pub ready_count: AtomicUsize,
    /// Last tick of the ready buffer.
    pub ready_tick: AtomicUsize,

    _buffer_a: PinnedBuffer<u32>,
    _buffer_b: PinnedBuffer<u32>,
}

impl TelemetrySwapchain {
    pub fn new(capacity: usize) -> anyhow::Result<Self> {
        let buffer_a = PinnedBuffer::new(capacity).map_err(anyhow::Error::msg)?;
        let buffer_b = PinnedBuffer::new(capacity).map_err(anyhow::Error::msg)?;
        let count_buffer = PinnedBuffer::new(1).map_err(anyhow::Error::msg)?;
        
        Ok(Self {
            ready_for_export: AtomicPtr::new(buffer_a.as_ptr() as *mut u32),
            back_buffer: AtomicPtr::new(buffer_b.as_ptr() as *mut u32),
            count_buffer,
            active_clients: AtomicUsize::new(0),
            ready_count: AtomicUsize::new(0),
            ready_tick: AtomicUsize::new(0),
            _buffer_a: buffer_a,
            _buffer_b: buffer_b,
        })
    }

    /// Toggles the buffers. Called by the Orchestrator after a successful DMA copy.
    pub fn swap_and_ready(&self, count: usize, tick: u64) {
        let back = self.back_buffer.load(Ordering::Relaxed);
        
        // Update metadata for the new ready buffer
        self.ready_count.store(count, Ordering::Relaxed);
        self.ready_tick.store(tick as usize, Ordering::Relaxed);

        // Atomic swap: Release ensures the DMA data is visible to the Telemetry thread.
        let old_ready = self.ready_for_export.swap(back, Ordering::Release);
        
        // Update the pointer for the next DMA.
        self.back_buffer.store(old_ready, Ordering::Relaxed);
    }

    /// Returns true if there are connected clients.
    pub fn is_active(&self) -> bool {
        self.active_clients.load(Ordering::Acquire) > 0
    }
}

pub struct TelemetryServer {
    pub swapchain: Arc<TelemetrySwapchain>,
}

impl TelemetryServer {
    pub fn new(capacity: usize) -> anyhow::Result<Self> {
        Ok(Self {
            swapchain: Arc::new(TelemetrySwapchain::new(capacity)?),
        })
    }

    /// Compatibility method for existing main.rs (if needed as a static-like start)
    pub async fn start(port: u16) -> Arc<TelemetrySwapchain> {
        let server = Arc::new(Self::new(1024 * 1024).unwrap());
        let server_clone = server.clone();
        let addr = format!("0.0.0.0:{}", port);
        
        tokio::spawn(async move {
            let _ = server_clone.run(&addr).await;
        });
        
        server.swapchain.clone()
    }

    /// Starts the WebSocket server on the given address.
    pub async fn run(self: Arc<Self>, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let app = Router::new()
            .route("/ws", get(move |ws| handle_ws(ws, self.clone())));

        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("[Telemetry Server] Listening on WS {}", addr);
        axum::serve(listener, app).await?;
        Ok(())
    }
}

async fn handle_ws(ws: WebSocketUpgrade, server: Arc<TelemetryServer>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| websocket_stream(socket, server))
}

async fn websocket_stream(mut socket: WebSocket, server: Arc<TelemetryServer>) {
    server.swapchain.active_clients.fetch_add(1, Ordering::SeqCst);
    println!("[Telemetry] Client connected. Active: {}", server.swapchain.active_clients.load(Ordering::Relaxed));

    let mut last_processed_tick = 0;

    loop {
        let current_tick = server.swapchain.ready_tick.load(Ordering::Acquire);
        
        if current_tick > last_processed_tick {
            let count = server.swapchain.ready_count.load(Ordering::Relaxed);
            let ptr = server.swapchain.ready_for_export.load(Ordering::Acquire);
            
            // Binary Frame Packing (Header + Array of u32)
            let header = TelemetryFrameHeader {
                magic: TELE_MAGIC,
                tick: current_tick as u32,
                spikes_count: count as u32,
                _padding: 0,
            };

            let header_bytes = bytemuck::bytes_of(&header);
            let spikes_slice = unsafe { std::slice::from_raw_parts(ptr, count) };
            let spikes_bytes = bytemuck::cast_slice(spikes_slice);

            let mut frame = Vec::with_capacity(header_bytes.len() + spikes_bytes.len());
            frame.extend_from_slice(header_bytes);
            frame.extend_from_slice(spikes_bytes);

            if let Err(_) = socket.send(Message::Binary(frame)).await {
                break; 
            }
            last_processed_tick = current_tick;
        } else {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }
    }

    server.swapchain.active_clients.fetch_sub(1, Ordering::SeqCst);
    println!("[Telemetry] Client disconnected.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::ipc::TELE_MAGIC;

    #[test]
    fn test_telemetry_frame_packing() {
        // Mock spikes: 10, 20, 30
        let spikes = vec![10u32, 20u32, 30u32];
        let tick = 1337u64;
        
        let header = TelemetryFrameHeader {
            magic: TELE_MAGIC,
            tick: tick as u32,
            spikes_count: spikes.len() as u32,
            _padding: 0,
        };
        
        let header_bytes = bytemuck::bytes_of(&header);
        let spikes_bytes = bytemuck::cast_slice(&spikes);
        
        let mut payload = Vec::new();
        payload.extend_from_slice(header_bytes);
        payload.extend_from_slice(spikes_bytes);
        
        assert_eq!(payload.len(), 16 + 3 * 4);
        assert_eq!(u32::from_le_bytes(payload[0..4].try_into().unwrap()), TELE_MAGIC);
        assert_eq!(u32::from_le_bytes(payload[4..8].try_into().unwrap()), 1337);
        assert_eq!(u32::from_le_bytes(payload[8..12].try_into().unwrap()), 3);
    }
}
