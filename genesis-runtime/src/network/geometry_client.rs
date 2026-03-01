use anyhow::{Context, Result};
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};

use super::slow_path::{GeometryRequest, GeometryResponse};

/// Connects to a GeometryServer, sends a request, and waits for a response.
pub async fn send_geometry_request(
    target_addr: SocketAddr,
    req: &GeometryRequest,
) -> Result<GeometryResponse> {
    let mut stream = TcpStream::connect(target_addr).await
        .with_context(|| format!("Failed to connect to GeometryServer at {}", target_addr))?;

    // Serialize payload
    let encoded = bincode::serialize(req).context("Failed to serialize GeometryRequest")?;

    // Send Length-prefix (4 bytes, little-endian)
    let len = encoded.len() as u32;
    stream.write_all(&len.to_le_bytes()).await?;

    // Send payload
    stream.write_all(&encoded).await?;

    // Read Length-prefix of response
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await
        .context("Failed to read length prefix from GeometryResponse")?;
    let resp_len = u32::from_le_bytes(len_buf) as usize;

    // Read payload
    let mut resp_buf = vec![0u8; resp_len];
    stream.read_exact(&mut resp_buf).await
        .context("Failed to read payload from GeometryResponse")?;

    let resp: GeometryResponse = bincode::deserialize(&resp_buf)
        .context("Failed to deserialize GeometryResponse")?;

    Ok(resp)
}

/// A TCP Server that accepts structural graph updates (GeometryRequest)
/// from neighboring shards and passes them via MPSC to the Orchestrator.
pub struct GeometryServer {
    listener: TcpListener,
}

impl GeometryServer {
    /// Binds the server to the provided SocketAddr.
    pub async fn bind(addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        Ok(Self { listener })
    }

    /// Returns the active local socket address
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Spawns the server loop in a Tokio task which continuously serves GEOM frames to connected IDEs.
    pub fn spawn(self) {
        // Generate a dense grid of Mock PackedPositions (1024 neurons)
        // PackedPosition: Type(4b) | Z(8b) | Y(10b) | X(10b)
        let num_neurons = 1024;
        let mut positions = Vec::with_capacity(num_neurons);
        let grid_size = 10;
        
        for i in 0..num_neurons {
            let x = (i % grid_size) as u32 * 2;
            let y = ((i / grid_size) % grid_size) as u32 * 2;
            let z = (i / (grid_size * grid_size)) as u32 * 2;
            // Type is 0..3 cyclically
            let t = (i % 4) as u32; 
            
            let packed = (x & 0x3FF) | ((y & 0x3FF) << 10) | ((z & 0xFF) << 20) | ((t & 0xF) << 28);
            positions.push(packed);
        }

        let payload_size = num_neurons * 4;
        let mut buf = Vec::with_capacity(12 + payload_size);
        buf.extend_from_slice(&0x47454F4Du32.to_le_bytes()); // "GEOM"
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&(num_neurons as u32).to_le_bytes()); // num_neurons
        
        let pos_bytes = unsafe {
            std::slice::from_raw_parts(positions.as_ptr() as *const u8, payload_size)
        };
        buf.extend_from_slice(pos_bytes);

        let shared_payload = std::sync::Arc::new(buf);

        tokio::spawn(async move {
            loop {
                let (mut stream, _) = match self.listener.accept().await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                
                let data = shared_payload.clone();
                tokio::spawn(async move {
                    let _ = stream.write_all(&data).await;
                });
            }
        });
    }
}
