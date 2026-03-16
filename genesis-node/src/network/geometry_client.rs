use anyhow::{Context, Result};
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use std::sync::Arc;

use super::slow_path::{GeometryRequest, GeometryResponse, SlowPathQueues};

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

pub struct GeometryServer {
    listener: TcpListener,
    pub slow_path_queues: Arc<SlowPathQueues>,
}

impl GeometryServer {
    pub async fn bind(addr: SocketAddr, shared_acks_queue: Arc<crossbeam::queue::SegQueue<super::slow_path::AxonHandoverAck>>) -> Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        let mut queues = SlowPathQueues::new();
        queues.incoming_ack = shared_acks_queue;
        Ok(Self { 
            listener,
            slow_path_queues: Arc::new(queues),
        })
    }

    /// Spawns the server loop serving the provided u32 neuron data and parsing requests.
    pub fn spawn(self, geometry_data: Vec<u32>) {
        let num_neurons = geometry_data.len();
        
        // Prepare legacy IDE telemetry payload
        let mut buf = Vec::with_capacity(8 + num_neurons * 4);
        buf.extend_from_slice(b"GEOM"); 
        buf.extend_from_slice(&(num_neurons as u32).to_le_bytes());
        for &u in &geometry_data {
            buf.extend_from_slice(&u.to_le_bytes());
        }
        let shared_payload = Arc::new(buf);

        let slow_path_queues = self.slow_path_queues.clone();

        tokio::spawn(async move {
            println!("[Geometry Server] Listening on TCP {}", self.listener.local_addr().unwrap());
            loop {
                let (mut stream, _) = match self.listener.accept().await {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                
                let data = shared_payload.clone();
                let queues = slow_path_queues.clone();
                
                tokio::spawn(async move {
                    let mut len_buf = [0u8; 4];
                    if stream.read_exact(&mut len_buf).await.is_err() { return; }

                    // Legacy GEOM request from IDE
                    if &len_buf == b"GEOM" {
                        let _ = stream.write_all(&data).await;
                        return;
                    }

                    // BulkHandover request
                    let req_len = u32::from_le_bytes(len_buf) as usize;
                    if req_len > 10_000_000 { return; } // Safety limit 10MB
                    
                    let mut req_buf = vec![0u8; req_len];
                    if stream.read_exact(&mut req_buf).await.is_err() { return; }

                    if let Ok(req) = bincode::deserialize::<GeometryRequest>(&req_buf) {
                        match req {
                            GeometryRequest::BulkHandover(events) => {
                                for ev in events {
                                    queues.incoming_grow.push(ev);
                                }
                            }
                            GeometryRequest::BulkAck(acks) => {
                                for ack in acks {
                                    // Пушим в Lock-Free очередь Оркестратора, которую мы создали на прошлом шаге!
                                    queues.incoming_ack.push(ack); 
                                }
                            }
                            GeometryRequest::Prune(_) => {}
                        }

                        // Send Ack
                        let resp = GeometryResponse::Ok;
                        if let Ok(resp_encoded) = bincode::serialize(&resp) {
                            let resp_len = resp_encoded.len() as u32;
                            let _ = stream.write_all(&resp_len.to_le_bytes()).await;
                            let _ = stream.write_all(&resp_encoded).await;
                        }
                    }
                });
            }
        });
    }
}
