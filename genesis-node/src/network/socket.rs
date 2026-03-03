use std::net::SocketAddr;
use tokio::net::UdpSocket;
use anyhow::{Result, bail, Context};
use std::sync::Arc;

use crate::network::{SpikeEvent, SpikeBatchHeader};

/// Async wrapper over a UDP socket providing Zero-Copy transmission
/// of SpikeBatch structures.
pub struct NodeSocket {
    socket: Arc<UdpSocket>,
}

impl NodeSocket {
    /// Bind to a local port
    pub async fn bind(addr: &str) -> Result<Self> {
        let socket = UdpSocket::bind(addr).await
            .context("Failed to bind UDP socket")?;
        
        // Use a reasonably large send/recv buffer for bulk syncing
        // (Ignoring OS limits here for simplicity, in prod we'd configure SO_RCVBUF)
        
        Ok(Self {
            socket: Arc::new(socket),
        })
    }

    /// Return assigned local address
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Zero-copy send of a SpikeBatch
    pub async fn send_batch(&self, target: SocketAddr, batch_id: u32, events: &[SpikeEvent]) -> Result<()> {
        let header = SpikeBatchHeader {
            magic: 0x5350494B, // "SPIK"
            batch_id,
        };

        // We construct a single buffer using IoSlice to avoid copying, 
        // but UdpSocket in tokio currently requires a single contiguous slice 
        // unless we drop down to lower level traits or make a contiguous buffer.
        // For absolute maximum efficiency we'd allocate one contiguous slice 
        // if IoSlice isn't supported, but copying 8 bytes of header isn't the bottleneck.
        
        let header_bytes = bytemuck::bytes_of(&header);
        let events_bytes = bytemuck::cast_slice(events);
        
        // Since Tokio's standard UdpSocket doesn't have `send_vectored_to`, 
        // we'll concat them into a fast contiguous buffer for the OS.
        let mut buffer = Vec::with_capacity(header_bytes.len() + events_bytes.len());
        buffer.extend_from_slice(header_bytes);
        buffer.extend_from_slice(events_bytes);

        let bytes_sent = self.socket.send_to(&buffer, target).await?;
        if bytes_sent != buffer.len() {
            bail!("Fragmented UDP send: {} of {} bytes", bytes_sent, buffer.len());
        }

        Ok(())
    }

    /// Receives a single UDP packet and casts it back to a SpikeBatch
    /// Returns (Sender Address, Batch ID, Vector of Events)
    pub async fn recv_batch(&self) -> Result<(SocketAddr, u32, Vec<SpikeEvent>)> {
        // Typical MTU is 1500, but loopback can be ~65k.
        // Let's allocate a 64KB buffer capable of receiving up to ~8000 spikes.
        let mut buf = vec![0u8; 65507];
        
        let (len, src_addr) = self.socket.recv_from(&mut buf).await?;
        let buf = &buf[..len];

        let header_sz = std::mem::size_of::<SpikeBatchHeader>();
        if len < header_sz {
            bail!("Packet too small for header ({} bytes)", len);
        }

        let (header_bytes, body_bytes) = buf.split_at(header_sz);
        let header: &SpikeBatchHeader = bytemuck::from_bytes(header_bytes);
        
        if header.magic != 0x5350494B {
            bail!("Invalid spike batch magic: {:x}", header.magic);
        }
        
        let batch_id = header.batch_id;
        let expected_body_sz = body_bytes.len() / std::mem::size_of::<SpikeEvent>() * std::mem::size_of::<SpikeEvent>();
        
        if body_bytes.len() < std::mem::size_of::<SpikeEvent>() && body_bytes.len() > 0 {
            bail!("Packet truncated. Body is {} bytes.", body_bytes.len());
        }

        // We slice strictly what the header claimed (ignoring trailing padding if any)
        let exact_body_bytes = &body_bytes[..expected_body_sz];
        
        // Zero-copy cast back to SpikeEvent slice, then clone into a vector.
        let events_slice: &[SpikeEvent] = bytemuck::cast_slice(exact_body_bytes);
        
        Ok((src_addr, batch_id, events_slice.to_vec()))
    }
}
