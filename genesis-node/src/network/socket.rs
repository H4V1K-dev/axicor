use std::net::SocketAddr;
use tokio::net::UdpSocket;
use anyhow::{Result, bail, Context};
use std::sync::Arc;

use crate::network::{SpikeEvent, SpikeBatchHeaderV3};

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
    pub async fn send_batch(
        &self, 
        target: SocketAddr, 
        batch_id: u32, 
        events: &[SpikeEvent],
        tx_buffer: &mut Vec<u8> // [DOD] Переиспользуемый буфер
    ) -> Result<()> {
        let header = SpikeBatchHeaderV3 {
            src_zone_hash: 0,
            dst_zone_hash: 0,
            epoch: batch_id,
            is_last: 0,
            tick: 0,
            _padding: 0,
        };

        let header_bytes = bytemuck::bytes_of(&header);
        let events_bytes = bytemuck::cast_slice(events);
        
        tx_buffer.clear();
        tx_buffer.extend_from_slice(header_bytes);
        tx_buffer.extend_from_slice(events_bytes);

        let bytes_sent = self.socket.send_to(tx_buffer, target).await?;
        if bytes_sent != tx_buffer.len() {
            bail!("Fragmented UDP send: {} of {} bytes", bytes_sent, tx_buffer.len());
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

        let header_sz = std::mem::size_of::<SpikeBatchHeaderV3>();
        if len < header_sz {
            bail!("Packet too small for header ({} bytes)", len);
        }

        let (header_bytes, body_bytes) = buf.split_at(header_sz);
        let header: &SpikeBatchHeaderV3 = bytemuck::from_bytes(header_bytes);
        
        // magic removed in V3, using tick/epoch for validation if needed,
        // but for legacy receiver we just check size.
        
        let batch_id = header.epoch; // Use epoch for batch_id in V3
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
