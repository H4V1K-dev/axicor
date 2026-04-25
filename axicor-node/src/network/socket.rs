use anyhow::{bail, Context, Result};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::UdpSocket;

use crate::network::{SpikeBatchHeader, SpikeEvent};

/// Async wrapper over a UDP socket providing Zero-Copy transmission
/// of SpikeBatch structures.
pub struct NodeSocket {
    socket: Arc<UdpSocket>,
}

impl NodeSocket {
    /// Bind to a local port
    pub async fn bind(addr: &str) -> Result<Self> {
        let socket = UdpSocket::bind(addr)
            .await
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
        tx_buffer: &mut Vec<u8>, // [DOD] Reusable buffer
    ) -> Result<()> {
        let header = SpikeBatchHeader {
            magic: 0x5350494B, // "SPIK"
            batch_id,
        };

        let header_bytes = bytemuck::bytes_of(&header);
        let events_bytes = bytemuck::cast_slice(events);

        tx_buffer.clear();
        tx_buffer.extend_from_slice(header_bytes);
        tx_buffer.extend_from_slice(events_bytes);

        let bytes_sent = self.socket.send_to(tx_buffer, target).await?;
        if bytes_sent != tx_buffer.len() {
            bail!(
                "Fragmented UDP send: {} of {} bytes",
                bytes_sent,
                tx_buffer.len()
            );
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
        let mut hdr_buf = [0u8; 8];
        hdr_buf.copy_from_slice(header_bytes);
        let header: &SpikeBatchHeader = bytemuck::from_bytes(&hdr_buf);

        if header.magic != 0x5350494B {
            bail!("Invalid spike batch magic: {:x}", header.magic);
        }

        let batch_id = header.batch_id;
        let expected_body_sz = body_bytes.len() / std::mem::size_of::<SpikeEvent>()
            * std::mem::size_of::<SpikeEvent>();

        if body_bytes.len() < std::mem::size_of::<SpikeEvent>() && !body_bytes.is_empty() {
            bail!("Packet truncated. Body is {} bytes.", body_bytes.len());
        }

        // We slice strictly what the header claimed (ignoring trailing padding if any)
        let exact_body_bytes = &body_bytes[..expected_body_sz];

        // Safe parse: network buffer may be unaligned
        let events_slice: Vec<SpikeEvent> = exact_body_bytes
            .chunks_exact(8)
            .map(|c| SpikeEvent {
                ghost_axon_id: u32::from_le_bytes(c[0..4].try_into().unwrap()),
                tick_offset: u32::from_le_bytes(c[4..8].try_into().unwrap()),
            })
            .collect();

        Ok((src_addr, batch_id, events_slice))
    }
}
