use crate::network::{SpikeEvent, SpikeBatchHeader};
use std::collections::HashMap;
use tokio::net::UdpSocket;
use std::net::SocketAddr;
use std::sync::Arc;

pub fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash_value: u32 = 0x811c9dc5;
    for &byte in data {
        hash_value ^= byte as u32;
        hash_value = hash_value.wrapping_mul(0x01000193);
    }
    hash_value
}

/// Defines a destination for a spike. 
/// A single local neuron might project to multiple Ghost Targets across different nodes.
#[derive(Clone, Debug)]
pub struct GhostTarget {
    pub node_id: u32,
    pub ghost_id: u32,
    pub tick_offset: u8,
}

/// The SpikeRouter maps a local dense neuron ID (extracted from the GPU) 
/// into one or more `SpikeEvent`s destined for remote nodes.
pub struct SpikeRouter {
    /// Maps generic Local Neuron Dense ID -> Array of Ghost Targets
    pub routing_table: HashMap<u32, Vec<GhostTarget>>,
    
    /// The accumulated batches per node, ready for flushing at the end of the Day Phase.
    pub outgoing_spikes: HashMap<u32, Vec<SpikeEvent>>,
}

impl SpikeRouter {
    pub fn new() -> Self {
        Self {
            routing_table: HashMap::new(),
            outgoing_spikes: HashMap::new(),
        }
    }

    /// Add a manual subscription mapping (useful for testing and slow-path geography setup).
    pub fn add_route(&mut self, local_id: u32, target: GhostTarget) {
        self.routing_table.entry(local_id).or_default().push(target);
    }

    /// Called natively per-tick by the Day Phase orchestration.
    pub fn route_spikes(&mut self, local_spikes: &[u32], current_tick_offset: u32) {
        for &nid in local_spikes {
            if let Some(targets) = self.routing_table.get(&nid) {
                // Fan-out: One neuron might send spikes to multiple remote locations
                for t in targets {
                    let total_offset = current_tick_offset as u32 + t.tick_offset as u32;
                    // We must clamp the offset mapping to u8 (assuming batch_size < 255)
                    let final_offset = std::cmp::min(total_offset, 255) as u8;

                    let event = SpikeEvent {
                        ghost_id: t.ghost_id,
                        tick_offset: final_offset as u32,
                    };

                    self.outgoing_spikes.entry(t.node_id).or_default().push(event);
                }
            }
        }
    }

    /// Fetches and clears the finalized outgoing buffers. 
    /// Intended for use primarily by the BspBarrier at the end of a Day Batch.
    pub fn flush_outgoing(&mut self) -> HashMap<u32, Vec<SpikeEvent>> {
        let current = std::mem::take(&mut self.outgoing_spikes);
        self.outgoing_spikes = HashMap::new(); // Ensure re-initialization
        current
    }
}

/// Статическая таблица маршрутизации (Загружается из конфига)
pub struct RoutingTable {
    // zone_hash -> IP:Port целевой ноды
    pub peers: HashMap<u32, SocketAddr>, 
}

pub struct InterNodeRouter {
    pub socket: Arc<UdpSocket>,
    pub routing_table: Arc<RoutingTable>,
}

impl InterNodeRouter {
    pub async fn new(bind_addr: &str, routing_table: RoutingTable) -> Self {
        let std_sock = std::net::UdpSocket::bind(bind_addr).expect("Fatal: Failed to bind InterNode UDP");
        // Максимальный буфер ОС для UDP (требует socket2 или nix crate, пропускаем для MVP)
        std_sock.set_nonblocking(true).expect("Failed to set non-blocking");
        let socket = UdpSocket::from_std(std_sock).expect("Failed to convert to tokio UdpSocket");

        Self {
            socket: Arc::new(socket),
            routing_table: Arc::new(routing_table),
        }
    }

    /// Вызывается в фоновом Tokio-потоке на каждый батч
    pub async fn flush_outgoing_batch(&self, target_zone_hash: u32, events: &[SpikeEvent]) {
        if events.is_empty() { return; }

        let Some(&addr) = self.routing_table.peers.get(&target_zone_hash) else {
            return; // Зона не найдена, drop packet (легально)
        };

        // Формируем строгий бинарный пакет (Zero-Cost сериализация)
        let header = SpikeBatchHeader {
            zone_hash: target_zone_hash,
            count: events.len() as u32,
        };

        // Трансмутация структур в байты. 
        // Безопасно, т.к. структуры #[repr(C, packed)]
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const _ as *const u8,
                std::mem::size_of::<SpikeBatchHeader>()
            )
        };
        let payload_bytes = unsafe {
            std::slice::from_raw_parts(
                events.as_ptr() as *const u8,
                events.len() * std::mem::size_of::<SpikeEvent>()
            )
        };

        // TODO: В идеале использовать send_vectored, но для простоты MVP копируем в один буфер
        let mut packet = Vec::with_capacity(header_bytes.len() + payload_bytes.len());
        packet.extend_from_slice(header_bytes);
        packet.extend_from_slice(payload_bytes);

        // Бросаем в сеть (не блокирует CPU)
        let _ = self.socket.send_to(&packet, addr).await;
    }

    pub fn spawn_receiver_loop(
        socket: Arc<UdpSocket>, 
        zone_ping_pongs: HashMap<u32, Arc<crate::network::bsp::PingPongSchedule>>
    ) {
        tokio::spawn(async move {
            let mut buf = [0u8; 65535]; // Max UDP packet
            
            loop {
                if let Ok((size, _)) = socket.recv_from(&mut buf).await {
                    if size < std::mem::size_of::<SpikeBatchHeader>() { continue; }
                    
                    let header_ptr = buf.as_ptr() as *const SpikeBatchHeader;
                    let header = unsafe { std::ptr::read_unaligned(header_ptr) };
                    
                    let expected_size = std::mem::size_of::<SpikeBatchHeader>() 
                        + (header.count as usize) * std::mem::size_of::<SpikeEvent>();
                        
                    if size < expected_size { continue; } // Corrupted packet
                    
                    let zone_hash = header.zone_hash; // Copy to avoid unaligned reference
                    
                    if let Some(ping_pong) = zone_ping_pongs.get(&zone_hash) {
                        let events_ptr = unsafe { header_ptr.add(1) as *const SpikeEvent };
                        let events = unsafe { std::slice::from_raw_parts(events_ptr, header.count as usize) };
                        
                        // Атомарно вкидываем спайки в спящий буфер VRAM
                        for event in events {
                            unsafe { ping_pong.ingest_spike(event) };
                        }
                    }
                }
            }
        });
    }
}
