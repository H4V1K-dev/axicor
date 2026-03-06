use crate::node::NodeRuntime;
use crate::network::replication::ShadowBufferManager;
use genesis_core::ipc::{ShardStateHeader, SNAP_MAGIC, ROUT_MAGIC, RouteUpdate};
 
impl NodeRuntime {
    /// Вызывается, если BspBarrier зафиксировал смерть соседа.
    /// Загружает веса из реплики и перехватывает управление зоной.
    pub async unsafe fn resurrect_shard(&self, dead_zone_hash: u32) {
        let manager = ShadowBufferManager::new(dead_zone_hash);
        let mmap = match manager.mmap_for_resurrection() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("🔴 [Recovery] Failed to mmap shadow buffer for 0x{:08X}: {}", dead_zone_hash, e);
                return;
            }
        };
 
        let header = &*(mmap.as_ptr() as *const ShardStateHeader);
        if header.magic != SNAP_MAGIC {
            eprintln!("🔴 [Recovery] Invalid SNAP_MAGIC in replica for 0x{:08X}", dead_zone_hash);
            return;
        }
 
        let padded_n = (header.payload_size / 256) as usize; // weights = padded_n * 128 * 2 bytes
        println!("🟢 [Recovery] Resurrection 0x{:08X} starting from tick {} (padded_n={})...", 
            dead_zone_hash, header.tick, padded_n);
 
        // [Architect] broadcast_route_update обязана быть вызвана СРАЗУ
        self.broadcast_route_update(dead_zone_hash).await;
    }
 
    pub async fn broadcast_route_update(&self, zone_hash: u32) {
        let update = RouteUpdate {
            magic: ROUT_MAGIC,
            zone_hash,
            new_ipv4: u32::from_be_bytes(self.local_ip.octets()),
            new_port: self.local_port,
            _padding1: 0,
            _padding2: [0; 2],
        };
        
        let bytes = bytemuck::bytes_of(&update);
        
        // Рассылаем всем пирам в RoutingTable через RCU
        let ptr = self.services.routing_table.get_map_ptr();
        if !ptr.is_null() {
            unsafe {
                let map = &*ptr;
                for addr in map.values() {
                    let _ = self.network.inter_node_router.socket.send_to(bytes, addr).await;
                }
            }
        }
    }
}
