use crate::node::NodeRuntime;
use genesis_core::ipc::{RouteUpdate, ROUT_MAGIC};

impl NodeRuntime {
    /// Broadcasts the new route of a resurrected shard to all known peers.
    /// Peers will execute an atomic RCU swap of their Egress tables upon receipt.
    pub async fn broadcast_route_update(&self, zone_hash: u32) {
        let new_ipv4 = u32::from_be_bytes(self.local_ip.octets());
        let new_port = self.local_port;

        let update = RouteUpdate {
            magic: ROUT_MAGIC,
            zone_hash,
            new_ipv4,
            new_port,
            mtu: genesis_core::constants::MAX_UDP_PAYLOAD as u16, // [DOD FIX]
            cluster_secret: self.cluster_secret, // [DOD FIX]
        };
        let packet = bytemuck::bytes_of(&update);

        let table = unsafe { &*self.services.routing_table.get_map_ptr() };
        for (&peer_hash, &(addr, _mtu)) in table.iter() {
            if peer_hash != zone_hash {
                let _ = self.network.inter_node_router.socket.send_to(packet, addr).await;
            }
        }
        println!("📡 [Recovery] Broadcasted RouteUpdate for 0x{:08X} to {} peers", zone_hash, table.len());
    }

    /// Вызывается, если BspBarrier зафиксировал смерть соседа.
    pub async unsafe fn resurrect_shard(&self, dead_zone_hash: u32) {
        println!("⚕️ [Recovery] Initiating The Great Resurrection for 0x{:08X}", dead_zone_hash);
        
        // 1. Broadcast new route to all peers via RCU
        self.broadcast_route_update(dead_zone_hash).await;

        // 2. Command the local shard thread to enter Warmup (Voltage Stabilization)
        if let Some(tx) = self.compute_dispatchers.get(&dead_zone_hash) {
            let _ = tx.send(crate::node::ComputeCommand::Resurrect);
        }
    }
}
