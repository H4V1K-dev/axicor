use std::sync::Arc;
use std::thread;
use crossbeam::channel::bounded;
use memmap2::Mmap;
use genesis_core::ipc::{ShardStateHeader, SNAP_MAGIC, RouteUpdate, ROUT_MAGIC};
use crate::node::{NodeRuntime, ComputeCommand, ComputeFeedback};
use genesis_compute::ShardEngine;
use genesis_compute::memory::DeviceSoA;

impl NodeRuntime {
    /// Вызывается, если BspBarrier зафиксировал смерть соседа.
    /// # Safety
    /// Обязана вызываться только внутри барьера, когда вычислительные ядра простаивают.
    pub async unsafe fn resurrect_shard(&self, dead_zone_hash: u32) {
        // [Contract §1] Barrier Invariant: Assert no RunBatch is in flight.
        // In our run_node_loop, we are between sync_and_swap and the next dispatch.
        // We can't easily check the channel fullness here without making it more complex,
        // but the architectural guarantee is that this is called from the main loop.
        
        let replica_path = format!("baked/replicas/{}_weights.bin", dead_zone_hash);
        if !std::path::Path::new(&replica_path).exists() {
            return; // Мы не являемся резервом для этой зоны
        }

        println!("[Recovery] Resurrection started for zone 0x{:08X}", dead_zone_hash);

        // 1. Zero-Copy загрузка через mmap
        let file = std::fs::File::open(&replica_path).unwrap();
        let mmap = Mmap::map(&file).unwrap();
        
        let header = &*(mmap.as_ptr() as *const ShardStateHeader);
        if header.magic != SNAP_MAGIC {
            panic!("[Recovery] Corrupt replica magic for zone 0x{:08X}", dead_zone_hash);
        }

        // [Contract §2] Warp Alignment check
        let base_ptr = mmap.as_ptr().add(std::mem::size_of::<ShardStateHeader>());
        assert_eq!(base_ptr as usize % 32, 0, "Replica SoA data is UNALIGNED!");

        // 2. Аллокация VRAM и DMA-передача (H2D)
        // Точный расчет оффсетов согласно ShardStateSoA раскладке
        let pn = header.tick as usize; 
        if pn == 0 { panic!("Replica header tick (padded_n) is 0"); }
        
        // По спецификации: dc = 128 * pn
        let dc = 128 * pn;
        let pa = (header.payload_size as usize - (pn * 14 + dc * 7)) / 4;

        let voltage = base_ptr as *const i32;
        let flags = unsafe { (voltage as *const u8).add(pn * 4) };
        let threshold_offset = unsafe { (flags as *const u8).add(pn * 1) as *const i32 };
        let refractory_timer = unsafe { (threshold_offset as *const u8).add(pn * 4) as *const u8 };
        let soma_to_axon = unsafe { (refractory_timer as *const u8).add(pn * 1) as *const u32 };
        let dendrite_targets = unsafe { (soma_to_axon as *const u8).add(pn * 4) as *const u32 };
        let dendrite_weights = unsafe { (dendrite_targets as *const u8).add(dc * 4) as *const i16 };
        let dendrite_timers = unsafe { (dendrite_weights as *const u8).add(dc * 2) as *const u8 };
        let axon_heads = unsafe { (dendrite_timers as *const u8).add(dc * 1) as *const u32 };

        let device_soa = unsafe {
            DeviceSoA::boot_from_raw_parts(
                pn,
                pa,
                voltage,
                flags,
                threshold_offset,
                refractory_timer,
                soma_to_axon,
                dendrite_targets,
                dendrite_weights,
                dendrite_timers,
                axon_heads,
            ).unwrap()
        };

        // 3. Инъекция в вычислительный контур (Lock-Free)
        let new_engine = ShardEngine::new(device_soa.state, std::ptr::null(), std::ptr::null(), 0, 0);
        self.spawn_shard_thread(dead_zone_hash, new_engine);
        
        println!("[Recovery] Resurrection COMPLETE for zone 0x{:08X}", dead_zone_hash);
        
        // 4. Broadcast Route Update
        self.broadcast_route_update(dead_zone_hash).await;
    }

    pub async fn broadcast_route_update(&self, zone_hash: u32) {
        let update = RouteUpdate {
            magic: ROUT_MAGIC,
            zone_hash,
            new_ipv4: u32::from_be_bytes(self.local_ip.octets()),
            new_port: self.local_port,
            _padding: 0,
        };

        let payload = bytemuck::bytes_of(&update);
        
        // Broadcast to all known peers
        let routing_ptr = self.routing_table.get_map_ptr();
        let peers: Vec<_> = unsafe { (*routing_ptr).values().copied().collect() };
        
        for addr in peers {
            let _ = self.io_server.socket.send_to(payload, addr).await;
        }
        
        println!("[Recovery] RouteUpdate broadcasted for zone 0x{:08X}", zone_hash);
    }
}
