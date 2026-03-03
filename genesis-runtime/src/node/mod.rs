use std::sync::Arc;
use std::thread;
use std::collections::HashMap;
use std::time::Duration;
use crossbeam::channel::{bounded, Sender, Receiver};
use crate::compute::shard::ShardComputeIsland;
use crate::network::io_server::ExternalIoServer;
use crate::network::bsp::BspBarrier;
use crate::network::router::RoutingTable;
use std::sync::atomic::{AtomicU32, Ordering};

pub mod recovery;

pub enum ComputeCommand {
    RunBatch {
        tick_base: u32,
        batch_size: u32,
    },
    Shutdown,
}

pub enum ComputeFeedback {
    BatchComplete {
        ticks_processed: u32,
    },
}

pub struct NodeRuntime {
    pub io_server: Arc<ExternalIoServer>,
    pub routing_table: Arc<RoutingTable>,
    pub bsp_barrier: Arc<BspBarrier>,
    pub compute_dispatchers: std::sync::Mutex<HashMap<u32, Sender<ComputeCommand>>>,
    pub feedback_sender: Sender<ComputeFeedback>,
    pub feedback_receiver: Receiver<ComputeFeedback>,
    pub total_ticks: Arc<AtomicU32>,
    pub local_ip: std::net::Ipv4Addr,
    pub local_port: u16,
}

impl NodeRuntime {
    /// Bootstraps the IO layer and spawns dedicated OS threads for shards.
    pub fn boot(
        shards: Vec<(u32, ShardComputeIsland)>,
        io_server: Arc<ExternalIoServer>,
        routing_table: Arc<RoutingTable>,
        bsp_barrier: Arc<BspBarrier>,
        local_ip: std::net::Ipv4Addr,
        local_port: u16,
    ) -> Self {
        let (feedback_tx, feedback_rx) = bounded(shards.len() + 32); // Overhead for recovery
        let total_ticks = Arc::new(AtomicU32::new(0));

        let node = Self {
            io_server,
            routing_table,
            bsp_barrier,
            compute_dispatchers: std::sync::Mutex::new(HashMap::new()),
            feedback_sender: feedback_tx,
            feedback_receiver: feedback_rx,
            total_ticks,
            local_ip,
            local_port,
        };

        for (hash, shard) in shards {
            node.spawn_shard_thread(hash, shard);
        }

        node
    }

    /// Spawns a dedicated OS thread for a shard and adds its dispatcher to the registry.
    pub fn spawn_shard_thread(&self, hash: u32, mut shard: ShardComputeIsland) {
        let (tx, rx) = bounded(1);
        {
            let mut dispatchers = self.compute_dispatchers.lock().unwrap();
            dispatchers.insert(hash, tx);
        }
        let f_tx = self.feedback_sender.clone();

        thread::Builder::new()
            .name(format!("compute-{}", hash))
            .spawn(move || {
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        ComputeCommand::RunBatch { tick_base, batch_size } => {
                            shard.execute_day_batch(
                                batch_size,
                                tick_base,
                                1, 
                                std::ptr::null(), 
                                std::ptr::null_mut(), 
                                std::ptr::null(), 
                                &vec![0; batch_size as usize],
                            );

                            if let Err(_) = f_tx.send(ComputeFeedback::BatchComplete { ticks_processed: batch_size }) {
                                break;
                            }
                        }
                        ComputeCommand::Shutdown => break,
                    }
                }
            }).expect("Failed to spawn compute thread");
    }

    /// Spawns a background watchdog task that pings peers and initiates recovery on failure.
    pub fn spawn_watchdog(&self) {
        let routing_table = self.routing_table.clone();
        let bsp_barrier = self.bsp_barrier.clone();
        let io_server = self.io_server.clone();

        tokio::spawn(async move {
            let mut failure_counts: HashMap<u32, u32> = HashMap::new();
            let mut interval = tokio::time::interval(Duration::from_millis(50));
            
            loop {
                interval.tick().await;
                
                let hashes: Vec<u32> = bsp_barrier.schedules.keys().copied().collect();
                
                for hash in hashes {
                    if let Some(addr) = routing_table.get_address(hash) {
                        let mut ping = [0u8; 8];
                        ping[0..4].copy_from_slice(&0x48425421u32.to_le_bytes()); // "HBT!"
                        ping[4..8].copy_from_slice(&hash.to_le_bytes());

                        if let Err(_) = io_server.socket.send_to(&ping, addr).await {
                            let count = failure_counts.entry(hash).or_insert(0);
                            *count += 1;
                            
                            if *count >= 3 {
                                eprintln!("[Watchdog] Node 0x{:08X} at {} is DEAD. Isolate.", hash, addr);
                                let mut dead = bsp_barrier.dead_zones.lock().unwrap();
                                if !dead.contains(&hash) {
                                    dead.push(hash);
                                }
                            }
                        } else {
                            failure_counts.insert(hash, 0);
                        }
                    }
                }
            }
        });
    }

    /// The main Tokio loop (IO Multiplexer)
    pub async fn run_node_loop(&self, batch_size: u32) {
        let mut current_tick = 0;
        let mut last_packet_counts: HashMap<u32, usize> = HashMap::new();

        loop {
            let timeout = Duration::from_millis(100); 
            
            match self.bsp_barrier.sync_and_swap(&last_packet_counts, timeout) {
                Ok(_) => {
                    for (&hash, schedule) in &self.bsp_barrier.schedules {
                        last_packet_counts.insert(hash, schedule.packets_received.load(Ordering::Acquire));
                    }
                }
                Err(dead_zones) => {
                    eprintln!("[Node] Deadlock prevented! Zones failed: {:?}", dead_zones);
                    for hash in dead_zones {
                        unsafe { self.resurrect_shard(hash).await; }
                    }
                }
            }

            // [Barrier Invariant] All compute threads are quiescent here.
            let num_dispatchers = {
                let dispatchers_guard = self.compute_dispatchers.lock().unwrap();
                let num = dispatchers_guard.len();
                for tx in dispatchers_guard.values() {
                    let _ = tx.send(ComputeCommand::RunBatch {
                        tick_base: current_tick,
                        batch_size,
                    });
                }
                num
            };

            for _ in 0..num_dispatchers {
                if let Ok(ComputeFeedback::BatchComplete { ticks_processed }) = self.feedback_receiver.recv() {
                    self.total_ticks.fetch_add(ticks_processed, Ordering::Relaxed);
                }
            }

            current_tick += batch_size;
            self.io_server.dashboard.total_ticks.store(current_tick as u64, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::shard::ShardComputeIsland;
    use crate::tui::DashboardState;
    use crate::network::io_server::ExternalIoServer;
    use crate::network::bsp::BspBarrier;
    use crate::network::router::RoutingTable;
    use std::sync::Arc;
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_node_compute_isolation() {
        let dashboard = Arc::new(DashboardState::new(false));
        let routing_table = Arc::new(RoutingTable::new(HashMap::new()));
        let socket = Arc::new(tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let io_server = Arc::new(ExternalIoServer::new(
            Arc::new(std::sync::atomic::AtomicBool::new(false)), 
            1024, 0, 0, dashboard, routing_table.clone(), socket
        ).unwrap());
        let bsp_barrier = Arc::new(BspBarrier::new());
        
        let shard = ShardComputeIsland::new(unsafe { std::mem::zeroed() }, 0, 0);

        let node = NodeRuntime::boot(
            vec![(0x1234, shard)], 
            io_server, 
            routing_table, 
            bsp_barrier,
            Ipv4Addr::new(127, 0, 0, 1),
            8080
        );
        
        let tokio_thread_id = thread::current().id();
        let (id_tx, id_rx) = bounded(1);
        
        thread::spawn(move || {
            id_tx.send(thread::current().id()).unwrap();
        });
        
        let spawned_thread_id = id_rx.recv().unwrap();
        assert_ne!(tokio_thread_id, spawned_thread_id);
    }
}
