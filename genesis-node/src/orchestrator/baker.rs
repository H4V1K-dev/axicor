use std::os::unix::net::UnixStream;
use std::io::{Read, Write};
use genesis_core::ipc::{BakeRequest, BAKE_MAGIC, BAKE_READY_MAGIC};

pub struct BakerClient {
    socket_path: String,
}

impl BakerClient {
    pub fn new(socket_path: String) -> Self {
        Self { socket_path }
    }

    /// Triggers the Baker Daemon to perform structural reorganization (Sprouting/Spatial logic).
    /// Contract: This is called AFTER the NightPhaseRunner has dumped weights to disk.
    /// The Baker Daemon will read the files, update them, and then this call returns BKOK.
    pub fn trigger_baker(&self, zone_hash: u32, tick: u32, prune_threshold: i16) {
        let mut stream = UnixStream::connect(&self.socket_path)
            .expect("CRITICAL: Baker Daemon socket not found. Is it running?");

        let req = BakeRequest {
            magic: BAKE_MAGIC,
            zone_hash,
            current_tick: tick,
            prune_threshold,
            _padding: 0,
        };

        // 1. Send the 16-byte trigger
        unsafe {
            let bytes = std::slice::from_raw_parts(
                &req as *const _ as *const u8, 
                std::mem::size_of::<BakeRequest>()
            );
            stream.write_all(bytes).unwrap();
        }

        // 2. Block the OS thread waiting forBKOK (This is Night Phase, blocking is intentional)
        let mut ack = [0u8; 4];
        stream.read_exact(&mut ack).expect("CRITICAL: Baker Daemon disconnected during baking");
        
        let ack_magic = u32::from_le_bytes(ack);
        assert_eq!(ack_magic, BAKE_READY_MAGIC, "CRITICAL: Invalid Baker ACK magic");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use std::thread;
    use tempfile::tempdir;

    #[test]
    fn test_baker_uds_trigger() {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("baker_test.sock");
        let socket_path_str = socket_path.to_str().unwrap().to_string();
        
        // Mock Baker Daemon
        let listener = UnixListener::bind(&socket_path).unwrap();
        let server_thread = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 16];
            stream.read_exact(&mut buf).unwrap();
            
            let req = unsafe { &*(buf.as_ptr() as *const BakeRequest) };
            assert_eq!(req.magic, BAKE_MAGIC);
            assert_eq!(req.zone_hash, 0x1234);
            
            // Send BKOK
            let ack = BAKE_READY_MAGIC.to_le_bytes();
            stream.write_all(&ack).unwrap();
        });

        let client = BakerClient::new(socket_path_str);
        client.trigger_baker(0x1234, 0, 10);
        
        server_thread.join().unwrap();
    }
}
