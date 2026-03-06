use tokio::io::{AsyncWriteExt, AsyncReadExt, AsyncSeekExt};
use std::path::{Path, PathBuf};
use genesis_core::ipc::{ShardStateHeader, SNAP_MAGIC};
use std::fs::OpenOptions;
use tokio::net::{TcpListener, TcpStream};
use bytemuck;

pub struct ShadowBufferManager {
    pub zone_hash: u32,
    pub shm_path: PathBuf,
}

impl ShadowBufferManager {
    pub fn new(zone_hash: u32) -> Self {
        Self {
            zone_hash,
            shm_path: PathBuf::from(format!("/dev/shm/0x{:08X}.shadow", zone_hash)),
        }
    }

    /// Atomic write of weights to /dev/shm
    pub async fn atomic_save(&self, data: &[u8]) -> std::io::Result<()> {
        let tmp_path = self.shm_path.with_extension("tmp");
        tokio::fs::write(&tmp_path, data).await?;
        tokio::fs::rename(&tmp_path, &self.shm_path).await?;
        Ok(())
    }

    /// Mmap the shadow buffer for resurrection (Read-only for loader)
    pub fn mmap_for_resurrection(&self) -> std::io::Result<memmap2::Mmap> {
        let file = OpenOptions::new().read(true).open(&self.shm_path)?;
        unsafe { memmap2::Mmap::map(&file) }
    }
}

pub struct ReplicationServer {
    listen_addr: String,
    // replica_dir: PathBuf, // Removed
}

impl ReplicationServer {
    pub fn new(listen_addr: &str) -> Self { // replica_dir parameter removed
        Self {
            listen_addr: listen_addr.to_string(),
            // replica_dir: PathBuf::from(replica_dir), // Removed
        }
    }

    pub async fn run(&self) -> std::io::Result<()> {
        let listener = TcpListener::bind(&self.listen_addr).await?;
        println!("[Replication] Listening on TCP {}", self.listen_addr);

        // if !self.replica_dir.exists() { // Removed
        //     std::fs::create_dir_all(&self.replica_dir)?; // Removed
        // }

        loop {
            let (socket, _) = listener.accept().await?;
            // let replica_dir = self.replica_dir.clone(); // Removed
            tokio::spawn(async move {
                if let Err(e) = handle_replication_stream(socket).await { // replica_dir argument removed
                    eprintln!("[Replication] Error handling stream: {}", e);
                }
            });
        }
    }
}

async fn handle_replication_stream(mut socket: TcpStream) -> anyhow::Result<()> { // replica_dir parameter removed, socket type specified
    // 1. Read ShardStateHeader
    let mut header_buf = [0u8; 32];
    socket.read_exact(&mut header_buf).await?; // Changed from tokio::io::AsyncReadExt::read_exact

    let header = unsafe { &*(header_buf.as_ptr() as *const ShardStateHeader) };
    if header.magic != SNAP_MAGIC {
        anyhow::bail!("Invalid snapshot magic: 0x{:08X}", header.magic);
    }

    let manager = ShadowBufferManager::new(header.zone_hash); // Use ShadowBufferManager
    let tmp_path = manager.shm_path.with_extension("tmp"); // Use temporary path for atomic write

    // 2. Direct write to /dev/shm
    let mut file = tokio::fs::File::create(&tmp_path).await?; // Create temporary file
    file.write_all(&header_buf).await?;

    // 3. Optimized copy from socket to file
    // On Linux, tokio::io::copy uses splice internally if possible, providing zero-copy-ish performance.
    tokio::io::copy(&mut socket, &mut file).await?;

    file.flush().await?;

    // 4. Atomic swap
    tokio::fs::rename(&tmp_path, &manager.shm_path).await?; // Atomic rename

    // println!("[Replication] Saved replica for zone 0x{:08X} to {:?}", zone_hash, file_path); // Removed
    // println!("[Replication] Saved shadow replica for zone 0x{:08X}", header.zone_hash); // Added for context, but commented out in instruction
    Ok(())
}

/// Helper to send a checkpoint using zero-copy mechanism from a local SHM file.
pub async fn send_replica_from_shm(
    target_addr: &str,
    header: ShardStateHeader,
    shm_path: &Path,
    offset: u64,
    size: u64,
) -> anyhow::Result<()> {
    let mut stream = tokio::net::TcpStream::connect(target_addr).await?;
    
    // 1. Send header
    stream.write_all(bytemuck::bytes_of(&header)).await?;
    
    // 2. Open SHM file and seek to weights
    let mut file = tokio::fs::File::open(shm_path).await?;
    file.seek(std::io::SeekFrom::Start(offset)).await?;
    
    // 3. Send weights slice using zero-copy-ish tokio::io::copy
    let mut reader = file.take(size);
    tokio::io::copy(&mut reader, &mut stream).await?;
    
    stream.flush().await?;
    Ok(())
}
