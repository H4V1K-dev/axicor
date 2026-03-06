use tokio::net::{TcpListener, TcpStream};
use tokio::io::AsyncWriteExt;
use std::path::PathBuf;
use genesis_core::ipc::{ShardStateHeader, SNAP_MAGIC};

pub struct ReplicationServer {
    listen_addr: String,
    replica_dir: PathBuf,
}

impl ReplicationServer {
    pub fn new(listen_addr: &str, replica_dir: &str) -> Self {
        Self {
            listen_addr: listen_addr.to_string(),
            replica_dir: PathBuf::from(replica_dir),
        }
    }

    pub async fn run(&self) -> std::io::Result<()> {
        let listener = TcpListener::bind(&self.listen_addr).await?;
        println!("[Replication] Listening on TCP {}", self.listen_addr);

        if !self.replica_dir.exists() {
            std::fs::create_dir_all(&self.replica_dir)?;
        }

        loop {
            let (socket, _) = listener.accept().await?;
            let replica_dir = self.replica_dir.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_replication_stream(socket, replica_dir).await {
                    eprintln!("[Replication] Error handling stream: {}", e);
                }
            });
        }
    }
}

async fn handle_replication_stream(mut socket: TcpStream, replica_dir: PathBuf) -> anyhow::Result<()> {
    // 1. Read ShardStateHeader
    let mut header_buf = [0u8; 32];
    tokio::io::AsyncReadExt::read_exact(&mut socket, &mut header_buf).await?;
    
    let header = unsafe { &*(header_buf.as_ptr() as *const ShardStateHeader) };
    if header.magic != SNAP_MAGIC {
        anyhow::bail!("Invalid snapshot magic: 0x{:08X}", header.magic);
    }

    let zone_hash = header.zone_hash;
    let file_path = replica_dir.join(format!("{}_weights.bin", zone_hash));
    
    // 2. Direct write to disk
    let mut file = tokio::fs::File::create(&file_path).await?;
    file.write_all(&header_buf).await?;

    // 3. Optimized copy from socket to file
    // On Linux, tokio::io::copy uses splice internally if possible, providing zero-copy-ish performance.
    tokio::io::copy(&mut socket, &mut file).await?;
    
    file.flush().await?;
    println!("[Replication] Saved replica for zone 0x{:08X} to {:?}", zone_hash, file_path);
    
    Ok(())
}

/// Helper to send a checkpoint using zero-copy sendfile.
pub async fn send_checkpoint_zerocopy(
    target_addr: &str,
    checkpoint_path: PathBuf,
) -> anyhow::Result<()> {
    use tokio::fs::File;
    
    let mut stream = TcpStream::connect(target_addr).await?;
    let mut file = File::open(&checkpoint_path).await?;
    
    // tokio::io::copy on Linux will use sendfile/splice for TcpStream <-> File
    tokio::io::copy(&mut file, &mut stream).await?;
    
    stream.flush().await?;
    Ok(())
}
