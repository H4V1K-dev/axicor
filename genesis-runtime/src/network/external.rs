// genesis-runtime/src/network/external.rs
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::net::UdpSocket;
use std::ptr;

/// Заголовок UDP-пакета (§08_io_matrix.md)
#[repr(C, packed)]
pub struct ExternalIoHeader {
    pub zone_hash: u32,
    pub matrix_hash: u32,
    pub payload_size: u32,
    pub global_reward: i16,
    pub _padding: u16,
}

const IO_HEADER_SIZE: usize = std::mem::size_of::<ExternalIoHeader>();

pub struct ExternalIoServer {
    socket: Arc<UdpSocket>,
    // Сырой указатель на Pinned RAM (cudaHostAlloc), переданный из ZoneMemory
    pinned_input_ptr: *mut u32, 
    max_payload_bytes: usize,
    
    // Атомик для сигнализации оркестратору, что пришел новый кадр
    pub new_frame_ready: Arc<AtomicUsize>, 
    
    // R-STDP Dopamine Modulator (Global Reward Broadcast)
    pub global_dopamine: Arc<std::sync::atomic::AtomicI32>, 

    pub dashboard: Arc<crate::tui::DashboardState>,
    
    // Mapping: matrix_hash -> offset_in_pinned_words
    pub matrix_offsets: std::collections::HashMap<u32, u32>,
}

// Легализуем передачу сырого указателя между потоками (мы гарантируем безопасность логикой)
unsafe impl Send for ExternalIoServer {}
unsafe impl Sync for ExternalIoServer {}

impl ExternalIoServer {
    pub async fn new(bind_addr: &str, pinned_input_ptr: *mut u32, max_payload_bytes: usize, dashboard: Arc<crate::tui::DashboardState>) -> Self {
        let socket = UdpSocket::bind(bind_addr).await.expect("Fatal: Failed to bind UDP I/O socket");
        
        Self {
            socket: Arc::new(socket),
            pinned_input_ptr,
            max_payload_bytes,
            new_frame_ready: Arc::new(AtomicUsize::new(0)),
            global_dopamine: Arc::new(std::sync::atomic::AtomicI32::new(0)),
            dashboard,
            matrix_offsets: std::collections::HashMap::new(),
        }
    }

    /// Запуск бесконечного цикла прослушивания
    pub async fn run_rx_loop(&self) {
        println!("🚀 [ExternalIO] UDP Receiver Loop Started on {}", self.socket.local_addr().unwrap());
        let mut buf = [0u8; 65536]; // Хард-лимит UDP пакета. Обошлись без аллокаций в куче.

        loop {
            match self.socket.recv_from(&mut buf).await {
                Ok((len, _addr)) => {
                    if len < IO_HEADER_SIZE {
                        continue; // Мусорный пакет
                    }

                    unsafe {
                        let header = &*(buf.as_ptr() as *const ExternalIoHeader);
                        
                        // Защита от переполнения Pinned Memory (Security & Stability Constraint)
                        let payload_bytes = header.payload_size as usize;
                        if payload_bytes == 0 || payload_bytes > self.max_payload_bytes || (IO_HEADER_SIZE + payload_bytes) > len {
                            // Drop Oversized packet. Метрики добавим позже.
                            continue; 
                        }

                        // Update global dopamine reward for R-STDP
                        self.global_dopamine.store(header.global_reward as i32, Ordering::Relaxed);
                        if header.global_reward != 0 {
                            let reward = header.global_reward;
                            println!("💉 [Dopamine] Reward Received: {}", reward);
                        }

                        // Find offset for this matrix
                        let matrix_hash_val = header.matrix_hash;
                        let offset = self.matrix_offsets.get(&matrix_hash_val).copied().unwrap_or(0);
                        
                        // ZERO-COPY PATH: Прямое копирование из стека в DMA-память хоста
                        let payload_src = buf.as_ptr().add(IO_HEADER_SIZE);
                        let dest_ptr = (self.pinned_input_ptr as *mut u8).add(offset as usize * 4);
                        
                        ptr::copy_nonoverlapping(
                            payload_src, 
                            dest_ptr, 
                            payload_bytes
                        );

                        // Сигнализируем CPU-Оркестратору, что Pinned RAM обновлена
                        self.new_frame_ready.store(1, Ordering::Release);
                        
                        // Debug log for tracking input
                        self.dashboard.udp_in_packets.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_) => {
                    // Игнорируем ошибки сети. Мозг не должен падать, если оторвали камеру.
                }
            }
        }
    }

    /// Отправка Output_History (Вызывается оркестратором после RecordReadout)
    pub async fn send_output_batch(&self, target_addr: &str, zone_hash: u32, matrix_hash: u32, pinned_output_addr: usize, output_bytes: usize) {
        if output_bytes + IO_HEADER_SIZE > 65535 {
            panic!("Output batch exceeds UDP MTU. Needs fragmentation.");
        }

        let mut out_buf = vec![0u8; IO_HEADER_SIZE + output_bytes]; // Fast enough for outbound
        
        unsafe {
            let header = out_buf.as_mut_ptr() as *mut ExternalIoHeader;
            (*header).zone_hash = zone_hash;
            (*header).matrix_hash = matrix_hash;
            (*header).payload_size = output_bytes as u32;

            ptr::copy_nonoverlapping(
                pinned_output_addr as *const u8,
                out_buf.as_mut_ptr().add(IO_HEADER_SIZE),
                output_bytes
            );
        }


        let _ = self.socket.send_to(&out_buf, target_addr).await;
        
        eprintln!("[ExternalIO] Sent output batch to {}, {} bytes", target_addr, output_bytes);
        self.dashboard.udp_out_packets.fetch_add(1, Ordering::Relaxed);
    }
}
