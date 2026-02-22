use axum::{
    extract::{ws::{Message, WebSocket, WebSocketUpgrade}, State},
    response::IntoResponse,
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::sync::broadcast;

/// Sent from the Orchestrator to the TelemetryServer at the end of each BSP batch.
#[derive(Clone, Debug)]
pub struct TelemetryPayload {
    pub tick: u64,
    pub active_spikes: Vec<u32>,
}

/// The Header structure sent exactly as-is over the WebSocket before the raw spike payload.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TelemetryFrameHeader {
    pub magic: u32,
    pub tick: u64,
    pub spikes_count: u32,
}

impl TelemetryFrameHeader {
    const MAGIC: u32 = u32::from_le_bytes(*b"GNSS"); // Genesis

    pub fn new(tick: u64, spikes_count: u32) -> Self {
        Self {
            magic: Self::MAGIC,
            tick,
            spikes_count,
        }
    }
    
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

pub struct TelemetryServer {
    tx: broadcast::Sender<TelemetryPayload>,
}

impl TelemetryServer {
    /// Starts the telemetry server on a given port. 
    /// Returns the Sender channel that the DayPhase orchestrator will push payloads into.
    pub async fn start(port: u16) -> broadcast::Sender<TelemetryPayload> {
        let (tx, _rx) = broadcast::channel(16); // Buffer size 16 frames
        let app_state = tx.clone();

        let app = Router::new()
            .route("/ws", get(ws_handler))
            .with_state(app_state);

        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        println!("🚀 Telemetry Server (Zero-Copy Binary) listening on ws://{}/ws", addr);

        tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
            axum::serve(listener, app)
                .await
                .unwrap();
        });

        tx
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(tx): State<broadcast::Sender<TelemetryPayload>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, tx))
}

async fn handle_socket(socket: WebSocket, tx: broadcast::Sender<TelemetryPayload>) {
    let (mut sender, mut _receiver) = socket.split();
    let mut rx = tx.subscribe();

    println!("👁️ Telemetry Client connected");

    tokio::spawn(async move {
        while let Ok(payload) = rx.recv().await {
            let header = TelemetryFrameHeader::new(payload.tick, payload.active_spikes.len() as u32);
            
            // Build absolute Zero-Copy styled flatbuffer.
            // Bevy / Unity / Blender clients reading this can just cast it instantly.
            let mut binary_frame = Vec::with_capacity(std::mem::size_of::<TelemetryFrameHeader>() + payload.active_spikes.len() * 4);
            
            // 1. Write Header
            binary_frame.extend_from_slice(header.as_bytes());
            
            // 2. Write Spikes payload (u32 array converted to bytes)
            let spikes_bytes = unsafe {
                std::slice::from_raw_parts(
                    payload.active_spikes.as_ptr() as *const u8,
                    payload.active_spikes.len() * 4,
                )
            };
            binary_frame.extend_from_slice(spikes_bytes);

            // Send binary packet
            if sender.send(Message::Binary(binary_frame)).await.is_err() {
                println!("Telemetry Client disconnected");
                break; // Client disconnected
            }
        }
    });
}
