use bevy::{
    prelude::*,
    tasks::IoTaskPool,
};
use crossbeam_channel::{bounded, Receiver};
use futures_util::StreamExt;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use std::convert::TryInto;

const WS_URL: &str = "ws://127.0.0.1:8002";

/// ECS Event, который стреляет каждый раз, когда приходит батч спайков из Runtime.
#[derive(Event, Clone)]
pub struct SpikeFrame {
    pub tick: u64,
    pub spike_ids: Vec<u32>,
}

/// Lock-Free ресивер в качестве ресурса ECS. 
/// Позволяет читать данные из асинхронного таска без блокировок.
#[derive(Resource)]
pub struct TelemetryBridge {
    pub rx: Receiver<SpikeFrame>,
}

pub struct TelemetryPlugin;

impl Plugin for TelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<SpikeFrame>()
           .add_systems(Startup, spawn_telemetry_client)
           .add_systems(Update, poll_telemetry_channel);
    }
}

fn spawn_telemetry_client(mut commands: Commands) {
    // Используем ограниченный канал. Если IDE тормозит и не успевает рисовать кадры, 
    // старые спайки лучше дропать, чем забивать оперативу до OOM.
    let (tx, rx) = bounded::<SpikeFrame>(60); 

    commands.insert_resource(TelemetryBridge { rx });

    let thread_pool = IoTaskPool::get();
    
    // Спавним асинхронную задачу вне Main Thread
    thread_pool.spawn(async move {
        // Важно: Bevy IoTaskPool не имеет tokio runtime,
        // поэтому приходится создавать локальный для WebSocket I/O
        use tokio::runtime::Builder;
        
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");
        
        rt.block_on(async {
            info!("Connecting to Genesis Telemetry at {}...", WS_URL);
            
            let ws_stream = match connect_async(WS_URL).await {
                Ok((stream, _)) => stream,
                Err(e) => {
                    error!("Telemetry WS connection failed: {}", e);
                    return;
                }
            };
            
            info!("Telemetry connected. Awaiting frames...");
            let (_, mut read) = ws_stream.split();

            while let Some(message) = read.next().await {
                if let Ok(Message::Binary(data)) = message {
                    if let Some(frame) = decode_telemetry_frame(&data) {
                        // Пытаемся пропихнуть кадр в канал. 
                        // Если Main Thread отстает, кадр дропается (TrySendError). Это норма для real-time.
                        let _ = tx.try_send(frame);
                    }
                }
            }
            
            warn!("Telemetry connection closed.");
        });
    }).detach();
}

/// Декодирование бинарного фрейма (08_ide.md §2.3)
/// Формат:
/// [0..4] Magic (b"SPIK")
/// [4..12] Tick (u64, LE)
/// [12..16] Spikes Count (u32, LE)
/// [16..] Array of u32 (Dense Indices / Spike IDs)
#[inline]
fn decode_telemetry_frame(data: &[u8]) -> Option<SpikeFrame> {
    if data.len() < 16 {
        return None;
    }

    // Проверка Magic (Fast Fail)
    let magic = &data[0..4];
    if magic != b"SPIK" {
        return None;
    }

    let tick = u64::from_le_bytes(data[4..12].try_into().unwrap());
    let count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;

    let payload = &data[16..];
    if payload.len() < count * 4 {
        error!("Corrupted telemetry frame: payload smaller than count");
        return None;
    }

    // Zero-cost cast среза байт в слайс u32, затем аллокация в Vec.
    // Выравнивание гарантируется безопасным копированием:
    let spike_ids = payload
        .chunks_exact(4)
        .take(count)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<u32>>();

    Some(SpikeFrame { tick, spike_ids })
}

/// Система (Hot Loop), опрашивающая канал каждый кадр.
/// Zero overhead, если канал пуст.
fn poll_telemetry_channel(
    bridge: Res<TelemetryBridge>,
    mut event_writer: EventWriter<SpikeFrame>,
) {
    // Вычитываем все накопленные фреймы с прошлого кадра (обычно 1-2)
    for frame in bridge.rx.try_iter() {
        event_writer.send(frame);
    }
}
