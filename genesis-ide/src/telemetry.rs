use bevy::prelude::*;
use crossbeam_channel::{unbounded, Receiver, Sender};
use futures_util::StreamExt;
use std::thread;
use tokio_tungstenite::connect_async;

// Контракт с genesis-runtime (байт-в-байт)
const MAGIC_GNSS: u32 = 0x474E5353;
const HEADER_SIZE: usize = 16;

/// Событие ECS, которое будет триггерить свечение (glow) в 3D
#[derive(Event, Clone)]
pub struct SpikeFrame {
    pub tick: u64,
    pub spikes: Vec<u32>, // Flat array of local IDs
}

// Ресурс для хранения канала связи
#[derive(Resource)]
struct TelemetryReceiver(Receiver<SpikeFrame>);

pub struct TelemetryPlugin;

impl Plugin for TelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<SpikeFrame>()
           .add_systems(Startup, start_network_thread)
           .add_systems(Update, drain_telemetry_channel);
    }
}

/// Поднимаем выделенный поток ОС для Tokio, чтобы не душить рендер Bevy
fn start_network_thread(mut commands: Commands) {
    let (tx, rx) = unbounded::<SpikeFrame>();
    commands.insert_resource(TelemetryReceiver(rx));

    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let url = "ws://127.0.0.1:8002/ws";
            println!("IDE: Connecting to Telemetry Stream at {}...", url);
            
            match connect_async(url).await {
                Ok((ws_stream, _)) => {
                    println!("IDE: Connected to Genesis Runtime!");
                    let (_, mut read) = ws_stream.split();

                    while let Some(msg) = read.next().await {
                        if let Ok(tokio_tungstenite::tungstenite::Message::Binary(bin)) = msg {
                            parse_and_send(&bin, &tx);
                        }
                    }
                }
                Err(e) => eprintln!("IDE: Failed to connect telemetry: {}", e),
            }
        });
    });
}

/// Хардкорный парсинг бинарника без сериализаторов
fn parse_and_send(data: &[u8], tx: &Sender<SpikeFrame>) {
    if data.len() < HEADER_SIZE { return; }

    // Little-Endian распаковка
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != MAGIC_GNSS { return; }

    let tick = u64::from_le_bytes(data[4..12].try_into().unwrap());
    let spikes_count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;

    let expected_len = HEADER_SIZE + spikes_count * 4;
    if data.len() < expected_len { return; }

    // Cast raw bytes to u32 vector (Zero-cost каст на CPU)
    let payload_bytes = &data[HEADER_SIZE..expected_len];
    let mut spikes = vec![0u32; spikes_count];
    
    unsafe {
        std::ptr::copy_nonoverlapping(
            payload_bytes.as_ptr(),
            spikes.as_mut_ptr() as *mut u8,
            payload_bytes.len(),
        );
    }

    let _ = tx.send(SpikeFrame { tick, spikes });
}

/// Перекачиваем данные из канала в ECS Event-шину
fn drain_telemetry_channel(
    receiver: Res<TelemetryReceiver>,
    mut events: EventWriter<SpikeFrame>,
) {
    // Выгребаем всё, что пришло по сети за время отрисовки кадра
    for frame in receiver.0.try_iter() {
        events.send(frame);
    }
}
