use bevy::{
    prelude::*,
    render::{
        storage::ShaderStorageBuffer,
    },
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};
use std::thread;
use tokio::runtime::Builder;
use futures_util::StreamExt;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use std::convert::TryInto;
use crate::config::IdeConfig;

#[derive(Event, Clone)]
pub struct SpikeFrameEvent {
    pub tick: u64,
    pub spike_ids: Vec<u32>,
}

#[derive(Resource)]
pub struct TelemetryReceiver(pub UnboundedReceiver<(u64, Vec<u32>)>);


#[derive(Resource, Default)]
pub struct GpuSpikeBuffer {
    pub handle: Handle<ShaderStorageBuffer>,
    pub intensities: Vec<f32>,
}

pub struct TelemetryPlugin;

impl Plugin for TelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<SpikeFrameEvent>()
            .init_resource::<GpuSpikeBuffer>()
            .add_systems(Startup, setup_telemetry_socket)
            .add_systems(Update, (drain_telemetry_socket, apply_telemetry_spikes).chain());
    }
}

pub fn setup_telemetry_socket(mut commands: Commands, config: Res<IdeConfig>) {
    let (tx, rx) = unbounded_channel();

    let ws_url = format!("ws://{}:{}/ws", config.target_ip, config.telemetry_port);

    // Создаем выделенный поток ОС для сетевого реактора
    thread::spawn(move || {
        // Поднимаем изолированный Tokio Runtime
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("FATAL: Failed to build isolated Tokio runtime for Telemetry");

        rt.block_on(async move {
            println!("[Telemetry] Connecting to Genesis Telemetry at {}...", ws_url);
            let mut ws_stream = match connect_async(&ws_url).await {
                Ok((stream, _)) => stream,
                Err(e) => {
                    eprintln!("[Telemetry] FATAL: WS connection failed at {}: {}", ws_url, e);
                    return;
                }
            };

            println!("[Telemetry] Connected. Awaiting frames...");
            while let Some(msg) = ws_stream.next().await {
                if let Ok(Message::Binary(data)) = msg {
                    if let Some((tick, spike_ids)) = decode_telemetry_frame(&data) {
                        let _ = tx.send((tick, spike_ids));
                    }
                }
            }
        });
    });

    commands.insert_resource(TelemetryReceiver(rx));
}

fn decode_telemetry_frame(data: &[u8]) -> Option<(u64, Vec<u32>)> {
    if data.len() < 16 { return None; }
    
    // SPIK magic check
    if &data[0..4] != b"SPIK" { return None; }
    
    let tick = u64::from_le_bytes(data[4..12].try_into().unwrap());
    let count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    
    if data.len() < 16 + count * 4 { return None; }
    
    let spikes = data[16..]
        .chunks_exact(4)
        .take(count)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
        
    Some((tick, spikes))
}

pub fn drain_telemetry_socket(
    mut receiver: ResMut<TelemetryReceiver>,
    mut ev_writer: EventWriter<SpikeFrameEvent>,
) {
    while let Ok((tick, spike_ids)) = receiver.0.try_recv() {
        if !spike_ids.is_empty() {
            bevy::log::info!("Telemetry tick: {}, Spikes: {}", tick, spike_ids.len());
        }
        ev_writer.send(SpikeFrameEvent { tick, spike_ids });
    }
}

pub fn apply_telemetry_spikes(
    mut events: EventReader<SpikeFrameEvent>,
    mut gpu_buffer: ResMut<GpuSpikeBuffer>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    if gpu_buffer.intensities.is_empty() {
        return;
    }

    let mut is_dirty = false;
    {
        let intensities = &mut gpu_buffer.intensities;

        // 1. Быстрый Decay (линейный проход L1 Cache)
        for glow in intensities.iter_mut() {
            if *glow > 0.0 {
                *glow = (*glow - 0.15).max(0.0);
                is_dirty = true;
            }
        }

        // 2. Инъекция (O(K))
        for frame in events.read() {
            for &id in &frame.spike_ids {
                if let Some(glow) = intensities.get_mut(id as usize) {
                    *glow = 1.0;
                    is_dirty = true;
                }
            }
        }
    }

    // 3. Обновление ассета
    if is_dirty {
        if let Some(buffer) = buffers.get_mut(&gpu_buffer.handle) {
            buffer.set_data(gpu_buffer.intensities.as_slice());
        }
    }
}
