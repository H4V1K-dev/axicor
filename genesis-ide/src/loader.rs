// genesis-ide/src/loader.rs
use bevy::prelude::*;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::thread;
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;

const MAGIC_GEOM: u32 = 0x47454F4D; // "GEOM"

#[derive(Event)]
pub struct GeometryLoaded {
    pub positions: Vec<u32>, // Сырые PackedPosition
}

#[derive(Resource)]
struct GeometryReceiver(Receiver<GeometryLoaded>);

pub struct GeometryLoaderPlugin;

impl Plugin for GeometryLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<GeometryLoaded>()
           .add_systems(Startup, start_tcp_loader)
           .add_systems(Update, handle_geometry_load);
    }
}

fn start_tcp_loader(mut commands: Commands) {
    let (tx, rx) = unbounded();
    commands.insert_resource(GeometryReceiver(rx));

    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async move {
            println!("IDE: Requesting Geometry from TCP 8001...");
            let mut stream = match TcpStream::connect("127.0.0.1:8001").await {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("IDE: Failed to connect to GeometryServer: {}", e);
                    return;
                }
            };

            let mut header = [0u8; 12];
            if stream.read_exact(&mut header).await.is_err() { return; }

            let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
            if magic != MAGIC_GEOM { return; }

            let num_neurons = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
            let payload_size = num_neurons * 4;
            
            let mut payload = vec![0u8; payload_size];
            if stream.read_exact(&mut payload).await.is_err() { return; }

            let mut positions = vec![0u32; num_neurons];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    payload.as_ptr(),
                    positions.as_mut_ptr() as *mut u8,
                    payload_size,
                );
            }

            println!("IDE: Loaded {} neuron positions.", num_neurons);
            let _ = tx.send(GeometryLoaded { positions });
        });
    });
}

fn handle_geometry_load(
    receiver: Res<GeometryReceiver>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for event in receiver.0.try_iter() {
        // Базовая сфера для всех инстансов
        let mesh = meshes.add(Sphere::new(0.5).mesh());
        
        // 4 базовых цвета для типов 0..3 (можно расширить до 16)
        let colors = [
            materials.add(StandardMaterial { base_color: Color::rgb(0.2, 0.8, 0.2), unlit: true, ..default() }), // Type 0 Excitatory
            materials.add(StandardMaterial { base_color: Color::rgb(0.8, 0.2, 0.2), unlit: true, ..default() }), // Type 1 Inhibitory
            materials.add(StandardMaterial { base_color: Color::rgb(0.2, 0.2, 0.8), unlit: true, ..default() }), // Type 2 Relay
            materials.add(StandardMaterial { base_color: Color::rgb(0.8, 0.8, 0.2), unlit: true, ..default() }), // Type 3 Burst
        ];

        for (_dense_id, &packed) in event.positions.iter().enumerate() {
            // Распаковка согласно спецификации
            let x = (packed & 0x3FF) as f32;
            let y = ((packed >> 10) & 0x3FF) as f32;
            let z = ((packed >> 20) & 0xFF) as f32;
            let t = ((packed >> 28) & 0xF) as usize;

            let mat = colors[t % 4].clone();

            commands.spawn(PbrBundle {
                mesh: mesh.clone(),
                material: mat,
                transform: Transform::from_xyz(x, z, y), // Bevy Y is up, our Z is vertical depth
                ..default()
            });
        }
    }
}
