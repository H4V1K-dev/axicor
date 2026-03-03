use bevy::{prelude::*, tasks::IoTaskPool};
use bytemuck::{Pod, Zeroable};
use std::io::Read;
use crate::world::{NeuronInstance, NeuronLayerData, GlobalSpikeMap, SpikeRoute};
use crate::connectome::{AxonInstance, GhostAxonLayerData, AxonLayerData};

const GEOM_URL: &str = "127.0.0.1:8001";

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct GeometryHeader {
    pub magic: [u8; 4],
    pub total_neurons: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct NeuronGeomData {
    pub packed_pos: u32,
    pub flags: u32,
}

impl NeuronGeomData {
    fn type_id(&self) -> u8 {
        ((self.flags >> 4) & 0xF) as u8
    }
}

#[derive(Event)]
pub struct GeometryLoadedEvent {
    pub chunks: Vec<Vec<NeuronInstance>>,
    pub routing_map: Vec<SpikeRoute>,
    pub axons: Vec<AxonInstance>,
    pub ghost_axons: Vec<AxonInstance>,
}

pub struct LoaderPlugin;

impl Plugin for LoaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<GeometryLoadedEvent>()
           .add_systems(Startup, fetch_real_geometry)
           .add_systems(Update, apply_real_geometry);
    }
}

fn fetch_real_geometry(mut commands: Commands) {
    let pool = IoTaskPool::get();
    let (tx, rx) = crossbeam_channel::bounded(1);
    commands.insert_resource(GeometryReceiver(rx));

    pool.spawn(async move {
        info!("Connecting to GeometryServer at {}", GEOM_URL);
        
        let mut stream = match std::net::TcpStream::connect(GEOM_URL) {
            Ok(s) => s,
            Err(e) => {
                // По умолчанию НЕ подхватываем мок. Требуется явный opt-in.
                if std::env::var("GENESIS_IDE_USE_MOCK_GEOMETRY").as_deref() == Ok("1") {
                    error!("GeometryServer unavailable ({}), falling back to MOCK geometry (GENESIS_IDE_USE_MOCK_GEOMETRY=1)", e);
                    let ev = generate_mock_geometry();
                    let _ = tx.try_send(ev);
                } else {
                    error!("GeometryServer unavailable ({}), and GENESIS_IDE_USE_MOCK_GEOMETRY not set. Geometry will NOT be loaded.", e);
                }
                return;
            }
        };

        if let Err(e) = parse_geometry_stream(&mut stream, &tx) {
            error!("Geometry parse failed: {}", e);
            if std::env::var("GENESIS_IDE_USE_MOCK_GEOMETRY").as_deref() == Ok("1") {
                error!("Falling back to MOCK geometry (GENESIS_IDE_USE_MOCK_GEOMETRY=1)");
                let ev = generate_mock_geometry();
                let _ = tx.try_send(ev);
            } else {
                error!("GENESIS_IDE_USE_MOCK_GEOMETRY not set, geometry will NOT be loaded.");
            }
        }
    }).detach();
}

fn parse_geometry_stream(
    stream: &mut std::net::TcpStream,
    tx: &crossbeam_channel::Sender<GeometryLoadedEvent>,
) -> std::io::Result<()> {
    let mut header_bytes = [0u8; 8];
    stream.read_exact(&mut header_bytes)?;
    let header: GeometryHeader = bytemuck::cast(header_bytes);

    if &header.magic != b"GEOM" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid GEOM magic",
        ));
    }

    let total = header.total_neurons as usize;
    info!("Geometry: {} neurons", total);

    let geom_size = total * 8;
    let mut geom_bytes = vec![0u8; geom_size];
    stream.read_exact(&mut geom_bytes)?;

    let mut chunks: Vec<Vec<NeuronInstance>> = vec![Vec::new(); 16];
    let mut routing_map = Vec::with_capacity(total);
    let mut axons: Vec<AxonInstance> = Vec::with_capacity(total * 3);
    let mut raw_neurons: Vec<NeuronGeomData> = Vec::with_capacity(total);

    for chunk in geom_bytes.chunks_exact(8) {
        let geom: NeuronGeomData = bytemuck::cast_slice(chunk)[0];
        let type_id = geom.type_id();
        let local_idx = chunks[type_id as usize].len() as u32;

        chunks[type_id as usize].push(NeuronInstance {
            packed_pos: geom.packed_pos,
            emissive: 0.0,
            selected: 0,
        });

        routing_map.push(SpikeRoute { type_id, local_idx });
        raw_neurons.push(geom);

        // MOCK КОННЕКТОМ: соединяем каждый нейрон с 2-3 последующими
        let start_pos = geom.packed_pos;
        for j in 1..=3 {
            let neighbor_index = routing_map.len() + j - 1;
            if neighbor_index < total {
                // берем следующий geom по плотному индексу
                let neighbor_offset = neighbor_index * 8;
                if neighbor_offset + 8 <= geom_bytes.len() {
                    let neighbor_geom: NeuronGeomData = bytemuck::cast_slice(&geom_bytes[neighbor_offset..neighbor_offset + 8])[0];
                    axons.push(AxonInstance {
                        packed_start: start_pos,
                        packed_end: neighbor_geom.packed_pos,
                    });
                }
            }
        }
    }

    // Генерация Ghost Axons: длинные связи с шагом +500 по глобальному индексу
    let mut ghost_axons = Vec::new();
    for i in (0..total).step_by(16) {
        if i + 500 < total {
            let start = raw_neurons[i].packed_pos;
            let end = raw_neurons[i + 500].packed_pos;
            ghost_axons.push(AxonInstance {
                packed_start: start,
                packed_end: end,
            });
        }
    }

    info!("SpikeRouter built: {} entries ({} ghost axons)", total, ghost_axons.len());
    let _ = tx.try_send(GeometryLoadedEvent { chunks, routing_map, axons, ghost_axons });
    Ok(())
}

fn generate_mock_geometry() -> GeometryLoadedEvent {
    const TOTAL: usize = 160_000;
    const PER_TYPE: usize = TOTAL / 16;

    let mut chunks: Vec<Vec<NeuronInstance>> = vec![Vec::new(); 16];
    let mut routing_map = Vec::with_capacity(TOTAL);
    let mut axons: Vec<AxonInstance> = Vec::with_capacity(TOTAL * 3);
    let mut ghost_axons: Vec<AxonInstance> = Vec::new();

    for global_idx in 0..TOTAL {
        let type_id = (global_idx / PER_TYPE) as u8;
        let local_idx = chunks[type_id as usize].len() as u32;

        let x = (global_idx % 50) as u32;
        let y = ((global_idx / 50) % 50) as u32;
        let z = (global_idx / 2500) as u32;
        let packed_pos = x | (y << 11) | (z << 22);

        chunks[type_id as usize].push(NeuronInstance {
            packed_pos,
            emissive: 0.0,
            selected: 0,
        });

        routing_map.push(SpikeRoute { type_id, local_idx });

        // MOCK КОННЕКТОМ: соединяем каждый нейрон с 2-3 последующими
        for j in 1..=3 {
            let neighbor = global_idx + j;
            if neighbor < TOTAL {
                let nx = (neighbor % 50) as u32;
                let ny = ((neighbor / 50) % 50) as u32;
                let nz = (neighbor / 2500) as u32;
                let n_packed = nx | (ny << 11) | (nz << 22);
                axons.push(AxonInstance {
                    packed_start: packed_pos,
                    packed_end: n_packed,
                });
            }
        }
    }

    // Генерация Mock Ghost Axons: длинные связи с шагом +500
    for i in (0..TOTAL).step_by(16) {
        if i + 500 < TOTAL {
            let x0 = (i % 50) as u32;
            let y0 = ((i / 50) % 50) as u32;
            let z0 = (i / 2500) as u32;
            let start_packed = x0 | (y0 << 11) | (z0 << 22);

            let j = i + 500;
            let x1 = (j % 50) as u32;
            let y1 = ((j / 50) % 50) as u32;
            let z1 = (j / 2500) as u32;
            let end_packed = x1 | (y1 << 11) | (z1 << 22);

            ghost_axons.push(AxonInstance {
                packed_start: start_packed,
                packed_end: end_packed,
            });
        }
    }

    info!("Mock SpikeRouter: {} neurons ({} ghost axons)", TOTAL, ghost_axons.len());
    GeometryLoadedEvent { chunks, routing_map, axons, ghost_axons }
}

#[derive(Resource)]
struct GeometryReceiver(crossbeam_channel::Receiver<GeometryLoadedEvent>);

fn apply_real_geometry(
    receiver: Res<GeometryReceiver>,
    mut q_layers: Query<&mut NeuronLayerData>,
    mut q_axons: Query<&mut AxonLayerData>,
    mut q_ghost_axons: Query<&mut GhostAxonLayerData>,
    mut commands: Commands,
) {
    if let Ok(ev) = receiver.0.try_recv() {
        for mut layer in q_layers.iter_mut() {
            let t = layer.type_id as usize;
            layer.instances = ev.chunks[t].clone();
            layer.needs_buffer_update = true;
        }

        if let Ok(mut axon_layer) = q_axons.get_single_mut() {
            axon_layer.instances = ev.axons;
            axon_layer.needs_buffer_update = true;
        }

        if let Ok(mut ghost_layer) = q_ghost_axons.get_single_mut() {
            ghost_layer.instances = ev.ghost_axons;
            ghost_layer.needs_buffer_update = true;
        }

        commands.insert_resource(GlobalSpikeMap {
            map: ev.routing_map,
        });
        info!("Geometry and Connectome applied to layers");
    }
}
