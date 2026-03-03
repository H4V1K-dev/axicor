#![allow(dead_code)]

use bevy::{
    prelude::*,
    render::{render_resource::*, storage::ShaderStorageBuffer},
};
use bytemuck::{Pod, Zeroable};
use crate::telemetry::SpikeFrame;
use crate::hud::SelectionState;

#[repr(C)]
#[allow(dead_code)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, ShaderType)]
pub struct NeuronInstance {
    pub packed_pos: u32,
    pub emissive: f32,
    pub selected: u32, // 1 = выделен, 0 = нет
}

#[derive(Component)]
pub struct NeuronLayerData {
    pub type_id: u8,
    pub instances: Vec<NeuronInstance>,
    pub needs_buffer_update: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum ViewMode {
    #[default]
    Solid = 0,
    Activity = 1,
}

#[derive(Resource)]
pub struct RenderSettings {
    pub view_mode: ViewMode,
    // xyz - нормаль плоскости, w - смещение
    pub clip_plane: Vec4,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            view_mode: ViewMode::default(),
            clip_plane: Vec4::new(0.0, 1.0, 0.0, 10000.0),
        }
    }
}

/// Global spike routing: Dense Index → (Type ID, Local Instance Index)
/// One-time lookup cost per spike in hot loop.
#[derive(Clone, Copy)]
pub struct SpikeRoute {
    pub type_id: u8,
    pub local_idx: u32,
}

#[derive(Resource, Default)]
pub struct GlobalSpikeMap {
    pub map: Vec<SpikeRoute>,
}

#[derive(Clone, Copy, ShaderType, Debug, Default)]
pub struct MaterialUniforms {
    pub base_color: LinearRgba,
    pub clip_plane: Vec4,
    pub view_mode: u32,
    pub _padding: Vec3,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct NeuronInstancedMaterial {
    #[uniform(0)]
    pub uniforms: MaterialUniforms,

    #[storage(1, read_only)]
    pub instances: Handle<ShaderStorageBuffer>,
}

impl Material for NeuronInstancedMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/neuron_instanced.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/neuron_instanced.wgsl".into()
    }
}

pub struct WorldViewPlugin;

impl Plugin for WorldViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<NeuronInstancedMaterial>::default())
            .init_resource::<RenderSettings>()
            .add_systems(Startup, setup_world_rendering)
            .add_systems(
                Update,
                (
                    apply_telemetry_spikes,
                    sync_selection_to_instances,
                    sync_vram_buffers,
                    handle_view_mode_toggle,
                    handle_clipping_plane,
                    sync_neuron_vram_buffers,
                )
                    .chain(),
            );
    }
}

fn setup_world_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut materials: ResMut<Assets<NeuronInstancedMaterial>>,
    settings: Res<RenderSettings>,
) {
    let mesh_handle = meshes.add(Sphere::new(0.5).mesh().ico(2).unwrap());

    for type_id in 0..16 {
        let color = get_type_color(type_id);
        let instances = buffers.add(ShaderStorageBuffer::from(Vec::<NeuronInstance>::new()));
        let material = materials.add(NeuronInstancedMaterial {
            uniforms: MaterialUniforms {
                base_color: color,
                clip_plane: settings.clip_plane,
                view_mode: ViewMode::Solid as u32,
                _padding: Vec3::ZERO,
            },
            instances,
        });

        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material),
            Transform::from_xyz(0., 0., 0.),
            NeuronLayerData {
                type_id: type_id as u8,
                instances: Vec::new(),
                needs_buffer_update: false,
            },
        ));
        
        println!("[world] Initialized neuron layer {}", type_id);
    }
    
    println!("[world] 16 neuron layers ready for GPU Instancing");
}

fn get_type_color(type_id: u8) -> LinearRgba {
    Color::hsl((type_id as f32) * 22.5, 0.8, 0.5).into()
}

fn apply_telemetry_spikes(
    mut query: Query<&mut NeuronLayerData>,
    mut spike_events: EventReader<SpikeFrame>,
    spike_map: Option<Res<GlobalSpikeMap>>,
) {
    let Some(global_map) = spike_map else { return };

    // 1. Плоский массив преаллоцированных батчей (Zero-Cost allocations)
    // Вместо HashMap — прямая индексация в array за O(1) без хэширования
    let mut batches: [Vec<u32>; 16] = std::array::from_fn(|_| Vec::with_capacity(1000));

    // 2. Быстрая диспетчеризация через прямой доступ по type_id
    for ev in spike_events.read() {
        for &dense_id in &ev.spike_ids {
            if let Some(route) = global_map.map.get(dense_id as usize) {
                batches[route.type_id as usize].push(route.local_idx);
            }
        }
    }

    // 3. Обновление слоёв без mutual borrow
    for mut layer in query.iter_mut() {
        let t_id = layer.type_id as usize;
        let active_local_ids = &batches[t_id];
        let mut dirty = false;

        // Fade out
        for instance in layer.instances.iter_mut() {
            if instance.emissive > 0.0 {
                instance.emissive = (instance.emissive - 0.05).max(0.0);
                dirty = true;
            }
        }

        // Инъекция спайков
        for &local_idx in active_local_ids {
            if let Some(instance) = layer.instances.get_mut(local_idx as usize) {
                instance.emissive = 1.0;
                dirty = true;
            }
        }

        if dirty {
            layer.needs_buffer_update = true;
        }
    }
}

/// Заливка из ECS-компонента в GPU-буфер.
/// Выполняется ТОЛЬКО если был спайк или изменилась геометрия.
/// Change Detection в Bevy автоматически триггерит RenderQueue::write_buffer
/// в фазе Prepare для всех изменённых компонентов.
fn sync_vram_buffers(
    mut query: Query<&mut NeuronLayerData, Changed<NeuronLayerData>>,
) {
    for mut layer in query.iter_mut() {
        if layer.needs_buffer_update {
            // Флаг сброшен. Bevy Change Detection при доступе через iter_mut()
            // автоматически пометит эту сущность для обновления в RenderQueue.
            layer.needs_buffer_update = false;
        }
    }
}

fn sync_neuron_vram_buffers(
    mut query: Query<(&mut NeuronLayerData, &MeshMaterial3d<NeuronInstancedMaterial>)>,
    materials: Res<Assets<NeuronInstancedMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    for (mut layer, mat_handle) in query.iter_mut() {
        if layer.needs_buffer_update {
            if let Some(material) = materials.get(&mat_handle.0) {
                if let Some(buffer) = buffers.get_mut(&material.instances) {
                    buffer.set_data(layer.instances.as_slice());
                }
            }
            layer.needs_buffer_update = false;
        }
    }
}

/// Система обновляет флаги выделения в инстансах при изменении SelectionState.
pub fn sync_selection_to_instances(
    selection: Res<SelectionState>,
    mut q_layers: Query<&mut NeuronLayerData>,
) {
    if !selection.is_changed() { return; }

    // Предподготовка: группируем выбранные локальные индексы по type_id
    let mut per_type: [Vec<u32>; 16] = std::array::from_fn(|_| Vec::new());
    for &(t_id, l_idx) in &selection.selected_neurons {
        if (t_id as usize) < per_type.len() {
            per_type[t_id as usize].push(l_idx);
        }
    }

    // 1. Быстрый сброс O(N) и 2. Точечная установка O(M) в одном проходе по слоям
    for mut layer in q_layers.iter_mut() {
        // Сброс
        for instance in layer.instances.iter_mut() {
            instance.selected = 0;
        }

        // Установка флагов для выбранных локальных индексов этого типа
        let t_idx = layer.type_id as usize;
        if t_idx < per_type.len() {
            for &local_idx in &per_type[t_idx] {
                if let Some(instance) = layer.instances.get_mut(local_idx as usize) {
                    instance.selected = 1;
                }
            }
        }

        layer.needs_buffer_update = true;
    }
}

pub fn handle_view_mode_toggle(
    keys: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<RenderSettings>,
    mut materials: ResMut<Assets<NeuronInstancedMaterial>>,
    q_materials: Query<&MeshMaterial3d<NeuronInstancedMaterial>>,
) {
    if !keys.just_pressed(KeyCode::KeyZ) {
        return;
    }

    settings.view_mode = match settings.view_mode {
        ViewMode::Solid => ViewMode::Activity,
        ViewMode::Activity => ViewMode::Solid,
    };

    info!("Switched View Mode to: {:?}", settings.view_mode);

    let mode_u32 = settings.view_mode as u32;
    for mat_handle in q_materials.iter() {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.uniforms.view_mode = mode_u32;
            mat.uniforms.clip_plane = settings.clip_plane;
        }
    }
}

use bevy::input::mouse::MouseWheel;

pub fn handle_clipping_plane(
    keys: Res<ButtonInput<KeyCode>>,
    mut ev_scroll: EventReader<MouseWheel>,
    q_camera: Query<&GlobalTransform, With<crate::camera::IdeCamera>>,
    mut settings: ResMut<RenderSettings>,
    q_neuron_mat: Query<&MeshMaterial3d<NeuronInstancedMaterial>>,
    mut neuron_materials: ResMut<Assets<NeuronInstancedMaterial>>,
    q_axon_mat: Query<&MeshMaterial3d<crate::connectome::AxonInstancedMaterial>>,
    mut axon_materials: ResMut<Assets<crate::connectome::AxonInstancedMaterial>>,
    q_ghost_mat: Query<&MeshMaterial3d<crate::connectome::GhostAxonMaterial>>,
    mut ghost_materials: ResMut<Assets<crate::connectome::GhostAxonMaterial>>,
) {
    let Ok(cam_transform) = q_camera.get_single() else { return };

    if keys.pressed(KeyCode::KeyC) {
        let forward = cam_transform.forward();

        let mut scroll_delta = 0.0;
        for ev in ev_scroll.read() {
            scroll_delta += ev.y;
        }

        settings.clip_plane.x = forward.x;
        settings.clip_plane.y = forward.y;
        settings.clip_plane.z = forward.z;

        if scroll_delta != 0.0 {
            settings.clip_plane.w += scroll_delta * 100.0;
        }

        for mat_handle in q_neuron_mat.iter() {
            if let Some(mat) = neuron_materials.get_mut(&mat_handle.0) {
                mat.uniforms.clip_plane = settings.clip_plane;
            }
        }
        for mat_handle in q_axon_mat.iter() {
            if let Some(mat) = axon_materials.get_mut(&mat_handle.0) {
                mat.uniforms.clip_plane = settings.clip_plane;
            }
        }
        for mat_handle in q_ghost_mat.iter() {
            if let Some(mat) = ghost_materials.get_mut(&mat_handle.0) {
                mat.uniforms.clip_plane = settings.clip_plane;
            }
        }
    }

    if keys.pressed(KeyCode::AltLeft) && keys.just_pressed(KeyCode::KeyC) {
        settings.clip_plane.w = 10000.0;
        for mat_handle in q_neuron_mat.iter() {
            if let Some(mat) = neuron_materials.get_mut(&mat_handle.0) {
                mat.uniforms.clip_plane = settings.clip_plane;
            }
        }
        for mat_handle in q_axon_mat.iter() {
            if let Some(mat) = axon_materials.get_mut(&mat_handle.0) {
                mat.uniforms.clip_plane = settings.clip_plane;
            }
        }
        for mat_handle in q_ghost_mat.iter() {
            if let Some(mat) = ghost_materials.get_mut(&mat_handle.0) {
                mat.uniforms.clip_plane = settings.clip_plane;
            }
        }
    }
}


