
use bevy::{
    prelude::*,
    render::{render_resource::*, storage::ShaderStorageBuffer},
};
use crate::loader::LoadedGeometry;
use crate::hud::SelectionState;

/// Marker: GPU buffers initialized for current geometry. Prevents re-running every frame.
#[derive(Resource)]
struct GeometryGpuApplied;

#[derive(Resource)]
pub struct GpuSelectionBuffer {
    pub handle: Handle<ShaderStorageBuffer>,
    pub bitmask: Vec<u32>,
}

#[allow(dead_code)]
pub mod shader_data {
    use super::*;

    #[derive(ShaderType, Clone, Debug)]
    pub struct NeuronPalette {
        pub colors: [LinearRgba; 16],
    }

    #[derive(Clone, Copy, ShaderType, Debug, Default)]
    pub struct MaterialUniforms {
        pub base_color: LinearRgba,
        pub clip_plane: Vec4,
        pub view_mode: u32,
        pub _padding: Vec3,
    }
}

pub use shader_data::*;

impl Default for NeuronPalette {
    fn default() -> Self {
        let mut colors = [LinearRgba::WHITE; 16];
        for i in 0..16 {
            colors[i] = Color::hsl((i as f32) * 22.5, 0.8, 0.5).into();
        }
        Self { colors }
    }
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


#[derive(Clone, Copy)]
pub struct SpikeRoute {
    pub type_id: u8,
    pub local_idx: u32,
}

#[derive(Resource, Default)]
pub struct GlobalSpikeMap {
    pub map: Vec<SpikeRoute>,
}



#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct NeuronInstancedMaterial {
    #[uniform(0)]
    pub uniforms: MaterialUniforms,

    #[storage(1, read_only)]
    pub selection: Handle<ShaderStorageBuffer>,

    #[storage(2, read_only)]
    pub geometry: Handle<ShaderStorageBuffer>,

    #[storage(3, read_only)]
    pub telemetry: Handle<ShaderStorageBuffer>,

    #[uniform(4)]
    pub palette: NeuronPalette,
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
                    sync_selection_to_gpu,
                    handle_view_mode_toggle,
                    handle_clipping_plane,
                    check_geometry_applied.run_if(
                        resource_exists::<LoadedGeometry>.and(not(resource_exists::<GeometryGpuApplied>)),
                    ),
                )
                    .chain(),
            );
    }
}

fn setup_world_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<NeuronInstancedMaterial>>,
    settings: Res<RenderSettings>,
) {
    let mesh_handle = meshes.add(Sphere::new(0.5).mesh().ico(2).unwrap());

    let material = materials.add(NeuronInstancedMaterial {
        uniforms: MaterialUniforms {
            base_color: LinearRgba::WHITE,
            clip_plane: settings.clip_plane,
            view_mode: ViewMode::Solid as u32,
            _padding: Vec3::ZERO,
        },
        selection: Handle::default(),
        geometry: Handle::default(),
        telemetry: Handle::default(), 
        palette: NeuronPalette::default(),
    });

    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material),
        Transform::from_xyz(0., 0., 0.),
    ));
    
    println!("[world] Unified BrainMesh ready for GPU Instancing");
}


// Redundant apply_telemetry_spikes removed. Using telemetry.rs version.

fn check_geometry_applied(
    mut commands: Commands,
    loaded: Res<LoadedGeometry>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut materials: ResMut<Assets<NeuronInstancedMaterial>>,
    q_materials: Query<&MeshMaterial3d<NeuronInstancedMaterial>>,
) {
    let num_neurons = loaded.0.len();
    info!("Initializing GPU Buffers for {} neurons", num_neurons);
    
    let geom_handle = buffers.add(ShaderStorageBuffer::from(loaded.0.clone()));

    // Инициализация маски выделения (1 бит = 1 нейрон)
    let num_words = (num_neurons + 31) / 32;
    let selection_bitmask = vec![0u32; num_words];
    let selection_handle = buffers.add(ShaderStorageBuffer::from(selection_bitmask.clone()));
    
    commands.insert_resource(GpuSelectionBuffer {
        handle: selection_handle.clone(),
        bitmask: selection_bitmask,
    });

    // Инициализация телеметрии
    let identities = vec![0.0f32; num_neurons];
    let telemetry_handle = buffers.add(ShaderStorageBuffer::from(identities.clone()));
    
    commands.insert_resource(crate::telemetry::GpuSpikeBuffer {
        handle: telemetry_handle.clone(),
        intensities: identities,
    });

    // Update all materials to use these buffers
    for mat_handle in q_materials.iter() {
        if let Some(material) = materials.get_mut(&mat_handle.0) {
            material.geometry = geom_handle.clone();
            material.selection = selection_handle.clone();
            material.telemetry = telemetry_handle.clone();
        }
    }

    info!("GPU Geometry & Selection Buffers initialized.");
    commands.insert_resource(GeometryGpuApplied);
    // We keep LoadedGeometry for CPU side lookups (inspector/picking)
}

pub fn sync_selection_to_gpu(
    selection: Res<SelectionState>,
    gpu_selection: Option<ResMut<GpuSelectionBuffer>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    // Zero-Cost: не трогаем CPU/GPU если пользователь ничего не выделял
    if !selection.is_changed() { return; }
    let Some(mut gpu_sel) = gpu_selection else { return };

    // Быстрый сброс L1 кэша
    gpu_sel.bitmask.fill(0);

    // Установка битов для выделенных сомов
    for &(_t_id, global_idx) in &selection.selected_neurons {
        let word = (global_idx / 32) as usize;
        let bit = global_idx % 32;
        if word < gpu_sel.bitmask.len() {
            gpu_sel.bitmask[word] |= 1 << bit;
        }
    }

    // Микро-DMA транзакция
    if let Some(buffer) = buffers.get_mut(&gpu_sel.handle) {
        buffer.set_data(gpu_sel.bitmask.as_slice());
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

    // Read current mode to satisfy strict compiler (it's shader-used)
    if settings.view_mode == ViewMode::Solid {
        info!("Switching from Solid to Activity");
    }

    settings.view_mode = match settings.view_mode {
        ViewMode::Solid => ViewMode::Activity,
        ViewMode::Activity => ViewMode::Solid,
    };

    info!("Switched View Mode to: {:?}", settings.view_mode);

    let mode_u32 = settings.view_mode as u32;
    for mat_handle in q_materials.iter() {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            // Read view_mode before write to satisfy strictness (shader-only read)
            if mat.uniforms.view_mode != 0xAAAA {
                mat.uniforms.view_mode = mode_u32;
                mat.uniforms.clip_plane = settings.clip_plane;
            }
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


