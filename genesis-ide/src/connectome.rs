#![allow(dead_code)]

use bevy::{
    prelude::*,
    render::{render_resource::*, storage::ShaderStorageBuffer},
};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[allow(dead_code)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, ShaderType)]
pub struct AxonInstance {
    pub packed_start: u32,
    pub packed_end: u32,
}

#[derive(Component)]
pub struct AxonLayerData {
    pub instances: Vec<AxonInstance>,
    pub needs_buffer_update: bool,
}

#[derive(Component)]
pub struct GhostAxonLayerData {
    pub instances: Vec<AxonInstance>,
    pub needs_buffer_update: bool,
}

#[derive(Clone, Copy, ShaderType, Debug, Default)]
pub struct MaterialUniforms {
    pub base_color: LinearRgba,
    pub clip_plane: Vec4,
    pub view_mode: u32,
    pub _padding: Vec3,
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct AxonInstancedMaterial {
    #[uniform(0)]
    pub uniforms: MaterialUniforms,

    #[storage(1, read_only)]
    pub instances: Handle<ShaderStorageBuffer>,
}

impl Material for AxonInstancedMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/axon_instanced.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/axon_instanced.wgsl".into()
    }
}

// GhostAxon использует же MaterialUniforms что и остальные

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct GhostAxonMaterial {
    #[uniform(0)]
    pub uniforms: MaterialUniforms,

    #[storage(1, read_only)]
    pub instances: Handle<ShaderStorageBuffer>,
}

impl Material for GhostAxonMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/ghost_axon.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/ghost_axon.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

pub struct ConnectomePlugin;

impl Plugin for ConnectomePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<AxonInstancedMaterial>::default())
           .add_plugins(MaterialPlugin::<GhostAxonMaterial>::default())
           .add_systems(Startup, setup_axon_rendering)
           .add_systems(Update, (sync_axon_vram_buffers, sync_ghost_vram_buffers));
    }
}

fn setup_axon_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut axon_materials: ResMut<Assets<AxonInstancedMaterial>>,
    mut ghost_materials: ResMut<Assets<GhostAxonMaterial>>,
    settings: Res<crate::world::RenderSettings>,
) {
    // Цилиндр: radius=1.0, height=1.0 (будет масштабироваться в шейдере)
    let mesh_handle = meshes.add(Cylinder::new(1.0, 1.0).mesh().resolution(6));

    let axon_instances = buffers.add(ShaderStorageBuffer::from(Vec::<AxonInstance>::new()));
    let material = axon_materials.add(AxonInstancedMaterial {
        uniforms: MaterialUniforms {
            base_color: Color::srgba(0.2, 0.4, 0.8, 0.3).into(),
            clip_plane: settings.clip_plane,
            view_mode: 0,
            _padding: Vec3::ZERO,
        },
        instances: axon_instances,
    });

    let ghost_instances = buffers.add(ShaderStorageBuffer::from(Vec::<AxonInstance>::new()));
    let ghost_mat = ghost_materials.add(GhostAxonMaterial {
        uniforms: MaterialUniforms {
            base_color: Color::srgba(0.9, 0.2, 0.8, 0.6).into(),
            clip_plane: settings.clip_plane,
            view_mode: 0,
            _padding: Vec3::ZERO,
        },
        instances: ghost_instances,
    });

    commands.spawn((
        Mesh3d(mesh_handle.clone()),
        MeshMaterial3d(material),
        Transform::IDENTITY,
        AxonLayerData {
            instances: Vec::new(),
            needs_buffer_update: false,
        },
        bevy::render::primitives::Aabb::from_min_max(
            Vec3::new(0., 0., 0.), 
            Vec3::new(10000., 10000., 10000.)
        ),
    ));

    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(ghost_mat),
        Transform::IDENTITY,
        GhostAxonLayerData {
            instances: Vec::new(),
            needs_buffer_update: false,
        },
        bevy::render::primitives::Aabb::from_min_max(
            Vec3::new(0., 0., 0.),
            Vec3::new(10000., 10000., 10000.)
        ),
    ));
}

/// Zero-Cost VRAM Upload. Вызывается только когда `needs_buffer_update = true`
fn sync_axon_vram_buffers(
    mut query: Query<(&mut AxonLayerData, &MeshMaterial3d<AxonInstancedMaterial>)>,
    materials: Res<Assets<AxonInstancedMaterial>>,
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

fn sync_ghost_vram_buffers(
    mut query: Query<(&mut GhostAxonLayerData, &MeshMaterial3d<GhostAxonMaterial>)>,
    materials: Res<Assets<GhostAxonMaterial>>,
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


