#![allow(dead_code)]

use bevy::{
    prelude::*,
    render::{render_resource::*, storage::ShaderStorageBuffer},
};
use bytemuck::{Pod, Zeroable};

/// 16 байт на пиксель I/O матрицы
#[repr(C)]
#[allow(dead_code)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, ShaderType)]
pub struct IoPixelInstance {
    pub position: Vec3,
    pub state: f32, // Активность 0.0 .. 1.0
}

#[derive(Component)]
pub struct IoMatrixData {
    pub instances: Vec<IoPixelInstance>,
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
pub struct IoInstancedMaterial {
    #[uniform(0)]
    pub uniforms: MaterialUniforms,

    #[storage(1, read_only)]
    pub instances: Handle<ShaderStorageBuffer>,
}

impl Material for IoInstancedMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/io_pixel.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/io_pixel.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

pub struct IoMatrixPlugin;

impl Plugin for IoMatrixPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<IoInstancedMaterial>::default())
            .add_systems(Startup, setup_mock_io_matrices)
            .add_systems(
                Update,
                (
                    animate_mock_io_matrices,
                    sync_io_vram_buffers,
                ),
            );
    }
}

fn setup_mock_io_matrices(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut materials: ResMut<Assets<IoInstancedMaterial>>,
) {
    // Пиксель — плоский квадрат 10x10 мкм
    let quad_handle = meshes.add(Rectangle::new(10.0, 10.0));

    // 1. Входная матрица (Сенсоры, например 32x32)
    let mut input_instances = Vec::with_capacity(32 * 32);
    for y in 0..32 {
        for x in 0..32 {
            input_instances.push(IoPixelInstance {
                // Располагаем над корой (Z = 500)
                position: Vec3::new(x as f32 * 12.0, y as f32 * 12.0, 500.0),
                state: 0.0,
            });
        }
    }

    let input_instances_buf =
        buffers.add(ShaderStorageBuffer::from(Vec::<IoPixelInstance>::new()));
    let input_mat = materials.add(IoInstancedMaterial {
        uniforms: MaterialUniforms {
            base_color: Color::srgba(0.2, 0.9, 0.4, 0.8).into(),
            clip_plane: Vec4::ZERO,
            view_mode: 0,
            _padding: Vec3::ZERO,
        },
        instances: input_instances_buf,
    });

    commands.spawn((
        Mesh3d(quad_handle.clone()),
        MeshMaterial3d(input_mat),
        Transform::IDENTITY,
        IoMatrixData {
            instances: input_instances,
            needs_buffer_update: true,
        },
        bevy::render::primitives::Aabb::from_min_max(
            Vec3::new(0., 0., 0.),
            Vec3::new(1000., 1000., 1000.),
        ),
    ));

    // 2. Выходная матрица (Моторы, например 16x16)
    let mut output_instances = Vec::with_capacity(16 * 16);
    for y in 0..16 {
        for x in 0..16 {
            output_instances.push(IoPixelInstance {
                // Располагаем сбоку
                position: Vec3::new(x as f32 * 12.0 - 300.0, y as f32 * 12.0, 400.0),
                state: 0.0,
            });
        }
    }

    let output_instances_buf =
        buffers.add(ShaderStorageBuffer::from(Vec::<IoPixelInstance>::new()));
    let output_mat = materials.add(IoInstancedMaterial {
        uniforms: MaterialUniforms {
            base_color: Color::srgba(0.9, 0.3, 0.2, 0.8).into(),
            clip_plane: Vec4::ZERO,
            view_mode: 0,
            _padding: Vec3::ZERO,
        },
        instances: output_instances_buf,
    });

    commands.spawn((
        Mesh3d(quad_handle),
        MeshMaterial3d(output_mat),
        Transform::IDENTITY,
        IoMatrixData {
            instances: output_instances,
            needs_buffer_update: true,
        },
        bevy::render::primitives::Aabb::from_min_max(
            Vec3::new(-1000., 0., 0.),
            Vec3::new(1000., 1000., 1000.),
        ),
    ));
}

/// Zero-Cost VRAM Upload. Вызывается только при needs_buffer_update = true
fn sync_io_vram_buffers(
    mut query: Query<(&mut IoMatrixData, &MeshMaterial3d<IoInstancedMaterial>)>,
    materials: Res<Assets<IoInstancedMaterial>>,
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

/// Процедурная анимация матриц для стресс-теста VRAM аплоада
pub fn animate_mock_io_matrices(
    time: Res<Time>,
    mut q_matrices: Query<&mut IoMatrixData>,
) {
    let t = time.elapsed_secs();
    
    for mut matrix in q_matrices.iter_mut() {
        let len = matrix.instances.len();
        if len == 0 {
            continue;
        }

        let width = (len as f32).sqrt() as usize;
        if width == 0 {
            continue;
        }
        
        for (i, instance) in matrix.instances.iter_mut().enumerate() {
            let x = (i % width) as f32;
            let y = (i / width) as f32;
            
            // Генерируем "бегущие волны" (имитация паттернов активности коры)
            let wave1 = (x * 0.3 + t * 4.0).sin();
            let wave2 = (y * 0.4 - t * 2.5).cos();
            let noise = (wave1 * wave2 * 0.5 + 0.5).powf(3.0); // Делаем пики более резкими
            
            // Дискретный LED-эффект: пиксель либо горит ярко, либо тлеет
            instance.state = if noise > 0.6 { 1.0 } else { 0.05 }; 
        }
        
        // Триггерим Zero-Cost Upload (sync_io_vram_buffers перехватит это)
        matrix.needs_buffer_update = true;
    }
}


