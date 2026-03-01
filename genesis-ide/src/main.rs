use bevy::prelude::*;
mod camera;
mod loader;
mod telemetry;

fn main() {
    App::new()
        // Настройки окна (Blender-like)
        .insert_resource(ClearColor(Color::rgb(0.05, 0.05, 0.05))) // Тёмно-серый фон
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Genesis AGI - V1 Core Viewport".into(),
                resolution: (1280., 720.).into(),
                ..default()
            }),
            ..default()
        }))
        // Наш сетевой плагин
        .add_plugins(loader::GeometryLoaderPlugin)
        .add_plugins(telemetry::TelemetryPlugin)
        .add_systems(Update, debug_spike_events)
        .add_systems(Startup, setup_viewport)
        .add_plugins(camera::CameraPlugin)
        .run();
}

/// Инициализация 3D-сцены
fn setup_viewport(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // 1. Освещение
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // 2. Декоративная сетка (Grid) для оценки масштаба
    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().size(50.0, 50.0)),
        material: materials.add(StandardMaterial {
            base_color: Color::rgb(0.1, 0.1, 0.15),
            unlit: true,
            ..default()
        }),
        ..default()
    });

    // 3. Камера
    commands.spawn((
        Camera3dBundle::default(),
        camera::OrbitCamera {
            radius: 150.0,
            center: Vec3::new(50.0, 0.0, 50.0), // Смещение к центру шарда
            ..default()
        },
    ));
}

/// Временная система для дебага: проверяем, что ECS видит спайки
fn debug_spike_events(mut events: EventReader<telemetry::SpikeFrame>) {
    for frame in events.read() {
        if !frame.spikes.is_empty() {
            println!(
                "IDE Render Tick: Received {} spikes from GPU Batch #{}",
                frame.spikes.len(),
                frame.tick
            );
        }
    }
}
