use bevy::prelude::*;
mod camera;
mod loader;
mod telemetry;
mod world;
mod hud;
mod layout;
mod config_editor;
mod signal_scope;
mod neuron_inspector;
mod connectome;
mod timeline;
mod log_console;
mod bake_panel;
mod shard_map;
mod io_matrix;

fn main() {
    App::new()
        // Настройки окна (Blender-like)
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.05))) // Тёмно-серый фон
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Genesis AGI - V1 Core Viewport".into(),
                resolution: (1280., 720.).into(),
                ..default()
            }),
            ..default()
        }))
        // Наш сетевой плагин
        .add_plugins(layout::LayoutPlugin)
        .add_plugins(config_editor::ConfigEditorPlugin)
        .add_plugins(loader::LoaderPlugin)
        .add_plugins(telemetry::TelemetryPlugin)
        .add_plugins(world::WorldViewPlugin)
        .add_plugins(connectome::ConnectomePlugin)
        .add_plugins(signal_scope::SignalScopePlugin)
        .add_plugins(neuron_inspector::NeuronInspectorPlugin)
        .add_plugins(timeline::TimelinePlugin)
        .add_plugins(log_console::LogConsolePlugin)
        .add_plugins(bake_panel::BakePanelPlugin)
        .add_plugins(shard_map::ShardMapPlugin)
        .add_plugins(io_matrix::IoMatrixPlugin)
        .add_plugins(hud::HudPlugin)
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
    commands.spawn((
        PointLight {
            intensity: 1500.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // 2. Декоративная сетка (Grid) для оценки масштаба
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.1, 0.15),
            unlit: true,
            ..default()
        })),
        Transform::IDENTITY,
    ));
}

/// Временная система для дебага: проверяем, что ECS видит спайки
fn debug_spike_events(mut events: EventReader<telemetry::SpikeFrame>) {
    for frame in events.read() {
        if !frame.spike_ids.is_empty() {
            println!(
                "IDE Render Tick: Received {} spikes from GPU Batch #{}",
                frame.spike_ids.len(),
                frame.tick
            );
        }
    }
}
