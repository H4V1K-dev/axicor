#![deny(warnings)]
#![deny(unused_variables)]
#![deny(dead_code)]
use bevy::prelude::*;
mod camera;
mod config;
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
        // Вставляем дефолтный конфиг ДО регистрации plugins,
        // чтобы parse_cli_config мог его перезаписать в PreStartup
        .insert_resource(config::IdeConfig::default())
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
        .add_systems(PreStartup, config::parse_cli_config)
        .add_systems(Startup, camera::setup_camera)
        .add_systems(Update, (debug_spike_events, camera::camera_controller))
        .run();
}


/// Временная система для дебага: проверяем, что ECS видит спайки
fn debug_spike_events(mut events: EventReader<telemetry::SpikeFrameEvent>) {
    for frame in events.read() {
        if !frame.spike_ids.is_empty() {
            println!(
                "IDE Render Tick: Received {} spikes",
                frame.spike_ids.len(),
            );
        }
    }
}
