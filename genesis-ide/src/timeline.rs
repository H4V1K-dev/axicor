use bevy::prelude::*;
use crate::{
    layout::{AreaBody, EditorType},
    telemetry::SpikeFrame,
    log_console::SystemLog,
};

/// Ресурс, хранящий локальное представление времени симуляции
#[derive(Resource, Default)]
pub struct SimulationTime {
    pub current_tick: u64,
    pub is_paused: bool, // Задел для отправки управляющих команд в Runtime
}

/// Маркер для текста с текущим тиком
#[derive(Component)]
pub struct TimelineTickBinding;

/// Маркер для кнопки Play/Pause
#[derive(Component)]
pub struct PlayPauseButton;

pub struct TimelinePlugin;

impl Plugin for TimelinePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationTime>()
            .add_systems(Update, (
                build_timeline_ui,
                update_simulation_time,
                sync_timeline_ui,
                handle_timeline_controls,
            ).chain());
    }
}

/// Построение Blender-like UI. Выполняется один раз при появлении панели Timeline.
fn build_timeline_ui(
    mut commands: Commands,
    q_bodies: Query<(Entity, &AreaBody), Added<AreaBody>>,
) {
    for (entity, body) in q_bodies.iter() {
        if body.0 != EditorType::Timeline { continue; }

        commands.entity(entity).with_children(|parent| {
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    padding: UiRect::all(Val::Px(10.0)),
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.12, 0.12, 0.12)),
            )).with_children(|row| {
                // Кнопка Play/Pause (UI-заглушка для будущего Control Socket'а)
                row.spawn((
                    Node {
                        width: Val::Px(30.0),
                        height: Val::Px(30.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        margin: UiRect::right(Val::Px(15.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor(Color::srgb(0.3, 0.3, 0.3)),
                    BackgroundColor(Color::srgb(0.18, 0.18, 0.18)),
                    Interaction::None,
                    PlayPauseButton,
                )).with_children(|btn| {
                    btn.spawn((
                        Text::new("▶"),
                        TextFont { font_size: 16.0, ..default() },
                        TextColor(Color::srgb(0.8, 0.8, 0.8)),
                    ));
                });

                // Текст: Глобальный Tick
                row.spawn((
                    Text::new("Tick: 0"),
                    TextFont { font_size: 15.0, ..default() },
                    TextColor(Color::srgb(0.9, 0.9, 0.9)),
                    TimelineTickBinding,
                ));
                
                // Track (Полоса прокрутки времени / визуализация батчей)
                row.spawn((
                    Node {
                        width: Val::Percent(100.0), // Занимает всё оставшееся место
                        height: Val::Px(4.0),
                        margin: UiRect::left(Val::Px(20.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.25, 0.25, 0.25)),
                ));
            });
        });
    }
}

/// Читает стрим телеметрии и обновляет часы. Сложность O(N_frames).
fn update_simulation_time(
    mut events: EventReader<SpikeFrame>,
    mut sim_time: ResMut<SimulationTime>,
) {
    let mut latest_tick = None;
    for ev in events.read() {
        latest_tick = Some(ev.tick);
    }
    
    // Мутируем ресурс только если тик реально вырос (избегаем ложных Change Detection)
    if let Some(tick) = latest_tick {
        if tick > sim_time.current_tick {
            sim_time.current_tick = tick;
        }
    }
}

/// Обновляет UI только при реальном изменении времени
fn sync_timeline_ui(
    sim_time: Res<SimulationTime>,
    mut q_text: Query<&mut Text, With<TimelineTickBinding>>,
) {
    if !sim_time.is_changed() { return; }

    for mut text in q_text.iter_mut() {
        let new_str = format!("Tick: {}", sim_time.current_tick);
        if text.0 != new_str {
            text.0 = new_str;
        }
    }
}

pub fn handle_timeline_controls(
    mut q_interactions: Query<
        (&Interaction, &mut BackgroundColor, &Children),
        (Changed<Interaction>, With<PlayPauseButton>),
    >,
    mut q_text: Query<&mut Text>,
    mut sim_time: ResMut<SimulationTime>,
    mut ev_log: EventWriter<SystemLog>,
) {
    for (interaction, mut bg_color, children) in q_interactions.iter_mut() {
        match *interaction {
            Interaction::Pressed => {
                *bg_color = Color::srgb(0.3, 0.3, 0.3).into();
                sim_time.is_paused = !sim_time.is_paused;

                // Меняем иконку
                for &child in children.iter() {
                    if let Ok(mut text) = q_text.get_mut(child) {
                        let new_str = if sim_time.is_paused { "▶" } else { "⏸" };
                        text.0 = new_str.to_string();
                        break;
                    }
                }

                let state_str = if sim_time.is_paused { "PAUSED" } else { "RESUMED" };
                ev_log.send(SystemLog {
                    message: format!("[Control] Simulation state changed to: {}", state_str),
                    is_error: false,
                });
            }
            Interaction::Hovered => {
                *bg_color = Color::srgb(0.25, 0.25, 0.25).into();
            }
            Interaction::None => {
                *bg_color = Color::srgb(0.18, 0.18, 0.18).into();
            }
        }
    }
}


