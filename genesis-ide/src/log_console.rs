use bevy::prelude::*;
use std::collections::VecDeque;
use crate::layout::{AreaBody, EditorType};

/// Событие для проброса логов из любых систем (например, загрузчика сети)
#[derive(Event)]
pub struct SystemLog {
    pub message: String,
    pub is_error: bool,
}

#[derive(Resource)]
pub struct LogHistory {
    pub lines: VecDeque<(String, bool)>,
    pub capacity: usize,
}

impl Default for LogHistory {
    fn default() -> Self {
        Self {
            lines: VecDeque::with_capacity(100),
            capacity: 100,
        }
    }
}

#[derive(Component)]
pub struct LogScrollContainer;

pub struct LogConsolePlugin;

impl Plugin for LogConsolePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LogHistory>()
           .add_event::<SystemLog>()
           .add_systems(Update, (
               receive_logs,
               build_log_ui,
               sync_log_ui,
           ).chain());
    }
}

/// O(1) добавление логов в кольцевой буфер
fn receive_logs(
    mut events: EventReader<SystemLog>,
    mut history: ResMut<LogHistory>,
) {
    let mut _changed = false;
    for ev in events.read() {
        if history.lines.len() >= history.capacity {
            history.lines.pop_front();
        }
        history.lines.push_back((ev.message.clone(), ev.is_error));
        _changed = true;
    }
    // Если были изменения, ResMut<LogHistory> уже помечен как изменённый
}

/// Строит контейнер консоли при сплите
fn build_log_ui(
    mut commands: Commands,
    q_bodies: Query<(Entity, &AreaBody), Added<AreaBody>>,
) {
    for (entity, body) in q_bodies.iter() {
        if body.0 != EditorType::LogConsole { continue; }

        commands.entity(entity).with_children(|parent| {
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    padding: UiRect::all(Val::Px(8.0)),
                    overflow: Overflow::clip_y(),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.06, 0.06, 0.06)),
                LogScrollContainer,
            ));
        });
    }
}

/// Перестраивает текст ТОЛЬКО если пришел новый лог
fn sync_log_ui(
    mut commands: Commands,
    history: Res<LogHistory>,
    q_containers: Query<Entity, With<LogScrollContainer>>,
) {
    if !history.is_changed() { return; }

    for container_entity in q_containers.iter() {
        // Удаляем старые строки
        commands.entity(container_entity).despawn_descendants();
        
        // Спавним новые
        commands.entity(container_entity).with_children(|parent| {
            for (msg, is_error) in history.lines.iter() {
                let color = if *is_error {
                    Color::srgb(0.9, 0.3, 0.3)
                } else {
                    Color::srgb(0.7, 0.7, 0.7)
                };

                parent.spawn((
                    Text::new(msg.clone()),
                    TextFont { font_size: 13.0, ..default() },
                    TextColor(color),
                    Node {
                        margin: UiRect::bottom(Val::Px(4.0)),
                        ..default()
                    },
                ));
            }
        });
    }
}

