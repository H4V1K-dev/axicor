use bevy::prelude::*;
use crate::layout::{AreaBody, EditorType};

#[allow(dead_code)]
#[derive(Component)]
pub struct ShardUiNode {
    pub shard_id: usize,
}

pub struct ShardMapPlugin;

impl Plugin for ShardMapPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, build_shard_map_ui);
    }
}

/// Строит 2D карту шардов при спавне панели
fn build_shard_map_ui(
    mut commands: Commands,
    q_bodies: Query<(Entity, &AreaBody), Added<AreaBody>>,
) {
    for (entity, body) in q_bodies.iter() {
        if body.0 != EditorType::ShardMap { continue; }

        commands.entity(entity).with_children(|parent| {
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    display: Display::Grid,
                    // Для MVP делаем жесткую сетку 2x2 (4 шарда).
                    // В будущем будем брать размеры из топологии (brain.toml)
                    grid_template_columns: vec![GridTrack::fr(1.0), GridTrack::fr(1.0)],
                    grid_template_rows: vec![GridTrack::fr(1.0), GridTrack::fr(1.0)],
                    padding: UiRect::all(Val::Px(10.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.05, 0.05, 0.05)),
            )).with_children(|grid| {
                for i in 0..4 {
                    grid.spawn((
                        Node {
                            width: Val::Percent(100.0),
                            height: Val::Percent(100.0),
                            border: UiRect::all(Val::Px(2.0)),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        BorderColor(Color::srgb(0.2, 0.2, 0.2)),
                        // Зеленый цвет = шард загружен и активен
                        BackgroundColor(Color::srgb(0.1, 0.3, 0.1)),
                        ShardUiNode { shard_id: i },
                    )).with_children(|cell| {
                        cell.spawn((
                            Text::new(format!("Shard {:02}\nStatus: ONLINE", i)),
                            TextFont { font_size: 14.0, ..default() },
                            TextColor(Color::srgb(0.8, 0.9, 0.8)),
                        ));
                    });
                }
            });
        });
    }
}

