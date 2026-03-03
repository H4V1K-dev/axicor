use bevy::{
    prelude::*,
    tasks::IoTaskPool,
};
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use crate::{
    layout::{AreaBody, EditorType},
    log_console::SystemLog,
};

#[derive(Resource)]
pub struct BakeState {
    pub is_baking: bool,
    pub log_rx: Receiver<String>,
    pub log_tx: Sender<String>,
}

impl Default for BakeState {
    fn default() -> Self {
        let (tx, rx) = unbounded();
        Self {
            is_baking: false,
            log_rx: rx,
            log_tx: tx,
        }
    }
}

#[derive(Component)]
pub struct BakeButton;

pub struct BakePanelPlugin;

impl Plugin for BakePanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BakeState>()
           .add_systems(Update, (
               build_bake_ui,
               handle_bake_button,
               poll_bake_logs,
           ).chain());
    }
}

fn build_bake_ui(
    mut commands: Commands,
    q_bodies: Query<(Entity, &AreaBody), Added<AreaBody>>,
) {
    for (entity, body) in q_bodies.iter() {
        if body.0 != EditorType::BakePanel { continue; }

        commands.entity(entity).with_children(|parent| {
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.08, 0.08, 0.08)),
            )).with_children(|container| {
                container.spawn((
                    Node {
                        width: Val::Px(200.0),
                        height: Val::Px(50.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(2.0)),
                        ..default()
                    },
                    BorderColor(Color::srgb(0.9, 0.4, 0.1)),
                    BackgroundColor(Color::srgb(0.15, 0.05, 0.0)),
                    Interaction::None,
                    BakeButton,
                )).with_children(|btn| {
                    btn.spawn((
                        Text::new("RUN BAKER"),
                        TextFont { font_size: 18.0, ..default() },
                        TextColor(Color::srgb(0.9, 0.5, 0.2)),
                    ));
                });
            });
        });
    }
}

fn handle_bake_button(
    mut q_interactions: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor, &Children),
        (Changed<Interaction>, With<BakeButton>),
    >,
    mut q_text: Query<&mut Text>,
    mut state: ResMut<BakeState>,
) {
    for (interaction, mut bg_color, mut border, children) in q_interactions.iter_mut() {
        if state.is_baking {
            *bg_color = Color::srgb(0.1, 0.1, 0.1).into();
            *border = Color::srgb(0.3, 0.3, 0.3).into();
            for &child in children.iter() {
                if let Ok(mut text) = q_text.get_mut(child) {
                    text.0 = "BAKING...".to_string();
                    break;
                }
            }
            continue;
        }

        match *interaction {
            Interaction::Pressed => {
                state.is_baking = true;
                *bg_color = Color::srgb(0.3, 0.1, 0.0).into();

                let tx = state.log_tx.clone();
                let pool = IoTaskPool::get();

                pool.spawn(async move {
                    let _ = tx.send("[Baker] Starting compilation pipeline...".to_string());

                    let mut child = match Command::new("cargo")
                        .args(["run", "-p", "genesis-baker"])
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx.send(format!("[Baker] Failed to spawn: {}", e));
                            let _ = tx.send("BAKE_COMPLETE_ERROR".to_string());
                            return;
                        }
                    };

                    if let Some(stdout) = child.stdout.take() {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines().map_while(Result::ok) {
                            let _ = tx.send(format!("[Baker] {}", line));
                        }
                    }

                    if let Ok(status) = child.wait() {
                        if status.success() {
                            let _ = tx.send("BAKE_COMPLETE_SUCCESS".to_string());
                        } else {
                            let _ = tx.send("BAKE_COMPLETE_ERROR".to_string());
                        }
                    } else {
                        let _ = tx.send("BAKE_COMPLETE_ERROR".to_string());
                    }
                }).detach();
            }
            Interaction::Hovered => {
                *bg_color = Color::srgb(0.25, 0.1, 0.0).into();
            }
            Interaction::None => {
                *bg_color = Color::srgb(0.15, 0.05, 0.0).into();
                *border = Color::srgb(0.9, 0.4, 0.1).into();
                for &child in children.iter() {
                    if let Ok(mut text) = q_text.get_mut(child) {
                        text.0 = "RUN BAKER".to_string();
                        break;
                    }
                }
            }
        }
    }
}

fn poll_bake_logs(
    mut state: ResMut<BakeState>,
    mut ev_log: EventWriter<SystemLog>,
) {
    // Собираем сообщения в локальный буфер, чтобы не держать заимствование state
    let mut messages: Vec<String> = state.log_rx.try_iter().collect();

    for msg in messages.drain(..) {
        if msg == "BAKE_COMPLETE_SUCCESS" {
            state.is_baking = false;
            ev_log.send(SystemLog {
                message: "[Baker] Pipeline finished successfully.".into(),
                is_error: false,
            });
        } else if msg == "BAKE_COMPLETE_ERROR" {
            state.is_baking = false;
            ev_log.send(SystemLog {
                message: "[Baker] Pipeline failed! Check logs.".into(),
                is_error: true,
            });
        } else {
            let lower = msg.to_lowercase();
            let is_err = lower.contains("error") || lower.contains("panic");
            ev_log.send(SystemLog {
                message: msg,
                is_error: is_err,
            });
        }
    }
}

