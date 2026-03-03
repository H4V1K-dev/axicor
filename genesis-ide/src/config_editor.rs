use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseButton};
use bevy::input::ButtonInput;
use bevy::window::{CursorGrabMode, PrimaryWindow};
use serde::{Deserialize, Serialize};
use std::fs;
use crate::layout::{AreaBody, EditorType};

/// Отражение структуры blueprints.toml
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct BlueprintVariant {
    pub threshold: i32,
    pub rest_potential: i32,
    pub gsop_potentiation: i32,
    pub gsop_depression: i32,
    // Остальные поля добавим по мере парсинга
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct BlueprintsConfig {
    #[serde(rename = "neuron_type")]
    pub neuron_types: Vec<BlueprintVariant>,
}

/// Компонент-маркер для привязки UI-ноды к конкретному полю TOML
#[derive(Component)]
pub struct ConfigFieldBinding {
    pub type_idx: usize,
    pub field: ConfigFieldType,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ConfigFieldType {
    Threshold,
    RestPotential,
    GsopPotentiation,
    GsopDepression,
}

pub struct ConfigEditorPlugin;

impl Plugin for ConfigEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveDragState>()
            .insert_resource(ConfigWritePending {
                timer: Timer::from_seconds(0.5, TimerMode::Once),
                needs_write: false,
            })
            .add_systems(Startup, load_config_from_disk)
            .add_systems(Update, (
                periodic_config_reload,
                build_config_ui,
                handle_drag_start,
                process_drag_operator,
                sync_config_ui_values,
                flush_config_to_disk,
            ));
    }
}

#[derive(Resource)]
pub struct LoadedConfig {
    pub blueprints: Option<BlueprintsConfig>,
    pub path: String,
    pub last_modified: Option<std::time::SystemTime>,
}

impl Default for LoadedConfig {
    fn default() -> Self {
        Self {
            blueprints: None,
            path: "config/brains/V1/blueprints.toml".to_string(),
            last_modified: None,
        }
    }
}

/// Ресурс для отслеживания активного Drag-to-Edit
#[derive(Resource, Default)]
pub struct ActiveDragState {
    pub is_dragging: bool,
    pub target_type_idx: usize,
    pub target_field: Option<ConfigFieldType>,
    pub accumulated_delta: f32, // Накопленная дельта мыши для плавности
}

/// Флаг для Debouncer'а: пора записать конфиг на диск
#[derive(Resource)]
pub struct ConfigWritePending {
    pub timer: Timer,
    pub needs_write: bool,
}

fn load_config_from_disk(mut commands: Commands) {
    let path = "config/brains/V1/blueprints.toml";
    
    match fs::read_to_string(path) {
        Ok(content) => {
            match toml::from_str::<BlueprintsConfig>(&content) {
                Ok(config) => {
                    info!("[ConfigEditor] Loaded blueprints.toml: {} neuron types", config.neuron_types.len());
                    for (idx, variant) in config.neuron_types.iter().enumerate() {
                        info!("  Type {}: threshold={}, rest={}, potentiation={}, depression={}", 
                            idx, variant.threshold, variant.rest_potential, 
                            variant.gsop_potentiation, variant.gsop_depression);
                    }
                    
                    if let Ok(metadata) = fs::metadata(path) {
                        let modified = metadata.modified().ok();
                        commands.insert_resource(LoadedConfig {
                            blueprints: Some(config),
                            path: path.to_string(),
                            last_modified: modified,
                        });
                    }
                }
                Err(e) => {
                    error!("[ConfigEditor] Failed to parse blueprints.toml: {}", e);
                    commands.insert_resource(LoadedConfig::default());
                }
            }
        }
        Err(e) => {
            error!("[ConfigEditor] Failed to read blueprints.toml: {}", e);
            commands.insert_resource(LoadedConfig::default());
        }
    }
}

/// Периодическая проверка изменений файла (простой Hot-Reload)
fn periodic_config_reload(
    mut config: ResMut<LoadedConfig>,
) {
    if let Ok(metadata) = fs::metadata(&config.path) {
        if let Ok(modified) = metadata.modified() {
            // Проверяем если файл был изменён
            if let Some(last_modified) = config.last_modified {
                if modified > last_modified {
                    info!("[ConfigEditor] Detected file change, reloading...");
                    if let Ok(content) = fs::read_to_string(&config.path) {
                        if let Ok(new_config) = toml::from_str::<BlueprintsConfig>(&content) {
                            info!("[ConfigEditor] Reloaded: {} neuron types", new_config.neuron_types.len());
                            config.blueprints = Some(new_config);
                            config.last_modified = Some(modified);
                        }
                    }
                }
            } else {
                config.last_modified = Some(modified);
            }
        }
    }
}

/// Ловит новые панели ConfigEditor и строит внутри них UI
/// Zero-Cost: выполняется только один раз при Added<AreaBody>
pub fn build_config_ui(
    mut commands: Commands,
    q_bodies: Query<(Entity, &AreaBody), Added<AreaBody>>,
    config: Option<Res<LoadedConfig>>,
) {
    let Some(config) = config else { return };

    for (entity, body) in q_bodies.iter() {
        if body.0 != EditorType::ConfigEditor { continue; }

        commands.entity(entity).with_children(|parent| {
            // Корневой скроллируемый/тянущийся контейнер редактора
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    padding: UiRect::all(Val::Px(12.0)),
                    overflow: Overflow::clip_y(), // Подготовка к скроллу
                    ..default()
                },
            )).with_children(|content| {
                // Согласно 01_foundations.md §2, у нас максимум 16 типов (обычно 4).
                if let Some(blueprints) = &config.blueprints {
                    for (i, variant) in blueprints.neuron_types.iter().enumerate() {
                        spawn_variant_panel(content, i, variant);
                    }
                }
            });
        });
    }
}

fn spawn_variant_panel(parent: &mut ChildBuilder, idx: usize, variant: &BlueprintVariant) {
    parent.spawn((
        Node {
            flex_direction: FlexDirection::Column,
            width: Val::Percent(100.0),
            border: UiRect::all(Val::Px(1.0)),
            padding: UiRect::all(Val::Px(8.0)),
            margin: UiRect::bottom(Val::Px(10.0)),
            ..default()
        },
        BorderColor(Color::srgb(0.25, 0.25, 0.25)),
        BackgroundColor(Color::srgb(0.08, 0.08, 0.08)),
    )).with_children(|panel| {
        // Заголовок типа (Type 0, Type 1...)
        panel.spawn((
            Text::new(format!("Neuron Type {}", idx)),
            TextFont { font_size: 16.0, ..default() },
            TextColor(Color::srgb(0.9, 0.6, 0.1)),
        ));

        // Ряды ключ-значение
        spawn_property_row(panel, "Threshold", idx, ConfigFieldType::Threshold, variant.threshold);
        spawn_property_row(panel, "Rest Potential", idx, ConfigFieldType::RestPotential, variant.rest_potential);
        spawn_property_row(panel, "GSOP Potent.", idx, ConfigFieldType::GsopPotentiation, variant.gsop_potentiation);
        spawn_property_row(panel, "GSOP Depress.", idx, ConfigFieldType::GsopDepression, variant.gsop_depression);
    });
}

fn spawn_property_row(
    parent: &mut ChildBuilder, 
    label: &str, 
    type_idx: usize, 
    field_type: ConfigFieldType, 
    initial_val: i32
) {
    parent.spawn((
        Node {
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            width: Val::Percent(100.0),
            margin: UiRect::top(Val::Px(6.0)),
            ..default()
        },
    )).with_children(|row| {
        row.spawn((
            Text::new(label.to_string()),
            TextFont { font_size: 14.0, ..default() },
            TextColor(Color::srgb(0.6, 0.6, 0.6)),
        ));

        // Само значение. Вешаем на него биндинг и Interaction (для будущего драга)
        row.spawn((
            Text::new(initial_val.to_string()),
            TextFont { font_size: 14.0, ..default() },
            TextColor(Color::srgb(0.9, 0.9, 0.9)),
            ConfigFieldBinding { type_idx, field: field_type },
            Interaction::None, // Подготовка к Blender-style редактированию
        ));
    });
}

/// Система обновляет UI-текст при мутации ресурса LoadedConfig
/// Zero-Cost: не делаем ничего, пока файл не обновился (is_changed)
pub fn sync_config_ui_values(
    config: Res<LoadedConfig>,
    mut q_text: Query<(&mut Text, &ConfigFieldBinding)>,
) {
    // Zero-Cost: не делаем ничего, пока файл не обновился
    if !config.is_changed() { return; }

    let Some(blueprints) = &config.blueprints else { return; };

    for (mut text, binding) in q_text.iter_mut() {
        if let Some(variant) = blueprints.neuron_types.get(binding.type_idx) {
            let new_val = match binding.field {
                ConfigFieldType::Threshold => variant.threshold,
                ConfigFieldType::RestPotential => variant.rest_potential,
                ConfigFieldType::GsopPotentiation => variant.gsop_potentiation,
                ConfigFieldType::GsopDepression => variant.gsop_depression,
            };

            // Избегаем аллокации строк впустую
            let new_str = new_val.to_string();
            if text.0 != new_str {
                text.0 = new_str;
            }
        }
    }
}

/// Ловит зажатие ЛКМ на текстовой ноде с биндингом
pub fn handle_drag_start(
    q_interactions: Query<(&Interaction, &ConfigFieldBinding)>,
    mut drag_state: ResMut<ActiveDragState>,
    mut q_windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    if drag_state.is_dragging { return; } // Уже тянем

    for (interaction, binding) in q_interactions.iter() {
        if *interaction == Interaction::Pressed {
            drag_state.is_dragging = true;
            drag_state.target_type_idx = binding.type_idx;
            drag_state.target_field = Some(binding.field);
            drag_state.accumulated_delta = 0.0;

            // Прячем курсор и лочим его, как в Blender
            if let Ok(mut window) = q_windows.get_single_mut() {
                window.cursor_options.visible = false;
                window.cursor_options.grab_mode = CursorGrabMode::Locked;
            }
            return;
        }
    }
}

/// Выполняется каждый кадр. Переводит дельту мыши в мутацию TOML-данных
pub fn process_drag_operator(
    mut drag_state: ResMut<ActiveDragState>,
    mut mouse_motion: EventReader<MouseMotion>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut config: ResMut<LoadedConfig>,
    mut q_windows: Query<&mut Window, With<PrimaryWindow>>,
    mut pending_write: ResMut<ConfigWritePending>,
) {
    if !drag_state.is_dragging { return; }

    // Отпускание ЛКМ -> выход из режима
    if mouse.just_released(MouseButton::Left) {
        drag_state.is_dragging = false;
        drag_state.target_field = None;

        if let Ok(mut window) = q_windows.get_single_mut() {
                window.cursor_options.visible = true;
                window.cursor_options.grab_mode = CursorGrabMode::None;
        }
        
        // Триггерим запись на диск
        pending_write.needs_write = true;
        pending_write.timer.reset();
        return;
    }

    // Читаем сырое движение мыши
    let mut delta_x = 0.0;
    for ev in mouse_motion.read() {
        delta_x += ev.delta.x;
    }

    if delta_x != 0.0 {
        // Sensitvity. Накапливаем дельту. Каждые N пикселей = ±1 к значению
        drag_state.accumulated_delta += delta_x * 0.5; 
        
        if drag_state.accumulated_delta.abs() >= 1.0 {
            let step = drag_state.accumulated_delta.trunc() as i32;
            drag_state.accumulated_delta -= drag_state.accumulated_delta.trunc();

            if let Some(blueprints) = config.blueprints.as_mut() {
                if let Some(variant) = blueprints.neuron_types.get_mut(drag_state.target_type_idx) {
                    if let Some(target_field) = drag_state.target_field {
                        // Мутируем данные. Bevy автоматически пометит ResMut<LoadedConfig> как changed,
                        // что вызовет твою систему `sync_config_ui_values` для обновления текста!
                        match target_field {
                            ConfigFieldType::Threshold => variant.threshold += step,
                            ConfigFieldType::RestPotential => variant.rest_potential += step,
                            ConfigFieldType::GsopPotentiation => variant.gsop_potentiation += step,
                            ConfigFieldType::GsopDepression => variant.gsop_depression += step,
                        }
                    }
                }
            }
        }
    }
}

/// Сбрасывает изменённый конфиг обратно в TOML без блокировки UI-потока надолго
pub fn flush_config_to_disk(
    time: Res<Time>,
    mut pending_write: ResMut<ConfigWritePending>,
    config: Res<LoadedConfig>,
) {
    if !pending_write.needs_write { return; }

    pending_write.timer.tick(time.delta());
    if pending_write.timer.finished() {
        // Сериализуем обратно в TOML
        if let Some(blueprints) = &config.blueprints {
            if let Ok(toml_string) = toml::to_string(blueprints) {
                // В идеале вынести в IoTaskPool, но для TOML-файла размером в 2КБ 
                // fs::write занимает микросекунды и допустим в Main Thread.
                if let Err(e) = fs::write(&config.path, toml_string) {
                    error!("[ConfigEditor] Failed to save blueprints.toml: {}", e);
                } else {
                    info!("[ConfigEditor] Saved blueprints.toml to disk.");
                }
            }
        }
        pending_write.needs_write = false;
    }
}
