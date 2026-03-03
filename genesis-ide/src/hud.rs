use bevy::{
    prelude::*,
    window::PrimaryWindow,
};
use crate::{
    camera::IdeCamera,
    world::NeuronLayerData,
};

/// Состояние выделенной популяции
#[derive(Resource, Default)]
pub struct SelectionState {
    // Храним векторы (Type ID, Local Index) для O(1) инспекции
    pub selected_neurons: Vec<(u8, u32)>, 
}

/// Машина состояний Blender-оператора "B"
#[derive(Resource, Default)]
pub struct BoxSelectTool {
    pub is_active: bool,
    pub start_pos: Vec2,
    pub current_pos: Vec2,
}

/// Маркер для UI-прямоугольника выделения
#[derive(Component)]
pub struct BoxSelectUi;

#[derive(Component)]
pub struct HudOverlay;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SelectionState>()
            .init_resource::<BoxSelectTool>()
            .add_systems(Startup, setup_hud)
            .add_systems(Update, (update_hud_text, handle_picking, handle_box_select).chain());
    }
}

fn setup_hud(mut commands: Commands) {
    // Text API для Bevy 0.15: одна строка с font/color компонентами
    commands.spawn((
        Text::new("Pos: 0.0, 0.0, 0.0\nChunk: [0, 0, 0]\nNeuron: None"),
        TextFont { font_size: 16.0, ..default() },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(15.0),
            left: Val::Px(15.0),
            ..default()
        },
        HudOverlay,
    ));

    // Скрытая рамка Box Select
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            display: Display::None, // Скрыта по умолчанию
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BorderColor(Color::srgb(0.8, 0.8, 1.0)),
        BackgroundColor(Color::srgba(0.3, 0.5, 0.8, 0.2)), // Полупрозрачный синий
        BoxSelectUi,
    ));
}

/// Zero-Cost Brute-force Raycast. Выполняется ТОЛЬКО при клике ЛКМ.
fn handle_picking(
    mouse: Res<ButtonInput<MouseButton>>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
    q_camera: Query<(&Camera, &GlobalTransform), With<IdeCamera>>,
    q_layers: Query<&NeuronLayerData>,
    mut selection: ResMut<SelectionState>,
) {
    // Обрабатываем только при клике ЛКМ
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = q_windows.get_single() else { return };
    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok((camera, cam_transform)) = q_camera.get_single() else { return };

    // Транслируем 2D курсор в 3D луч
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else { return };

    let mut closest_dist = f32::MAX;
    let mut best_match = None;
    let radius_sq = 0.5 * 0.5; // Радиус сферы 0.5

    // O(N) проход по всем батчам. Для 500k это доли миллисекунды.
    for layer in q_layers.iter() {
        for (local_idx, instance) in layer.instances.iter().enumerate() {
            // Распаковка PackedPosition (Voxel Size = 25.0)
            let x = (instance.packed_pos & 0x7FF) as f32 * 25.0;
            let y = ((instance.packed_pos >> 11) & 0x7FF) as f32 * 25.0;
            let z = ((instance.packed_pos >> 22) & 0x3FF) as f32 * 25.0;
            let center = Vec3::new(x, y, z);

            // Векторная математика пересечения луча и сферы
            let l = center - ray.origin;
            let tca = l.dot(ray.direction.normalize());
            
            if tca < 0.0 { continue; } // Сфера позади камеры
            
            let d2 = l.length_squared() - tca * tca;
            if d2 > radius_sq { continue; } // Луч прошел мимо

            let dist = tca - (radius_sq - d2).sqrt();
            if dist < closest_dist {
                closest_dist = dist;
                best_match = Some((layer.type_id, local_idx as u32));
            }
        }
    }

    selection.selected_neurons.clear();
    if let Some((t_id, l_idx)) = best_match {
        selection.selected_neurons.push((t_id, l_idx));
    }
}

/// Обновление текста HUD. Выполняется каждый кадр.
fn update_hud_text(
    q_camera: Query<&Transform, With<IdeCamera>>,
    selection: Res<SelectionState>,
    mut q_text: Query<&mut Text, With<HudOverlay>>,
) {
    let Ok(cam_transform) = q_camera.get_single() else { return };
    let Ok(mut text) = q_text.get_single_mut() else { return };

    let pos = cam_transform.translation;
    // Размер чанка 10x10x10 вокселей (250x250x250 мкм)
    let chunk_x = (pos.x / 250.0).floor() as i32;
    let chunk_y = (pos.y / 250.0).floor() as i32;
    let chunk_z = (pos.z / 250.0).floor() as i32;

    let selection_str = if selection.selected_neurons.is_empty() {
        "None".to_string()
    } else if selection.selected_neurons.len() == 1 {
        let (t, idx) = selection.selected_neurons[0];
        format!("Type: {}, Local ID: {}", t, idx)
    } else {
        format!("Selected: {} neurons", selection.selected_neurons.len())
    };

    // Обновление текста в первой секции
    let new_text = format!(
        "Pos: {:.1}, {:.1}, {:.1}\nChunk: [{}, {}, {}]\nNeuron: {}",
        pos.x, pos.y, pos.z,
        chunk_x, chunk_y, chunk_z,
        selection_str
    );
    
    text.0 = new_text;
}

/// Box Select (Blender-style). Тяжелая математика только при отпускании ЛКМ.
fn handle_box_select(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    q_windows: Query<&Window, With<PrimaryWindow>>,
    q_camera: Query<(&Camera, &GlobalTransform), With<IdeCamera>>,
    q_layers: Query<&NeuronLayerData>,
    mut tool: ResMut<BoxSelectTool>,
    mut selection: ResMut<SelectionState>,
    mut q_box_ui: Query<&mut Node, With<BoxSelectUi>>,
) {
    let Ok(window) = q_windows.get_single() else { return };
    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok(mut box_ui) = q_box_ui.get_single_mut() else { return };

    // Активация инструмента по 'B' (как в Blender)
    if keys.just_pressed(KeyCode::KeyB) {
        tool.is_active = true;
        info!("Box Select tool activated.");
    }

    if !tool.is_active { return; }

    // Начало рисования рамки
    if mouse.just_pressed(MouseButton::Left) {
        tool.start_pos = cursor_pos;
        tool.current_pos = cursor_pos;
        box_ui.display = Display::Flex;
    }

    // Обновление рамки каждый кадр (Zero-Cost для физики)
    if mouse.pressed(MouseButton::Left) {
        tool.current_pos = cursor_pos;
        
        let min_x = tool.start_pos.x.min(tool.current_pos.x);
        let max_x = tool.start_pos.x.max(tool.current_pos.x);
        let min_y = tool.start_pos.y.min(tool.current_pos.y);
        let max_y = tool.start_pos.y.max(tool.current_pos.y);

        box_ui.left = Val::Px(min_x);
        box_ui.top = Val::Px(min_y);
        box_ui.width = Val::Px(max_x - min_x);
        box_ui.height = Val::Px(max_y - min_y);
    }

    // Завершение выделения: O(N) проход по нейронам выполняется один раз
    if mouse.just_released(MouseButton::Left) {
        box_ui.display = Display::None;
        tool.is_active = false; // Выход из режима

        let min_x = tool.start_pos.x.min(tool.current_pos.x);
        let max_x = tool.start_pos.x.max(tool.current_pos.x);
        let min_y = tool.start_pos.y.min(tool.current_pos.y);
        let max_y = tool.start_pos.y.max(tool.current_pos.y);

        let Ok((camera, cam_transform)) = q_camera.get_single() else { return };
        let mut new_selection = Vec::with_capacity(1000);

        // Проходим по всем сырым VRAM-батчам
        for layer in q_layers.iter() {
            for (local_idx, instance) in layer.instances.iter().enumerate() {
                // Распаковка PackedPosition (Voxel Size = 25.0 um)
                let x = (instance.packed_pos & 0x7FF) as f32 * 25.0;
                let y = ((instance.packed_pos >> 11) & 0x7FF) as f32 * 25.0;
                let z = ((instance.packed_pos >> 22) & 0x3FF) as f32 * 25.0;
                let world_pos = Vec3::new(x, y, z);

                // Если нейрон сзади камеры / вне вьюпорта, получим Err
                if let Ok(screen_pos) = camera.world_to_viewport(cam_transform, world_pos) {
                    // 2D AABB Intersection
                    if screen_pos.x >= min_x && screen_pos.x <= max_x && 
                       screen_pos.y >= min_y && screen_pos.y <= max_y {
                        new_selection.push((layer.type_id, local_idx as u32));
                    }
                }
            }
        }

        info!("Box Select found {} neurons.", new_selection.len());
        selection.selected_neurons = new_selection;
    }
}


