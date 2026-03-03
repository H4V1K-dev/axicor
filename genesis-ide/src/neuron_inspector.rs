use bevy::prelude::*;
use crate::{
    hud::SelectionState,
    layout::{AreaBody, EditorType},
    config_editor::LoadedConfig,
    world::NeuronLayerData,
};

#[derive(Component)]
pub struct InspectorDataBinding;

pub struct NeuronInspectorPlugin;

impl Plugin for NeuronInspectorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            build_inspector_ui,
            update_inspector_ui,
        ).chain());
    }
}

/// Строит UI один раз при открытии панели
fn build_inspector_ui(
    mut commands: Commands,
    q_bodies: Query<(Entity, &AreaBody), Added<AreaBody>>,
) {
    for (entity, body) in q_bodies.iter() {
        if body.0 != EditorType::NeuronInspector { continue; }

        commands.entity(entity).with_children(|parent| {
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    padding: UiRect::all(Val::Px(15.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.08, 0.08, 0.08)),
            )).with_children(|panel| {
                panel.spawn((
                    Text::new("No neuron selected.\nClick on a sphere in 3D View."),
                    TextFont { font_size: 16.0, ..default() },
                    TextColor(Color::srgb(0.8, 0.8, 0.8)),
                    InspectorDataBinding, // Маркер для Hot Loop обновления
                ));
            });
        });
    }
}

/// Zero-Cost обновление. Выполняется только если изменился SelectionState.
fn update_inspector_ui(
    selection: Res<SelectionState>,
    config: Option<Res<LoadedConfig>>,
    q_layers: Query<&NeuronLayerData>,
    mut q_text: Query<&mut Text, With<InspectorDataBinding>>,
) {
    // Встроенный Change Detection Bevy. Если пользователь не кликал, выходим.
    if !selection.is_changed() { return; }

    let Some(config) = config else { return };
    let Some(blueprints) = &config.blueprints else { return };
    let Ok(mut text) = q_text.get_single_mut() else { return };

    // Берём первый нейрон из выделения (MVP для популяций)
    if let Some(&(t_id, l_idx)) = selection.selected_neurons.first() {
        // Достаем PackedPosition из VRAM snapshot'а (наш слой)
        let mut packed_pos = 0;
        for layer in q_layers.iter() {
            if layer.type_id == t_id {
                if let Some(instance) = layer.instances.get(l_idx as usize) {
                    packed_pos = instance.packed_pos;
                }
                break;
            }
        }

        // Декодируем координаты
        let x = (packed_pos & 0x7FF) as f32 * 25.0;
        let y = ((packed_pos >> 11) & 0x7FF) as f32 * 25.0;
        let z = ((packed_pos >> 22) & 0x3FF) as f32 * 25.0;

        // Берем физические параметры из загруженного Blueprint
        let profile = blueprints.neuron_types.get(t_id as usize);
        let thresh = profile.map_or(0, |p| p.threshold);
        let rest = profile.map_or(0, |p| p.rest_potential);

        let new_str = format!(
            "=== NEURON INSPECTOR ===\n\n\
            Local Index: {}\n\
            Type Index:  {} (4-bit mask)\n\n\
            [ Spatial Data ]\n\
            Position:    X:{:.1}  Y:{:.1}  Z:{:.1} um\n\
            Packed:      0x{:08X}\n\n\
            [ Membrane Physics (Live Config) ]\n\
            Threshold:   {} mV\n\
            Rest Pot.:   {} mV\n",
            l_idx, t_id, x, y, z, packed_pos, thresh, rest
        );

        text.0 = new_str;
    } else {
        text.0 = "No neuron selected.\nClick on a sphere in 3D View.".to_string();
    }
}

