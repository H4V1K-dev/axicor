use bevy::prelude::*;
use std::collections::VecDeque;
use crate::{
    hud::SelectionState,
    layout::{AreaBody, EditorType},
    telemetry::SpikeFrame,
    world::GlobalSpikeMap,
};

const HISTORY_FRAMES: usize = 60; // Храним историю за последние 60 кадров (~1 секунда)

/// Ресурс-агрегатор для осциллографа
#[derive(Resource)]
pub struct ScopeHistory {
    /// Кольцевой буфер. Каждый элемент — массив из 16 счетчиков спайков (по типам)
    pub frames: VecDeque<[u32; 16]>,
    /// Динамический максимум для нормализации высоты столбцов
    pub peak_spikes: u32, 
}

impl Default for ScopeHistory {
    fn default() -> Self {
        let mut frames = VecDeque::with_capacity(HISTORY_FRAMES);
        for _ in 0..HISTORY_FRAMES {
            frames.push_back([0; 16]);
        }
        Self { frames, peak_spikes: 1 }
    }
}

/// Компонент-маркер для столбца графика конкретного типа
#[derive(Component)]
pub struct ScopeBar {
    pub type_idx: usize,
}

pub struct SignalScopePlugin;

impl Plugin for SignalScopePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScopeHistory>()
           .add_systems(Update, (
               aggregate_spike_data,
               build_scope_ui,
               update_scope_ui,
           ).chain());
    }
}

/// Агрегация спайков в скользящее окно истории
fn aggregate_spike_data(
    mut events: EventReader<SpikeFrame>,
    spike_map: Option<Res<GlobalSpikeMap>>,
    selection: Res<SelectionState>,
    mut scope: ResMut<ScopeHistory>,
) {
    let Some(global_map) = spike_map else { return };

    let mut current_frame_counts = [0u32; 16];
    let mut has_spikes = false;

    // Режим фильтрации: если популяция выделена, строим O(1) Lookup Table (Dense Mask).
    let filter_mask = if !selection.selected_neurons.is_empty() {
        let mut mask = vec![false; global_map.map.len()];
        for (dense_id, route) in global_map.map.iter().enumerate() {
            if selection.selected_neurons.contains(&(route.type_id, route.local_idx)) {
                mask[dense_id] = true;
            }
        }
        Some(mask)
    } else {
        None
    };

    // Считаем спайки текущего кадра
    for ev in events.read() {
        for &dense_id in &ev.spike_ids {
            // Если включен фильтр, дропаем спайки вне выделения
            if let Some(mask) = &filter_mask {
                if !mask.get(dense_id as usize).copied().unwrap_or(false) {
                    continue;
                }
            }

            if let Some(route) = global_map.map.get(dense_id as usize) {
                current_frame_counts[route.type_id as usize] += 1;
                has_spikes = true;
            }
        }
    }

    if has_spikes || !scope.frames.is_empty() {
        scope.frames.pop_front();
        scope.frames.push_back(current_frame_counts);

        // Обновляем peak_spikes для автомасштабирования графика
        scope.peak_spikes = scope.frames.iter()
            .flat_map(|f| f.iter())
            .max()
            .copied()
            .unwrap_or(1)
            .max(10); // Минимальный потолок, чтобы график не скакал от 1 спайка
    }
}

/// Построение UI-инспектора осциллографа (раз в появление панели)
fn build_scope_ui(
    mut commands: Commands,
    q_bodies: Query<(Entity, &AreaBody), Added<AreaBody>>,
) {
    for (entity, body) in q_bodies.iter() {
        if body.0 != EditorType::SignalScope { continue; }

        commands.entity(entity).with_children(|parent| {
            // Контейнер графика
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::FlexEnd, // Выравнивание столбцов по низу
                    justify_content: JustifyContent::SpaceEvenly,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    padding: UiRect::all(Val::Px(10.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.05, 0.05, 0.05)),
            )).with_children(|chart| {
                // Создаем 16 столбцов для 16 типов нейронов
                for i in 0..16 {
                    chart.spawn((
                        Node {
                            width: Val::Percent(5.0),
                            height: Val::Percent(0.0), // Высота будет меняться динамически
                            ..default()
                        },
                        // Используем ту же палитру, что и в 3D (HSL)
                        BackgroundColor(Color::hsl((i as f32) * 22.5, 0.8, 0.5)),
                        ScopeBar { type_idx: i },
                    ));
                }
            });
        });
    }
}

/// Zero-Cost обновление высоты столбцов графика
fn update_scope_ui(
    scope: Res<ScopeHistory>,
    mut q_bars: Query<(&mut Node, &ScopeBar)>,
) {
    // Берем данные последнего (самого свежего) кадра
    let Some(latest_frame) = scope.frames.back() else { return };
    let peak = scope.peak_spikes as f32;

    for (mut node, bar) in q_bars.iter_mut() {
        let count = latest_frame[bar.type_idx] as f32;
        let percentage = (count / peak) * 100.0;
        
        // Мутируем высоту столбца через Style. Bevy UI сам перерисует layout.
        node.height = Val::Percent(percentage);
    }
}

