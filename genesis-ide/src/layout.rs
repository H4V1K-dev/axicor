use bevy::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

// Глобальный генератор ID для панелей (Lock-Free)
static AREA_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

pub fn generate_area_id() -> u32 {
    AREA_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EditorType {
    #[default]
    WorldView,
    ConfigEditor,
    Timeline,
    SignalScope,
    NeuronInspector,
    LogConsole,
    BakePanel,
    ShardMap,
}

impl EditorType {
    pub fn next(&self) -> Self {
        match self {
            EditorType::WorldView => EditorType::ConfigEditor,
            EditorType::ConfigEditor => EditorType::SignalScope,
            EditorType::SignalScope => EditorType::NeuronInspector,
            EditorType::NeuronInspector => EditorType::Timeline,
            EditorType::Timeline => EditorType::LogConsole,
            EditorType::LogConsole => EditorType::BakePanel,
            EditorType::BakePanel => EditorType::ShardMap,
            EditorType::ShardMap => EditorType::WorldView,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitDirection {
    Horizontal, // FlexDirection::Row
    Vertical,   // FlexDirection::Column
}

#[derive(Debug, Clone)]
pub enum AreaNode {
    Split {
        direction: SplitDirection,
        ratio: f32, // Процент разделения (0.0 .. 1.0), обычно 0.5
        a: Box<AreaNode>,
        b: Box<AreaNode>,
    },
    Leaf(EditorArea),
}

#[derive(Debug, Clone)]
pub struct EditorArea {
    pub area_id: u32,
    pub editor_type: EditorType,
}

/// Главный ресурс рабочего пространства
#[derive(Resource)]
pub struct WorkspaceLayout {
    pub root: AreaNode,
    pub needs_rebuild: bool, // Флаг для Zero-Cost обновления
}

impl Default for WorkspaceLayout {
    fn default() -> Self {
        Self {
            root: AreaNode::Split {
                direction: SplitDirection::Horizontal,
                ratio: 0.7, // 70% на 3D вид, 30% на конфиг
                a: Box::new(AreaNode::Leaf(EditorArea {
                    area_id: 1,
                    editor_type: EditorType::WorldView,
                })),
                b: Box::new(AreaNode::Leaf(EditorArea {
                    area_id: 2,
                    editor_type: EditorType::ConfigEditor, // Правая панель
                })),
            },
            needs_rebuild: true,
        }
    }
}

// Компоненты-маркеры для привязки ECS-сущностей к логике
#[derive(Component)]
pub struct WorkspaceRoot;

#[derive(Component)]
pub struct AreaHeader {
    pub area_id: u32,
    pub editor_type: EditorType,
}

#[derive(Component)]
pub struct AreaBody(pub EditorType);

/// Оператор сплита (Blender workflow)
#[derive(Event)]
pub struct SplitAreaEvent {
    pub target_area_id: u32,
    pub direction: SplitDirection,
    pub ratio: f32,
    pub new_editor: EditorType,
}

/// Zero-cost обработчик операторов сплита. Отрабатывает только если есть эвенты.
pub fn apply_split_operator(
    mut events: EventReader<SplitAreaEvent>,
    mut layout: ResMut<WorkspaceLayout>,
) {
    if events.is_empty() {
        return; // Ранний выход, не тратим такты Main Thread
    }

    let mut changed = false;
    for ev in events.read() {
        if split_node_in_tree(&mut layout.root, ev) {
            changed = true;
        }
    }

    // Триггерим Zero-Cost reconciliation
    if changed {
        layout.needs_rebuild = true;
    }
}

pub fn split_node_in_tree(node: &mut AreaNode, ev: &SplitAreaEvent) -> bool {
    match node {
        AreaNode::Split { a, b, .. } => {
            // Рекурсивно идем вглубь дерева
            split_node_in_tree(a, ev) || split_node_in_tree(b, ev)
        }
        AreaNode::Leaf(area) => {
            if area.area_id == ev.target_area_id {
                // Мутируем Leaf в Split узел
                let old_leaf = AreaNode::Leaf(area.clone());
                let new_leaf = AreaNode::Leaf(EditorArea {
                    area_id: generate_area_id(),
                    editor_type: ev.new_editor,
                });

                *node = AreaNode::Split {
                    direction: ev.direction,
                    ratio: ev.ratio,
                    a: Box::new(old_leaf),
                    b: Box::new(new_leaf),
                };
                return true;
            }
            false
        }
    }
}

pub struct WorkspaceOperatorPlugin;

impl Plugin for WorkspaceOperatorPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<SplitAreaEvent>()
            // Обрабатываем мутации дерева до перестроения UI
            .add_systems(Update, (handle_header_interactions, apply_split_operator.before(rebuild_workspace_ui), test_split_operator));
    }
}

/// Тестовая система: Ctrl+S дергает сплит первой панели
fn test_split_operator(
    keys: Res<ButtonInput<KeyCode>>,
    mut split_events: EventWriter<SplitAreaEvent>,
) {
    if keys.just_pressed(KeyCode::KeyS) && keys.pressed(KeyCode::ControlLeft) {
        split_events.send(SplitAreaEvent {
            target_area_id: 1,
            direction: SplitDirection::Vertical,
            ratio: 0.5,
            new_editor: EditorType::Timeline,
        });
        info!("[operator] Split triggered by Ctrl+S");
    }
}

/// Система ловит клик ПКМ по заголовку любой панели и кидает Event.
/// Zero-Cost: фильтр Changed<Interaction> гарантирует, что мы не проверяем ноды каждый кадр.
pub fn handle_header_interactions(
    mouse: Res<ButtonInput<MouseButton>>,
    q_interactions: Query<(&Interaction, &AreaHeader), Changed<Interaction>>,
    mut ev_split: EventWriter<SplitAreaEvent>,
    mut layout: ResMut<WorkspaceLayout>,
) {
    let mut changed_layout = false;

    for (interaction, header) in q_interactions.iter() {
        if *interaction == Interaction::Hovered && mouse.just_pressed(MouseButton::Right) {
            // ПКМ: Сплит окна (по умолчанию дублируем текущий тип)
            ev_split.send(SplitAreaEvent {
                target_area_id: header.area_id,
                direction: SplitDirection::Vertical, 
                ratio: 0.5,
                new_editor: header.editor_type,
            });
            info!("[operator] Split triggered by RMB on Area {}", header.area_id);
        } 
        else if *interaction == Interaction::Pressed && mouse.just_pressed(MouseButton::Left) {
            // ЛКМ: Меняем тип окна
            let next_type = header.editor_type.next();
            if change_node_type_in_tree(&mut layout.root, header.area_id, next_type) {
                changed_layout = true;
                info!("[layout] Area {} changed to {:?}", header.area_id, next_type);
            }
        }
    }

    if changed_layout {
        layout.needs_rebuild = true;
    }
}

// Рекурсивный поиск и замена типа в дереве
fn change_node_type_in_tree(node: &mut AreaNode, target_id: u32, new_type: EditorType) -> bool {
    match node {
        AreaNode::Split { a, b, .. } => {
            change_node_type_in_tree(a, target_id, new_type) || change_node_type_in_tree(b, target_id, new_type)
        }
        AreaNode::Leaf(area) => {
            if area.area_id == target_id {
                area.editor_type = new_type;
                return true;
            }
            false
        }
    }
}

pub struct LayoutPlugin;

impl Plugin for LayoutPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorkspaceLayout>()
            .add_plugins(WorkspaceOperatorPlugin)
            .add_systems(Startup, setup_workspace_root)
            .add_systems(Update, rebuild_workspace_ui);
    }
}

fn setup_workspace_root(mut commands: Commands) {
    // Корневой узел, занимающий 100% экрана
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            ..default()
        },
        WorkspaceRoot,
    ));
}

/// Рекурсивное построение Flexbox дерева
fn spawn_area_node(commands: &mut ChildBuilder, node: &AreaNode) {
    match node {
        AreaNode::Split { direction, ratio, a, b } => {
            let flex_dir = match direction {
                SplitDirection::Horizontal => FlexDirection::Row,
                SplitDirection::Vertical => FlexDirection::Column,
            };

            commands.spawn((
                Node {
                    flex_direction: flex_dir,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    ..default()
                },
            )).with_children(|parent| {
                // Дочерний узел A
                parent.spawn((
                    Node {
                        width: match direction {
                            SplitDirection::Horizontal => Val::Percent(*ratio * 100.0),
                            SplitDirection::Vertical => Val::Percent(100.0),
                        },
                        height: match direction {
                            SplitDirection::Horizontal => Val::Percent(100.0),
                            SplitDirection::Vertical => Val::Percent(*ratio * 100.0),
                        },
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                )).with_children(|p| spawn_area_node(p, a));

                // Дочерний узел B
                parent.spawn((
                    Node {
                        width: match direction {
                            SplitDirection::Horizontal => Val::Percent((1.0 - ratio) * 100.0),
                            SplitDirection::Vertical => Val::Percent(100.0),
                        },
                        height: match direction {
                            SplitDirection::Horizontal => Val::Percent(100.0),
                            SplitDirection::Vertical => Val::Percent((1.0 - ratio) * 100.0),
                        },
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                )).with_children(|p| spawn_area_node(p, b));
            });
        }
        AreaNode::Leaf(area) => {
            // Контейнер конкретного редактора (Header + Body)
            commands.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    border: UiRect::all(Val::Px(1.0)), // Граница между окнами
                    ..default()
                },
                BorderColor(Color::srgb(0.2, 0.2, 0.2)),
                BackgroundColor(Color::srgb(0.1, 0.1, 0.1)),
            )).with_children(|parent| {
                // Header (верхняя полоска для смены типа окна)
                parent.spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(24.0),
                        align_items: AlignItems::Center,
                        padding: UiRect::horizontal(Val::Px(8.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
                    AreaHeader {
                        area_id: area.area_id,
                        editor_type: area.editor_type,
                    },
                    Interaction::default(),
                )).with_children(|header| {
                    header.spawn((
                        Text::new(format!("{:?}", area.editor_type)),
                        TextFont { font_size: 14.0, ..default() },
                        TextColor(Color::srgb(0.8, 0.8, 0.8)),
                    ));
                });

                // Body (сам контент редактора)
                parent.spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0), // Оставшееся пространство
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                    AreaBody(area.editor_type),
                ));
            });
        }
    }
}

/// Система сверяет ресурс и перестраивает ECS-сущности интерфейса
fn rebuild_workspace_ui(
    mut commands: Commands,
    mut layout: ResMut<WorkspaceLayout>,
    q_root: Query<Entity, With<WorkspaceRoot>>,
) {
    if !layout.needs_rebuild {
        return;
    }

    let Ok(root_entity) = q_root.get_single() else { return };

    // Удаляем старое дерево UI
    commands.entity(root_entity).despawn_descendants();

    // Строим новое
    commands.entity(root_entity).with_children(|parent| {
        spawn_area_node(parent, &layout.root);
    });

    layout.needs_rebuild = false;
    info!("[layout] Workspace UI rebuilt.");
}
