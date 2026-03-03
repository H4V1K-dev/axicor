use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    window::{CursorGrabMode, PrimaryWindow},
};

#[derive(Component)]
pub struct IdeCamera {
    pub speed: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl Default for IdeCamera {
    fn default() -> Self {
        Self { 
            speed: 50.0, 
            pitch: 0.0, 
            yaw: 0.0 
        }
    }
}

// Ресурс-флаг активного режима (Blender-like toggle)
#[derive(Resource, Default, PartialEq, Eq)]
pub enum CameraMode {
    #[default]
    Free,
    Captured,
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraMode>()
           .add_systems(Startup, setup_camera)
           .add_systems(Update, (toggle_camera_mode, camera_movement_system));
    }
}

fn setup_camera(mut commands: Commands) {
    // 1. Спавним камеру для UI (ОБЯЗАТЕЛЬНО для Bevy)
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None,
            ..default()
        },
    ));

    // 2. Наша 3D FPS камера
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 500.0).looking_at(Vec3::ZERO, Vec3::Y),
        IdeCamera::default(),
    ));
}

fn toggle_camera_mode(
    keys: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<CameraMode>,
    mut q_windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    // Перехват по Alt или выход по Esc
    if keys.just_pressed(KeyCode::AltLeft) || keys.just_pressed(KeyCode::Escape) {
        let mut window = q_windows.single_mut();
        
        if *mode == CameraMode::Captured || keys.just_pressed(KeyCode::Escape) {
            *mode = CameraMode::Free;
            window.cursor_options.visible = true;
            window.cursor_options.grab_mode = CursorGrabMode::None;
        } else {
            *mode = CameraMode::Captured;
            window.cursor_options.visible = false;
            window.cursor_options.grab_mode = CursorGrabMode::Locked;
        }
    }
}

fn camera_movement_system(
    time: Res<Time>,
    mode: Res<CameraMode>,
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut q_camera: Query<(&mut Transform, &mut IdeCamera)>,
) {
    if *mode != CameraMode::Captured {
        return; // Ранний выход, не тратим такты
    }

    let (mut transform, mut cam) = q_camera.single_mut();
    let dt = time.delta_secs();

    // Вращение (Mouse)
    let mut mouse_delta = Vec2::ZERO;
    for ev in mouse_motion.read() {
        mouse_delta += ev.delta;
    }

    if mouse_delta != Vec2::ZERO {
        let sensitivity = 0.002;
        cam.yaw -= mouse_delta.x * sensitivity;
        cam.pitch -= mouse_delta.y * sensitivity;
        cam.pitch = cam.pitch.clamp(-1.54, 1.54); // Ограничение по вертикали (~88 градусов)

        transform.rotation = Quat::from_axis_angle(Vec3::Y, cam.yaw) 
                           * Quat::from_axis_angle(Vec3::X, cam.pitch);
    }

    // Движение (WASD + Space/Shift)
    let mut direction = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) { direction += *transform.forward(); }
    if keys.pressed(KeyCode::KeyS) { direction += *transform.back(); }
    if keys.pressed(KeyCode::KeyA) { direction += *transform.left(); }
    if keys.pressed(KeyCode::KeyD) { direction += *transform.right(); }
    if keys.pressed(KeyCode::Space) { direction += Vec3::Y; }
    if keys.pressed(KeyCode::ShiftLeft) { direction -= Vec3::Y; }

    if direction != Vec3::ZERO {
        let move_speed = cam.speed * dt;
        transform.translation += direction.normalize() * move_speed;
    }
}
