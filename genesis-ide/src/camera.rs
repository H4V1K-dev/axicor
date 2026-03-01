// genesis-ide/src/camera.rs
use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};

#[derive(Component)]
pub struct OrbitCamera {
    pub radius: f32,
    pub center: Vec3,
    pub alpha: f32, // Вращение вокруг Y
    pub beta: f32,  // Вращение вокруг X
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            radius: 100.0,
            center: Vec3::ZERO,
            alpha: std::f32::consts::PI / 4.0,
            beta: std::f32::consts::PI / 6.0,
        }
    }
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_orbit_camera);
    }
}

fn update_orbit_camera(
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let Ok((mut orbit, mut transform)) = query.get_single_mut() else { return };

    // Зум
    for event in mouse_wheel.read() {
        orbit.radius -= event.y * orbit.radius * 0.1;
        orbit.radius = orbit.radius.clamp(5.0, 1000.0);
    }

    // Собираем дельту мыши один раз за кадр
    let mut delta = Vec2::ZERO;
    for event in mouse_motion.read() {
        delta += event.delta;
    }

    // Вращение (Правая кнопка мыши)
    if mouse.pressed(MouseButton::Right) {
        orbit.alpha -= delta.x * 0.005;
        orbit.beta += delta.y * 0.005;
        // Ограничиваем зенит, чтобы камера не перевернулась
        orbit.beta = orbit.beta.clamp(-1.5, 1.5);
    } 
    // Панорамирование (Средняя кнопка мыши)
    else if mouse.pressed(MouseButton::Middle) {
        let right = transform.right() * -delta.x * orbit.radius * 0.002;
        let up = transform.up() * delta.y * orbit.radius * 0.002;
        orbit.center += right + up;
    }

    // Подняться/опуститься по глобальной высоте (Пробел / Shift)
    let move_speed = orbit.radius * 0.02;
    if keys.pressed(KeyCode::Space) {
        orbit.center.y += move_speed;
    }
    if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        orbit.center.y -= move_speed;
    }

    // Пересчет позиции из сферических координат в декартовы
    let cos_beta = orbit.beta.cos();
    let position = orbit.center + Vec3::new(
        orbit.radius * orbit.alpha.sin() * cos_beta,
        orbit.radius * orbit.beta.sin(),
        orbit.radius * orbit.alpha.cos() * cos_beta,
    );

    *transform = Transform::from_translation(position).looking_at(orbit.center, Vec3::Y);
}
