# Genesis IDE - Инструкции запуска

## Что готово?
✓ FPS-камера (Blender-style Alt-захват)
✓ Асинхронная загрузка геометрии (Mock Data в AsyncComputeTaskPool)
✓ 16 батчей для визуализации 500k нейронов
✓ Hot Loop для спайков (fade-out логика)

## Контроли

### Камера (Alt для захвата)
```
Alt              - Переключить режим захвата курсора
                   Captured: невидимый курсор, полная навигация
                   Free: видимый курсор, UI взаимодействие

Esc              - Выход из режима захвата

W/A/S/D          - Движение вперед/влево/назад/вправо
                   (относительно направления взгляда)

Space            - Движение вверх (глобальное Y)
Shift            - Движение вниз (глобальное -Y)

Mouse            - Вращение камеры (когда захвачен курсор)
                   X: yaw (поворот влево/вправо)
                   Y: pitch (поворот вверх/вниз)
                   Ограничена вертикаль: ±88°
```

## Сборка

```bash
cd /home/alex/Workflow/Genesis
cargo build -p genesis-ide
```

## Запуск

```bash
cargo run -p genesis-ide --release  # Release для лучшей производительности
```

или

```bash
cargo run -p genesis-ide  # Debug (медленнее)
```

## Ожидаемый вывод

```
[world] Initialized neuron layer 0
[world] Initialized neuron layer 1
[world] Initialized neuron layer 2
...
[world] Initialized neuron layer 15
[world] 16 neuron layers ready for GPU Instancing
[loader] Geometry fetch task completed.
[loader] Loaded 10000 instances for neuron type 0
[loader] Loaded 10000 instances for neuron type 1
...
```

После этого вы увидите окно с кубом из сфер. Нажмите Alt и летайте внутри!

## Архитектура

### AsyncComputeTaskPool (loader.rs)
- Спавнит фоновую задачу в пул потоков за O(1)
- Генерирует Mock Data: 16 типов × 10k инстансов = 160k нейронов
- Опрашивает Future каждый кадр (неблокирующий poll)
- Отправляет данные в ECS через Event
- **TODO Phase 2**: Заменить Mock на TCP 8001 запрос

### FPS Камера (camera.rs)
- IdeCamera компонент (speed, pitch, yaw)
- CameraMode ресурс (Free | Captured)
- toggle_camera_mode: управление захватом Alt/Esc
- camera_movement_system: WASD + Mouse навигация
- Transform обновления для движения и вращения

### World Rendering (world.rs)
- 16 Entity батчей (по одному на тип)
- NeuronLayerData компонент для каждого типа
- apply_telemetry_spikes: затухание спайков каждый кадр
- **TODO Phase 3**: Инжекция спайков из telemetry

## Производительность

### Текущий MVP (Mock Data: 160k нейронов)
- 16 Draw Calls (батчи по типам)
- Нет GPU Instancing (StandardMaterial)
- Ожидаемо ~30-60 FPS на современном GPU

### Phase 2+ (Реальные данные: 500k нейронов)
- GPU Instancing через DynamicStorageBuffer
- PackedPosition декодирование на GPU
- Ожидаемо ~60-144 FPS (в зависимости от GPU)

## Trouble Shooting

### "Окно не открывается"
- Проверьте DISPLAY переменную (X11/Wayland)
- На WSL может потребоваться X11-сервер

### "Camera не движется"
- Нажмите Alt (должен исчезнуть курсор)
- Затем WASD + Mouse

### "Мало нейронов видно"
- Используйте zoom или летайте поближе
- Mock Data: только 160k из запланированных 500k

## Что дальше? (Phase 3)

1. **telemetry.rs** - WebSocket спайки
   - `ws://127.0.0.1:8012/ws` от genesis-runtime
   - Инжекция spike_ids в apply_telemetry_spikes

2. **GPU Instancing**
   - DynamicStorageBuffer для PackedPosition
   - Кастомный render pass
   - PackedPosition декодирование в WGSL

3. **GPU оптимизации**
   - Bloom эффекты для спайков
   - LOD по дистанции
   - Frustum culling

##문의?

Все вопросы относительно архитектуры ищите в:
- [genesis-ide/ARCHITECTURE.md](ARCHITECTURE.md)
- [genesis-ide/src/camera.rs](src/camera.rs) - детали камеры
- [genesis-ide/src/loader.rs](src/loader.rs) - детали загрузки
- [genesis-ide/src/world.rs](src/world.rs) - батчинг и спайки
