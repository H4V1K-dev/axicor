// genesis-runtime/src/orchestrator/night_phase.rs
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::fs::{OpenOptions, File};
use std::io::{BufWriter, Write, Read};
use std::path::Path;
use genesis_core::constants::MAX_DENDRITE_SLOTS;

const CHKT_MAGIC: u32 = 0x43484B54; // "CHKT"

/// Выполняет прямой сброс Pinned RAM на диск без аллокаций.
unsafe fn save_hot_checkpoint(
    zone_name: &str,
    total_ticks: u64,
    padded_n: u32,
    pinned_targets: *const u32,
    pinned_weights: *const i16,
) {
    // В реальном проекте путь берется из конфига зоны (baked_dir)
    // Здесь мы временно захардкодим на папку baked/zone_name
    let path_str = format!("baked/{}/checkpoint_weights.bin", zone_name);
    let path = Path::new(&path_str);
    
    // Используем BufWriter с большим буфером (4MB) для утилизации скорости NVMe
    let file = OpenOptions::new().create(true).write(true).truncate(true).open(path).expect("Fatal: Failed to open checkpoint file");
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);

    // 1. Header (16 bytes)
    writer.write_all(&CHKT_MAGIC.to_le_bytes()).unwrap();
    writer.write_all(&total_ticks.to_le_bytes()).unwrap();
    writer.write_all(&padded_n.to_le_bytes()).unwrap();

    // 2. Рассчитываем размеры массивов в байтах
    let targets_bytes_len = (MAX_DENDRITE_SLOTS * padded_n as usize) * 4;
    let weights_bytes_len = (MAX_DENDRITE_SLOTS * padded_n as usize) * 2;

    // 3. Формируем сырые &[u8] слайсы прямо из Pinned RAM
    let targets_slice = std::slice::from_raw_parts(pinned_targets as *const u8, targets_bytes_len);
    let weights_slice = std::slice::from_raw_parts(pinned_weights as *const u8, weights_bytes_len);

    // 4. Zero-Copy I/O: Пишем на диск напрямую
    writer.write_all(targets_slice).unwrap();
    writer.write_all(weights_slice).unwrap();
    writer.flush().unwrap();
}

/// Прямое чтение с диска в Pinned RAM без промежуточных буферов.
pub unsafe fn load_hot_checkpoint(
    zone_name: &str,
    expected_padded_n: u32,
    pinned_targets: *mut u32,
    pinned_weights: *mut i16,
) -> Option<u64> {
    let path_str = format!("baked/{}/checkpoint_weights.bin", zone_name);
    let path = Path::new(&path_str);
    
    if !path.exists() {
        return None; // Чекпоинта нет, зона стартует с нуля
    }

    let mut file = File::open(path).expect("Fatal: Failed to open checkpoint");
    let mut header = [0u8; 16];
    if file.read_exact(&mut header).is_err() {
        return None;
    }

    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    assert_eq!(magic, CHKT_MAGIC, "Fatal: Checkpoint corrupted for zone {}", zone_name);

    let tick = u64::from_le_bytes(header[4..12].try_into().unwrap());
    let padded_n = u32::from_le_bytes(header[12..16].try_into().unwrap());
    
    assert_eq!(
        padded_n, expected_padded_n, 
        "Fatal: Topology changed (padded_n mismatch). Cannot load checkpoint for {}", zone_name
    );

    let targets_bytes_len = (MAX_DENDRITE_SLOTS * padded_n as usize) * 4;
    let weights_bytes_len = (MAX_DENDRITE_SLOTS * padded_n as usize) * 2;

    // Контракт unsafe: мы гарантируем, что размеры буферов в Pinned RAM 
    // в точности совпадают с ожидаемыми размерами слайсов.
    let targets_slice = std::slice::from_raw_parts_mut(pinned_targets as *mut u8, targets_bytes_len);
    let weights_slice = std::slice::from_raw_parts_mut(pinned_weights as *mut u8, weights_bytes_len);

    // ОС сама зальет страницы с диска прямо в Pinned RAM
    file.read_exact(targets_slice).unwrap();
    file.read_exact(weights_slice).unwrap();

    Some(tick)
}

/// Вызывается из главного потока, мгновенно возвращает управление
pub fn trigger_night_phase(
    zone_name: String,
    total_ticks: u64,
    vram_ptr: *mut crate::memory::VramState, 
    padded_n: u32,
    total_axons: u32,
    prune_threshold: i16,
    is_sleeping: Arc<AtomicBool>,
    master_seed: u64
) {
    // Уводим зону в сон
    is_sleeping.store(true, Ordering::Release);

    // Поднимаем выделенный OS-поток под тяжелую математику (Sprouting / Baking)
    let vram_addr = vram_ptr as usize;
    thread::spawn(move || {
        unsafe {
            let vram = &mut *(vram_addr as *mut crate::memory::VramState);
            let stream = std::ptr::null_mut(); // Для асинхронной ночи лучше создать отдельный cudaStream_t
            
            // Step 1: GPU Sort & Prune
            crate::ffi::launch_sort_and_prune(
                padded_n,
                vram.dendrite_targets as *mut std::ffi::c_void,
                vram.dendrite_weights as *mut std::ffi::c_void,
                vram.dendrite_refractory as *mut std::ffi::c_void,
                prune_threshold,
                stream
            );
            crate::ffi::gpu_stream_synchronize(stream);

            // Step 2: Download
            let targets_size = genesis_core::constants::MAX_DENDRITE_SLOTS * (padded_n as usize) * 4;
            let weights_size = genesis_core::constants::MAX_DENDRITE_SLOTS * (padded_n as usize) * 2;
            
            crate::ffi::gpu_memcpy_device_to_host_async(vram.pinned_host_targets as *mut _, vram.dendrite_targets as *const _, targets_size, stream);
            crate::ffi::gpu_memcpy_device_to_host_async(vram.pinned_host_weights as *mut _, vram.dendrite_weights as *const _, weights_size, stream);
            crate::ffi::gpu_stream_synchronize(stream);

            let zone_name_clone = zone_name.clone();
            save_hot_checkpoint(
                &zone_name_clone,
                total_ticks,
                padded_n,
                vram.pinned_host_targets as *const u32,
                vram.pinned_host_weights as *const i16
            );

            // Step 3: CPU Sprouting (DOD slice manipulation)
            let targets_slice = std::slice::from_raw_parts_mut(vram.pinned_host_targets as *mut u32, targets_size / 4);
            let weights_slice = std::slice::from_raw_parts_mut(vram.pinned_host_weights as *mut i16, weights_size / 2);
            
            crate::orchestrator::sprouting::run_cpu_sprouting(targets_slice, weights_slice, padded_n as usize, total_axons, master_seed);

            // Step 4: Upload back to VRAM
            crate::ffi::gpu_memcpy_host_to_device_async(vram.dendrite_targets as *mut _, vram.pinned_host_targets as *const _, targets_size, stream);
            crate::ffi::gpu_memcpy_host_to_device_async(vram.dendrite_weights as *mut _, vram.pinned_host_weights as *const _, weights_size, stream);
            crate::ffi::gpu_stream_synchronize(stream);
        }

        // Пробуждение зоны (Day Phase снова подхватит её в следующем батче)
        is_sleeping.store(false, Ordering::Release);
    });
}
