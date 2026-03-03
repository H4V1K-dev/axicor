use genesis_compute::ffi;
use genesis_compute::memory::VramState;
use genesis_core::constants::AXON_SENTINEL;

/// Интервал очистки: 1_800_000_000 тиков = 180 000 секунд = 50 часов (при 100мкс тике).
/// Sentinel переполняется через 2^31 тиков ≈ 59.6 часов. 50 часов даёт консервативный запас.
pub const SENTINEL_REFRESH_INTERVAL_TICKS: u64 = 1_800_000_000;

/// Допуск переполнения, при котором аксон считается «мёртвым» и сбрасывается.
/// 0x80000000 + 1_800_000_000 ≈ 0xEB9F_B000.
/// Мы сбрасываем всё, что больше 0xE000_0000.
pub const SENTINEL_OVERFLOW_THRESHOLD: u32 = 0xE000_0000;

pub struct SentinelManager {
    last_refresh_tick: u64,
}

impl Default for SentinelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SentinelManager {
    pub fn new() -> Self {
        Self {
            last_refresh_tick: 0,
        }
    }

    /// Проверяет, пришло ли время делать Sentinel Refresh, и если да — делает его.
    /// Это тяжелая операция (скачивание массива VRAM на хост и обратно),
    /// но она происходит крайне редко (раз в 50 часов).
    pub fn check_and_refresh(&mut self, vram: &VramState, current_tick: u64) {
        if current_tick - self.last_refresh_tick >= SENTINEL_REFRESH_INTERVAL_TICKS {
            println!(
                "[Sentinel] Refresh triggered at tick {}. Scanning {} axons...",
                current_tick,
                vram.total_axons
            );

            let start = std::time::Instant::now();
            self.perform_refresh(vram);
            let elapsed = start.elapsed();

            println!("[Sentinel] Refresh completed in {:?}", elapsed);
            self.last_refresh_tick = current_tick;
        }
    }

    fn perform_refresh(&self, vram: &VramState) {
        let total_axons = vram.total_axons;
        if total_axons == 0 {
            return;
        }

        // 1. Выделяем временный буфер на хосте для скачивания axon_heads
        let mut host_axon_heads = vec![0u32; total_axons];

        // 2. Скачиваем с GPU
        unsafe {
            ffi::gpu_device_synchronize();
            ffi::gpu_memcpy_device_to_host(
                host_axon_heads.as_mut_ptr() as *mut std::ffi::c_void,
                vram.axon_head_index as *const std::ffi::c_void,
                total_axons * std::mem::size_of::<u32>(),
            );
        }

        // 3. Сканируем на CPU и сбрасываем
        let mut reset_count = 0;
        for head in host_axon_heads.iter_mut() {
            if *head >= SENTINEL_OVERFLOW_THRESHOLD {
                *head = AXON_SENTINEL;
                reset_count += 1;
            }
        }

        // 4. Если были изменения — заливаем обратно
        if reset_count > 0 {
            unsafe {
                ffi::gpu_memcpy_host_to_device(
                    vram.axon_head_index as *mut std::ffi::c_void,
                    host_axon_heads.as_ptr() as *const std::ffi::c_void,
                    total_axons * std::mem::size_of::<u32>(),
                );
                ffi::gpu_device_synchronize();
            }
            // println!("[Sentinel] Reset {} overflowed axons to AXON_SENTINEL", reset_count);
        } else {
            // println!("[Sentinel] No axons needed reset. All safe.");
        }
    }
}
