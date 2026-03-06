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
        let total_axons = vram.total_axons as usize;
        if total_axons == 0 {
            return;
        }

        // 1. Выделяем 32-байтный выровненный буфер на хосте
        let mut host_axon_heads = vec![genesis_core::layout::BurstHeads8::empty(0); total_axons];

        // 2. Скачиваем с GPU
        unsafe {
            ffi::gpu_device_synchronize();
            ffi::gpu_memcpy_device_to_host(
                host_axon_heads.as_mut_ptr() as *mut std::ffi::c_void,
                vram.ptrs.axon_heads as *const std::ffi::c_void,
                total_axons * std::mem::size_of::<genesis_core::layout::BurstHeads8>(),
            );
        }

        // 3. Сканируем все 8 голов каждого аксона на CPU
        let mut reset_count = 0;
        for burst in host_axon_heads.iter_mut() {
            let mut changed = false;
            
            if burst.h0 >= SENTINEL_OVERFLOW_THRESHOLD { burst.h0 = AXON_SENTINEL; changed = true; }
            if burst.h1 >= SENTINEL_OVERFLOW_THRESHOLD { burst.h1 = AXON_SENTINEL; changed = true; }
            if burst.h2 >= SENTINEL_OVERFLOW_THRESHOLD { burst.h2 = AXON_SENTINEL; changed = true; }
            if burst.h3 >= SENTINEL_OVERFLOW_THRESHOLD { burst.h3 = AXON_SENTINEL; changed = true; }
            if burst.h4 >= SENTINEL_OVERFLOW_THRESHOLD { burst.h4 = AXON_SENTINEL; changed = true; }
            if burst.h5 >= SENTINEL_OVERFLOW_THRESHOLD { burst.h5 = AXON_SENTINEL; changed = true; }
            if burst.h6 >= SENTINEL_OVERFLOW_THRESHOLD { burst.h6 = AXON_SENTINEL; changed = true; }
            if burst.h7 >= SENTINEL_OVERFLOW_THRESHOLD { burst.h7 = AXON_SENTINEL; changed = true; }

            if changed {
                reset_count += 1;
            }
        }

        // 4. Заливаем обратно
        if reset_count > 0 {
            unsafe {
                ffi::gpu_memcpy_host_to_device(
                    vram.ptrs.axon_heads as *mut std::ffi::c_void,
                    host_axon_heads.as_ptr() as *const std::ffi::c_void,
                    total_axons * std::mem::size_of::<genesis_core::layout::BurstHeads8>(),
                );
                ffi::gpu_device_synchronize();
            }
        }
    }
}
