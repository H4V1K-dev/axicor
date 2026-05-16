import os
import sys
import numpy as np
from pathlib import Path

# [DOD] Строгое выравнивание по L2 кэш-линии (64 байта)
def align_64(offset):
    return (offset + 63) & ~63

def calc_offsets(padded_n):
    off = 0
    off = align_64(off + padded_n * 4)  # 1. voltage (i32)
    off = align_64(off + padded_n * 1)  # 2. flags (u8)
    off = align_64(off + padded_n * 4)  # 3. threshold (i32)
    off = align_64(off + padded_n * 1)  # 4. timers (u8)
    off = align_64(off + padded_n * 4)  # 5. soma_to_axon (u32)
    
    off_targets = off
    off = align_64(off + padded_n * 128 * 4) # 6. dendrite_targets (u32)
    
    off_weights = off
    off = align_64(off + padded_n * 128 * 4) # 7. dendrite_weights (i32)
    
    off_dtimers = off
    total_size = align_64(off + padded_n * 128 * 1) # 8. dendrite_timers (u8)
    
    return off_targets, off_weights, total_size

def check_shard(zone_name, state_path):
    if not os.path.exists(state_path):
        print(f"[!] Файл не найден: {state_path}")
        return

    # Загружаем блоб целиком в RAM
    blob = np.fromfile(state_path, dtype=np.uint8)
    blob_len = len(blob)

    # Реверс-инжиниринг padded_n через C-ABI макет
    padded_n = 32 # Шаг варпа (CUDA)
    while True:
        off_tgt, off_wgt, expected_size = calc_offsets(padded_n)
        if expected_size == blob_len:
            break
        padded_n += 32
        if padded_n > 5_000_000:
            print(f"[{zone_name}] FATAL: C-ABI Mismatch. Файл поврежден или нарушен 64-Byte Invariant.")
            return

    # Zero-Copy срезы прямо из бинарного блоба
    targets = np.frombuffer(blob, dtype=np.uint32, count=padded_n * 128, offset=off_tgt)
    weights = np.frombuffer(blob, dtype=np.int32, count=padded_n * 128, offset=off_wgt)

    # Hardware Early Exit Guard: target == 0 означает пустой (выжженный) слот
    active_mask = (targets != 0)
    active_count = np.sum(active_mask)
    total_slots = padded_n * 128
    density = (active_count / total_slots) * 100.0

    print(f"\n=== Аудит Зоны: {zone_name} ===")
    print(f"  Нейронов (padded): {padded_n}")
    print(f"  Плотность связей:  {density:.2f}% ({active_count}/{total_slots} слотов)")

    if active_count == 0:
        print("  [RED FLAG] СЕТЬ МЕРТВА. 0 активных связей.")
        return

    # Анализ дрейфа массы (Только для живых связей)
    active_weights = np.abs(weights[active_mask]) # GSOP оперирует абсолютной силой
    max_w = np.max(active_weights)
    min_w = np.min(active_weights)
    avg_w = np.mean(active_weights)

    print(f"  Mass Domain (Min): {min_w}")
    print(f"  Mass Domain (Avg): {avg_w:.1f}")
    print(f"  Mass Domain (Max): {max_w}")

    # Диагностика R-STDP
    if max_w > 1_500_000_000:
        print("  [OK] R-STDP Монументализация ЗАФИКСИРОВАНА. Веса пробили 15-й ранг инерции.")
    elif max_w < 10_000_000:
        print("  [FAIL] Веса в стагнации. GSOP мертв или Дофамин не доходит из агента.")
    
    if density == 100.0:
        print("  [FAIL] Плотность 100%. Фоновая депрессия (LTD) не выжигает мусор.")

if __name__ == "__main__":
    base_dir = "Axicor-Models/AntConnectome/baked"
    zones = ["SensoryCortex", "MotorCortex", "FL_Ganglion", "FR_Ganglion", "BL_Ganglion", "BR_Ganglion"]
    
    for z in zones:
        # Проверяем сначала чекпоинт, если его нет - базовый дамп
        chk_path = os.path.join(base_dir, z, "checkpoint.state")
        if not os.path.exists(chk_path):
            chk_path = os.path.join(base_dir, z, "shard.state")
        check_shard(z, chk_path)