import os
import sys
import time
import ctypes
import mmap
import numpy as np
import matplotlib.pyplot as plt

# [DOD] Strict C-ABI memory mapping for Zero-Copy Ephys
class EphysShm(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),              # 0..4
        ("state", ctypes.c_uint32),              # 4..8   (0: Idle, 1: Trigger, 2: Busy, 3: Done)
        ("count", ctypes.c_uint32),              # 8..12  (Number of neurons <= 16)
        ("max_ticks", ctypes.c_uint32),          # 12..16 (e.g. 10000)
        ("current_tick", ctypes.c_uint32),       # 16..20 (Progress)
        ("target_tids", ctypes.c_uint32 * 16),   # 20..84 (Dense IDs to probe)
        ("injection_uv", ctypes.c_int32 * 16),   # 84..148 (Current injection)
        ("_padding", ctypes.c_uint8 * 44),       # 148..192 (64-byte L2 Cache Line Alignment)
        ("out_trace", ctypes.c_int32 * (16 * 10000)) # 192..640192 (Flat 2D Projection)
    ]

# FNV-1a Hash for SensoryCortex (from axicor_core::hash::fnv1a_32)
ZONE_HASH = 0x273FD103  
MAX_TICKS = 10000 # 1 секунда симуляции (при 100 мкс на тик)

def get_shm_path(zone_hash: int) -> str:
    if os.name == 'nt':
        import tempfile
        return os.path.join(tempfile.gettempdir(), f"axicor_ephys_{zone_hash:08x}.shm")
    return f"/dev/shm/axicor_ephys_{zone_hash:08x}.shm"

def run_probe():
    shm_path = get_shm_path(ZONE_HASH)
    print(f"[*] Locating Ephys SHM block: {shm_path}")
    
    if not os.path.exists(shm_path):
        print(f"[!] SHM file not found. Ensure axicor-node is running and Warmup is complete.")
        sys.exit(1)

    with open(shm_path, "r+b") as f:
        # Zero-Copy memory mapping
        mm = mmap.mmap(f.fileno(), ctypes.sizeof(EphysShm), access=mmap.ACCESS_WRITE)
        ephys = EphysShm.from_buffer(mm)

        if ephys.state == 1 or ephys.state == 2:
            print("[!] Probe is already busy. Wait or restart node.")
            sys.exit(1)

        # Выбираем нейроны для зондирования (Dense IDs)
        # В SensoryCortex Dense ID 0..63 обычно плотно заселены входными сигналами
        target_neurons = [1, 2, 4, 8] # Выбираем степени двойки для теста
        
        ephys.count = len(target_neurons)
        ephys.max_ticks = MAX_TICKS
        
        for i, tid in enumerate(target_neurons):
            ephys.target_tids[i] = tid
            ephys.injection_uv[i] = 0  # Без искусственного тока, читаем чистую биологию

        print(f"[*] Injecting probe into neurons: {target_neurons}")
        
        # State Machine: Trigger Node Orchestrator
        ephys.state = 1
        
        print("[*] Recording...", end="")
        while ephys.state != 3:
            if ephys.state == 2:
                # Читаем прогресс без блокировок
                pct = (ephys.current_tick / MAX_TICKS) * 100
                print(f"\r[*] Recording... {pct:.1f}%", end="")
            time.sleep(0.05)
            
        print("\r[*] Recording... DONE. Unpacking V(t) trace.      ")

        # Извлекаем и нормализуем данные
        # Массив out_trace плоский [tid * max_ticks + tick]
        traces = []
        for i in range(ephys.count):
            start = i * MAX_TICKS
            end = start + MAX_TICKS
            # Извлекаем микроВольты и переводим в миллиВольты для графика
            trace_mv = np.array(ephys.out_trace[start:end], dtype=np.float32) / 1000.0
            traces.append(trace_mv)

        # Очищаем состояние для будущих зондирований
        ephys.state = 0
        del ephys
        mm.close()

        # Рендеринг кардиограммы
        fig, axes = plt.subplots(len(target_neurons), 1, figsize=(12, 8), sharex=True)
        fig.suptitle("Axicor Ephys Probe: GLIF Membrane Voltage $V(t)$", fontsize=14)
        
        time_axis = np.arange(MAX_TICKS) * 0.1  # 100 us = 0.1 ms
        
        for i, (tid, trace) in enumerate(zip(target_neurons, traces)):
            ax = axes[i] if len(target_neurons) > 1 else axes
            ax.plot(time_axis, trace, color='cyan', linewidth=0.8)
            ax.set_ylabel(f"ID {tid} (mV)")
            ax.grid(True, alpha=0.2, color='gray')
            # Закрашиваем фон для киберпанк-стиля
            ax.set_facecolor('#1e1e1e')
            
        axes[-1].set_xlabel("Time (ms)")
        fig.patch.set_facecolor('#121212')
        for ax in axes:
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#333333')

        plt.tight_layout()
        # Save to artifacts for viewing
        artifact_path = "ephys_probe_results.png"
        plt.savefig(artifact_path)
        print(f"[*] Plot saved to {artifact_path}")
        # plt.show() # Disabled for headless execution

if __name__ == "__main__":
    run_probe()