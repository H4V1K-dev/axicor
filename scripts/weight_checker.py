import numpy as np
import sys
def check_shard(name, path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        # Рассчитываем N (14 байт заголовка + 128 слотов по 7 байт на дендрит)
        n = len(data) // (14 + 128 * 7)
        off_tgt = n * 14
        off_w = off_tgt + n * 128 * 4
        # Читаем веса (i16)
        w = np.frombuffer(data[off_w : off_w + n*128*2], dtype=np.int16)
        
        active = w[w != 0]
        if len(active) == 0:
            print(f"[{name}] 💀 All weights are ZERO")
            return
        exc_count = len(active[active > 0])
        inh_count = len(active[active < 0])
        
        # Биологическая цель: 80% Exc / 20% Inh => 4:1
        actual_ratio = exc_count / inh_count if inh_count > 0 else float('inf')
        target_ratio = 4.0  # (80/20)
        deviation = actual_ratio / target_ratio
        
        status = "✅ OK" if 0.5 <= deviation <= 2.0 else "⚠️ I-DOMINANCE" if deviation < 0.5 else "⚠️ E-DOMINANCE"
        
        print(f"[{name}] 🧠 Stats:")
        print(f"  - Active Synapses: {len(active)} ({(len(active)/(n*128))*100:.2f}% density)")
        print(f"  - Avg Weight:      {np.mean(np.abs(active)):.2f}")
        print(f"  - Max/Min:         {np.max(active)} / {np.min(active)}")
        print(f"  - Exc/Inh Ratio:   {exc_count} / {inh_count} ({actual_ratio:.4f}) | Target: {target_ratio:.2f} | {status}")
        print(f"  - Balance Health:  {deviation*100:.2f}% of target")
    except Exception as e:
        print(f"[{name}] Error: {e}")
check_shard("SensoryCortex", "examples/cartpole/baked/SensoryCortex/shard.state")
check_shard("HiddenCortex", "examples/cartpole/baked/HiddenCortex/shard.state")
check_shard("MotorCortex", "examples/cartpole/baked/MotorCortex/shard.state")