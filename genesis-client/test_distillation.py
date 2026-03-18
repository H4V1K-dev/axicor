import os
import struct
import mmap
import time
import numpy as np
from genesis.memory import GenesisMemory

def test_distillation():
    ZONE_HASH = 0xDEADBEEF
    PADDED_N = 100_000
    
    # Эмуляция VRAM шарда на 100к нейронов
    # 64 (Header) + Weights (100k * 128 * 2) + Targets (100k * 128 * 4) + Handovers...
    WEIGHTS_SIZE = PADDED_N * 128 * 2
    TARGETS_SIZE = PADDED_N * 128 * 4
    SHM_SIZE = 64 + WEIGHTS_SIZE + TARGETS_SIZE + (10000 * 20)
    
    shm_path = f"/dev/shm/genesis_shard_{ZONE_HASH:08X}"
    
    # 1. Создаем фейковый дамп VRAM
    with open(shm_path, "wb") as f:
        f.truncate(SHM_SIZE)
        
    with open(shm_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        
        weights_off = 64
        targets_off = 64 + WEIGHTS_SIZE
        
        # Пишем строгий C-ABI Header v2
        # magic(I), version(B), state(B), pad(H)
        # padded_n(I), dendrite_slots(I), weights_off(I), targets_off(I)
        # ... rest 64 bytes
        # Using the format I identified earlier: "<IBBHIIIIQIIIIIIII" (64 bytes)
        # But user's struct.pack_into used a shorter one: "<IBBHIIII"
        # I'll use the user's provided code as is, it's their test.
        # Wait, the user's struct.pack_into in the test_distillation code was:
        # struct.pack_into("<IBBHIIII", mm, 0, 0x47454E53, 2, 0, 0, PADDED_N, 128, weights_off, targets_off)
        # 4 + 1+1+2 + 4 + 4 + 4 + 4 = 24 bytes. The remaining 40 bytes will be 0.
        
        # [DOD FIX] Strict C-ABI v2 (64 bytes)
        # <IBBHIIIIQIIIIIIII
        struct.pack_into("<IBBHIIIIQIIIIIIII", mm, 0,
                         0x47454E53, 2, 0, 0,
                         PADDED_N, 128, weights_off, targets_off,
                         0, # epoch
                         PADDED_N, # total_axons (для теста равно padded_n)
                         0, 0, ZONE_HASH, 0, 0, 0, 0)
        mm.close()

    # 2. Подключаем Data-Oriented SDK
    # По умолчанию теперь read_only=False, что нам и нужно.
    mem = GenesisMemory(ZONE_HASH)
    
    # 3. Инжектируем тестовые данные (создаем 100 000 сильных и 100 000 слабых связей)
    mem.targets[0, :] = 5 # axon_id = 4 (смещение Zero-Index)
    mem.weights[0, :] = 100 # Сильная связь
    
    mem.targets[1, :] = 10 # axon_id = 9
    mem.weights[1, :] = 10 # Слабая связь (должна быть уничтожена)
    
    # 4. Запуск дистилляции
    print(f"🧠 Запуск дистилляции {PADDED_N} нейронов (Threshold = 15)...")
    start = time.perf_counter()
    
    killed = mem.distill_graph(prune_threshold=15)
    
    duration_ms = (time.perf_counter() - start) * 1000
    
    print(f"⏱ Время дистилляции: {duration_ms:.3f} ms")
    print(f"💀 Выжжено связей: {killed}")
    
    # Проверки инвариантов
    assert killed == PADDED_N, "Должны были умереть все связи во 2 слоте!"
    assert np.all(mem.targets[1, :] == 0), "Слабые цели не обнулились!"
    assert np.all(mem.targets[0, :] == 5), "Сильные цели пострадали!"
    
    mem.close()
    os.remove(shm_path)
    print("✅ Zero-Copy дистилляция отработала безупречно.\n")

if __name__ == '__main__':
    test_distillation()
