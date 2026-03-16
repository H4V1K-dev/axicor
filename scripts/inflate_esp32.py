#!/usr/bin/env python3
import os
import mmap
import sys
import numpy as np

ESP_SLOTS = 32
DESKTOP_SLOTS = 128
SOMA_STATIC_BYTES = 14  # 4(V) + 1(F) + 4(T) + 1(Ref) + 4(S2A)
ESP_BYTES_PER_NEURON = SOMA_STATIC_BYTES + (ESP_SLOTS * 7)

def inflate_blob(esp_path: str, out_path: str):
    file_size = os.path.getsize(esp_path)
    padded_n = file_size // ESP_BYTES_PER_NEURON
    
    assert padded_n * ESP_BYTES_PER_NEURON == file_size, \
        f"FATAL: Blob size {file_size} is not a multiple of {ESP_BYTES_PER_NEURON} bytes/neuron"
    
    print(f"[*] Inflating ESP32 State: {esp_path}")
    print(f"[+] Detected padded_n = {padded_n}")

    with open(esp_path, "rb") as f_in:
        mm_in = mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ)

        # 1. Zero-Copy экстракция статических SoA массивов сомы
        static_soma_size = padded_n * SOMA_STATIC_BYTES
        soma_data = mm_in[:static_soma_size]

        # 2. Натягиваем матрицы на дендриты ESP32
        off = static_soma_size
        tgt_esp = np.frombuffer(mm_in, dtype=np.uint32, count=padded_n * ESP_SLOTS, offset=off).reshape(ESP_SLOTS, padded_n)
        off += padded_n * ESP_SLOTS * 4

        wgt_esp = np.frombuffer(mm_in, dtype=np.int16, count=padded_n * ESP_SLOTS, offset=off).reshape(ESP_SLOTS, padded_n)
        off += padded_n * ESP_SLOTS * 2

        tmr_esp = np.frombuffer(mm_in, dtype=np.uint8, count=padded_n * ESP_SLOTS, offset=off).reshape(ESP_SLOTS, padded_n)

        # 3. Аллокация десктопных матриц (заполняются нулями по умолчанию)
        print(f"[*] Broadcasting SIMD matrices (32 -> 128 slots)...")
        tgt_desk = np.zeros((DESKTOP_SLOTS, padded_n), dtype=np.uint32)
        wgt_desk = np.zeros((DESKTOP_SLOTS, padded_n), dtype=np.int16)
        tmr_desk = np.zeros((DESKTOP_SLOTS, padded_n), dtype=np.uint8)

        # 4. Векторизованный перенос живых синапсов в верхние слоты
        tgt_desk[:ESP_SLOTS, :] = tgt_esp
        wgt_desk[:ESP_SLOTS, :] = wgt_esp
        tmr_desk[:ESP_SLOTS, :] = tmr_esp

        # 5. Сброс C-ABI блоба на диск
        print(f"[*] Packing Desktop Blob to {out_path}")
        with open(out_path, "wb") as f_out:
            f_out.write(soma_data)
            f_out.write(tgt_desk.tobytes())
            f_out.write(wgt_desk.tobytes())
            f_out.write(tmr_desk.tobytes())
            
        final_size = os.path.getsize(out_path)
        print(f"[+] Success! New blob size: {final_size / 1024 / 1024:.2f} MB")
        
        # Cleanup views before closing mmap
        del tgt_esp, wgt_esp, tmr_esp
        mm_in.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/inflate_esp32.py <esp32_in.blob> <desktop_out.blob>")
        sys.exit(1)
    
    inflate_blob(sys.argv[1], sys.argv[2])
