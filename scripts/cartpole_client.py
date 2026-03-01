import socket
import struct
import gymnasium as gym
import numpy as np
import threading
import time

def fnv1a_32(data: bytes) -> int:
    hash_value = 0x811c9dc5
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x01000193) & 0xFFFFFFFF
    return hash_value

ZONE_HASH = fnv1a_32(b"SensoryCortex") # Шлем в сенсорную кору
MATRIX_IN_HASH = fnv1a_32(b"cartpole_in")
GENESIS_IP = "127.0.0.1"
PORT_OUT = 8081
PORT_IN = 8082

# Разделяемое состояние без блокировок (GIL Python спасает от Data Race)
class State:
    def __init__(self):
        self.obs = np.zeros(4)
        self.action = 0
        self.running = True

state = State()

def encode_population(value, min_val, max_val, neurons=16):
    norm = np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    center_idx = int(norm * (neurons - 1))
    bitmask = 0
    for i in range(max(0, center_idx - 1), min(neurons, center_idx + 2)):
        bitmask |= (1 << i)
    return bitmask

def udp_hot_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT_IN))
    sock.settimeout(0.01) # Короткий таймаут чтобы не висеть при выходе
    
    while state.running:
        # 1. Быстрая упаковка
        obs = state.obs
        mask_pos = encode_population(obs[0], -2.4, 2.4)
        mask_vel = encode_population(obs[1], -3.0, 3.0)
        mask_ang = encode_population(obs[2], -0.41, 0.41)
        mask_ang_vel = encode_population(obs[3], -2.0, 2.0)
        
        word_0 = mask_pos | (mask_vel << 16)
        word_1 = mask_ang | (mask_ang_vel << 16)
        payload = struct.pack("<IIIII", ZONE_HASH, MATRIX_IN_HASH, 8, word_0, word_1)
        
        sock.sendto(payload, (GENESIS_IP, PORT_OUT))
        
        # 2. Быстрое чтение ответа от MotorCortex
        try:
            data, _ = sock.recvfrom(65535)
            out_history = data[12:] # Пропуск хедера
            left_spikes = sum(out_history[0::2])
            right_spikes = sum(out_history[1::2])
            state.action = 0 if left_spikes > right_spikes else 1
        except socket.timeout:
            pass

import pygame

def main():
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    state.obs, _ = env.reset()
    
    pygame.init()
    # The default CartPole size is ~ 600x400
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Genesis CartPole")
    font = pygame.font.SysFont(None, 36)
    
    udp_thread = threading.Thread(target=udp_hot_loop)
    udp_thread.start()
    
    frame_count = 0
    try:
        while True:
            # Среда обновляется асинхронно от мозга
            state.obs, reward, terminated, truncated, _ = env.step(state.action)
            img = env.render()
            
            if img is not None:
                # Транспонируем изображение для pygame (H,W,C) -> (W,H,C)
                surf = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
                
                # Сероватый оттенок (по заявке пользователя)
                gray_overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                gray_overlay.fill((100, 100, 100, 80)) # RGBA
                surf.blit(gray_overlay, (0, 0))
                
                # Черный счетчик кадров
                text = font.render(f"Frame: {frame_count} | Action: {'Left' if state.action == 0 else 'Right'}", True, (0, 0, 0))
                surf.blit(text, (10, 10))
                
                # Масштабируем на экран и отрисовываем
                screen.blit(pygame.transform.scale(surf, (600, 400)), (0, 0))
                pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            
            if terminated or truncated:
                state.obs, _ = env.reset()
                frame_count = 0
                
            frame_count += 1
            
            # Опциональный sleep для ограничения FPS симулятора
            # time.sleep(1/60) 
    except (KeyboardInterrupt, SystemExit):
        state.running = False
        udp_thread.join()
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()
