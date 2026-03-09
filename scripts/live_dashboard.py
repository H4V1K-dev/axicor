#!/usr/bin/env python3
import socket
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

# Настройки
UDP_IP = "127.0.0.1"
UDP_PORT = 8100
MAX_HISTORY = 10000
VIEW_WINDOW = 500

# Глобальные контейнеры данных
episodes = []
scores = []
tps_values = []
sma_25 = []
sma_100 = []
sma_300 = []

# Состояние
view_offset = 0
is_paused = False

def calculate_sma(data, window):
    if not data: return 0.0
    arr = data[-window:]
    return float(np.mean(arr))

# UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.bind((UDP_IP, UDP_PORT))
except Exception as e:
    print(f"❌ Bind error: {e}")
    sys.exit(1)
sock.setblocking(False)

# UI Setup
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 7))
fig.canvas.manager.set_window_title('Genesis Neural Dashboard v2.2')
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# Цвета
C_SCORE  = '#444444'
C_SMA25  = '#00ff88'
C_SMA100 = '#00ccff'
C_SMA300 = '#ff00ff'
C_TPS    = '#ffff00'

# Объекты графиков
line_score, = ax1.plot([], [], color=C_SCORE, alpha=0.3, label='Raw Score', linewidth=1)
line_sma25, = ax1.plot([], [], color=C_SMA25, alpha=0.9, label='SMA-25', linewidth=1.5)
line_sma100, = ax1.plot([], [], color=C_SMA100, alpha=0.9, label='SMA-100', linewidth=2)
line_sma300, = ax1.plot([], [], color=C_SMA300, alpha=0.9, label='SMA-300', linewidth=2.5)
line_tps, = ax2.plot([], [], color=C_TPS, alpha=0.1, label='TPS', linewidth=1)

# Стилизация
ax1.set_facecolor('#050505')
fig.patch.set_facecolor('#050505')
ax1.set_title('Neural Network Evolution (CartPole)', color='white', pad=20, fontsize=15)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score', color=C_SMA25)
ax2.set_ylabel('TPS', color=C_TPS, alpha=0.5)
ax1.grid(True, color='#111', linestyle='--')

# Текстовые блоки (Индивидуальные для каждого цвета)
# Используем один bbox для всей группы
info_items = {}
y_pos = 0.96
def add_info_line(key, label, color):
    global y_pos
    t = ax1.text(0.02, y_pos, f'{label}: ---', 
                 transform=ax1.transAxes, color=color, 
                 family='monospace', fontsize=10, fontweight='bold',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#000', alpha=0.6, ec='none'))
    info_items[key] = t
    y_pos -= 0.035

add_info_line('status', 'MODE', '#fff')
add_info_line('ep',     'EPISODE', '#aaa')
add_info_line('cur',    'CURRENT', '#fff')
add_info_line('s25',    'SMA-25 ', C_SMA25)
add_info_line('s100',   'SMA-100', C_SMA100)
add_info_line('s300',   'SMA-300', C_SMA300)
add_info_line('tps',    'TPS    ', C_TPS)

ax1.text(0.98, 0.02, 'Arrows: Scroll | P: Pause | R: Reset | S: Screenshot', 
         transform=ax1.transAxes, color='#444', ha='right', fontsize=8)

def on_key(event):
    global view_offset, is_paused
    if event.key == 'left':
        view_offset += 50
    elif event.key == 'right':
        view_offset = max(0, view_offset - 50)
    elif event.key == 'p':
        is_paused = not is_paused
    elif event.key == 'r':
        view_offset = 0
        is_paused = False
    elif event.key == 's':
        plt.savefig(f"snapshot_ep{len(episodes)}.png", dpi=150)
        print("📸 Captured!")

fig.canvas.mpl_connect('key_press_event', on_key)

def update(frame):
    global episodes, scores, tps_values, sma_25, sma_100, sma_300
    
    latest_score = scores[-1] if scores else 0.0
    latest_tps = tps_values[-1] if tps_values else 0.0
    latest_ep = episodes[-1] if episodes else 0
    
    # Слив UDP буфера
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            if len(data) == 16:
                ep, score, tps, is_done = struct.unpack("<Ifff", data)
                
                latest_ep = int(ep)
                latest_score = float(score)
                latest_tps = float(tps)
                
                if is_done > 0.5:
                    episodes.append(latest_ep)
                    scores.append(latest_score)
                    tps_values.append(latest_tps)
                    
                    if len(episodes) > MAX_HISTORY:
                        episodes.pop(0); scores.pop(0); tps_values.pop(0)
                        if sma_25: sma_25.pop(0)
                        if sma_100: sma_100.pop(0)
                        if sma_300: sma_300.pop(0)

                    sma_25.append(calculate_sma(scores, 25))
                    sma_100.append(calculate_sma(scores, 100))
                    sma_300.append(calculate_sma(scores, 300))
        except BlockingIOError:
            break
        except Exception as e:
            break

    # Обновление текста (всегда)
    st = "LIVE" if not is_paused and view_offset == 0 else "HISTORY/PAUSED"
    info_items['status'].set_text(f"MODE:    {st}")
    info_items['ep'].set_text(f"EPISODE: {latest_ep}")
    info_items['cur'].set_text(f"CURRENT: {latest_score:.0f}")
    
    cur_s25 = sma_25[-1] if sma_25 else 0.0
    cur_s100 = sma_100[-1] if sma_100 else 0.0
    cur_s300 = sma_300[-1] if sma_300 else 0.0
    
    info_items['s25'].set_text(f"SMA-25:  {cur_s25:.1f}")
    info_items['s100'].set_text(f"SMA-100: {cur_s100:.1f}")
    info_items['s300'].set_text(f"SMA-300: {cur_s300:.1f}")
    info_items['tps'].set_text(f"TPS:     {int(latest_tps)}")

    if not episodes:
        return line_score, line_sma25, line_sma100, line_sma300, line_tps

    # Слайсинг для отображения
    limit = len(episodes) - view_offset
    start = max(0, limit - VIEW_WINDOW)
    
    if limit <= 0 or start >= limit:
        return line_score, line_sma25, line_sma100, line_sma300, line_tps

    x = np.array(episodes[start:limit])
    y_raw = np.array(scores[start:limit])
    y_25 = np.array(sma_25[start:limit])
    y_100 = np.array(sma_100[start:limit])
    y_300 = np.array(sma_300[start:limit])
    y_tps = np.array(tps_values[start:limit])

    line_score.set_data(x, y_raw)
    line_sma25.set_data(x, y_25)
    line_sma100.set_data(x, y_100)
    line_sma300.set_data(x, y_300)
    line_tps.set_data(x, y_tps)

    # Авто-масштабирование
    ax1.set_xlim(x[0], x[-1] + 1)
    
    visible_scores = y_raw
    max_s = np.max(visible_scores) if visible_scores.size > 0 else 50
    ax1.set_ylim(-5, max(50, max_s * 1.1))
    
    visible_tps = y_tps
    if visible_tps.size > 0:
        ax2.set_ylim(0, np.max(visible_tps) * 2)

    return line_score, line_sma25, line_sma100, line_sma300, line_tps

# Запуск
ani = animation.FuncAnimation(fig, update, interval=250, blit=False)
plt.tight_layout()
print(f"🚀 Dashboard 2.2 Online ({UDP_IP}:{UDP_PORT})")
plt.show()
