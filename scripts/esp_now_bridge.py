#!/usr/bin/env python3
import serial
import socket
import sys

# [DOD] HFT Serial-to-UDP Bridge
# Никаких строк, JSON или декодирования. Читаем 16 байт -> шлем 16 байт.

def run_bridge(serial_port="/dev/ttyUSB0", baudrate=115200):
    udp_ip = "127.0.0.1"
    udp_port = 8100
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[*] Bridging ESP-NOW ({serial_port}) -> UDP ({udp_ip}:{udp_port})")
    
    try:
        with serial.Serial(serial_port, baudrate) as ser:
            while True:
                # В реальности донгл ESP32 должен печатать маркер начала кадра, 
                # но для HFT мы предполагаем, что он просто пишет сырые байты (fwrite).
                raw_bytes = ser.read(16)
                if len(raw_bytes) == 16:
                    sock.sendto(raw_bytes, (udp_ip, udp_port))
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)

if __name__ == '__main__':
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    run_bridge(port)
