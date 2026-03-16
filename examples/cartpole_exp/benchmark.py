#!/usr/bin/env python3
import os
import sys
import time
import socket
import numpy as np

# Добавляем путь к SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "genesis-client")))

from genesis.client import GenesisMultiClient
from genesis.memory import GenesisMemory
from genesis.brain import fnv1a_32

#============================================================
#                   BENCHMARK CONFIG
#============================================================
DURATION = 10.0               # Длительность теста (секунды)
WALL_DELAY = 0.01             # Симуляция "стены" (задержка среды, например 0.02 для 20мс)
IDLE_MODE = False             # Если True, не обновляем дофамин (чистый пульс)
#============================================================

def run_benchmark():
    print("🚀 Genesis HFT Benchmark: Stress Testing Environment...")
    
    # 1. Параметры идентичные CartPole примеру
    BATCH_SIZE = 20
    zone_hash = fnv1a_32(b"SensoryCortex")
    matrix_hash = fnv1a_32(b"cartpole_sensors")
    input_payload_size = (64 * BATCH_SIZE) // 8
    
    # 2. Инициализация
    try:
        client = GenesisMultiClient(
            addr=("127.0.0.1", 8081),
            matrices=[{'zone_hash': zone_hash, 'matrix_hash': matrix_hash, 'payload_size': input_payload_size}]
        )
        # [DOD FIX] Биндим порт 8092 (строго по манифесту ноды)
        client.sock.bind(("0.0.0.0", 8092))
        client.sock.settimeout(2.0) # Защита от вечного ожидания
    except Exception as e:
        print(f"❌ FATAL: Could not connect to Node. Is it running? {e}")
        return

    # 3. Подключение к Shared Memory для получения параметров графа
    try:
        memory = GenesisMemory(zone_hash, read_only=True)
        stats = memory.get_network_stats()
        neurons_count = memory.padded_n # Извлекаем из заголовка SHM
        synapses_count = stats["active_synapses"]
        print(f"📊 Current Graph: {neurons_count:,} Neurons | {synapses_count:,} Synapses")
    except Exception as e:
        print(f"⚠️ Warning: Could not read graph stats from SHM: {e}")
        neurons_count = 0
        synapses_count = 0

    # 4. Stress Test Loop (10 seconds)
    print(f"🔥 Running 10s stress test (Lockstep BATCH_SIZE={BATCH_SIZE})...")
    
    start_time = time.time()
    packet_count = 0
    
    while time.time() - start_time < DURATION:
        try:
            # 1. Симуляция задержки среды (WALL)
            if WALL_DELAY > 0:
                time.sleep(WALL_DELAY)

            # 2. Шлем сигнал. В IDLE_MODE просто пульсируем.
            client.step(0 if IDLE_MODE else 0) # По умолчанию 0 в обоих случаях, но логика разделена для будущего
            packet_count += 1
        except socket.timeout:
            print("\n❌ TIMEOUT: Node is not responding on port 8092!")
            print("   Check if Genesis Node is running and manifest.toml has 'external_udp_out_target = 127.0.0.1:8092'")
            break
        except Exception as e:
            print(f"\n❌ Error during step: {e}")
            break
        
    end_time = time.time()
    actual_duration = end_time - start_time
    total_ticks = packet_count * BATCH_SIZE
    tps = total_ticks / actual_duration
    
    print("\n" + "="*50)
    print(f"🏁 Benchmark Results ({'IDLE' if IDLE_MODE else 'STRESS'} mode):")
    if WALL_DELAY > 0:
        print(f"   Simulated Latency: {WALL_DELAY*1000:.1f} ms (Wall Clock)")
    print(f"   Packets Sent:   {packet_count:,}")
    print(f"   Total Ticks:    {total_ticks:,}")
    print(f"   Actual Time:    {actual_duration:.2f} s")
    print(f"🚀 THROUGHPUT:     {tps:,.0f} TPS (Ticks Per Second)")
    
    target_rt = 500 # 2ms RT target
    rel_rt = tps / target_rt
    print(f"   Relative to RT: {rel_rt:.2f}x ({'Faster' if rel_rt > 1 else 'Slower'} than 2ms RT)")
    
    if WALL_DELAY > 0:
        target_wall = 1.0 / (WALL_DELAY + (BATCH_SIZE * 0.0001)) # Примерный расчет
        print(f"   Wall Limit:     {1.0/WALL_DELAY * BATCH_SIZE:,.0f} TPS (theoretical max)")
    print("="*50)

    # 5. Экстраполяция (C-ABI Scaling Projection)
    # Потребление ресурсов Genesis Node растет линейно от количества синапсов (Synaptic Integration).
    # Мы предполагаем, что текущий TPS ограничен либо GPU (если граф большой), либо оверхедом UDP/Python.
    
    if synapses_count > 0:
        target_neurons = 1_000_000
        target_synapses = 128_000_000
        
        # Вычисляем текущую "пропускную способность синапсов в секунду"
        synaptic_throughput = tps * synapses_count
        
        # Прогнозируемый TPS для целевого графа (учитывая линейную сложность интеграции)
        projected_tps = synaptic_throughput / target_synapses
        
        print(f"\n📈 EXTRAPOLATION (1M Neurons / 128M Synapses):")
        if projected_tps > 500: # "Реальное время" условно 500 TPS (2ms шаг)
            print(f"   Projected: {projected_tps:,.0f} TPS (Real-time capable)")
        else:
            print(f"   Projected: {projected_tps:,.1f} TPS ({500/projected_tps:.1f}x slower than real-time)")
        
        print(f"   Note: Calculation based on synaptic density of {synapses_count/neurons_count:.1f} syn/neuron")
        print("   Physical constraint: Memory bandwidth becomes the bottleneck at 128M synapses.")

if __name__ == '__main__':
    run_benchmark()
