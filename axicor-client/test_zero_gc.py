import gc
import numpy as np
from unittest.mock import patch
from axicor.encoders import PopulationEncoder, PwmEncoder
from axicor.decoders import PwmDecoder
from axicor.client import AxicorMultiClient, HEADER_SIZE, GSOO_MAGIC, HEADER_FMT
import struct
import os
import sys

# Ensure axicor package is found
sys.path.append(os.getcwd())

class ManualMock:
    def __init__(self, return_val):
        self.return_val = return_val
        self.call_count = 0
    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.return_val

class RecvMock:
    def __init__(self, fake_packet, addr):
        self.fake_packet = fake_packet
        self.addr = addr
    def __call__(self, buf, nbytes):
        buf[:len(self.fake_packet)] = self.fake_packet
        return len(self.fake_packet), self.addr

def test_encoder_zero_gc():
    batch_size = 10
    variables_count = 5
    neurons_per_var = 16
    encoder = PopulationEncoder(variables_count, neurons_per_var, batch_size)
    
    states = np.random.rand(variables_count).astype(np.float32)
    tx_arena = bytearray(encoder.total_bytes)
    tx_view = memoryview(tx_arena)
    
    # Warmup
    encoder.encode_into(states, tx_view)
    
    gc.collect()
    gc.disable()
    try:
        # Run once to clear any first-time call allocations
        encoder.encode_into(states, tx_view)
        count_before = gc.get_count()[0]
        for _ in range(10000):
            encoder.encode_into(states, tx_view)
        count_after = gc.get_count()[0]
        diff = count_after - count_before
        assert diff <= 1, f"Allocations detected: {diff} over 10000 iterations"
    finally:
        gc.enable()

def test_pwm_encoder_zero_gc():
    batch_size = 10
    num_sensors = 32
    encoder = PwmEncoder(num_sensors, batch_size)
    
    sensors = np.random.rand(num_sensors).astype(np.float32)
    tx_arena = bytearray(encoder.total_bytes)
    tx_view = memoryview(tx_arena)
    
    # Warmup
    encoder.encode_into(sensors, tx_view)
    
    gc.collect()
    gc.disable()
    try:
        encoder.encode_into(sensors, tx_view)
        count_before = gc.get_count()[0]
        for _ in range(10000):
            encoder.encode_into(sensors, tx_view)
        count_after = gc.get_count()[0]
        diff = count_after - count_before
        assert diff <= 1, f"Allocations detected: {diff} over 10000 iterations"
    finally:
        gc.enable()

def test_decoder_zero_gc():
    num_outputs = 10
    batch_size = 10
    decoder = PwmDecoder(num_outputs, batch_size)
    
    rx_data = bytearray(num_outputs * batch_size)
    rx_view = memoryview(rx_data)
    
    # Warmup
    decoder.decode_from(rx_view)
    
    gc.collect()
    gc.disable()
    try:
        decoder.decode_from(rx_view)
        count_before = gc.get_count()[0]
        for _ in range(10000):
            decoder.decode_from(rx_view)
        count_after = gc.get_count()[0]
        diff = count_after - count_before
        assert diff <= 1, f"Allocations detected: {diff} over 10000 iterations"
    finally:
        gc.enable()

def test_client_step_zero_gc():
    addr = ("127.0.0.1", 9000)
    matrices = [{'zone_hash': 1, 'matrix_hash': 100, 'payload_size': 64}]
    rx_layout = [{'matrix_hash': 200, 'size': 32}]
    
    # Patch socket.socket before creating the client
    with patch('socket.socket') as mock_socket_class:
        mock_sock = mock_socket_class.return_value
        client = AxicorMultiClient(addr, matrices, rx_layout)
        
        # Prepare a fake response packet
        fake_packet = bytearray(HEADER_SIZE + 32)
        struct.pack_into(HEADER_FMT, fake_packet, 0, GSOO_MAGIC, 1, 200, 32, 0, 0)
        
        mock_sock.recvfrom_into = RecvMock(fake_packet, addr)
        mock_sock.sendto = ManualMock(len(fake_packet))
        
        # Warmup
        client.step(reward=10)
        
        gc.collect()
        gc.disable()
        try:
            client.step(reward=10)
            count_before = gc.get_count()[0]
            for _ in range(10000):
                client.step(reward=10)
            count_after = gc.get_count()[0]
            diff = count_after - count_before
            assert diff <= 1, f"Allocations detected: {diff} over 10000 iterations"
        finally:
            gc.enable()

if __name__ == "__main__":
    print("Running test_encoder_zero_gc...")
    try:
        test_encoder_zero_gc()
        print("test_encoder_zero_gc passed!")
    except AssertionError as e:
        print(f"test_encoder_zero_gc FAILED: {e}")

    print("Running test_pwm_encoder_zero_gc...")
    try:
        test_pwm_encoder_zero_gc()
        print("test_pwm_encoder_zero_gc passed!")
    except AssertionError as e:
        print(f"test_pwm_encoder_zero_gc FAILED: {e}")

    print("Running test_decoder_zero_gc...")
    try:
        test_decoder_zero_gc()
        print("test_decoder_zero_gc passed!")
    except AssertionError as e:
        print(f"test_decoder_zero_gc FAILED: {e}")

    print("Running test_client_step_zero_gc...")
    try:
        test_client_step_zero_gc()
        print("test_client_step_zero_gc passed!")
    except AssertionError as e:
        print(f"test_client_step_zero_gc FAILED: {e}")
