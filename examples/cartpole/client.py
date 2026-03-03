import socket
import struct
import numpy as np
import gymnasium as gym
import time

# --- Genesis Binary Contract §12 ---
GSIO_MAGIC = 0x4F495347 # "GSIO" (Input)
GSOO_MAGIC = 0x4F4F5347 # "GSOO" (Output)

# Cluster Hashes (Extracted from manifest/GXI/GXO)
SENSORY_ZONE_HASH = 658493699
SENSORY_MATRIX_HASH = 780390696
MOTOR_ZONE_HASH = 3597008999
MOTOR_MATRIX_HASH = 1590622057

# Network Config
ENGINE_IP = "127.0.0.1"
SENSOR_PORT = 8081
MOTOR_PORT = 8082

# Hyperparameters
NUM_VARS = 4
NEURONS_PER_VAR = 16
SIGMA_SQ = 0.15 ** 2
BATCH_TICKS = 100

class GenesisCartPoleClient:
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to motor port to receive spikes from Genesis
        self.sock.bind(("0.0.0.0", MOTOR_PORT))
        self.sock.settimeout(1.0) # 1s failover timeout

        # Precompute Gaussian Centers
        # CartPole observation space: 
        # [cart pos (-4.8, 4.8), cart vel (-inf, inf), pole angle (-.418, .418), pole vel (-inf, inf)]
        # We normalize centers to expected ranges
        self.centers = np.zeros((NUM_VARS, NEURONS_PER_VAR))
        ranges = [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.0, 3.0)]
        for i, (low, high) in enumerate(ranges):
            self.centers[i] = np.linspace(low, high, NEURONS_PER_VAR)

    def encode_state(self, state):
        """Gaussian Population Coding (Vectorized)"""
        active_axons = []
        for i in range(NUM_VARS):
            val = state[i]
            # Gaussian Tuning Curve: exp(- (x - mu)^2 / (2 * sigma^2))
            activations = np.exp(-((val - self.centers[i])**2) / (2 * SIGMA_SQ))
            
            # Select top-3 neurons (distributed representation)
            top_indices = np.argsort(activations)[-3:]
            for idx in top_indices:
                global_id = i * NEURONS_PER_VAR + idx
                active_axons.append(global_id)
        return active_axons

    def pack_input(self, active_axons):
        """Strict C-FFI Binary Packing: 4x u32 Header + 1x u64 Bitmask"""
        bitmask = 0
        for axon_id in active_axons:
            if 0 <= axon_id < 64:
                bitmask |= (1 << axon_id)
        
        # Header: magic, zone, matrix, payload_size
        return struct.pack('<4IQ', GSIO_MAGIC, SENSORY_ZONE_HASH, SENSORY_MATRIX_HASH, 8, bitmask)

    def decode_action(self, payload):
        """Population Decoding: WTA (Winner-Takes-All) on Spike Counts"""
        if len(payload) < 16:
            return 0
        
        header = struct.unpack('<4I', payload[:16])
        if header[0] != GSOO_MAGIC:
            print(f"Warning: Invalid GSOO magic 0x{header[0]:08X}")
            return 0
        
        # Raw spike data [Ticks, Neurons]
        # MotorCortex has 32 output channels (16 Left, 16 Right)
        raw_data = np.frombuffer(payload[16:], dtype=np.uint8)
        num_channels = len(raw_data) // BATCH_TICKS
        
        if num_channels < 32:
            return 0
            
        history = raw_data.reshape((BATCH_TICKS, num_channels))
        
        # Sum spikes over populations
        left_spikes = np.sum(history[:, :16])
        right_spikes = np.sum(history[:, 16:32])
        
        return 0 if left_spikes > right_spikes else 1

    def run(self):
        state, _ = self.env.reset()
        print(f"--- Genesis CartPole Bridge Started ---")
        print(f"Sending to {ENGINE_IP}:{SENSOR_PORT}, Waiting on {MOTOR_PORT}")
        
        try:
            while True:
                # 1. Encode & Pack
                axons = self.encode_state(state)
                packet = self.pack_input(axons)
                
                # 2. Synchronous Pacing: Send & Wait
                self.sock.sendto(packet, (ENGINE_IP, SENSOR_PORT))
                
                try:
                    data, _ = self.sock.recvfrom(8192) # Wait for Motor Output
                    action = self.decode_action(data)
                except socket.timeout:
                    print("Timeout: Genesis missed a batch!")
                    continue

                # 3. Environment Step
                state, reward, terminated, truncated, _ = self.env.step(action)
                
                if terminated or truncated:
                    state, _ = self.env.reset()
                    
        except KeyboardInterrupt:
            print("\nShutting down bridge...")
        finally:
            self.env.close()

if __name__ == "__main__":
    client = GenesisCartPoleClient()
    client.run()
