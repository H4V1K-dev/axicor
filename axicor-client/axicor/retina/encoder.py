import math
import numpy as np
import cv2

class RetinaEncoder:
    """
    [DOD] Event-Driven Vision Pipeline.
    Converts RGB frames into sparse feature bitmasks (DoG, Motion, Color) 
    without any heap allocations.
    Strictly uses uint8/int16 to avoid FPU overhead and L2 cache thrashing.
    """
    def __init__(self, width: int, height: int, batch_size: int, 
                 center_sigma: float = 1.0, surround_sigma: float = 2.0, base_threshold: float = 15.0):
        self.W = width
        self.H = height
        self.N = width * height
        self.B = batch_size

        self.center_sigma = center_sigma
        self.surround_sigma = surround_sigma
        self.base_threshold = int(base_threshold)

        # C-ABI Warp Alignment (strictly multiple of 64 bits for AMD/L2)
        self.padded_N = math.ceil(self.N / 64) * 64
        self.bytes_per_tick = self.padded_N // 8
        self.total_bytes = self.bytes_per_tick * self.B

        # [DOD] Pre-allocated buffers (Zero-Garbage Integer Math)
        self._frame_u8 = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self._gray = np.zeros((self.H, self.W), dtype=np.uint8)
        self._gray_i16 = np.zeros((self.H, self.W), dtype=np.int16)
        self._prev_gray = np.zeros((self.H, self.W), dtype=np.uint8)
        
        # Intermediate Gaussian buffers (signed for DoG)
        self._center = np.zeros((self.H, self.W), dtype=np.int16)
        self._surround = np.zeros((self.H, self.W), dtype=np.int16)
        self._dog = np.zeros((self.H, self.W), dtype=np.int16)
        self._motion = np.zeros((self.H, self.W), dtype=np.uint8)

        self._b = np.zeros((self.H, self.W), dtype=np.int16)
        self._g = np.zeros((self.H, self.W), dtype=np.int16)
        self._r = np.zeros((self.H, self.W), dtype=np.int16)
        self._rg_opp = np.zeros((self.H, self.W), dtype=np.int16)
        self._by_opp = np.zeros((self.H, self.W), dtype=np.int16)
        self._yellow = np.zeros((self.H, self.W), dtype=np.int16)

        self._bool_buffer = np.zeros(self.padded_N, dtype=np.bool_)
        self._batch_bool_buffer = np.zeros((self.B, self.padded_N), dtype=np.bool_)

    def encode_into(self, frame_bgr: np.ndarray, tx_view: memoryview) -> int:
        if frame_bgr.shape[:2] != (self.H, self.W):
            raise ValueError(f"Fovea size mismatch: expected {(self.H, self.W)}, got {frame_bgr.shape[:2]}")

        # 0. Zero-allocation copy
        self._frame_u8[:] = frame_bgr
        self._batch_bool_buffer.fill(False)

        # 1. Grayscale & Mean-Based Inhibition (Dynamic Threshold)
        cv2.cvtColor(self._frame_u8, cv2.COLOR_BGR2GRAY, dst=self._gray)
        mean_illum = cv2.mean(self._gray)[0]
        dynamic_thresh = self.base_threshold + int(mean_illum * 0.1)
        
        # Pre-cast grayscale to int16 for subsequent signed operations
        self._gray_i16[:] = self._gray

        # --- TICK 0: Difference of Gaussians (Contours) ---
        cv2.GaussianBlur(self._gray_i16, (0, 0), self.center_sigma, dst=self._center)
        cv2.GaussianBlur(self._gray_i16, (0, 0), self.surround_sigma, dst=self._surround)
        
        np.subtract(self._center, self._surround, out=self._dog)
        np.greater(self._dog.ravel(), dynamic_thresh, out=self._bool_buffer[:self.N])
        self._batch_bool_buffer[0, :] = self._bool_buffer

        # --- TICK 1: Frame Delta (Motion) ---
        cv2.absdiff(self._gray, self._prev_gray, dst=self._motion)
        np.greater(self._motion.ravel(), dynamic_thresh, out=self._bool_buffer[:self.N])
        self._batch_bool_buffer[1, :] = self._bool_buffer
        np.copyto(self._prev_gray, self._gray)

        # --- TICK 2 & 3: Chromatic Opponents ---
        # Fast extraction directly into int16 buffers
        self._b[:] = self._frame_u8[:, :, 0]
        self._g[:] = self._frame_u8[:, :, 1]
        self._r[:] = self._frame_u8[:, :, 2]

        # Tick 2: R-G (Red-Green)
        np.subtract(self._r, self._g, out=self._rg_opp)
        np.greater(self._rg_opp.ravel(), dynamic_thresh, out=self._bool_buffer[:self.N])
        self._batch_bool_buffer[2, :] = self._bool_buffer

        # Tick 3: B-Y (Blue-Yellow) -> Y = (R+G)/2
        np.add(self._r, self._g, out=self._yellow)
        np.right_shift(self._yellow, 1, out=self._yellow) # Fast divide by 2
        np.subtract(self._b, self._yellow, out=self._by_opp)
        np.greater(self._by_opp.ravel(), dynamic_thresh, out=self._bool_buffer[:self.N])
        self._batch_bool_buffer[3, :] = self._bool_buffer

        # 4. Strict C-ABI Packing (Little-Endian)
        packed = np.packbits(self._batch_bool_buffer, bitorder='little', axis=1)
        tx_view[:self.total_bytes] = packed.ravel()

        return self.total_bytes
