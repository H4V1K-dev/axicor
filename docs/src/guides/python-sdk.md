# HFT Client & Network (Python SDK)

## 1. The 10ms Rule & Zero-Garbage Law

> ⚠️ **WARNING: The Biological Real-Time Boundary**
> Axicor's runtime operates in strict physical lockstep. Your observation-action cycle (`while True`) must execute in under 10ms. Dropping frames or delaying the cycle breaks the physics synchronization.

In the hot HFT loop, Python's Garbage Collector is a fatal enemy. A single `list.append()`, `np.array()` creation, or a standard `socket.recv()` allocation will stall the thread, destroy timing predictability, and ultimately break the BSP (Bulk Synchronous Parallel) barrier.

### The Zero-Allocation Contract
To survive in the 10ms window, you must adhere to three laws:
1. **Pre-allocate everything** before the loop starts.
2. **Write in-place** using `out=` parameters for any mathematical operations.
3. **Read memory directly** using `memoryview` without copying byte arrays.

### Example: The Pure HFT Loop

```python
import numpy as np

# 1. Pre-allocate memory OUTSIDE the hot loop
action_buffer = np.zeros(8, dtype=np.float32)
norm_state = np.zeros(27, dtype=np.float16)

bounds_min = np.full(27, -5.0, dtype=np.float16)
range_diff = np.full(27, 10.0, dtype=np.float16)

# 2. HFT Hot Loop
while True:
    # 💡 NOTE: In-place math operations. Zero allocations.
    np.subtract(raw_state, bounds_min, out=norm_state)
    np.divide(norm_state, range_diff, out=norm_state)
    np.clip(norm_state, 0.0, 1.0, out=norm_state)
    
    encoder.encode_into(norm_state, client.payload_views)
    
    # client.step() is strictly synchronous and blocking.
    # It executes zero-copy recv_into under the hood.
    rx_raw = client.step(reward=0)
    
    # 💡 NOTE: Zero-copy read via memoryview. 
    # Calling np.array(rx_raw) here is a fatal violation!
    rx_view = memoryview(rx_raw)
    
    # Slice the view directly into the decoder
    fl_out = dec_fl.decode_from(rx_view[0 : dec_fl.payload_size])
```

## 2. Synchronous BSP Lockstep (Ping-Pong Lockstep)

The Axicor engine operates on a strict Bulk Synchronous Parallel (BSP) model. Your Python client and the hardware runtime must remain in perfect, frame-by-frame lockstep.

The `client.step()` method is strictly **synchronous and blocking**. It sends your encoded input matrix to the network and immediately blocks the thread, waiting until the GPU computes the tick and returns the output matrix.

> ⚠️ **WARNING: The Async/Await Fatality**
> Never wrap `client.step()` in `asyncio` or attempt to spam the runtime with non-blocking UDP transmissions. 
> **The Physical Reason:** Asynchronous spam of `GSIO` packets physically overwrites the `InputSwapchain` ring buffer in the Rust orchestrator *before* the GPU has a chance to read the data for the current tick. This silently destroys the BSP synchronization barrier, causing inputs to bleed across ticks and corrupting the neural state.

### The Lockstep Pattern
There is only one legal way to advance time in the Axicor engine. The thread must halt until the hardware acknowledges the state transition:

```python
# HFT Hot Loop
while True:
    # 1. Encode local state into pre-allocated memory
    encoder.encode_into(norm_state, client.payload_views)
    
    # 2. Block and wait for the hardware (Ping)
    # The thread sleeps here until the GPU finishes the tick.
    rx_raw = client.step(reward=0)
    
    # 3. Hardware returned (Pong). Safe to decode and act.
    rx_view = memoryview(rx_raw)
    
    # ... zero-copy decoding ...
```

## 3. C-ABI Network Contracts

### ExternalIoHeader
Every Data Plane UDP packet requires a strict 20-byte Little-Endian header. Python `ctypes.Structure` classes MUST include `_pack_ = 1` to prevent CPython from inserting hidden padding.

```python
import ctypes

class ExternalIoHeader(ctypes.Structure):
    _pack_ = 1  # CRITICAL: Disables CPython padding
    _fields_ = [
        ("magic", ctypes.c_uint32),         # 0x4F495347 ("GSIO") or 0x4F4F5347 ("GSOO")
        ("zone_hash", ctypes.c_uint32),     # FNV-1a hash of the zone
        ("matrix_hash", ctypes.c_uint32),   # FNV-1a hash of the I/O matrix
        ("payload_size", ctypes.c_uint32),  # Strictly the size of the bitmask payload
        ("global_reward", ctypes.c_int16),  # R-STDP Dopamine Modulator (-32768..32767)
        ("_padding", ctypes.c_uint16),      # Alignment to exactly 20 bytes
    ]
```

> ⚠️ **WARNING: The Payload Size Trap**
> The `payload_size` field dictates the size of the bitmask payload in bytes, excluding the 20 bytes of the header itself. Including the header size in this field causes the `axicor-node` to silently drop the packet.

### Network Limits & L7-Chunking
The strict UDP MTU limit for the payload is 65507 bytes. If an I/O matrix exceeds this limit, the matrix is fragmented using L7-Chunking (aligned to 64 bytes). The `AxicorMultiClient` handles reconstruction transparently into a pre-allocated bytearray.

### Zero-Copy Transpose (Decoding)
The `Output_History` (GSOO) payload arrives over the network already transposed by the GPU into `[Pixel][Tick]` format. Python `for` loops for matrix reconstruction are forbidden. You must unpack the payload in O(1):

```python
reshaped = memoryview(raw_payload).cast('B').reshape((num_pixels, batch_ticks))
```

## 4. Zero-Garbage Hot Loop Example
This is the canonical implementation of the HFT cycle.

```python
import numpy as np
from axicor.client import AxicorMultiClient

# 1. Pre-Allocation Phase (Cold Start)
# Allocate buffers once.
action = np.zeros(8, dtype=np.float32)
norm_state = np.zeros(27, dtype=np.float16)

bounds = np.tile([-5.0, 5.0], (27, 1)).astype(np.float16)
range_diff = bounds[:, 1] - bounds[:, 0]

# Client encapsulates the UDP socket pool and pre-allocates MTU buffers internally.
# It strictly uses socket.recv_into() under the hood.
client = AxicorMultiClient(addr=("127.0.0.1", 8081), matrices=...)

while True:
    # 2. Environment Step
    state = env.step(action)
    
    # 3. Vectorized Normalization (Strict In-Place)
    # FORBIDDEN: norm_state = np.clip((state - bounds[:,0]) / range_diff, 0, 1)
    np.subtract(state, bounds[:, 0], out=norm_state)
    np.divide(norm_state, range_diff, out=norm_state)
    np.clip(norm_state, 0.0, 1.0, out=norm_state)
    
    # 4. Zero-Cost Facade
    # Writes directly into the pre-allocated UDP buffer via memoryview offsets.
    encoder.encode_into(norm_state, client.payload_views)
    
    # 5. Blocking Ping-Pong Lockstep
    # Synchronously sends GSIO with dopamine (-2) and blocks until GSOO is received.
    client.step(dopamine_signal=-2) 
```

## 5. Offline Surgery (AxicorSurgeon)
The `AxicorSurgeon` module provides direct mmap access to `.state` and `.axons` binary files. Calling the Surgeon inside the HFT loop is strictly prohibited.

### The Zero-Index Trap
When working with `dendrite_targets`, `target == 0` is a hardware trigger for GPU Early Exit.

- **Packing (Writing to VRAM):** The real `axon_id` MUST be shifted by +1 (`axon_id + 1`). Packing an empty slot (`axon_id = -1`) with a non-zero segment offset will cause a GPU Segmentation Fault.
- **Unpacking (Reading from VRAM):** The actual ID is extracted as `(target_packed & 0x00FFFFFF) - 1`.

### Physics Domains (Mass vs Charge)
Synaptic weights in VRAM are stored in the Mass Domain as 32-bit integers (`i32`, up to 2.14B). For analytical reading and human-readable representation, the weight must be divided by `65536.0` to transition into the Charge Domain (microvolts). This is the only valid use of floats for synaptic weights in the SDK API.