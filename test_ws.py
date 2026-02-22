import asyncio
import websockets
import struct

async def test_telemetry():
    uri = "ws://127.0.0.1:8002/ws"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected! Waiting for DayPhase broadcast...")
        
        # We just want to receive the very first frame to verify the header
        frame = await websocket.recv()
        print(f"Received frame of size {len(frame)} bytes")
        
        if len(frame) < 16:
            print("Error: Frame too small for header")
            exit(1)
            
        # Parse the 16-byte header: magic (u32), tick (u64), spikes_count (u32)
        # Note: Rust struct repr(C) uses padding!
        # u32 (4) + padding (4) + u64 (8) + u32 (4) = 20 bytes natively in C?
        # Let's inspect raw bytes first to be sure!
        print(f"Raw Header Bytes: {frame[:24].hex()}")
        
        # Struct format: 
        # I = u32
        # q = i64 / Q = u64
        magic = struct.unpack_from("<I", frame, 0)[0]
        
        # 'GNSS' in little endian Hex is 0x53534E47.
        is_gnss = magic == int.from_bytes(b"GNSS", byteorder="little")
        print(f"Magic Number: {hex(magic)} (Valid GNSS: {is_gnss})")
        
        if not is_gnss:
            print("ERROR: Invalid Magic Number!")
            exit(1)
            
        print("SUCCESS! Binary Telemetry stream is working.")

if __name__ == "__main__":
    asyncio.run(test_telemetry())
