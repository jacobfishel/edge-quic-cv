#!/usr/bin/env python3
"""Test script: Send a fake JPEG frame over QUIC to the server."""

import cv2
import asyncio
import struct
import numpy as np
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

async def test_quic():
    """Send a test frame over QUIC."""
    # Create a fake test image (640x480, blue)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:, :] = [255, 0, 0]  # Blue
    
    # Encode as JPEG
    _, jpeg_data = cv2.imencode('.jpg', test_frame)
    frame_bytes = jpeg_data.tobytes()
    
    # QUIC config
    config = QuicConfiguration(is_client=True, alpn_protocols=["hq-29"])
    config.verify_mode = 0
    
    try:
        async with connect("localhost", 4433, configuration=config) as connection:
            stream = await connection.create_stream()
            print("Connected to QUIC server")
            
            # Send frame size + data
            size = len(frame_bytes)
            await stream.send(struct.pack("!I", size))
            await stream.send(frame_bytes)
            
            print(f"Sent test frame: {size} bytes")
            await asyncio.sleep(0.5)  # Give server time to process
            
    except Exception as e:
        print(f"Error: {e!r}")

if __name__ == "__main__":
    asyncio.run(test_quic())

