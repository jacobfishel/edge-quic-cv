#!/usr/bin/env python3
"""Simple Raspberry Pi client that streams webcam frames over QUIC."""

import asyncio
import struct
import ssl

import cv2
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

SERVER_HOST = "localhost"
SERVER_PORT = 4433
CAMERA_INDEX = 0

async def send_frame(stream, frame_data):
    """Send a single frame over QUIC stream."""
    # Send frame size first (4 bytes)
    size = len(frame_data)
    await stream.send(struct.pack("!I", size))
    # Send frame data
    await stream.send(frame_data)

async def main():
    """Main function: capture frames and send over QUIC."""
    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return
    
    # QUIC configuration
    config = QuicConfiguration(is_client=True, alpn_protocols=["hq-29"])
    config.verify_mode = ssl.CERT_NONE  # Disable certificate verification for simplicity
    config.server_name = SERVER_HOST
    
    try:
        # Connect to server
        async with connect(SERVER_HOST, SERVER_PORT, configuration=config) as connection:
            stream = await connection.create_stream()
            print(f"Connected to {SERVER_HOST}:{SERVER_PORT}")
            
            # Capture and send frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Encode frame as JPEG
                _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Send frame
                await send_frame(stream, jpeg_data.tobytes())
                
                # Small delay to avoid overwhelming the connection
                await asyncio.sleep(0.033)  # ~30 FPS
                
    except Exception as e:
        print(f"Error: {e!r}")
    finally:
        cap.release()
        print("Camera released")

if __name__ == "__main__":
    asyncio.run(main())

