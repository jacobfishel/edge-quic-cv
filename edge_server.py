import asyncio
import cv2
import numpy as np
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

# Cloud server address & port
CLOUD_HOST = "your.cloud.server.ip"  # replace with your cloud server IP or DNS
CLOUD_PORT = 4433

# QUIC TLS configuration (for testing, can skip verification)
quic_config = QuicConfiguration(is_client=True)
quic_config.verify_mode = False  # only for self-signed certs

async def send_video():
    # Connect to the cloud QUIC server
    async with connect(CLOUD_HOST, CLOUD_PORT, configuration=quic_config) as client:
        stream_id = client._quic.get_next_available_stream_id()
        
        # OpenCV webcam capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        print("Streaming video to cloud... Press Ctrl+C to stop.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame for faster transmission (optional)
                frame = cv2.resize(frame, (320, 240))

                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                data = buffer.tobytes()

                # Send frame over QUIC stream
                client._quic.send_stream_data(stream_id, data, end_stream=False)
                await client._loop.run_in_executor(None, client._quic.transmit)

                # Wait for server response (inference results)
                events = await client._loop.run_in_executor(None, client._quic.poll)
                for event in events:
                    if hasattr(event, 'data') and event.data:
                        print("Received result:", event.data.decode())

        except KeyboardInterrupt:
            print("Stopping video stream...")
        finally:
            cap.release()

if __name__ == "__main__":
    asyncio.run(send_video())
