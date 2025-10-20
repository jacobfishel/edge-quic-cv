import asyncio
import cv2
import struct
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

CLOUD_HOST = "127.0.0.1"
CLOUD_PORT = 6000

quic_config = QuicConfiguration(is_client=True)
quic_config.verify_mode = False  # allow self-signed certs

async def capture_frame(cap):
    """Read a frame in a non-blocking way."""
    return await asyncio.to_thread(cap.read)

async def send_video():
    async with connect(CLOUD_HOST, CLOUD_PORT, configuration=quic_config) as client:
        # Create a new bidirectional stream
        stream = await client.create_stream()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        print("Streaming video to server (press Ctrl+C to stop)...")

        try:
            while True:
                ret, frame = await capture_frame(cap)
                if not ret:
                    break

                # Resize + compress frame
                _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)))
                data = buffer.tobytes()

                # Prefix with length for framing
                packet = struct.pack("!I", len(data)) + data

                # Send over QUIC
                await stream.send_data(packet, end_stream=False)

                # Optional: reduce CPU/bandwidth
                await asyncio.sleep(0.03)

        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            cap.release()
            await stream.send_data(b"", end_stream=True)

if __name__ == "__main__":
    asyncio.run(send_video())
