import asyncio
import struct
import cv2
import numpy as np
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from cvzone.FaceDetectionModule import FaceDetector

detector = FaceDetector()

# QUIC configuration
quic_config = QuicConfiguration(is_client=False)
quic_config.verify_mode = False  # allow self-signed certs

async def handle_stream(reader, writer):
    """Handle a single QUIC stream from the client."""
    print("[+] Client connected over QUIC stream.")

    buffer = b""
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break  # client disconnected
            buffer += data

            while len(buffer) >= 4:
                frame_len = struct.unpack("!I", buffer[:4])[0]
                if len(buffer) < 4 + frame_len:
                    break  # incomplete frame

                jpg_data = buffer[4:4 + frame_len]
                buffer = buffer[4 + frame_len:]

                # Decode JPEG to image
                np_data = np.frombuffer(jpg_data, np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Detect faces in a thread to avoid blocking
                frame, _ = await asyncio.to_thread(detector.findFaces, frame, True)

                # Display frame
                cv2.imshow("QUIC Stream - Face Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting server display.")
                    return

    except Exception as e:
        print(f"[!] Stream error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        cv2.destroyAllWindows()
        print("[*] Client stream closed.")

async def main():
    server = await serve(
        host="127.0.0.1",
        port=6000,
        configuration=quic_config,
        stream_handler=handle_stream
    )
    print("[*] QUIC Face Detection Server running on port 6000.")

    try:
        await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
