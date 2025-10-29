import asyncio
import struct
import cv2
import numpy as np
import threading
import queue
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from cvzone.FaceDetectionModule import FaceDetector

detector = FaceDetector()

# Thread-safe queue to pass frames to display thread
frame_queue = queue.Queue(maxsize=5)

def display_loop():
    """Runs in a separate thread to handle cv2.imshow()"""
    print("[*] Display thread started.")
    while True:
        frame = frame_queue.get()
        if frame is None:  # signal to stop
            break
        cv2.imshow("QUIC Stream - Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[!] Quitting display.")
            break
    cv2.destroyAllWindows()
    print("[*] Display thread ended.")

async def handle_stream(reader, writer):
    print("[+] Client connected over QUIC stream.")
    buffer = b""

    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            buffer += data

            while len(buffer) >= 4:
                frame_len = int.from_bytes(buffer[:4], "big")
                if len(buffer) < 4 + frame_len:
                    break

                frame_data = buffer[4:4 + frame_len]
                buffer = buffer[4 + frame_len:]

                width, height = 640, 480  # must match client
                np_frame = np.frombuffer(frame_data, dtype=np.uint8)

                try:
                    frame = np_frame.reshape((height, width, 3))
                except ValueError:
                    continue

                # Detect faces (offload to thread)
                frame, _ = await asyncio.to_thread(detector.findFaces, frame, True)

                # Push frame to queue for display thread
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    # drop frame if display is behind
                    pass

    except Exception as e:
        print(f"[!] Stream error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        frame_queue.put(None)
        print("[*] Client stream closed.")

async def main():
    quic_config = QuicConfiguration(is_client=False, alpn_protocols=["hq-29"])
    quic_config.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

    server = await serve(
        host="127.0.0.1",
        port=6000,
        configuration=quic_config,
        stream_handler=handle_stream
    )
    print("[*] QUIC Face Detection Server running on port 6000.")

    # Start display thread
    display_thread = threading.Thread(target=display_loop, daemon=True)
    display_thread.start()

    try:
        await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        pass
    finally:
        frame_queue.put(None)
        display_thread.join()

if __name__ == "__main__":
    asyncio.run(main())
