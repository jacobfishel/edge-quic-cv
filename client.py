import cv2
import socket
import time

# Azure VM public IP and port
CLOUD_HOST = "74.179.82.115"
UDP_PORT = 6000

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_counter = 0

    async with connect(CLOUD_HOST, CLOUD_PORT, configuration=config) as client:
        # create a bidirectional stream
        reader, writer = await client.create_stream()
        print("Connected. Sending raw frames... (Ctrl+C to stop)")

        try:
            while True:
                ret, frame = cap.read()
                #print("reading frame")
                if not ret:
                    break
                    
                # Optional downscaling for performance
                frame = cv2.resize(frame, (640, 480))

                # JPEG encode the frame to dramatically reduce size
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    continue

                data = encoded.tobytes()

                # 4-byte big-endian length prefix
                length = len(data).to_bytes(4, "big")

                # write framed, compressed data to the QUIC stream
                writer.write(length + data)

                frame_counter += 1
                # Only drain occasionally to avoid stalling the stream
                if frame_counter % 10 == 0:
                    await writer.drain()

                await asyncio.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopped streaming.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        sock.close()
        print("Released resources")

if __name__ == "__main__":
    main()