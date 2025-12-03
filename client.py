import asyncio
import cv2
import struct
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

CLOUD_HOST = "74.179.82.115"  # Azure VM IP
CLOUD_PORT = 6000
COMPRESSION_QUALITY = 50

async def main():
    # QUIC configuration
    config = QuicConfiguration(is_client=True, alpn_protocols=["hq-29"])
    config.verify_mode = False  # for self-signed certs during testing

    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

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

                # Encode frame with compression
                success, encoded = cv2.imencode(".webp", frame, [int(cv2.IMWRITE_WEBP_QUALITY), COMPRESSION_QUALITY])
                if not success:
                    # Fallback to JPEG if WEBP fails
                    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, COMPRESSION_QUALITY])
                if not success:
                    continue
                
                data = encoded.tobytes()
                length_prefix = struct.pack('>I', len(data))

                # write to the QUIC stream
                writer.write(length_prefix + data)
                #print("writing to QUIC stream")
                await writer.drain()  # ensure data is sent
                #print("ensure data is sent")
                await asyncio.sleep(0.03)

        except KeyboardInterrupt:
            print("\nStopped streaming.")
        finally:
            cap.release()
            writer.close()
            await writer.wait_closed()
            await client.wait_closed()
            print("release resources")
if __name__ == "__main__":
    asyncio.run(main())
