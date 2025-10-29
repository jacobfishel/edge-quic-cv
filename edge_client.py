# import asyncio
# import cv2
# import struct
# from aioquic.asyncio import connect
# from aioquic.quic.configuration import QuicConfiguration

# CLOUD_HOST = "127.0.0.1"
# CLOUD_PORT = 6000

# quic_config = QuicConfiguration(is_client=True, alpn_protocols="hq-29")
# quic_config.verify_mode = False  # allow self-signed certs

# async def capture_frame(cap):
#     return await asyncio.to_thread(cap.read)

# async def send_video():
#     async with connect(CLOUD_HOST, CLOUD_PORT, configuration=quic_config) as client:
#         stream_id = client._quic.get_next_available_stream_id(is_unidirectional=False)
#         client._quic.send_stream_data(stream_id, b"")  # open the stream

#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             print("Cannot open webcam")
#             return

#         print("Streaming video to server (press Ctrl+C to stop)...")

#         try:
#             while True:
#                 ret, frame = await capture_frame(cap)
#                 if not ret:
#                     break

#                 _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)))
#                 data = buffer.tobytes()
#                 packet = struct.pack("!I", len(data)) + data

#                 client._quic.send_stream_data(stream_id, packet)
#                 await asyncio.sleep(0.03)

#                 # flush QUIC packets
#                 await client._network_loop()

#         except KeyboardInterrupt:
#             print("Stopped.")
#         finally:
#             cap.release()

# if __name__ == "__main__":
#     asyncio.run(send_video())


import asyncio
import cv2
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

CLOUD_HOST = "127.0.0.1"
CLOUD_PORT = 6000

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
                if not ret:
                    break

                # convert frame to bytes (uncompressed raw data)
                data = frame.tobytes()
                length = len(data).to_bytes(4, 'big')  # send 4-byte length prefix

                # write to the QUIC stream
                writer.write(length + data)
                await writer.drain()  # ensure data is sent

                await asyncio.sleep(0.03)

        except KeyboardInterrupt:
            print("\nStopped streaming.")
        finally:
            cap.release()
            writer.close()
            await writer.wait_closed()
            await client.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
