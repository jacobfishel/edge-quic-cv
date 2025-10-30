import asyncio
import struct
import cv2
import numpy as np
import threading
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from flask import Flask, Response

# Flask setup
app = Flask(__name__)

latest_frame = None  # shared variable between QUIC and Flask

@app.route("/")
def index():
    return "<h1>QUIC Stream Viewer</h1><img src='/video'>"

@app.route("/video")
def video_feed():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                # encode the frame as JPEG
                _, jpeg = cv2.imencode(".jpg", latest_frame)
                frame = jpeg.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                # wait briefly if no frame yet
                asyncio.run(asyncio.sleep(0.05))
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# QUIC handler
async def handle_stream(reader, writer):
    global latest_frame
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

                width, height = 640, 480  # match client
                np_frame = np.frombuffer(frame_data, dtype=np.uint8)

                try:
                    frame = np_frame.reshape((height, width, 3))
                    latest_frame = frame
                except ValueError:
                    continue

    except Exception as e:
        print(f"[!] Stream error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print("[*] Client stream closed.")

async def start_quic_server():
    quic_config = QuicConfiguration(is_client=False, alpn_protocols=["hq-29"])
    quic_config.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

    await serve(
        host="127.0.0.1",
        port=6000,
        configuration=quic_config,
        stream_handler=handle_stream
    )
    print("[*] QUIC Face Detection Server running on port 6000.")
    await asyncio.Future()  # keep running

def start_flask():
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

if __name__ == "__main__":
    # Run Flask in a separate thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    # Run QUIC server (main thread)
    asyncio.run(start_quic_server())
