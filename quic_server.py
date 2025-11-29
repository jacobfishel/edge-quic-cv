import asyncio
import cv2
import numpy as np
import threading
import queue
import base64
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from flask import Flask, send_from_directory
from flask_cors import CORS
import websockets
import json

# Thread-safe queue for frames
frame_queue = queue.Queue(maxsize=5)
# Queue for WebSocket clients (thread-safe)
websocket_clients = set()
websocket_clients_lock = threading.Lock()
# Event loop for WebSocket server
ws_loop = None

# Match the client's capture size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_CHANNELS = 3
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS

# Flask app for serving frontend
app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('frontend/build', 'index.html')

@app.route('/detections')
def detections():
    return {
        'faces': [],
        'count': 0,
        'timestamp': None
    }

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend/build', path)


async def handle_stream(reader, writer):
    """Handle QUIC stream - process incoming video frames."""
    print("[+] Client connected over QUIC stream.")
    buffer = b""
    frame_count = 0

    try:
        while True:
            # Read incoming data
            data = await reader.read(65536)
            if not data:
                break
            buffer += data

            # Process complete frames
            while len(buffer) >= FRAME_SIZE:
                frame_data = buffer[:FRAME_SIZE]
                buffer = buffer[FRAME_SIZE:]

                # Convert bytes → numpy frame
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                    (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
                )

                # Push to display queue
                try:
                    frame_queue.put_nowait(frame)
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"[*] Processed {frame_count} frames")
                except queue.Full:
                    pass  # Drop frame if display thread is behind

    except asyncio.IncompleteReadError:
        print("[!] Client disconnected.")
    except Exception as e:
        print(f"[!] Stream error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"[*] Client stream closed. Total frames: {frame_count}")

def create_stream_handler():
    """Create a stream handler that properly awaits the async function."""
    async def handler(reader, writer):
        await handle_stream(reader, writer)
    return handler


def frame_broadcaster():
    """Broadcasts frames to all WebSocket clients."""
    global ws_loop
    broadcast_count = 0
    print("[*] Frame broadcaster thread started")
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:  # signal to exit
                break
            
            # Convert frame to JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                print("[!] Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
            
            # Send to all connected WebSocket clients
            message = json.dumps({'type': 'frame', 'data': frame_b64})
            message_size = len(message)
            disconnected = set()
            
            with websocket_clients_lock:
                clients_copy = list(websocket_clients)
            
            if len(clients_copy) > 0:
                for client in clients_copy:
                    try:
                        if ws_loop and ws_loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                client.send(message), 
                                ws_loop
                            )
                            # Wait a bit to catch errors
                            future.result(timeout=0.1)
                    except Exception as e:
                        print(f"[!] Error sending to client: {e}")
                        import traceback
                        traceback.print_exc()
                        disconnected.add(client)
                
                # Remove disconnected clients
                with websocket_clients_lock:
                    websocket_clients -= disconnected
                
                broadcast_count += 1
                if broadcast_count % 30 == 0:
                    print(f"[*] Broadcasted {broadcast_count} frames ({message_size} bytes) to {len(websocket_clients)} clients")
            else:
                if broadcast_count % 100 == 0:
                    print(f"[!] No WebSocket clients connected. Frames in queue: {frame_queue.qsize()}")
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[!] Frame broadcaster error: {e}")
            import traceback
            traceback.print_exc()

async def websocket_handler(websocket, path):
    """Handle WebSocket connections for video streaming."""
    print("[+] WebSocket client connected.")
    with websocket_clients_lock:
        websocket_clients.add(websocket)
        print(f"[*] Total WebSocket clients: {len(websocket_clients)}")
    try:
        # Send a test message to verify connection
        await websocket.send(json.dumps({'type': 'test', 'message': 'WebSocket connected'}))
        print("[*] Sent test message to WebSocket client")
        await websocket.wait_closed()
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        with websocket_clients_lock:
            websocket_clients.discard(websocket)
        print("[*] WebSocket client disconnected.")

def run_flask():
    """Run Flask server in a separate thread."""
    app.run(host='127.0.0.1', port=8080, debug=False, use_reloader=False)

def run_websocket_server():
    """Run WebSocket server."""
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    start_server = websockets.serve(websocket_handler, "127.0.0.1", 8081)
    ws_loop.run_until_complete(start_server)
    print("[*] WebSocket server running on port 8081.")
    ws_loop.run_forever()

async def main():
    quic_config = QuicConfiguration(is_client=False, alpn_protocols=["hq-29"])
    quic_config.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

    # Start Flask server thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("[*] Flask server starting on port 8080...")

    # Start WebSocket server thread
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()

    # Start frame broadcaster thread
    broadcaster_thread = threading.Thread(target=frame_broadcaster, daemon=True)
    broadcaster_thread.start()
    print("[*] Frame broadcaster started.")

    # Give other servers time to start
    await asyncio.sleep(1)

    # Start QUIC server
    print("[*] QUIC Server starting on port 6000...")
    try:
        # Create stream handler wrapper
        stream_handler = create_stream_handler()
        # Start the QUIC server
        server_task = asyncio.create_task(serve(
            host="127.0.0.1",
            port=6000,
            configuration=quic_config,
            stream_handler=stream_handler,
        ))
        await asyncio.sleep(0.5)  # Give it a moment to start
        print("[*] QUIC Server running on port 6000.")
        print("[*] All servers running. Press Ctrl+C to stop.")
        
        # Keep the event loop running forever
        await asyncio.Future()
    except KeyboardInterrupt:
        print("\n[!] Shutting down...")
    except Exception as e:
        print(f"[!] QUIC Server error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())