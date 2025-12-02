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
import time
from ultralytics import YOLO

# Thread-safe queue for frames
frame_queue = queue.Queue(maxsize=5)
# Queue for WebSocket clients (thread-safe)
websocket_clients = set()
websocket_clients_lock = threading.Lock()
# Event loop for WebSocket server
ws_loop = None

# YOLOv8 model (loaded once globally)
yolo_model = YOLO("yolov8n.pt")

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


async def read_exact(reader, n: int):
    """Read exactly n bytes from the reader, or return None if EOF is reached early."""
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


async def process_frame_bytes(frame_bytes: bytes, frame_id: int):
    """Decode JPEG bytes, run YOLOv8 person detection, and push annotated frame to the queue."""
    try:
        # Decode JPEG to numpy array (BGR frame)
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[!] Failed to decode frame {frame_id}")
            return

        # Run YOLOv8 inference and keep only 'person' detections (COCO class 0)
        try:
            results = yolo_model(frame, verbose=False)
        except Exception as e:
            print(f"[!] YOLO inference error on frame {frame_id}: {e}")
            results = None

        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    # COCO class 0 is "person"
                    if cls_id != 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    # Draw bounding box and label for person detections
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                    label = f"person {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), max(int(y1) - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        # Push to display queue (non-blocking)
        try:
            frame_queue.put_nowait(frame)
            if frame_id % 30 == 0:  # Log every 30 frames
                print(f"[*] Processed {frame_id} frames, Queue size: {frame_queue.qsize()}")
        except queue.Full:
            # Drop oldest frame and add new one
            try:
                frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except Exception:
                pass
    except Exception as e:
        print(f"[!] Error processing frame {frame_id}: {e}")
        import traceback
        traceback.print_exc()


async def handle_stream(reader, writer):
    """Handle QUIC stream - process incoming video frames."""
    print("[+] Client connected over QUIC stream.")
    frame_count = 0

    try:
        while True:
            # Read 4-byte length prefix
            length_bytes = await read_exact(reader, 4)
            if not length_bytes:
                break
            frame_length = int.from_bytes(length_bytes, "big")
            if frame_length <= 0:
                continue

            # Read the exact frame payload
            frame_data = await read_exact(reader, frame_length)
            if frame_data is None:
                break

            frame_count += 1

            # Process frame in the background so the QUIC handler is not blocked
            asyncio.create_task(process_frame_bytes(frame_data, frame_count))

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
    def handler(reader, writer):
        # Get the current event loop and create a task
        loop = asyncio.get_event_loop()
        loop.create_task(handle_stream(reader, writer))
    return handler


def encode_frame(frame, quality=85):
    """Encode frame to base64 JPEG."""
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return None
    frame_bytes = buffer.tobytes()
    return base64.b64encode(frame_bytes).decode('utf-8')


def frame_broadcaster():
    """Broadcast a single processed video stream to all WebSocket clients."""
    global ws_loop
    broadcast_count = 0
    last_log_time = time.time()
    frame_id = 0
    print("[*] Frame broadcaster thread started")

    while True:
        try:
            # Use timeout to avoid blocking forever
            frame = frame_queue.get(timeout=1)
            frame_id += 1

            # Encode the single processed (annotated) frame
            processed_b64 = encode_frame(frame, quality=85)
            if not processed_b64:
                print("[!] Failed to encode frame")
                continue

            # Get a snapshot of clients
            with websocket_clients_lock:
                clients_copy = list(websocket_clients)

            if len(clients_copy) > 0:
                # Schedule sends without blocking
                for client in clients_copy:
                    try:
                        if ws_loop and ws_loop.is_running():
                            async def send_frame(client_ws, frame_b64, current_frame_id):
                                try:
                                    msg = json.dumps({
                                        'type': 'frame',
                                        'data': frame_b64,
                                        'frameId': current_frame_id,
                                    })
                                    await client_ws.send(msg)
                                    if current_frame_id == 1 or current_frame_id % 30 == 0:
                                        print(f"[*] Sent frame {current_frame_id} to WebSocket client")
                                except Exception as e:
                                    print(f"[!] Error sending frame to client: {e}")
                                    raise

                            asyncio.run_coroutine_threadsafe(
                                send_frame(client, processed_b64, frame_id),
                                ws_loop,
                            )
                    except Exception as e:
                        print(f"[!] Error scheduling send to client: {e}")

                broadcast_count += 1

                # Log periodically (every 2 seconds)
                current_time = time.time()
                if current_time - last_log_time >= 2.0:
                    fps = broadcast_count / (current_time - last_log_time) if broadcast_count > 0 else 0
                    print(f"[*] Broadcasting at ~{fps:.1f} fps to {len(clients_copy)} client(s)")
                    broadcast_count = 0
                    last_log_time = current_time
            else:
                # Only log occasionally when no clients
                if broadcast_count % 100 == 0:
                    print(f"[!] No WebSocket clients connected. Frames in queue: {frame_queue.qsize()}")
                broadcast_count += 1

        except queue.Empty:
            # No frames available, just continue
            continue
        except Exception as e:
            print(f"[!] Frame broadcaster error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

async def websocket_handler(websocket):
    """Handle WebSocket connections for video streaming."""
    print("[+] WebSocket client connected.")
    
    # Add client to set
    with websocket_clients_lock:
        websocket_clients.add(websocket)
        print(f"[*] Total WebSocket clients: {len(websocket_clients)}")
    
    try:
        # Send initial connection confirmation
        await websocket.send(json.dumps({
            'type': 'test', 
            'message': 'WebSocket connected'
        }))
        print("[*] Sent connection confirmation to WebSocket client")
        print(f"[*] WebSocket client ready to receive video feeds")
        
        # Keep connection alive and handle any incoming messages
        async for message in websocket:
            # Echo back any messages (for debugging)
            try:
                data = json.loads(message)
                print(f"[*] Received from client: {data}")
            except:
                pass
                
    except websockets.exceptions.ConnectionClosed:
        print("[*] WebSocket client disconnected normally.")
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Remove client from set
        with websocket_clients_lock:
            websocket_clients.discard(websocket)
            print(f"[*] WebSocket client removed. Remaining clients: {len(websocket_clients)}")

def run_flask():
    """Run Flask server in a separate thread."""
    print("[*] Starting Flask server on http://127.0.0.1:8080")
    app.run(host='127.0.0.1', port=8080, debug=False, use_reloader=False)

def run_websocket_server():
    """Run WebSocket server."""
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    print("[*] Starting WebSocket server on ws://127.0.0.1:8081")
    
    async def start_ws_server():
        server = await websockets.serve(websocket_handler, "127.0.0.1", 8081)
        print("[*] WebSocket server running on port 8081.")
        await asyncio.Future()  # Run forever
    
    ws_loop.run_until_complete(start_ws_server())

async def main():
    print("=" * 50)
    print("Starting Edge QUIC CV Server")
    print("=" * 50)
    
    # Load QUIC configuration
    quic_config = QuicConfiguration(is_client=False, alpn_protocols=["hq-29"])
    quic_config.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    print("[✓] QUIC configuration loaded")

    # Start Flask server thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    await asyncio.sleep(0.5)  # Give Flask time to start
    print("[✓] Flask server started")

    # Start WebSocket server thread
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()
    await asyncio.sleep(0.5)  # Give WebSocket time to start
    print("[✓] WebSocket server started")

    # Start frame broadcaster thread
    broadcaster_thread = threading.Thread(target=frame_broadcaster, daemon=True)
    broadcaster_thread.start()
    print("[✓] Frame broadcaster started")

    # Start QUIC server
    print("[*] Starting QUIC server on udp://127.0.0.1:6000...")
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
        print("[✓] QUIC server started")
        
        print("\n" + "=" * 50)
        print("All servers running!")
        print("=" * 50)
        print("  • Frontend:  http://127.0.0.1:8080")
        print("  • WebSocket: ws://127.0.0.1:8081")
        print("  • QUIC:      udp://127.0.0.1:6000")
        print("=" * 50)
        print("\nPress Ctrl+C to stop.\n")
        
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