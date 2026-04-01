import asyncio
import cv2
import numpy as np
import threading
import queue
import base64
import struct
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from flask import Flask, send_from_directory
from flask_cors import CORS
import websockets
import json
import time
from ultralytics import YOLO
from deepface import DeepFace

# Network configuration
AZURE_VM_IP = "74.179.82.115"
FLASK_HOST = "0.0.0.0"  # Bind to all interfaces to accept external connections
FLASK_PORT = 8080
WEBSOCKET_HOST = "0.0.0.0"  # Bind to all interfaces to accept external connections
WEBSOCKET_PORT = 8081
QUIC_HOST = "0.0.0.0"  # Bind to all interfaces to accept external connections
QUIC_PORT = 6000

# Thread-safe queue for frames
frame_queue = queue.Queue(maxsize=5)
# Queue for WebSocket clients (thread-safe)
websocket_clients = set()
websocket_clients_lock = threading.Lock()
# Event loop for WebSocket server
ws_loop = None

# YOLO model (loaded once at startup)
yolo_model = None
VERIFIED_IMAGE_PATH = "verified.jpg"
DEEPFACE_INTERVAL = 10
last_person_statuses = []
detection_frame_counter = 0

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
    """Handle QUIC stream - process incoming compressed video frames."""
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

            # Process complete frames (length prefix + compressed data)
            while len(buffer) >= 4:
                # Read length prefix
                length = struct.unpack('>I', buffer[:4])[0]
                if len(buffer) < 4 + length:
                    break
                
                # Extract compressed frame data
                compressed_data = buffer[4:4+length]
                buffer = buffer[4+length:]

                # Decode compressed frame
                frame = cv2.imdecode(np.frombuffer(compressed_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Push to display queue (non-blocking)
                try:
                    frame_queue.put_nowait(frame)
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"[*] Processed {frame_count} frames, Queue size: {frame_queue.qsize()}")
                except queue.Full:
                    # Drop oldest frame and add new one
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait(frame)
                    except:
                        pass

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

def detect_persons(frame):
    """Run YOLOv8 person detection and verify faces against verified.jpg."""
    global yolo_model, detection_frame_counter, last_person_statuses
    if yolo_model is None:
        return frame

    detection_frame_counter += 1
    should_verify = (detection_frame_counter % DEEPFACE_INTERVAL) == 0

    results = yolo_model(frame, verbose=False)
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        person_index = 0
        for box in boxes:
            # Only detect person class (class 0)
            if int(box.cls) == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                # Clamp coordinates to valid frame bounds before cropping
                x1i = max(0, int(x1))
                y1i = max(0, int(y1))
                x2i = min(frame.shape[1], int(x2))
                y2i = min(frame.shape[0], int(y2))

                if person_index >= len(last_person_statuses):
                    missing = person_index - len(last_person_statuses) + 1
                    last_person_statuses.extend([False] * missing)

                if should_verify and x2i > x1i and y2i > y1i:
                    person_crop = frame[y1i:y2i, x1i:x2i]
                    try:
                        result = DeepFace.verify(
                            img1_path=person_crop,
                            img2_path=VERIFIED_IMAGE_PATH,
                            enforce_detection=False
                        )
                        last_person_statuses[person_index] = bool(result.get("verified", False))
                    except Exception:
                        last_person_statuses[person_index] = False

                is_verified = last_person_statuses[person_index]
                color = (0, 255, 0) if is_verified else (0, 0, 255)
                label = "Verified" if is_verified else "Unknown"

                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1i, max(15, y1i - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                person_index += 1

        if person_index < len(last_person_statuses):
            last_person_statuses = last_person_statuses[:person_index]

    return frame

def frame_broadcaster():
    """Broadcasts single video stream to all WebSocket clients."""
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
            
            # Run person detection
            frame = detect_persons(frame)
            
            # Encode frame
            frame_b64 = encode_frame(frame, quality=85)
            if not frame_b64:
                print("[!] Failed to encode frame")
                continue
            
            # Get a snapshot of clients
            with websocket_clients_lock:
                clients_copy = list(websocket_clients)
            
            if len(clients_copy) > 0:
                # Send single feed
                msg = json.dumps({
                    'type': 'frame',
                    'feed': 'original',
                    'data': frame_b64
                })
                
                # Schedule sends without blocking
                for client in clients_copy:
                    try:
                        if ws_loop and ws_loop.is_running():
                            async def send_frame(client_ws, message):
                                try:
                                    await client_ws.send(message)
                                except Exception as e:
                                    print(f"[!] Error sending frame to client: {e}")
                                    raise
                            
                            asyncio.run_coroutine_threadsafe(
                                send_frame(client, msg),
                                ws_loop
                            )
                    except Exception as e:
                        print(f"[!] Error scheduling send to client: {e}")
                
                broadcast_count += 1
                
                # Log periodically (every 2 seconds instead of every 30 frames)
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
            # Add a small delay to prevent tight error loop
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
    print(f"[*] Starting Flask server on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)

def run_websocket_server():
    """Run WebSocket server."""
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    print(f"[*] Starting WebSocket server on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    
    async def start_ws_server():
        server = await websockets.serve(websocket_handler, WEBSOCKET_HOST, WEBSOCKET_PORT)
        print(f"[*] WebSocket server running on port {WEBSOCKET_PORT}.")
        await asyncio.Future()  # Run forever
    
    ws_loop.run_until_complete(start_ws_server())

async def main():
    global yolo_model
    print("=" * 50)
    print("Starting Edge QUIC CV Server")
    print("=" * 50)
    
    # Load YOLO model once at startup
    try:
        yolo_model = YOLO('yolov8n.pt')
        print("[✓] YOLOv8 model loaded")
    except Exception as e:
        print(f"[!] Failed to load YOLO model: {e}")
        yolo_model = None
    
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
    print(f"[*] Starting QUIC server on udp://{QUIC_HOST}:{QUIC_PORT}...")
    try:
        # Create stream handler wrapper
        stream_handler = create_stream_handler()
        
        # Start the QUIC server
        server_task = asyncio.create_task(serve(
            host=QUIC_HOST,
            port=QUIC_PORT,
            configuration=quic_config,
            stream_handler=stream_handler,
        ))
        
        await asyncio.sleep(0.5)  # Give it a moment to start
        print("[✓] QUIC server started")
        
        print("\n" + "=" * 50)
        print("All servers running!")
        print("=" * 50)
        print(f"  • Frontend:  http://{AZURE_VM_IP}:{FLASK_PORT}")
        print(f"  • WebSocket: ws://{AZURE_VM_IP}:{WEBSOCKET_PORT}")
        print(f"  • QUIC:      udp://{AZURE_VM_IP}:{QUIC_PORT}")
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
