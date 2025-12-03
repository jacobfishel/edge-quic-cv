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

# Thread-safe queue for frames
# ============================================================================
frame_queue = queue.Queue(maxsize=20)  # Increased from 5 to reduce frame drops
# Queue for WebSocket clients (thread-safe)
websocket_clients = set()
websocket_clients_lock = threading.Lock()
# Event loop for WebSocket server
ws_loop = None

# YOLOv8 model (loaded once globally)
yolo_model = YOLO("yolov8n.pt")

# YOLO model (loaded once at startup)
yolo_model = None

# Flask app for serving frontend
app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)


@app.route('/')
def index():
    return send_from_directory('frontend/build', 'index.html')


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
    """Handle QUIC stream - process incoming compressed video frames."""
    print("[+] Client connected over QUIC stream.")
    frame_count = 0
    skip_counter = 0
    chunk_size = 60000  # Match client chunk size
    
    try:
        while True:
            # Read 4-byte length prefix
            length_bytes = await read_exact(reader, 4)
            if not length_bytes:
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
        print(f"[!] UDP receiver error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()
        print(f"[*] UDP receiver closed. Total frames: {frame_count}")




def encode_frame(frame, quality=JPEG_QUALITY, resize_factor=None):
    """Encode frame to base64 JPEG.
    
    Note: Resizing should be done BEFORE calling this function for detection feeds.
    For non-detection feeds, resize_factor can be provided to resize here.
    """
    # Resize frame if resize_factor is provided and < 1.0
    if resize_factor is not None and resize_factor < 1.0:
        new_width = int(frame.shape[1] * resize_factor)
        new_height = int(frame.shape[0] * resize_factor)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return None
    frame_bytes = buffer.tobytes()
    return base64.b64encode(frame_bytes).decode('utf-8')

def detect_persons(frame):
    """Run YOLOv8 person detection and draw bounding boxes."""
    global yolo_model
    if yolo_model is None:
        return frame
    
    results = yolo_model(frame, verbose=False)
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            # Only detect person class (class 0)
            if int(box.cls) == 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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


    # Start UDP frame receiver thread
    udp_thread = threading.Thread(target=udp_frame_receiver, daemon=True)
    udp_thread.start()
    await asyncio.sleep(0.5)  # Give UDP time to start
    print("[✓] UDP frame receiver started")

    # Start frame broadcaster thread
    broadcaster_thread = threading.Thread(target=frame_broadcaster, daemon=True)
    broadcaster_thread.start()
    print("[✓] Frame broadcaster started")

    print("\n" + "=" * 50)
    print("All servers running!")
    print("=" * 50)
    print("NETWORK CONFIGURATION SUMMARY:")
    print("=" * 50)
    print(f"  • Flask Server:     http://{FLASK_HOST}:{FLASK_PORT} (bind: 0.0.0.0)")
    print(f"  • WebSocket Server: ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT} (bind: 0.0.0.0)")
    print(f"  • UDP Receiver:     udp://{UDP_HOST}:{UDP_PORT} (bind: 0.0.0.0)")
    print("")
    print("EXTERNAL CONNECTIONS:")
    print("=" * 50)
    print(f"  • Frontend URL:     http://{AZURE_VM_IP}:{FLASK_PORT}")
    print(f"  • WebSocket URL:   ws://{AZURE_VM_IP}:{WEBSOCKET_PORT}")
    print(f"  • Client UDP:       {AZURE_VM_IP}:{UDP_PORT}")
    print("=" * 50)
    print("\nPress Ctrl+C to stop.\n")
   
    # Keep the event loop running forever
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        print("\n[!] Shutting down...")




if __name__ == "__main__":
    asyncio.run(main())
