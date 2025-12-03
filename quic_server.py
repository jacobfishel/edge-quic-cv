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

# Network configuration
AZURE_VM_IP = "127.0.0.1"
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

# YOLO models (loaded once at startup)
YOLO_DET_MODEL = None   # Object detection
YOLO_SEG_MODEL = None   # Instance segmentation
YOLO_POSE_MODEL = None  # Pose estimation

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


def run_detection(frame):
    """Run YOLOv8 object detection and draw bounding boxes."""
    global YOLO_DET_MODEL
    if YOLO_DET_MODEL is None:
        return frame

    results = YOLO_DET_MODEL(frame, verbose=False)
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            class_id = int(box.cls)
            class_name = YOLO_DET_MODEL.names.get(class_id, "")
            # Only draw detections for the "person" class
            if class_name == "person":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f'Person {conf:.2f}',
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
    return frame


def run_segmentation(frame):
    """Run YOLOv8 instance segmentation and return annotated frame."""
    global YOLO_SEG_MODEL
    if YOLO_SEG_MODEL is None:
        return frame

    results = YOLO_SEG_MODEL(frame, verbose=False)
    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            # Filter to only "person" class based on model names
            cls_tensor = boxes.cls
            keep_indices = []
            for idx, cls_value in enumerate(cls_tensor):
                class_id = int(cls_value)
                class_name = YOLO_SEG_MODEL.names.get(class_id, "")
                if class_name == "person":
                    keep_indices.append(idx)

            if not keep_indices:
                return frame

            # Apply filtering to boxes and masks before plotting
            boxes.data = boxes.data[keep_indices]
            if r.masks is not None and r.masks.data is not None:
                r.masks.data = r.masks.data[keep_indices]

        # ultralytics plot() returns an annotated image
        annotated = r.plot()
        return annotated
    return frame


def run_pose(frame):
    """Run YOLOv8 pose estimation and return annotated frame."""
    global YOLO_POSE_MODEL
    if YOLO_POSE_MODEL is None:
        return frame

    results = YOLO_POSE_MODEL(frame, verbose=False)
    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes
        keypoints = getattr(r, "keypoints", None)

        if boxes is not None and len(boxes) > 0:
            # Filter to only "person" class based on model names
            cls_tensor = boxes.cls
            keep_indices = []
            for idx, cls_value in enumerate(cls_tensor):
                class_id = int(cls_value)
                class_name = YOLO_POSE_MODEL.names.get(class_id, "")
                if class_name == "person":
                    keep_indices.append(idx)

            if not keep_indices:
                return frame

            # Apply filtering to boxes and keypoints before plotting
            boxes.data = boxes.data[keep_indices]
            if keypoints is not None and keypoints.data is not None:
                keypoints.data = keypoints.data[keep_indices]

        annotated = r.plot()
        return annotated
    return frame

def frame_broadcaster():
    """Broadcasts video stream to all WebSocket clients and shows combined YOLO views."""
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

            # Run all three YOLOv8 tasks on copies of the same frame
            detect_frame = run_detection(frame.copy())
            segment_frame = run_segmentation(frame.copy())
            pose_frame = run_pose(frame.copy())

            # Ensure frames have the same height before concatenation, if needed
            try:
                h = min(detect_frame.shape[0], segment_frame.shape[0], pose_frame.shape[0])
                detect_frame = cv2.resize(detect_frame, (int(detect_frame.shape[1] * h / detect_frame.shape[0]), h))
                segment_frame = cv2.resize(segment_frame, (int(segment_frame.shape[1] * h / segment_frame.shape[0]), h))
                pose_frame = cv2.resize(pose_frame, (int(pose_frame.shape[1] * h / pose_frame.shape[0]), h))
                combined_frame = cv2.hconcat([detect_frame, segment_frame, pose_frame])
            except Exception:
                combined_frame = detect_frame

            # Local OpenCV display disabled for headless/remote operation
            # Previously used cv2.imshow and cv2.waitKey here.

            # Encode each YOLO view frame (use detection as the primary stream)
            detect_b64 = encode_frame(detect_frame, quality=85)
            segment_b64 = encode_frame(segment_frame, quality=85)
            pose_b64 = encode_frame(pose_frame, quality=85)

            if not detect_b64:
                print("[!] Failed to encode detection frame")
                continue
            
            # Get a snapshot of clients
            with websocket_clients_lock:
                clients_copy = list(websocket_clients)
            
            if len(clients_copy) > 0:
                # Send multi-feed frame payload (preserve original 'data' field for compatibility)
                msg = json.dumps({
                    'type': 'frame',
                    'feed': 'original',
                    'data': detect_b64,
                    'detect': detect_b64,
                    'segment': segment_b64,
                    'pose': pose_b64
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
    global YOLO_DET_MODEL, YOLO_SEG_MODEL, YOLO_POSE_MODEL
    print("=" * 50)
    print("Starting Edge QUIC CV Server")
    print("=" * 50)
    
    # Load YOLO models once at startup
    try:
        YOLO_DET_MODEL = YOLO('yolov8n.pt')
        print("[✓] YOLOv8 detection model loaded")
    except Exception as e:
        print(f"[!] Failed to load YOLO detection model: {e}")
        YOLO_DET_MODEL = None

    try:
        YOLO_SEG_MODEL = YOLO('yolov8n-seg.pt')
        print("[✓] YOLOv8 segmentation model loaded")
    except Exception as e:
        print(f"[!] Failed to load YOLO segmentation model: {e}")
        YOLO_SEG_MODEL = None

    try:
        YOLO_POSE_MODEL = YOLO('yolov8n-pose.pt')
        print("[✓] YOLOv8 pose model loaded")
    except Exception as e:
        print(f"[!] Failed to load YOLO pose model: {e}")
        YOLO_POSE_MODEL = None
    
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
        print(f"Running in LOCAL MODE on 127.0.0.1:{FLASK_PORT}")
        print(f"Running in LOCAL MODE on 127.0.0.1:{WEBSOCKET_PORT}")
        print(f"Running in LOCAL MODE on 127.0.0.1:{QUIC_PORT}")
        
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
