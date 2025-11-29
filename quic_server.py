import asyncio
import cv2
import numpy as np
import threading
import queue
import base64
import socket
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


# YOLOv8 model (loaded once, thread-safe for inference)
yolo_model = 'yolov8n.pt'
yolo_model_lock = threading.Lock()


# Frame skipping: process every Nth frame (1 = every frame, 3 = every 3rd frame)
FRAME_SKIP = 3  # Process every 3rd frame to improve FPS

# UDP configuration
UDP_HOST = "0.0.0.0"
UDP_PORT = 5005

# Match the client's capture size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_CHANNELS = 3
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS

# Performance settings
JPEG_QUALITY = 40  # Lower quality for better performance
RESIZE_FACTOR = 0.75  # Resize frames to 75% for faster processing (optional, set to 1.0 to disable)


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
    skip_counter = 0  # Counter for frame skipping


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


                # Frame skipping: only process every Nth frame
                skip_counter += 1
                if skip_counter % FRAME_SKIP != 0:
                    continue  # Skip this frame


                # Convert bytes → numpy frame
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                    (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
                )


                # Push to display queue (non-blocking)
                try:
                    frame_queue.put_nowait(frame)
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        print(f"[*] Processed {frame_count} frames (skipping {FRAME_SKIP-1} out of {FRAME_SKIP}), Queue size: {frame_queue.qsize()}")
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

def udp_frame_receiver():
    """UDP receiver thread - receives frames from client via UDP (handles chunked frames)."""
    print(f"[*] Starting UDP frame receiver on {UDP_HOST}:{UDP_PORT}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind((UDP_HOST, UDP_PORT))
        print(f"[✓] UDP receiver bound to {UDP_HOST}:{UDP_PORT}")
    except Exception as e:
        print(f"[!] Failed to bind UDP socket: {e}")
        return
    
    # Current frame being assembled
    current_frame_buffer = None
    frame_count = 0
    skip_counter = 0
    chunk_size = 60000  # Match client chunk size
    
    try:
        while True:
            try:
                data, addr = sock.recvfrom(65507)
                if not data or len(data) < 8:  # Need at least 8 bytes for header
                    continue
                
                # Parse header: total_size (4 bytes) + chunk_index (4 bytes)
                total_size = int.from_bytes(data[0:4], 'big')
                chunk_index = int.from_bytes(data[4:8], 'big')
                chunk_data = data[8:]
                
                # Calculate number of chunks
                num_chunks = (total_size + chunk_size - 1) // chunk_size
                
                # Start new frame if chunk_index is 0 (first chunk of new frame)
                if chunk_index == 0:
                    current_frame_buffer = {
                        'chunks': {},
                        'total_size': total_size,
                        'num_chunks': num_chunks
                    }
                
                # Skip if we don't have a current frame buffer
                if current_frame_buffer is None:
                    continue
                
                # Verify this chunk belongs to current frame
                if total_size != current_frame_buffer['total_size']:
                    # New frame started, reset
                    current_frame_buffer = {
                        'chunks': {},
                        'total_size': total_size,
                        'num_chunks': num_chunks
                    }
                
                # Store chunk
                current_frame_buffer['chunks'][chunk_index] = chunk_data
                
                # Check if we have all chunks
                if len(current_frame_buffer['chunks']) == num_chunks:
                    # Reassemble frame
                    frame_data = b""
                    for i in range(num_chunks):
                        if i not in current_frame_buffer['chunks']:
                            # Missing chunk, discard this frame
                            current_frame_buffer = None
                            break
                        frame_data += current_frame_buffer['chunks'][i]
                    else:
                        # All chunks received, process frame
                        if len(frame_data) == total_size and total_size == FRAME_SIZE:
                            # Frame skipping: only process every Nth frame
                            skip_counter += 1
                            if skip_counter % FRAME_SKIP != 0:
                                current_frame_buffer = None
                                continue  # Skip this frame
                            
                            # Convert bytes → numpy frame
                            try:
                                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                                    (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
                                )
                                
                                # Push to display queue (non-blocking)
                                try:
                                    frame_queue.put_nowait(frame)
                                    frame_count += 1
                                    if frame_count % 30 == 0:
                                        print(f"[*] UDP: Processed {frame_count} frames (skipping {FRAME_SKIP-1} out of {FRAME_SKIP}), Queue size: {frame_queue.qsize()}")
                                except queue.Full:
                                    # Drop oldest frame and add new one
                                    try:
                                        frame_queue.get_nowait()
                                        frame_queue.put_nowait(frame)
                                    except:
                                        pass
                            except Exception as e:
                                print(f"[!] UDP frame processing error: {e}")
                        
                        # Reset for next frame
                        current_frame_buffer = None
                        
            except socket.error as e:
                print(f"[!] UDP socket error: {e}")
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"[!] UDP receiver error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
                continue
                
    except KeyboardInterrupt:
        print("\n[!] UDP receiver stopping...")
    except Exception as e:
        print(f"[!] UDP receiver error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()
        print(f"[*] UDP receiver closed. Total frames: {frame_count}")




def encode_frame(frame, quality=JPEG_QUALITY, resize_factor=RESIZE_FACTOR):
    """Encode frame to base64 JPEG with optional resizing."""
    # Resize frame if needed for better performance
    if resize_factor < 1.0:
        new_width = int(frame.shape[1] * resize_factor)
        new_height = int(frame.shape[0] * resize_factor)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return None
    frame_bytes = buffer.tobytes()
    return base64.b64encode(frame_bytes).decode('utf-8')


def run_yolo_inference(frame):
    """Run YOLOv8 inference on a frame and return annotated frame and results.
    Only returns 'person' class detections (class_id == 0)."""
    global yolo_model
    if yolo_model is None:
        return frame.copy(), []
   
    try:
        with yolo_model_lock:
            # Run inference
            results = yolo_model(frame, verbose=False)
           
            # Draw detections on frame
            annotated_frame = results[0].plot()
            detections = []
           
            # Person class ID in COCO dataset is 0
            PERSON_CLASS_ID = 0
           
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Filter: only process 'person' class (class_id == 0)
                    if cls != PERSON_CLASS_ID:
                        continue  # Skip non-person detections
                    
                    class_name = yolo_model.names[cls]  # Should be 'person'
                   
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                   
                    # Draw label
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                   
                    # Store detection
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class': class_name
                    })
           
            return annotated_frame, detections
    except Exception as e:
        print(f"[!] YOLOv8 inference error: {e}")
        return frame.copy(), []


def frame_broadcaster():
    """Broadcasts video streams to all WebSocket clients."""
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
           
            # Feed 1: Original (raw frame) - with throttling
            original_frame = frame.copy()
            original_b64 = encode_frame(original_frame, quality=JPEG_QUALITY, resize_factor=RESIZE_FACTOR)
           
            # Feed 2: Processed (with potential annotations)
            processed_frame = frame.copy()
            cv2.putText(processed_frame, 'Processed Feed', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            processed_b64 = encode_frame(processed_frame, quality=JPEG_QUALITY, resize_factor=RESIZE_FACTOR)
           
            # Feed 3: Detection overlay (visualization view)
            overlay_frame = frame.copy()
            cv2.putText(overlay_frame, 'Detection Overlay', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            overlay_b64 = encode_frame(overlay_frame, quality=JPEG_QUALITY, resize_factor=RESIZE_FACTOR)
           
            # Feed 4: YOLOv8 Detection feed (NEW)
            detected_frame, detections = run_yolo_inference(frame)
            detected_b64 = encode_frame(detected_frame, quality=JPEG_QUALITY, resize_factor=RESIZE_FACTOR)
           
            if not original_b64 or not processed_b64 or not overlay_b64 or not detected_b64:
                print("[!] Failed to encode frame")
                continue
           
            # Get a snapshot of clients
            with websocket_clients_lock:
                clients_copy = list(websocket_clients)
           
            if len(clients_copy) > 0:
                # Send each feed as a separate message with frame ID for synchronization
                feeds = [
                    ('original', original_b64),
                    ('processed', processed_b64),
                    ('overlay', overlay_b64),
                    ('detected', detected_b64)  # NEW: YOLOv8 detection feed
                ]
               
                # Schedule sends without blocking
                for client in clients_copy:
                    try:
                        if ws_loop and ws_loop.is_running():
                            # Create async function with proper closure - send all three feeds
                            async def send_feeds(client_ws, feed_list, current_frame_id):
                                try:
                                    for feed_name, feed_data in feed_list:
                                        msg = json.dumps({
                                            'type': 'frame',
                                            'feed': feed_name,
                                            'data': feed_data,
                                            'frameId': current_frame_id
                                        })
                                        await client_ws.send(msg)
                                    # Log first frame of each batch for debugging
                                    if current_frame_id == 1 or current_frame_id % 30 == 0:
                                        print(f"[*] Sent frame {current_frame_id} with all 4 feeds to WebSocket client")
                                except Exception as e:
                                    print(f"[!] Error sending feeds to client: {e}")
                                    raise  # Re-raise to remove client
                           
                            # Fire and forget - don't wait for result
                            asyncio.run_coroutine_threadsafe(
                                send_feeds(client, feeds, frame_id),
                                ws_loop
                            )
                    except Exception as e:
                        print(f"[!] Error scheduling send to client: {e}")
                        # Mark for removal in the websocket handler itself
               
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
   
    # Load YOLOv8 model
    global yolo_model
    try:
        print("[*] Loading YOLOv8 model...")
        yolo_model = YOLO('yolov8n-seg.pt')  # Use nano model for speed
        print("[✓] YOLOv8 model loaded successfully")
    except Exception as e:
        print(f"[!] Failed to load YOLOv8 model: {e}")
        print("[!] Continuing without YOLOv8 inference")
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


    # Start QUIC server (kept for backward compatibility, but UDP is primary)
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
        print("  • UDP:       udp://0.0.0.0:5005 (PRIMARY)")
        print("  • QUIC:      udp://127.0.0.1:6000 (legacy)")
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
