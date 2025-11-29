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
    async def handler(reader, writer):
        await handle_stream(reader, writer)
    return handler


def frame_broadcaster():
    """Broadcasts frames to all WebSocket clients."""
    global ws_loop
    broadcast_count = 0
    last_log_time = time.time()
    print("[*] Frame broadcaster thread started")
    
    while True:
        try:
            # Use timeout to avoid blocking forever
            frame = frame_queue.get(timeout=1)
            
            # Convert frame to JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                print("[!] Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
            
            # Prepare message
            message = json.dumps({'type': 'frame', 'data': frame_b64})
            
            # Get a snapshot of clients
            with websocket_clients_lock:
                clients_copy = list(websocket_clients)
            
            if len(clients_copy) > 0:
                # Schedule sends without blocking
                for client in clients_copy:
                    try:
                        if ws_loop and ws_loop.is_running():
                            # Fire and forget - don't wait for result
                            asyncio.run_coroutine_threadsafe(
                                client.send(message), 
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

async def websocket_handler(websocket, path):
    """Handle WebSocket connections for video streaming."""
    print("[+] WebSocket client connected.")
    
    # Add client to set
    with websocket_clients_lock:
        websocket_clients.add(websocket)
        print(f"[*] Total WebSocket clients: {len(websocket_clients)}")
    
    try:
        # Send initial connection confirmation
        await websocket.send(json.dumps({
            'type': 'connected', 
            'message': 'WebSocket connected successfully'
        }))
        print("[*] Sent connection confirmation to WebSocket client")
        
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
    start_server = websockets.serve(websocket_handler, "127.0.0.1", 8081)
    ws_loop.run_until_complete(start_server)
    print("[*] WebSocket server running on port 8081.")
    ws_loop.run_forever()

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
