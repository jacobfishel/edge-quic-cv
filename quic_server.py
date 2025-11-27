#!/usr/bin/env python3
"""Minimal QUIC server that receives frames and stores them for YOLO processing."""

import cv2
import os
import asyncio
import struct
import time
import threading
from datetime import datetime, timedelta
import numpy as np
from aioquic.asyncio import serve
from aioquic.quic.configuration import QuicConfiguration
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# Global variables
latest_frame = None
detection_results = {"faces": [], "count": 0, "timestamp": None}
model = None
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"

# Flask app
app = Flask(__name__)

def ensure_certificate():
    """Create a self-signed certificate if none exists."""
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "edge-quic"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow() - timedelta(days=1))
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(private_key=key, algorithm=hashes.SHA256())
    )

    with open(KEY_FILE, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with open(CERT_FILE, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

@app.route("/video")
def video_feed():
    """Stream current frame as MJPEG."""
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                _, jpeg = cv2.imencode(".jpg", latest_frame)
                frame = jpeg.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.1)
    
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def detections():
    """Return latest YOLO results as JSON."""
    return jsonify(detection_results)

async def handle_stream(reader, writer):
    """Handle incoming QUIC stream data."""
    global latest_frame
    
    buffer = b""
    print("Client connected")
    
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            
            buffer += data
            
            # Process complete frames
            while len(buffer) >= 4:
                # Read frame size (4 bytes, big-endian)
                frame_size = struct.unpack("!I", buffer[:4])[0]
                
                # Check if we have complete frame
                if len(buffer) < 4 + frame_size:
                    break
                
                # Extract frame data
                frame_data = buffer[4:4 + frame_size]
                buffer = buffer[4 + frame_size:]
                
                # Decode JPEG frame
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    latest_frame = frame
                    process_frame(frame)
                    print(f"Received frame: {frame.shape}")
    
    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print("Client disconnected")

def process_frame(frame):
    """Process frame with YOLOv8 and update detection results."""
    global detection_results, model
    
    if model is None or frame is None:
        return
    
    # Run YOLOv8 detection
    results = model(frame, verbose=False)
    
    # Extract face detections
    faces = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            
            faces.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": confidence
            })
    
    # Update global detection results
    detection_results = {
        "faces": faces,
        "count": len(faces),
        "timestamp": time.time()
    }

async def start_server():
    """Start QUIC server."""
    global model
    
    # Load YOLOv8 model once at startup
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # Use yolov8n-face.pt for face-specific model if available
    print("Model loaded")
    
    ensure_certificate()
    config = QuicConfiguration(is_client=False, alpn_protocols=["hq-29"])
    # For simplicity, disable certificate requirements
    config.verify_mode = 0
    config.load_cert_chain(CERT_FILE, KEY_FILE)
    
    host = "0.0.0.0"
    port = 4433
    
    print(f"Starting QUIC server on {host}:{port}")
    
    await serve(
        host=host,
        port=port,
        configuration=config,
        stream_handler=handle_stream
    )
    
    # Keep server running
    await asyncio.Future()

def start_flask():
    """Start Flask server in a separate thread."""
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

if __name__ == "__main__":
    # Start Flask in background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    print("Flask server starting on port 8080")
    
    # Run QUIC server in main thread
    asyncio.run(start_server())

