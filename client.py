import cv2
import socket
import time

CLOUD_HOST = "127.0.0.1"
UDP_PORT = 5005

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Connected. Sending raw frames via UDP to {CLOUD_HOST}:{UDP_PORT}... (Ctrl+C to stop)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to bytes (uncompressed raw data)
            data = frame.tobytes()
            
            # Send via UDP
            # Note: UDP max packet size is ~65507 bytes
            # For frames larger than this, we'd need fragmentation
            # Current frame size: 640*480*3 = 921600 bytes (too large for single UDP packet)
            # We'll send in chunks
            chunk_size = 60000  # Safe chunk size for UDP
            total_size = len(data)
            
            # Send frame in chunks with header
            # Header: 4 bytes for total size, 4 bytes for chunk index
            num_chunks = (total_size + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, total_size)
                chunk = data[start:end]
                
                # Prepend header: total_size (4 bytes) + chunk_index (4 bytes) + chunk_data
                header = total_size.to_bytes(4, 'big') + i.to_bytes(4, 'big')
                sock.sendto(header + chunk, (CLOUD_HOST, UDP_PORT))
            
            # Small delay to avoid overwhelming the network
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopped streaming.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        sock.close()
        print("Released resources")

if __name__ == "__main__":
    main()