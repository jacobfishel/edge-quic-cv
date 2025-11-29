#!/usr/bin/env python3
"""Test script: Verify /video endpoint returns MJPEG stream."""

import urllib.request

def test_video():
    """Check if /video returns MJPEG stream."""
    try:
        req = urllib.request.Request("http://localhost:8080/video")
        with urllib.request.urlopen(req, timeout=5) as response:
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            print(f"Content-Type: {content_type}")
            
            if 'multipart/x-mixed-replace' in content_type:
                print("✓ MJPEG stream detected")
            else:
                print("✗ Not an MJPEG stream")
            
            # Read first chunk to verify data
            chunk = response.read(1024)
            if chunk:
                print(f"✓ Received data: {len(chunk)} bytes")
                if chunk.startswith(b'--frame'):
                    print("✓ MJPEG boundary found")
                else:
                    print("⚠ Unexpected data format")
            else:
                print("✗ No data received")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_video()

