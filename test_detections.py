#!/usr/bin/env python3
"""Test script: Hit /detections endpoint and print JSON."""

import urllib.request
import json

def test_detections():
    """Fetch and print detection results."""
    try:
        with urllib.request.urlopen("http://localhost:8080/detections") as response:
            data = json.loads(response.read())
            print("Detection Results:")
            print(json.dumps(data, indent=2))
            
            print(f"\nFaces detected: {data.get('count', 0)}")
            if data.get('faces'):
                for i, face in enumerate(data['faces']):
                    print(f"  Face {i+1}: confidence={face['confidence']:.2f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_detections()

