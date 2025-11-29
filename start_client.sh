#!/bin/bash

# Simple script to start the QUIC client

cd "$(dirname "$0")"

echo "=================================================="
echo "Starting Edge QUIC CV Client"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[!] Virtual environment not found. Run start_server.sh first."
    exit 1
fi

source venv/bin/activate

echo "[*] Starting client..."
echo ""
python client.py

