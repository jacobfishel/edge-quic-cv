#!/bin/bash

# Simple script to start the QUIC server

cd "$(dirname "$0")"

echo "=================================================="
echo "Starting Edge QUIC CV Server"
echo "=================================================="

# Kill any existing processes on the ports
echo "[*] Checking for existing processes..."
lsof -ti:8080,8081,6000 | xargs kill -9 2>/dev/null
pkill -f "quic_server.py" 2>/dev/null
sleep 1

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[!] Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "[*] Installing dependencies..."
    source venv/bin/activate
    pip install -r requirments.txt > /dev/null 2>&1
else
    source venv/bin/activate
fi

# Check if SSL certificates exist
if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
    echo "[*] Generating SSL certificates..."
    openssl req -new -x509 -days 365 -nodes -out cert.pem -keyout key.pem -subj "/CN=localhost" > /dev/null 2>&1
fi

# Check if frontend is built
if [ ! -d "frontend/build" ]; then
    echo "[*] Building frontend..."
    npm install > /dev/null 2>&1
    npm run build > /dev/null 2>&1
fi

echo "[*] Starting server..."
echo ""
python quic_server.py

