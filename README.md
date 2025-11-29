# edge-quic-cv
This project utilizes QUIC protocol to send image feed from an edge device (raspberry pi 3) to a model on the cloud to handle the heavy computations and send back to the device. 

Setup steps:
-------------------------------------------------------------------------------------------------------------
1. Create virtual environment
    python3 -m venv venv

-------------------------------------------------------------------------------------------------------------
2. Install dependencies
    pip install -r requirements.txt

-------------------------------------------------------------------------------------------------------------
3. For windows devices:
    ssl is needed to run the quic server
    go to https://slproweb.com/products/Win32OpenSSL.html and install Win64 OpenSSL v3.6.0 Light EXE
    run the downloaded executable to install the CLI

    
    Open the OpenSSL terminal, cd into your project directory and run:
        openssl req -new -x509 -days 365 -nodes -out cert.pem -keyout key.pem -subj "/CN=localhost"

        This will generate two files: cert.pem and key.pem in your project directory. These two files are needed to run the quic server

-------------------------------------------------------------------------------------------------------------
4. For ubuntu simply open your terminal, cd to the project directory and run:
    openssl req -new -x509 -days 365 -nodes -out cert.pem -keyout key.pem -subj "/CN=localhost"

-------------------------------------------------------------------------------------------------------------
5. Run edge_server.py  

-------------------------------------------------------------------------------------------------------------
6. Run edge_client.py

## Quick Start

### Start the Server:
```bash
./start_server.sh
```

This script automatically:
- Kills any existing processes on ports 8080, 8081, and 6000
- Creates virtual environment if needed
- Installs Python dependencies
- Generates SSL certificates if needed
- Builds the React frontend if needed
- Starts the QUIC server

The server starts:
- QUIC server on port 6000 (receives video from client)
- Flask server on port 8080 (serves React frontend)
- WebSocket server on port 8081 (streams video to frontend)

### Start the Client (in another terminal):
```bash
./start_client.sh
```

### View the Dashboard:
Open your browser and navigate to **http://localhost:8080**

The dashboard displays three video feeds:
- **Original Feed** - Raw video from the client
- **Processed Feed** - Processed video with annotations
- **Detection Overlay** - Detection visualization

---

## Manual Setup (if needed)

Install Python dependencies:
   pip install -r requirments.txt
Install Node.js dependencies (if not already installed):
   npm install
Build the React frontend:
   npm run build
Start the QUIC server manually:
   source venv/bin/activate
   python quic_server.py
Start the client manually:
   source venv/bin/activate
   python client.py
