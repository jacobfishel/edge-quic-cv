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

Install Python dependencies:
   pip install -r requirments.txt
Install Node.js dependencies (if not already installed):
   npm install
Build the React frontend:
   npm run build
This creates the frontend/build directory with the compiled React app.
Start the QUIC server:
   python quic_server.py
This starts:
QUIC server on port 6000 (receives video from client)
Flask server on port 8080 (serves React frontend)
WebSocket server on port 8081 (streams video to frontend)
Start the client (in another terminal):
   python client.py
Open your browser:
Navigate to http://localhost:8080 to see the video feed in the React dashboard.
The video stream should appear in the browser once the client connects and starts sending frames. The dashboard shows connection status and will automatically reconnect if the WebSocket connection drops.
