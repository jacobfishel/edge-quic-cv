Project Overview:
This project is a simple end-to-end security system that streams webcam video from a Raspberry Pi client to a cloud server using QUIC. The cloud server runs YOLOv8 face detection on each frame and exposes the detection results and processed video frames through a small Flask API. A React dashboard displays the video and results. The server will be containerized with Docker and deployed to Azure Container Apps.
Architecture
Client (Raspberry Pi, Python)
Uses OpenCV to capture webcam frames
Uses QUIC to stream frames to the cloud server
Keeps code simple — no unnecessary abstractions
Server (Azure Container App, Python)
Receives QUIC video frames
Runs YOLOv8 or YOLOv8n face detection
Uses Flask to expose:
/video → MJPEG stream
/detections → JSON results
Code must be small, readable, and follow the KISS principle
React Dashboard
Displays MJPEG feed from Flask
Displays detection information
Minimal UI, zero unnecessary dependencies
Deployment
Single lightweight Dockerfile
No Kubernetes, no microservices, no unnecessary configs
Only the essentials needed for Azure Container Apps
Coding Style Requirements
You must always follow these rules:
1. Keep It Simple (KISS)
Write minimal, readable code
Avoid unnecessary abstractions, complex class structures, or “best practice” overengineering
Do not introduce patterns like dependency injection, service layers, or elaborate directory trees
Avoid premature optimization
2. Keep Files Small
Prefer one file per component unless clearly needed
No more than ~200 lines per file unless unavoidable
3. Use Only Necessary Dependencies
Allowed Python libs:
opencv-python
ultralytics (YOLOv8)
aioquic or quicly (whichever keeps the code simpler)
flask
uvicorn (optional, only if needed)
numpy
Allowed React libs:
React
Axios
Nothing else unless absolutely required
4. Straightforward, Procedural Code Is OK
Do NOT force classes
Do NOT build “services,” “managers,” or “controllers” unless essential
5. No Magic
No AI-driven fanciness
No unused helper functions
No scaffolding
No boilerplate frameworks
6. Minimal Docker
One Dockerfile
No docker-compose unless explicitly asked
Keep container small and readable
Output Style Requirements
Whenever you generate code:
Keep explanations short and practical
Include only the needed files
Keep filenames consistent with the project architecture
Use clear comments sparingly — only where helpful
No overly advanced patterns
If the user asks for something complex
Simplify it
Recommend the simplest possible implementation
Avoid introducing complexity