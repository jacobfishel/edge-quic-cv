import React, { useState, useEffect, useRef } from 'react';

const Dashboard: React.FC = () => {
  const [detections, setDetections] = useState<DetectionResults>({
    faces: [],
    count: 0,
    timestamp: null,
  });
  const [videoSrc, setVideoSrc] = useState<string>('');
  const [connectionStatus, setConnectionStatus] = useState<string>('Disconnected');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connectWebSocket = () => {
      try {
        // Connect to Azure VM WebSocket server
        // Use window.location.hostname for same-origin, or hardcode VM IP for external access
        const wsHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
          ? '74.179.82.115'  // Azure VM IP when accessing remotely
          : window.location.hostname;  // Use current hostname when served from VM
        const ws = new WebSocket(`ws://${wsHost}:8081`);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnectionStatus('Connected');
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            if (message.type === 'frame' && message.data) {
              // Handle frame message (support both feed and non-feed formats)
              if (message.feed === 'original' || !message.feed) {
                setVideoSrc(`data:image/jpeg;base64,${message.data}`);
              } else if (message.feed) {
                // Handle other feed types by updating the single video element
                setVideoSrc(`data:image/jpeg;base64,${message.data}`);
              }
            } else if (message.type === 'test') {
              setDebugData(`Test message received: ${message.message}\n\nWaiting for frame data...`);
              console.log('Test message:', message);
            } else {
              setDebugData(`Unknown message type: ${JSON.stringify(message).substring(0, 200)}`);
              console.log('Received message:', message);
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };


        ws.onerror = () => {
          setConnectionStatus('Error');
        };


        ws.onclose = () => {
          setConnectionStatus('Disconnected');
          setTimeout(connectWebSocket, 3000);
        };
      } catch (error) {
        console.error('Error connecting WebSocket:', error);
        setConnectionStatus('Error');
      }
    };


    connectWebSocket();

    const fetchDetections = async () => {
      try {
        const response = await fetch('http://localhost:8080/detections');
        const data: DetectionResults = await response.json();
        setDetections(data);
      } catch (error) {
        console.error('Error fetching detections:', error);
      }
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);


  const containerStyle: React.CSSProperties = {
    fontFamily: 'Arial, sans-serif',
    margin: '0',
    padding: '20px',
    minHeight: '100vh',
    background: '#1a1a1a',
    color: '#e0e0e0',
  };


  const sectionStyle: React.CSSProperties = {
    background: '#2d2d2d',
    padding: '20px',
    borderRadius: '8px',
    marginBottom: '20px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
  };

  const statusStyle: React.CSSProperties = {
    display: 'inline-block',
    padding: '4px 12px',
    borderRadius: '12px',
    fontSize: '14px',
    marginLeft: '10px',
    background: connectionStatus === 'Connected' ? '#4CAF50' : '#f44336',
    color: 'white',
  };


  return (
    <div style={containerStyle}>
      <h1 style={{ color: '#e0e0e0', marginBottom: '20px' }}>
        QUIC YOLOv8 Person Detection
        <span style={statusStyle}>{connectionStatus}</span>
      </h1>


      <div style={sectionStyle}>
        <h2 style={{ color: '#e0e0e0', marginTop: '0' }}>Video Feed</h2>
        {videoSrc ? (
          <img 
            src={videoSrc} 
            alt="Video stream" 
            style={{ 
              width: '50%', 
              height: 'auto',
              border: '2px solid #4CAF50', 
              borderRadius: '4px', 
              display: 'block' 
            }} 
          />
        ) : (
          <div style={{ 
            padding: '60px 20px', 
            textAlign: 'center', 
            color: '#888',
            background: '#2d2d2d',
            borderRadius: '4px'
          }}>
            Waiting for video feed...
          </div>
        )}
      </div>
    </div>
  );
};


export default Dashboard;


