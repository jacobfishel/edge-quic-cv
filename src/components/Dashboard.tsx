import React, { useState, useEffect, useRef } from 'react';

interface Face {
  bbox: [number, number, number, number];
  confidence: number;
}

interface DetectionResults {
  faces: Face[];
  count: number;
  timestamp: number | null;
}

const Dashboard: React.FC = () => {
  const [detections, setDetections] = useState<DetectionResults>({
    faces: [],
    count: 0,
    timestamp: null,
  });
  const [videoSrc, setVideoSrc] = useState<string>('');
  const [connectionStatus, setConnectionStatus] = useState<string>('Disconnected');
  const [debugData, setDebugData] = useState<string>('');
  const [messageCount, setMessageCount] = useState<number>(0);
  const [lastMessage, setLastMessage] = useState<string>('');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8081');
        wsRef.current = ws;

        ws.onopen = () => {
          setConnectionStatus('Connected');
          setDebugData('WebSocket connection opened. Waiting for messages...');
        };

        ws.onmessage = (event) => {
          setMessageCount(prev => prev + 1);
          setLastMessage(event.data);
          
          // Show raw data preview (first 200 chars)
          const preview = event.data.length > 200 
            ? event.data.substring(0, 200) + '...' 
            : event.data;
          
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
            setDebugData(`Parse error: ${error}\nRaw data: ${preview}`);
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

  const detectionItemStyle: React.CSSProperties = {
    padding: '10px',
    margin: '10px 0',
    background: '#3a3a3a',
    borderLeft: '3px solid #4CAF50',
    borderRadius: '4px',
    color: '#e0e0e0',
  };

  const countBadgeStyle: React.CSSProperties = {
    display: 'inline-block',
    background: '#4CAF50',
    color: 'white',
    padding: '4px 12px',
    borderRadius: '12px',
    fontSize: '14px',
    marginLeft: '10px',
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
        QUIC Video Dashboard
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

