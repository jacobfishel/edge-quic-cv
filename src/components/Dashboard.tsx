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
  const [originalVideoSrc, setOriginalVideoSrc] = useState<string>('');
  const [processedVideoSrc, setProcessedVideoSrc] = useState<string>('');
  const [overlayVideoSrc, setOverlayVideoSrc] = useState<string>('');
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
            if (message.type === 'frame' && message.data && message.feed) {
              // Handle individual feed messages
              const feedType = message.feed;
              const frameId = message.frameId || 0;
              
              console.log(`Received ${feedType} feed, frameId: ${frameId}, data length: ${message.data.length}`);
              
              if (feedType === 'original') {
                setOriginalVideoSrc(`data:image/jpeg;base64,${message.data}`);
              } else if (feedType === 'processed') {
                setProcessedVideoSrc(`data:image/jpeg;base64,${message.data}`);
              } else if (feedType === 'overlay') {
                setOverlayVideoSrc(`data:image/jpeg;base64,${message.data}`);
              }
              
              setDebugData(`Frame ID: ${frameId}, Feed: ${feedType}, Data length: ${message.data.length}`);
            } else if (message.type === 'frames') {
              // Legacy: Handle all three feeds in one message
              if (message.original) {
                setOriginalVideoSrc(`data:image/jpeg;base64,${message.original}`);
              }
              if (message.processed) {
                setProcessedVideoSrc(`data:image/jpeg;base64,${message.processed}`);
              }
              if (message.overlay) {
                setOverlayVideoSrc(`data:image/jpeg;base64,${message.overlay}`);
              }
              setDebugData(`Message type: ${message.type}\nReceived all three video feeds`);
            } else if (message.type === 'frame' && message.data && !message.feed) {
              // Legacy single frame support (no feed type)
              const dataPreview = message.data.length > 100 
                ? message.data.substring(0, 100) + '...' 
                : message.data;
              setDebugData(`Message type: ${message.type}\nData length: ${message.data.length}\nData preview: ${dataPreview}`);
              setOriginalVideoSrc(`data:image/jpeg;base64,${message.data}`);
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

    fetchDetections();
    const interval = setInterval(fetchDetections, 500);
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      clearInterval(interval);
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
        <h2 style={{ color: '#e0e0e0', marginTop: '0' }}>Video Feeds</h2>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
          gap: '20px',
          marginTop: '20px'
        }}>
          {/* Original Feed */}
          <div style={{ background: '#1a1a1a', padding: '15px', borderRadius: '8px' }}>
            <h3 style={{ color: '#e0e0e0', marginTop: '0', marginBottom: '10px', fontSize: '18px' }}>
              Original Feed
            </h3>
            {originalVideoSrc ? (
              <img 
                src={originalVideoSrc} 
                alt="Original video stream" 
                style={{ 
                  width: '100%', 
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
                Waiting for original feed...
              </div>
            )}
          </div>

          {/* Processed Feed */}
          <div style={{ background: '#1a1a1a', padding: '15px', borderRadius: '8px' }}>
            <h3 style={{ color: '#e0e0e0', marginTop: '0', marginBottom: '10px', fontSize: '18px' }}>
              Processed Feed
            </h3>
            {processedVideoSrc ? (
              <img 
                src={processedVideoSrc} 
                alt="Processed video stream" 
                style={{ 
                  width: '100%', 
                  height: 'auto',
                  border: '2px solid #2196F3', 
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
                Waiting for processed feed...
              </div>
            )}
          </div>

          {/* Detection Overlay Feed */}
          <div style={{ background: '#1a1a1a', padding: '15px', borderRadius: '8px' }}>
            <h3 style={{ color: '#e0e0e0', marginTop: '0', marginBottom: '10px', fontSize: '18px' }}>
              Detection Overlay
            </h3>
            {overlayVideoSrc ? (
              <img 
                src={overlayVideoSrc} 
                alt="Detection overlay stream" 
                style={{ 
                  width: '100%', 
                  height: 'auto',
                  border: '2px solid #FF9800', 
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
                Waiting for overlay feed...
              </div>
            )}
          </div>
        </div>
      </div>

      <div style={sectionStyle}>
        <h2 style={{ color: '#e0e0e0', marginTop: '0' }}>Debug Info</h2>
        <div style={{ background: '#1a1a1a', padding: '15px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '12px', color: '#0f0', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
          <div style={{ marginBottom: '10px' }}>
            <strong>Messages Received:</strong> {messageCount}
          </div>
          <div style={{ marginBottom: '10px' }}>
            <strong>Last Message Length:</strong> {lastMessage.length} bytes
          </div>
          <div style={{ marginBottom: '10px' }}>
            <strong>Last Message Preview:</strong>
          </div>
          <div style={{ background: '#000', padding: '10px', borderRadius: '4px', maxHeight: '300px', overflow: 'auto' }}>
            {debugData || 'No data received yet...'}
          </div>
        </div>
      </div>

      <div style={sectionStyle}>
        <h2 style={{ color: '#e0e0e0', marginTop: '0' }}>
          Detections
          {detections.count > 0 && <span style={countBadgeStyle}>{detections.count}</span>}
        </h2>
        {detections.faces && detections.faces.length > 0 ? (
          <div>
            {detections.faces.map((face, index) => (
              <div key={index} style={detectionItemStyle}>
                <strong>Face {index + 1}</strong>
                <br />
                Confidence: {(face.confidence * 100).toFixed(1)}%
                <br />
                BBox: [{face.bbox[0].toFixed(0)}, {face.bbox[1].toFixed(0)}, {face.bbox[2].toFixed(0)}, {face.bbox[3].toFixed(0)}]
              </div>
            ))}
          </div>
        ) : (
          <div style={{ color: '#888', fontStyle: 'italic' }}>No faces detected</div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;

