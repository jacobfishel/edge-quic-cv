import React, { useState, useEffect, useRef } from 'react';


const Dashboard: React.FC = () => {
  const [detectedVideoSrc, setDetectedVideoSrc] = useState<string>('');
  const [segmentedVideoSrc, setSegmentedVideoSrc] = useState<string>('');
  const [poseVideoSrc, setPoseVideoSrc] = useState<string>('');
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
            if (message.type === 'frame' && message.data && message.feed) {
              // Handle individual feed messages
              const feedType = message.feed;
              
              if (feedType === 'detection') {
                setDetectedVideoSrc(`data:image/jpeg;base64,${message.data}`);
              } else if (feedType === 'segmentation') {
                setSegmentedVideoSrc(`data:image/jpeg;base64,${message.data}`);
              } else if (feedType === 'pose') {
                setPoseVideoSrc(`data:image/jpeg;base64,${message.data}`);
              }
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
        QUIC Video Dashboard
        <span style={statusStyle}>{connectionStatus}</span>
      </h1>


      <div style={sectionStyle}>
        <h2 style={{ color: '#e0e0e0', marginTop: '0' }}>YOLOv8 Model Feeds</h2>
        <div style={{
          display: 'flex',
          flexDirection: 'row',
          gap: '20px',
          marginTop: '20px',
          flexWrap: 'wrap'
        }}>
          {/* YOLOv8 Detection Feed */}
          <div style={{ background: '#1a1a1a', padding: '15px', borderRadius: '8px', flex: '1', minWidth: '300px' }}>
            <h3 style={{ color: '#e0e0e0', marginTop: '0', marginBottom: '10px', fontSize: '18px' }}>
              YOLOv8 Detection
            </h3>
            {detectedVideoSrc ? (
              <img
                src={detectedVideoSrc}
                alt="YOLOv8 detection stream"
                style={{
                  width: '100%',
                  height: 'auto',
                  border: '2px solid #9C27B0',
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
                Waiting for YOLOv8 detection feed...
              </div>
            )}
          </div>

          {/* YOLOv8 Segmentation Feed */}
          <div style={{ background: '#1a1a1a', padding: '15px', borderRadius: '8px', flex: '1', minWidth: '300px' }}>
            <h3 style={{ color: '#e0e0e0', marginTop: '0', marginBottom: '10px', fontSize: '18px' }}>
              YOLOv8 Segmentation
            </h3>
            {segmentedVideoSrc ? (
              <img
                src={segmentedVideoSrc}
                alt="YOLOv8 segmentation stream"
                style={{
                  width: '100%',
                  height: 'auto',
                  border: '2px solid #00BCD4',
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
                Waiting for YOLOv8 segmentation feed...
              </div>
            )}
          </div>

          {/* YOLOv8 Pose Estimation Feed */}
          <div style={{ background: '#1a1a1a', padding: '15px', borderRadius: '8px', flex: '1', minWidth: '300px' }}>
            <h3 style={{ color: '#e0e0e0', marginTop: '0', marginBottom: '10px', fontSize: '18px' }}>
              YOLOv8 Pose Estimation
            </h3>
            {poseVideoSrc ? (
              <img
                src={poseVideoSrc}
                alt="YOLOv8 pose estimation stream"
                style={{
                  width: '100%',
                  height: 'auto',
                  border: '2px solid #FF5722',
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
                Waiting for YOLOv8 pose estimation feed...
              </div>
            )}
          </div>
        </div>
      </div>

    </div>
  );
};


export default Dashboard;


