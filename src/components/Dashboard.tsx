import React, { useState, useEffect } from 'react';

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

  useEffect(() => {
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
    return () => clearInterval(interval);
  }, []);

  const containerStyle: React.CSSProperties = {
    fontFamily: 'Arial, sans-serif',
    margin: '20px',
    maxWidth: '1200px',
  };

  const sectionStyle: React.CSSProperties = {
    background: 'white',
    padding: '20px',
    borderRadius: '8px',
    marginBottom: '20px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  };

  const detectionItemStyle: React.CSSProperties = {
    padding: '10px',
    margin: '10px 0',
    background: '#f9f9f9',
    borderLeft: '3px solid #4CAF50',
    borderRadius: '4px',
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

  return (
    <div style={containerStyle}>
      <h1>QUIC Video Dashboard</h1>

      <div style={sectionStyle}>
        <h2>Video Stream</h2>
        <img src="http://localhost:8080/video" alt="Video stream" style={{ maxWidth: '100%' }} />
      </div>

      <div style={sectionStyle}>
        <h2>
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
          <div style={{ color: '#666', fontStyle: 'italic' }}>No faces detected</div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;

