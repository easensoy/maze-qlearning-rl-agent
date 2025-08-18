import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import './TrainingCharts.css';

function TrainingCharts({ trainingStats }) {
  if (!trainingStats) return null;
  
  const { rewards_history = [], steps_history = [] } = trainingStats;
  
  // Prepare data for charts
  const rewardData = rewards_history.map((reward, index) => ({
    episode: index + 1,
    reward: reward,
    steps: steps_history[index] || 0
  }));
  
  // Calculate moving averages
  const windowSize = 10;
  const rewardDataWithMA = rewardData.map((item, index) => {
    if (index < windowSize - 1) return { ...item, movingAverage: item.reward };
    
    const window = rewardData.slice(index - windowSize + 1, index + 1);
    const average = window.reduce((sum, w) => sum + w.reward, 0) / window.length;
    
    return { ...item, movingAverage: average };
  });
  
  // Recent performance (last 20 episodes)
  const recentData = rewardDataWithMA.slice(-20);
  
  // Performance metrics
  const avgReward = rewards_history.length > 0 
    ? (rewards_history.reduce((a, b) => a + b, 0) / rewards_history.length).toFixed(2)
    : '0.00';
  
  const avgSteps = steps_history.length > 0
    ? Math.round(steps_history.reduce((a, b) => a + b, 0) / steps_history.length)
    : 0;
  
  const bestReward = rewards_history.length > 0 ? Math.max(...rewards_history).toFixed(2) : '0.00';
  const bestSteps = steps_history.length > 0 ? Math.min(...steps_history) : 0;
  
  return (
    <div className="training-charts">
      <div className="charts-header">
        <h3>📊 Training Progress</h3>
        <div className="performance-metrics">
          <div className="metric">
            <span className="metric-label">Avg Reward:</span>
            <span className="metric-value">{avgReward}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Best Reward:</span>
            <span className="metric-value">{bestReward}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Avg Steps:</span>
            <span className="metric-value">{avgSteps}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Best Steps:</span>
            <span className="metric-value">{bestSteps}</span>
          </div>
        </div>
      </div>
      
      {rewardData.length > 0 && (
        <div className="charts-container">
          <div className="chart-section">
            <h4>Reward Over Episodes</h4>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={rewardDataWithMA} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="reward" 
                  stroke="#8884d8" 
                  strokeWidth={1}
                  dot={false}
                  name="Reward"
                />
                <Line 
                  type="monotone" 
                  dataKey="movingAverage" 
                  stroke="#ff7300" 
                  strokeWidth={2}
                  dot={false}
                  name="Moving Average"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="chart-section">
            <h4>Steps to Complete Episode</h4>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={rewardData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="steps" 
                  stroke="#82ca9d" 
                  strokeWidth={2}
                  dot={false}
                  name="Steps"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          {recentData.length > 0 && (
            <div className="chart-section">
              <h4>Recent Performance (Last 20 Episodes)</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={recentData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="reward" fill="#8884d8" name="Reward" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
      
      {rewardData.length === 0 && (
        <div className="no-data">
          <p>🎯 Start training to see performance charts!</p>
        </div>
      )}
    </div>
  );
}

export default TrainingCharts;