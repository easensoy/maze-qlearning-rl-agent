import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { 
  prepareRewardData, 
  calculateMovingAverage, 
  getRecentData, 
  calculatePerformanceMetrics,
  createChartColors 
} from '../utils/chartUtils';
import { CHART_CONFIG } from '../utils/constants';
import './TrainingCharts.css';

function TrainingCharts({ trainingStats }) {
  if (!trainingStats) return null;
  
  const { rewards_history = [], steps_history = [] } = trainingStats;
  
  const rewardData = prepareRewardData(rewards_history, steps_history);
  const rewardDataWithMA = calculateMovingAverage(rewardData);
  const recentData = getRecentData(rewardDataWithMA);
  const { avgReward, avgSteps, bestReward, bestSteps } = calculatePerformanceMetrics(rewards_history, steps_history);
  const colors = createChartColors();
  
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
              <LineChart data={rewardDataWithMA} margin={CHART_CONFIG.chartMargin}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="reward" 
                  stroke={colors.primary}
                  strokeWidth={1}
                  dot={false}
                  name="Reward"
                />
                <Line 
                  type="monotone" 
                  dataKey="movingAverage" 
                  stroke={colors.secondary}
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
              <LineChart data={rewardData} margin={CHART_CONFIG.chartMargin}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="steps" 
                  stroke={colors.success}
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
                <BarChart data={recentData} margin={CHART_CONFIG.chartMargin}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="reward" fill={colors.primary} name="Reward" />
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