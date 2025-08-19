import { CHART_CONFIG } from './constants';

export const prepareRewardData = (rewards_history = [], steps_history = []) => {
  return rewards_history.map((reward, index) => ({
    episode: index + 1,
    reward: reward,
    steps: steps_history[index] || 0
  }));
};

export const calculateMovingAverage = (data, windowSize = CHART_CONFIG.movingAverageWindow) => {
  return data.map((item, index) => {
    if (index < windowSize - 1) return { ...item, movingAverage: item.reward };
    
    const window = data.slice(index - windowSize + 1, index + 1);
    const average = window.reduce((sum, w) => sum + w.reward, 0) / window.length;
    
    return { ...item, movingAverage: average };
  });
};

export const getRecentData = (data, limit = CHART_CONFIG.recentDataLimit) => {
  return data.slice(-limit);
};

export const calculatePerformanceMetrics = (rewards_history = [], steps_history = []) => {
  const avgReward = rewards_history.length > 0 
    ? (rewards_history.reduce((a, b) => a + b, 0) / rewards_history.length).toFixed(2)
    : '0.00';
  
  const avgSteps = steps_history.length > 0
    ? Math.round(steps_history.reduce((a, b) => a + b, 0) / steps_history.length)
    : 0;
  
  const bestReward = rewards_history.length > 0 ? Math.max(...rewards_history).toFixed(2) : '0.00';
  const bestSteps = steps_history.length > 0 ? Math.min(...steps_history) : 0;
  
  return { avgReward, avgSteps, bestReward, bestSteps };
};

export const createChartColors = () => ({
  primary: "#8884d8",
  secondary: "#ff7300",
  success: "#82ca9d",
  warning: "#ffc658",
  danger: "#ff7c7c"
});