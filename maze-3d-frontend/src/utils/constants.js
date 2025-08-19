export const DEFAULT_MAZE_CONFIG = {
  width: 21,
  height: 21,
  depth: 1,
  wall_density: 0.65,
  branching_factor: 0.4,
  dead_end_percentage: 0.3
};

export const DEFAULT_TRAINING_PARAMS = {
  learning_rate: 0.1,
  epsilon: 0.1,
  speed: 1.0,
  max_episodes: 1000
};

export const MAZE_SIZES = {
  SMALL: { width: 15, height: 15, depth: 1, label: 'Small (15x15)' },
  MEDIUM: { width: 21, height: 21, depth: 1, label: 'Medium (21x21)' },
  LARGE: { width: 31, height: 31, depth: 1, label: 'Large (31x31)' }
};

export const VIEW_MODES = {
  MAZE: 'maze',
  CHARTS: 'charts',
  BOTH: 'both'
};

export const WEBSOCKET_CONFIG = {
  url: 'ws://localhost:8765',
  maxReconnectAttempts: 5,
  reconnectDelayMultiplier: 2000
};

export const CHART_CONFIG = {
  movingAverageWindow: 10,
  recentDataLimit: 20,
  chartMargin: { top: 5, right: 30, left: 20, bottom: 5 }
};