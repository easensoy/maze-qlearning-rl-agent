import React, { useState } from 'react';
import Maze3D from './components/Maze3D';
import TrainingControls from './components/TrainingControls';
import TrainingCharts from './components/TrainingCharts';
import useWebSocket from './hooks/useWebSocket';
import { VIEW_MODES, WEBSOCKET_CONFIG, DEFAULT_MAZE_CONFIG } from './utils/constants';
import './App.css';

function App() {
  const [showQValues, setShowQValues] = useState(false);
  const [selectedView, setSelectedView] = useState(VIEW_MODES.MAZE);
  
  const {
    isConnected,
    connectionError,
    mazeData,
    agentPosition,
    goalPosition,
    qValues,
    trainingStats,
    isTraining,
    startTraining,
    stopTraining,
    resetMaze,
    updateParameters,
    initializeMaze
  } = useWebSocket(WEBSOCKET_CONFIG.url);
  
  const handleUpdateParams = (params) => {
    if (params.maze_width || params.maze_height || params.maze_depth || params.regenerate) {
      const complexityParams = ['wall_density', 'branching_factor', 'dead_end_percentage']
        .reduce((acc, key) => {
          if (params[key] !== undefined) acc[key] = params[key];
          return acc;
        }, {});
      
      initializeMaze(
        params.maze_width || DEFAULT_MAZE_CONFIG.width,
        params.maze_height || DEFAULT_MAZE_CONFIG.height, 
        params.maze_depth || DEFAULT_MAZE_CONFIG.depth,
        complexityParams
      );
    } else {
      updateParameters(params);
    }
  };
  
  return (
    <div className="App">
      <div className="app-header">
        <div className="header-left">
          <h1>🌿 3D Maze RL Training</h1>
          <div className="connection-status">
            <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
              {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
            </div>
            {connectionError && (
              <div className="error-message">⚠️ {connectionError}</div>
            )}
          </div>
        </div>
        <div className="header-right">
          <div className="view-controls">
            <button 
              className={selectedView === VIEW_MODES.MAZE ? 'active' : ''}
              onClick={() => setSelectedView(VIEW_MODES.MAZE)}
            >
              🎮 Maze View
            </button>
            <button 
              className={selectedView === VIEW_MODES.CHARTS ? 'active' : ''}
              onClick={() => setSelectedView(VIEW_MODES.CHARTS)}
            >
              📊 Charts View
            </button>
            <button 
              className={selectedView === VIEW_MODES.BOTH ? 'active' : ''}
              onClick={() => setSelectedView(VIEW_MODES.BOTH)}
            >
              🔄 Both Views
            </button>
            <label className="q-values-toggle">
              <input
                type="checkbox"
                checked={showQValues}
                onChange={(e) => setShowQValues(e.target.checked)}
              />
              Show Q-Values
            </label>
          </div>
        </div>
      </div>

      <main className="app-main">

        <aside className="controls-panel">
          <TrainingControls
            onStartTraining={startTraining}
            onStopTraining={stopTraining}
            onResetMaze={resetMaze}
            onUpdateParams={handleUpdateParams}
            isTraining={isTraining}
            trainingStats={trainingStats}
            qValues={qValues}
          />
        </aside>
        
        <div className="content-area">
          {(selectedView === VIEW_MODES.MAZE || selectedView === VIEW_MODES.BOTH) && (
            <section className="maze-section">
              {isConnected ? (
                <Maze3D
                  mazeData={mazeData}
                  agentPosition={agentPosition}
                  goalPosition={goalPosition}
                  qValues={qValues}
                  showQValues={showQValues}
                />
              ) : (
                <div className="loading-state">
                  <p>🔄 Connecting to maze server...</p>
                </div>
              )}
            </section>
          )}
          
          {(selectedView === VIEW_MODES.CHARTS || selectedView === VIEW_MODES.BOTH) && (
            <section className="charts-section">
              <TrainingCharts trainingStats={trainingStats} />
            </section>
          )}
        </div>
      </main>
      
      {trainingStats && (
        <footer className="app-footer">
          <div className="quick-stats">
            Episode: {trainingStats.episode} | 
            Reward: {trainingStats.total_reward?.toFixed(2)} | 
            ε: {trainingStats.epsilon?.toFixed(3)}
          </div>
        </footer>
      )}
    </div>
  );
}

export default App;
