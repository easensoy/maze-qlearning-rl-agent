import React, { useState } from 'react';
import Maze3D from './components/Maze3D';
import TrainingControls from './components/TrainingControls';
import TrainingCharts from './components/TrainingCharts';
import useWebSocket from './hooks/useWebSocket';
import './App.css';

function App() {
  const [showQValues, setShowQValues] = useState(false);
  const [selectedView, setSelectedView] = useState('maze'); // 'maze', 'charts', 'both'
  
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
  } = useWebSocket('ws://localhost:8765');
  
  const handleUpdateParams = (params) => {
    // Handle maze regeneration or size changes with complexity
    if (params.maze_width || params.maze_height || params.maze_depth || params.regenerate) {
      const complexityParams = {};
      if (params.wall_density !== undefined) complexityParams.wall_density = params.wall_density;
      if (params.branching_factor !== undefined) complexityParams.branching_factor = params.branching_factor;
      if (params.dead_end_percentage !== undefined) complexityParams.dead_end_percentage = params.dead_end_percentage;
      
      initializeMaze(
        params.maze_width || 21,
        params.maze_height || 21, 
        params.maze_depth || 1,
        complexityParams
      );
    } else {
      updateParameters(params);
    }
  };
  
  return (
    <div className="App">
      <header className="app-header">
        <h1>🌿 3D Maze RL Training</h1>
        <div className="connection-status">
          <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
          </div>
          {connectionError && (
            <div className="error-message">⚠️ {connectionError}</div>
          )}
        </div>
        <div className="view-controls">
          <button 
            className={selectedView === 'maze' ? 'active' : ''}
            onClick={() => setSelectedView('maze')}
          >
            🎮 Maze View
          </button>
          <button 
            className={selectedView === 'charts' ? 'active' : ''}
            onClick={() => setSelectedView('charts')}
          >
            📊 Charts View
          </button>
          <button 
            className={selectedView === 'both' ? 'active' : ''}
            onClick={() => setSelectedView('both')}
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
      </header>
      
      <main className="app-main">
        <aside className="controls-panel">
          <TrainingControls
            onStartTraining={startTraining}
            onStopTraining={stopTraining}
            onResetMaze={resetMaze}
            onUpdateParams={handleUpdateParams}
            isTraining={isTraining}
            trainingStats={trainingStats}
          />
        </aside>
        
        <div className="content-area">
          {(selectedView === 'maze' || selectedView === 'both') && (
            <section className="maze-section">
              <h2>🎯 Interactive 3D Maze</h2>
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
          
          {(selectedView === 'charts' || selectedView === 'both') && (
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
