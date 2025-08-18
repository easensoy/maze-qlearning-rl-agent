import React, { useState } from 'react';
import './TrainingControls.css';

function TrainingControls({ 
  onStartTraining, 
  onStopTraining, 
  onResetMaze,
  onUpdateParams,
  isTraining,
  trainingStats,
  qValues 
}) {
  const [params, setParams] = useState({
    learning_rate: 0.1,
    epsilon: 0.1,
    speed: 1.0,
    max_episodes: 1000
  });
  
  const [mazeParams, setMazeParams] = useState({
    wall_density: 0.65,
    branching_factor: 0.4,
    dead_end_percentage: 0.3
  });
  
  const [currentMazeSize, setCurrentMazeSize] = useState({
    width: 21,
    height: 21,
    depth: 1
  });
  
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showMazeComplexity, setShowMazeComplexity] = useState(false);
  
  const handleParamChange = (param, value) => {
    const newParams = { ...params, [param]: parseFloat(value) };
    setParams(newParams);
    onUpdateParams(newParams);
  };
  
  const handleMazeSize = (width, height, depth) => {
    const newSize = { width, height, depth };
    setCurrentMazeSize(newSize);
    onUpdateParams({ 
      maze_width: width, 
      maze_height: height, 
      maze_depth: depth,
      ...mazeParams  // Include complexity parameters
    });
  };
  
  const handleMazeComplexityChange = (param, value) => {
    const newMazeParams = { ...mazeParams, [param]: parseFloat(value) };
    setMazeParams(newMazeParams);
    // Immediately regenerate maze with new complexity parameters
    onUpdateParams({
      regenerate: true,
      ...newMazeParams
    });
  };
  
  const generateNewMaze = () => {
    onUpdateParams({
      regenerate: true,
      ...mazeParams
    });
  };
  
  const exportQTableToCSV = () => {
    if (!qValues || !qValues.best_actions) {
      console.log('No Q-table data available for export');
      return;
    }
    
    // Convert Q-values to CSV format
    let csvContent = "State,Action,Q-Value\n";
    
    Object.entries(qValues.best_actions).forEach(([state, data]) => {
      csvContent += `"${state}","${data.action}",${data.value}\n`;
    });
    
    // Create and download CSV file
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `q_table_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    console.log('Q-table exported to CSV successfully');
  };
  
  return (
    <div className="training-controls">
      <div className="controls-section">
        <h3>Training Controls</h3>
        <div className="button-group">
          <button
            className={`btn ${isTraining ? 'btn-danger' : 'btn-success'}`}
            onClick={isTraining ? onStopTraining : onStartTraining}
          >
            {isTraining ? '⏹ Stop Training' : '▶ Start Training'}
          </button>
          <button className="btn btn-warning" onClick={onResetMaze}>
            🔄 Reset Maze
          </button>
        </div>
        <div className="button-group" style={{ marginTop: '12px' }}>
          <button 
            className="btn btn-primary" 
            onClick={exportQTableToCSV}
            disabled={!qValues || !qValues.best_actions}
          >
            💾 Save Q-Table CSV
          </button>
        </div>
      </div>
      
      <div className="controls-section">
        <h4>Training Statistics</h4>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Episode:</span>
            <span className="stat-value">{trainingStats?.episode || 0}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Step:</span>
            <span className="stat-value">{trainingStats?.step || 0}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Reward:</span>
            <span className="stat-value">{trainingStats?.total_reward?.toFixed(2) || '0.00'}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Epsilon:</span>
            <span className="stat-value">{trainingStats?.epsilon?.toFixed(3) || '0.100'}</span>
          </div>
        </div>
      </div>
      
      <div className="controls-section">
        <h4 onClick={() => setShowAdvanced(!showAdvanced)} className="collapsible">
          ⚙️ Parameters {showAdvanced ? '▼' : '▶'}
        </h4>
        {showAdvanced && (
          <div className="params-grid">
            <div className="param-item">
              <label>Learning Rate:</label>
              <input
                type="range"
                min="0.01"
                max="1.0"
                step="0.01"
                value={params.learning_rate}
                onChange={(e) => handleParamChange('learning_rate', e.target.value)}
              />
              <span>{params.learning_rate}</span>
            </div>
            
            <div className="param-item">
              <label>Epsilon (Exploration):</label>
              <input
                type="range"
                min="0.01"
                max="1.0"
                step="0.01"
                value={params.epsilon}
                onChange={(e) => handleParamChange('epsilon', e.target.value)}
              />
              <span>{params.epsilon}</span>
            </div>
            
            <div className="param-item">
              <label>Training Speed:</label>
              <input
                type="range"
                min="0.1"
                max="10.0"
                step="0.1"
                value={params.speed}
                onChange={(e) => handleParamChange('speed', e.target.value)}
              />
              <span>{params.speed}x</span>
            </div>
            
            <div className="param-item">
              <label>Max Episodes:</label>
              <input
                type="number"
                min="10"
                max="10000"
                value={params.max_episodes}
                onChange={(e) => handleParamChange('max_episodes', e.target.value)}
              />
            </div>
          </div>
        )}
      </div>
      
      <div className="controls-section">
        <h4>Maze Size</h4>
        <div className="maze-size-buttons">
          <button 
            className={currentMazeSize.width === 15 ? 'active' : ''}
            onClick={() => handleMazeSize(15, 15, 1)}
          >
            Small (15x15)
          </button>
          <button 
            className={currentMazeSize.width === 21 ? 'active' : ''}
            onClick={() => handleMazeSize(21, 21, 1)}
          >
            Medium (21x21)
          </button>
          <button 
            className={currentMazeSize.width === 31 ? 'active' : ''}
            onClick={() => handleMazeSize(31, 31, 1)}
          >
            Large (31x31)
          </button>
        </div>
        <div className="current-size-display">
          Current: {currentMazeSize.width}x{currentMazeSize.height}
        </div>
      </div>
      
      <div className="controls-section">
        <h4 onClick={() => setShowMazeComplexity(!showMazeComplexity)} className="collapsible">
          🏗️ Maze Complexity {showMazeComplexity ? '▼' : '▶'}
        </h4>
        {showMazeComplexity && (
          <div className="params-grid">
            <div className="param-item">
              <label>Wall Density:</label>
              <input
                type="range"
                min="0.60"
                max="0.85"
                step="0.01"
                value={mazeParams.wall_density}
                onChange={(e) => handleMazeComplexityChange('wall_density', e.target.value)}
              />
              <span>{(mazeParams.wall_density * 100).toFixed(0)}%</span>
            </div>
            
            <div className="param-item">
              <label>Branching Factor:</label>
              <input
                type="range"
                min="0.20"
                max="0.60"
                step="0.01"
                value={mazeParams.branching_factor}
                onChange={(e) => handleMazeComplexityChange('branching_factor', e.target.value)}
              />
              <span>{(mazeParams.branching_factor * 100).toFixed(0)}%</span>
            </div>
            
            
            <div className="param-item full-width">
              <button className="btn btn-primary" onClick={generateNewMaze}>
                🎲 Generate New Maze
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TrainingControls;