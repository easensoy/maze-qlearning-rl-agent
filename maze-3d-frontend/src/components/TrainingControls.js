import React, { useState } from 'react';
import { Button } from './common';
import { 
  DEFAULT_TRAINING_PARAMS, 
  DEFAULT_MAZE_CONFIG, 
  MAZE_SIZES 
} from '../utils/constants';
import { 
  createParameterHandler, 
  createMazeSizeHandler, 
  createComplexityHandler 
} from '../utils/parameterUtils';
import { exportQTableToCSV } from '../utils/fileUtils';
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
  const [params, setParams] = useState(DEFAULT_TRAINING_PARAMS);
  const [mazeParams, setMazeParams] = useState(DEFAULT_MAZE_CONFIG);
  const [currentMazeSize, setCurrentMazeSize] = useState(MAZE_SIZES.MEDIUM);
  
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showMazeComplexity, setShowMazeComplexity] = useState(false);
  
  const handleParamChange = createParameterHandler(params, setParams, onUpdateParams);
  const handleMazeSize = createMazeSizeHandler(setCurrentMazeSize, onUpdateParams);
  const handleMazeComplexityChange = createComplexityHandler(mazeParams, setMazeParams, onUpdateParams);
  
  const generateNewMaze = () => {
    onUpdateParams({
      regenerate: true,
      ...mazeParams
    });
  };
  
  const handleExportQTable = () => exportQTableToCSV(qValues);
  
  return (
    <div className="training-controls">
      <div className="controls-section">
        <h3>Training Controls</h3>
        <div className="button-group">
          <Button
            variant={isTraining ? 'danger' : 'success'}
            onClick={isTraining ? onStopTraining : onStartTraining}
          >
            {isTraining ? '⏹ Stop Training' : '▶ Start Training'}
          </Button>
          <Button variant="warning" onClick={onResetMaze}>
            🔄 Reset Maze
          </Button>
        </div>
        <div className="button-group" style={{ marginTop: '12px' }}>
          <Button 
            variant="primary"
            onClick={handleExportQTable}
            disabled={!qValues || !qValues.best_actions}
          >
            💾 Save Q-Table CSV
          </Button>
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
            {[
              { key: 'learning_rate', label: 'Learning Rate', type: 'range', min: 0.01, max: 1.0, step: 0.01 },
              { key: 'epsilon', label: 'Epsilon (Exploration)', type: 'range', min: 0.01, max: 1.0, step: 0.01 },
              { key: 'speed', label: 'Training Speed', type: 'range', min: 0.1, max: 10.0, step: 0.1, suffix: 'x' },
              { key: 'max_episodes', label: 'Max Episodes', type: 'number', min: 10, max: 10000 }
            ].map(({ key, label, type, min, max, step, suffix }) => (
              <div key={key} className="param-item">
                <label>{label}:</label>
                <input
                  type={type}
                  min={min}
                  max={max}
                  step={step}
                  value={params[key]}
                  onChange={(e) => handleParamChange(key, e.target.value)}
                />
                <span>{params[key]}{suffix || ''}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      
      <div className="controls-section">
        <h4>Maze Size</h4>
        <div className="maze-size-buttons">
          {Object.entries(MAZE_SIZES).map(([key, size]) => (
            <button 
              key={key}
              className={currentMazeSize.width === size.width ? 'active' : ''}
              onClick={() => handleMazeSize(size.width, size.height, size.depth, mazeParams)}
            >
              {size.label}
            </button>
          ))}
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
            {[
              { key: 'wall_density', label: 'Wall Density', min: 0.60, max: 0.85 },
              { key: 'branching_factor', label: 'Branching Factor', min: 0.20, max: 0.60 }
            ].map(({ key, label, min, max }) => (
              <div key={key} className="param-item">
                <label>{label}:</label>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step="0.01"
                  value={mazeParams[key]}
                  onChange={(e) => handleMazeComplexityChange(key, e.target.value)}
                />
                <span>{(mazeParams[key] * 100).toFixed(0)}%</span>
              </div>
            ))}
            <div className="param-item full-width">
              <Button variant="primary" onClick={generateNewMaze}>
                🎲 Generate New Maze
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TrainingControls;