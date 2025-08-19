# 🌿 Interactive 3D Maze RL Training System

A full-stack real-time reinforcement learning environment featuring:
- **Interactive 3D maze** with green grass cube walls
- **Real-time RL agent training** visualization
- **Live performance charts** and Q-value heatmaps
- **WebSocket communication** between Python backend and React frontend
- **Three.js** powered 3D graphics with smooth animations

## 🚀 Quick Start

### 1. Start the Backend Server
```bash
# Option 1: Use batch file
double-click start_backend.bat

# Option 2: Manual start
cd "C:\Users\alper\OneDrive\Documents\Maze Runner RL"
venv\Scripts\activate
python maze_runner_rl\backend\websocket_server.py
```

### 2. Start the Frontend
```bash
# Option 1: Use batch file  
double-click start_frontend.bat

# Option 2: Manual start
cd maze-3d-frontend
npm start
```

### 3. Open Browser
Navigate to: **http://localhost:3000**

## 🎮 Features

### 🌿 3D Visualization
- **Green grass cube walls** - Realistic textured wall blocks
- **Orange agent sphere** - Smooth movement animations
- **Gold goal cube** - Animated floating effect
- **Interactive camera** - Zoom, pan, rotate with mouse
- **Q-value heatmaps** - Toggle visualization of learned values

### 🤖 Real-time Training
- **Live agent movement** - Watch the RL agent explore in real-time
- **Training controls** - Start/stop/reset with one click
- **Parameter tuning** - Adjust learning rate, epsilon, speed on the fly
- **Multiple maze sizes** - 7x7x3, 11x11x5, 15x15x7 options

### 📊 Performance Analytics
- **Reward progression** - Line charts with moving averages
- **Episode statistics** - Steps to completion tracking
- **Real-time metrics** - Current episode, reward, epsilon display
- **Performance indicators** - Average and best performance tracking

### ⚙️ Interactive Controls
- **Training speed** - 0.1x to 10x speed adjustment
- **Hyperparameters** - Real-time learning rate and epsilon tuning
- **View modes** - Maze view, charts view, or both simultaneously
- **Q-value overlay** - Toggle to see learned state values

## 🛠️ Technical Stack

### Backend (Python)
- **WebSocket Server** - Real-time bidirectional communication
- **3D Maze Generator** - Depth-first search algorithm
- **Q-Learning Agent** - Epsilon-greedy exploration
- **Async Training Loop** - Non-blocking training execution

### Frontend (React + Three.js)
- **Three.js/React Fiber** - 3D rendering and animations
- **WebSocket Client** - Real-time data synchronization
- **Recharts** - Interactive performance visualizations
- **Responsive Design** - Works on desktop and mobile

## 🎯 How It Works

1. **Backend generates** a 3D maze using depth-first search
2. **WebSocket sends** maze data to frontend in real-time
3. **Three.js renders** green grass cubes, agent, and goal
4. **RL agent trains** using Q-learning with epsilon-greedy policy
5. **Frontend updates** agent position and statistics live
6. **Charts display** training progress and performance metrics
7. **User controls** training parameters and visualization options

## 🔧 Customization

### Maze Parameters
- **Size**: 7x7x3 (Small) to 15x15x7 (Large)
- **Algorithm**: Depth-first search with randomization
- **Walls**: 3D grass cube blocks with realistic textures

### Training Parameters
- **Learning Rate**: 0.01 - 1.0
- **Epsilon**: 0.01 - 1.0 (exploration rate)
- **Training Speed**: 0.1x - 10x real-time
- **Max Episodes**: 10 - 10,000

### Visualization Options
- **Camera Controls**: Orbit, zoom, pan
- **Q-value Heatmaps**: Toggle learned value overlays
- **View Modes**: Maze only, charts only, or split view
- **Real-time Stats**: Episode, reward, epsilon display

## 📁 Project Structure
```
├── maze_runner_rl/
│   ├── backend/
│   │   └── websocket_server.py    # WebSocket RL server
│   ├── src/
│   │   ├── environments/
│   │   │   ├── maze_3d_env.py     # 3D maze environment
│   │   │   └── maze_3d_generator.py # 3D maze generation
│   │   └── agents/
│   │       └── q_learning_agent.py # Q-learning implementation
├── maze-3d-frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Maze3D.js         # Three.js 3D visualization
│   │   │   ├── TrainingControls.js # Control panel
│   │   │   └── TrainingCharts.js  # Performance charts
│   │   ├── hooks/
│   │   │   └── useWebSocket.js    # WebSocket communication
│   │   └── App.js                 # Main application
├── start_backend.bat              # Backend startup script
└── start_frontend.bat             # Frontend startup script
```

## 🎉 Ready to Train!

The system is now ready for interactive 3D maze reinforcement learning! Watch your RL agent learn to navigate through green grass cube mazes in real-time with full visual feedback and performance analytics.

**Next Steps:**
1. Start both backend and frontend
2. Click "▶ Start Training" 
3. Watch the agent learn in real-time
4. Experiment with different parameters
5. Try different maze sizes
6. Toggle Q-value visualization
7. Analyze performance charts