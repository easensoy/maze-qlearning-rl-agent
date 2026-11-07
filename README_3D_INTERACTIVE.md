# 🌿 Interactive 3D Maze RL Training System

A full-stack real-time reinforcement learning environment featuring:
- **Interactive 3D maze** rendered with Three.js, including multi-level mazes
- **Real-time RL agent training** visualization
- **Live performance charts** and Q-value heatmaps
- **WebSocket communication** between Python backend and React frontend

## 🚀 Quick Start

### 1. Start the Backend Server
```bash
# Option 1: Use batch file (creates .venv on first run)
double-click start_backend.bat

# Option 2: Manual start
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r maze_runner_rl\requirements.txt
cd maze_runner_rl
..\.venv\Scripts\python.exe main.py
```
The server listens on `ws://localhost:8765` (see `maze_runner_rl/websocket/config.yaml`).

### 2. Start the Frontend
```bash
# Option 1: Use batch file (runs npm ci on first run)
double-click start_frontend.bat

# Option 2: Manual start
cd maze-3d-frontend
npm ci
npm start
```

### 3. Open Browser
Navigate to: **http://localhost:3001** (port set in `maze-3d-frontend/.env`)

### Docker (alternative)
`docker compose up --build` is meant to start the backend on port 8765 and the frontend on port 3001. The compose file validates, but the images have not been built and tested yet.

## 🎮 Features

### 🌿 3D Visualization
- **Blue wall blocks** on a dark floor; a multi-level maze is laid out level by level, side by side, with shaft tiles (amber = way up, cyan = way down)
- **Red agent sphere** and **green goal sphere**, drawn on top so walls never hide them
- **Interactive camera** - zoom and rotate with the mouse
- **Q-value heatmap** - floor tiles coloured by the best learned value per cell (red = lowest, green = highest)

### 🤖 Real-time Training
- **Live agent movement** - watch the RL agent explore in real-time
- **Training controls** - start/stop, reset maze (new maze and a fresh agent), save best actions as CSV
- **Parameter tuning** - adjust learning rate, epsilon, speed and episodes per run on the fly
- **Maze sizes** - Small 15x15, Medium 21x21, Large 31x31, and 21x21 with 3 levels

### 📊 Performance Analytics
- **Reward progression** - reward per episode for the last 100 episodes, with a 10-episode moving average
- **Episode statistics** - steps per episode
- **Real-time metrics** - episode, step, reward, epsilon and how often the goal was reached
- **Performance indicators** - average and best reward and steps

### ⚙️ Interactive Controls
- **Training speed** - logarithmic slider from 0.1 steps per second up to unthrottled (at the top, 1000, the delay between steps is removed)
- **Hyperparameters** - real-time learning rate and epsilon tuning
- **View modes** - maze view, charts view, or both simultaneously
- **Q-value overlay** - toggle to see learned state values

## 🛠️ Technical Stack

### Backend (Python)
- **WebSocket Server** - real-time bidirectional communication; per-step updates capped at 30 per second
- **Maze Generators** - depth-first search (recursive backtracker), plus loop corridors and dead-end shaping that never disconnects the maze
- **Gymnasium Environments** - 2D and multi-level 3D mazes (`reset() -> (obs, info)`, five-value `step()`)
- **Q-Learning Agent** - tabular, epsilon-greedy; untried actions count as 0
- **Training Loop** - runs in a worker thread and hands messages to the asyncio loop

### Frontend (React + Three.js)
- **Three.js/React Fiber** - 3D rendering
- **WebSocket Client** - one connection per tab, automatic reconnect
- **Recharts** - performance charts
- **Responsive Design** - layout adapts below 1200 px and 768 px

## 🎯 How It Works

1. **Backend generates** a maze; it stays the same for the whole training run
2. **WebSocket sends** the maze to the frontend when it changes
3. **Three.js renders** walls, agent and goal
4. **RL agent trains** using Q-learning with epsilon-greedy policy
5. **Frontend updates** agent position and statistics live
6. **Charts display** training progress and performance metrics
7. **User controls** training parameters and visualization options

## 🔧 Customization

### Maze Parameters
- **Size**: 15x15 (Small) to 31x31 (Large), or 21x21 with 3 levels
- **Algorithm**: Depth-first search with randomization
- **Wall density**: 0.60 - 0.85, the target share of wall cells; dead ends are trimmed towards it, and trimming stops early once none are left
- **Branching factor**: 0.20 - 0.60, how many extra loop corridors are added

### Training Parameters
- **Learning Rate**: 0.01 - 1.0
- **Epsilon**: 0.01 - 1.0 (exploration rate)
- **Training Speed**: 0.1 steps per second up to unthrottled
- **Max Episodes**: 10 - 10,000 per run (up to 1000 steps per episode)

### Visualization Options
- **Camera Controls**: Orbit and zoom
- **Q-value Heatmaps**: Toggle learned value overlays
- **View Modes**: Maze only, charts only, or split view
- **Real-time Stats**: Episode, reward, epsilon, goals reached

## 📁 Project Structure
```
├── maze_runner_rl/
│   ├── main.py                        # Entry point: python main.py
│   ├── requirements.txt
│   ├── websocket/
│   │   ├── config.yaml                # Server, training, agent and maze settings
│   │   └── websocket_server.py        # WebSocket RL server
│   ├── config/                        # 2D environment settings
│   ├── experiments/
│   │   └── training_experiment.py     # Offline training
│   └── src/
│       ├── environments/
│       │   ├── maze_env.py            # 2D maze environment
│       │   ├── maze_3d_env.py         # 3D maze environment
│       │   └── maze_3d_generator.py   # 3D maze generation
│       ├── agents/
│       │   └── q_learning_agent.py    # Q-learning implementation
│       └── algorithms/
│           └── policy_iteration.py    # Exact solver for comparison
├── maze-3d-frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Maze3D.js              # Three.js 3D visualization
│   │   │   ├── TrainingControls.js    # Control panel
│   │   │   └── TrainingCharts.js      # Performance charts
│   │   ├── hooks/
│   │   │   └── useWebSocket.js        # WebSocket communication
│   │   └── App.js                     # Main application
├── start_backend.bat                  # Backend startup script
├── start_frontend.bat                 # Frontend startup script
└── docker-compose.yml
```

## 🎉 Ready to Train!

**Next Steps:**
1. Start both backend and frontend
2. Click "▶ Start Training"
3. Watch the agent learn in real-time; raise the speed under ⚙️ Parameters to finish a run in seconds
4. Experiment with different parameters
5. Try different maze sizes, including the 3-level maze
6. Toggle Q-value visualization
7. Analyze performance charts
