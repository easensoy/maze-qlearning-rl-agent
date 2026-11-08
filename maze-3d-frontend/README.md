# maze-3d-frontend

React dashboard for the maze Q-learning project: a three.js view of the maze and the agent, live training statistics and charts, and controls that drive the Python training server in `../maze_runner_rl`.

## Run

Start the training server first (see the main README), then:

```bash
npm ci      # first time only
npm start   # http://localhost:3001
```

The port comes from `.env` (`PORT=3001`). The app connects to `ws://localhost:8765`; set `REACT_APP_WS_URL` to use another server.

## Test and build

```bash
npm test -- --watchAll=false
npm run build
```

## Container

The [`Dockerfile`](Dockerfile) is a two-stage build:

1. `npm ci` and `npm run build` on `node:22-alpine`.
2. The static bundle served by `nginx:1.28-alpine` on port 3001, configured in [`nginx.conf`](nginx.conf).

Hashed assets under `/static/` are cached as immutable, and `index.html` is revalidated on every load.

Create React App inlines `REACT_APP_*` variables when it builds, so the WebSocket URL is a build argument. The default is `ws://localhost:8765`:

```bash
docker build --build-arg REACT_APP_WS_URL=ws://localhost:8765 -t maze-frontend .
docker run --rm -p 3001:3001 maze-frontend
```

The browser opens the WebSocket, so the URL must be reachable from the browser, not from the container.

## Layout

| Path | Contents |
|---|---|
| `src/hooks/useWebSocket.js` | One WebSocket per tab: receives maze, training and Q-value messages and sends commands |
| `src/components/Maze3D.js`, `src/components/3d/` | Maze walls, agent, goal and the Q-value heatmap; multi-level mazes are laid out level by level |
| `src/components/TrainingControls.js` | Start/stop, maze size, complexity and training parameters |
| `src/components/TrainingCharts.js` | Reward and steps per episode |
| `src/utils/` | Constants, chart data, parameter handlers and CSV export |

Bootstrapped with Create React App (react-scripts 5).
