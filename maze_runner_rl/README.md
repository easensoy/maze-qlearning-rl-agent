# maze_runner_rl (backend)

Python side of the maze RL system: maze generators, Gymnasium environments, a tabular
Q-learning agent and the WebSocket server the React frontend talks to.

## Run

```
py -3.13 -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe main.py
```

Run from this folder: `config`, `src`, `experiments` and `websocket` are imported relative to it.
The server listens on `ws://localhost:8765`; override with `MAZE_SERVER_HOST` / `MAZE_SERVER_PORT`.

Offline training without the frontend:

```
..\.venv\Scripts\python.exe -m experiments.training_experiment --env 2d --seed 0
..\.venv\Scripts\python.exe -m experiments.training_experiment --env 3d --depth 3 --seed 0 --save results.json
```

Each run reports three things separately:

- how often training episodes reached the goal, while the agent was still exploring;
- how often the greedy policy (no exploration) reaches the goal;
- how often the greedy policy takes a shortest route. It is checked after every episode against a breadth-first search over the environment's own transition model.

A run counts as converged only if the greedy route is a shortest route at every check for at least the last `--window` episodes (default 100). `--save` writes the per-episode results, the summary and the run settings (git commit, config, library versions).

## Tests

```
..\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
..\.venv\Scripts\python.exe -m pytest
```

The suite covers:

- the Q-learning update rules;
- maze solvability and connectivity across sizes, and the density settings;
- the Gymnasium API, and that `step_from_state()` matches `step()`;
- the convergence rule, plus a learning regression on the UI's default maze;
- policy iteration against BFS;
- the server's stop/restart lifecycle and its WebSocket protocol, over a real socket.

CI runs it on every push (`.github/workflows/ci.yml`).

## Layout

| Path | Contents |
|---|---|
| `main.py` | Starts `websocket.websocket_server.MazeRLServer` |
| `websocket/config.yaml` | Server, training, agent and default maze settings |
| `config/config.yaml` | Actions, rewards and size of the 2D environment |
| `src/environments/maze_env.py` | `MazeEnvironment`: 2D maze, 4 actions |
| `src/environments/maze_3d_env.py` | `Maze3DEnvironment`: multi-level maze, 6 actions (±x, ±y, ±z) |
| `src/environments/maze_generator.py`, `maze_3d_generator.py` | Maze generation (always solvable, fully connected) |
| `src/agents/q_learning_agent.py` | `QLearningAgent`: tabular, epsilon-greedy |
| `src/algorithms/policy_iteration.py` | `PolicyIteration`: exact solver using the env's `step_from_state()` |
| `experiments/training_experiment.py` | `TrainingExperiment` loop, greedy-route checks, the convergence rule and a command-line entry point |
| `tests/` | pytest suite: agent, generators, environments, training, policy iteration and the server |

## Environment contract

Both environments follow the Gymnasium API: `reset(seed=None, options=None)` returns
`(observation, info)` and `step(action)` returns `(observation, reward, terminated, truncated, info)`.
The maze is generated on the first reset and kept afterwards so an agent can learn it; pass a
`seed` or `options={"new_maze": True}` for a new one. `info["state"]` is the agent's cell index,
the state used by the Q-table. Rewards: −0.1 per move, −1 for bumping into a wall, +100 at the goal.

## WebSocket messages

| Direction | `type` | Fields |
|---|---|---|
| client → server | `init_maze` | `width`, `height`, `depth`, optional `wall_density`, `branching_factor`, `dead_end_percentage` (new maze, fresh agent) |
| client → server | `start_training` / `stop_training` | |
| client → server | `reset_maze` | new maze of the same size, fresh agent |
| client → server | `update_params` | `params`: any of `speed` (steps/s), `max_episodes`, `max_steps_per_episode`, `learning_rate`, `epsilon` |
| server → client | `maze_state` | `maze` (depth × height × width, 1 = wall), `agent_position`, `start_position`, `goal_position`, `dimensions` |
| server → client | `training_update` | `stats` (episode, step, total_reward, epsilon, goals_reached, last-100 histories), `is_training`, optional `step_data` |
| server → client | `q_values` | `best_actions`: `{"x,y,z": {"action", "value"}}` |
| server → client | `training_started` / `training_stopped` | |
