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

## Evaluation

```
..\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
..\.venv\Scripts\python.exe -m experiments.evaluate              # 34 minutes on 20 worker processes
..\.venv\Scripts\python.exe -m experiments.plot_evaluation
```

`evaluate.py` trains the agent on 60 mazes of each type, with one exploration seed per maze. It repeats the runs with each repair undone (ablation) and with two follow-up settings (discount 0.99, three times the episodes), and tests every trained agent on 50 unseen mazes. It also computes two references exactly: a random walk and a planner that is given the maze. Results go to `docs/evaluation/results` with 95% intervals, the settings, the commit and a hash of the code that ran. `plot_evaluation.py` draws the figures into `docs/evaluation/figures`. `--quick` runs a small version in under a minute; `--configs` and `--kinds` run a subset. The report is [docs/evaluation](../docs/evaluation/README.md).

## Value propagation: Q-learning, Watkins's Q(λ) and Dyna-Q

```
..\.venv\Scripts\python.exe -m experiments.compare_methods tune      # development mazes -> frozen settings
..\.venv\Scripts\python.exe -m experiments.compare_methods final     # final mazes, frozen settings
..\.venv\Scripts\python.exe -m experiments.compare_methods tables
..\.venv\Scripts\python.exe -m experiments.plot_value_propagation
```

`compare_methods.py` tunes each new method's own setting on development mazes, by rules fixed in the code. It writes the choice to `frozen_settings.json`, then runs all three methods on the final mazes with the same exploration seeds. It records real steps, Q-value updates, planning updates and agent time per run, and a learned-depth diagnostic per episode. The report is [docs/value-propagation](../docs/value-propagation/README.md).

## Claude tool use over MCP

```
..\.venv\Scripts\python.exe -m pip install -r requirements-claude.txt
..\.venv\Scripts\python.exe -m mcp_maze.server                                  # the MCP server alone (stdio)
..\.venv\Scripts\python.exe -m experiments.claude_navigation references         # no API calls
..\.venv\Scripts\python.exe -m experiments.claude_navigation claude --max-usd 20 # needs ANTHROPIC_API_KEY
```

`mcp_maze/` serves isolated, seeded maze sessions through four MCP tools: `reset`, `observe`, `step` and `episode_status`. `claude_navigation.py` evaluates Claude on held-out mazes through those tools, with a depth-first explorer, breadth-first search and a random walk as references. It counts tokens, cost and latency, keeps every transcript and enforces a spending cap. The report is [docs/claude-tool-use](../docs/claude-tool-use/README.md).

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
- the server's stop/restart lifecycle and its WebSocket protocol, over a real socket;
- the evaluation suite: its confidence intervals, the exact random-walk and value-iteration references, the pre-fix agent rule used in the ablation, a deterministic end-to-end run and the figures;
- Watkins's Q(λ) and Dyna-Q, checked by hand:
  - traces carry rewards back, exploration cuts them and ties do not, and λ = 0 is Q-learning;
  - Dyna-Q's model holds only observed transitions, planning propagates value, and n = 0 is Q-learning;
- the value-propagation experiment:
  - its Q-learning loop reproduces the evaluation suite run for run;
  - the learned-depth diagnostic is exact;
  - the selection rules pick as written, and the final runs refuse to start without frozen settings;
- the MCP maze server and the Claude harness:
  - the session rules and the tool contract;
  - the harness's accounting, transcripts, stopping rules and spending cap, driven by a scripted client so no test calls the API;
- the README visuals: that the replay matches the suite run for run, and that the figures and the animation draw from the recorded data.

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
| `src/agents/q_lambda_agent.py`, `dyna_q_agent.py` | `WatkinsQLambdaAgent` and `DynaQAgent`: the same agent with eligibility traces, or with planning from a learned model |
| `src/algorithms/policy_iteration.py` | `PolicyIteration`: exact solver using the env's `step_from_state()` |
| `experiments/training_experiment.py` | `TrainingExperiment` loop, greedy-route checks, the convergence rule and a command-line entry point |
| `experiments/evaluate.py`, `stats.py` | Evaluation suite (seeds, references, ablation, unseen mazes) and its confidence intervals |
| `experiments/plot_evaluation.py` | Figures for `docs/evaluation` |
| `experiments/compare_methods.py`, `plot_value_propagation.py` | The value-propagation experiment and its figures ([docs/value-propagation](../docs/value-propagation/README.md)) |
| `experiments/claude_navigation.py`, `mcp_maze/` | The Claude tool-use evaluation and its MCP server ([docs/claude-tool-use](../docs/claude-tool-use/README.md)) |
| `experiments/capture_traces.py` | Replays the suite's runs for the README figures, checked against its records ([docs/assets](../docs/assets/README.md)) |
| `experiments/capture_hero.py`, `render_hero.py`, `iso_maze.py` | The README animation: recorded runs on one three-level maze, drawn in an exploded 3D view |
| `experiments/render_figures.py`, `asset_style.py` | The README's results and policy figures, and the style the visuals share |
| `tests/` | pytest suite: the agents, generators, environments, training, policy iteration, the server, the evaluation suite, the value-propagation experiment, the MCP server and Claude harness, and the README visuals |
| `requirements.txt`, `requirements-dev.txt`, `requirements-claude.txt` | Runtime; tests and figures; the Claude evaluation |

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
