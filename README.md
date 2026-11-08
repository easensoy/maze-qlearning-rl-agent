# Maze Q-learning

[![CI](https://github.com/easensoy/maze-qlearning-rl-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/easensoy/maze-qlearning-rl-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A tabular Q-learning agent recovered from a pipeline in which it never learned. On 180 seeded mazes, from single-level grids to three stacked levels, its greedy route now settles on a shortest route in 140 runs within 1,000 episodes, and in all 180 within 3,000.

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/hero-poster.png">
  <img src="docs/assets/hero.gif" width="800" alt="One recorded training run on a three-level maze, maze seed 12. Level 0 and level 2 are drawn side by side; level 1 holds only the 13 shafts between them. In its first training episode the agent wanders for 1,000 moves without reaching the goal. A time-lapse of checkpoints every 10 episodes shows the cells from which the greedy policy reaches the goal growing back from the goal along the route, until at episode 371 the greedy walk from the start reaches the goal. After 1,000 episodes the greedy route takes 72 moves, the same cells as a breadth-first-search shortest route, and changes level three times through shafts.">
</picture>

<sub>One recorded run, maze seed 12: of the 26 converged three-level runs, the one at the median convergence episode. Playback is compressed and episodes 505 to 1,000 are skipped; each frame says which. <a href="docs/assets/README.md">How the visuals were made</a>.</sub>

| 60 seeded mazes per type | Late training success | Greedy route converged | Moves to converge | In 3,000 episodes | Unseen mazes solved |
|---|---|---|---|---|---|
| 2D, 21&nbsp;×&nbsp;21 | 100% | **54/60** (80–95%) | 91k | 60/60 | 0 of 3,000 |
| One level, 21&nbsp;×&nbsp;21 | 100% | **60/60** (94–100%) | 75k | 60/60 | 0 of 3,000 |
| Three levels, 21&nbsp;×&nbsp;21&nbsp;×&nbsp;3 | 99% | **26/60** (32–56%) | 148k | 60/60 | 0 of 3,000 |

<sub><b>Late training success:</b> share of the last 100 of 1,000 training episodes that reached the goal, still exploring (ε-greedy). <b>Converged:</b> the greedy policy, with no exploration, took a shortest route (checked against breadth-first search) after every one of at least the last 100 episodes; brackets are 95% Wilson intervals. <b>Moves to converge:</b> median over converged runs only; counting the others as never converging, the medians are 98k, 75k and not reached. <b>Unseen mazes:</b> each trained agent's greedy route on 50 mazes it never saw. One level is the viewer's default.</sub>

**Run** the [live 3D viewer](docs/demo.md) · **Reproduce** the [evaluation](docs/evaluation/README.md#reproduce) and [these visuals](docs/assets/README.md#regenerate) · **Read** [what failed and how it was found](docs/evaluation/case-study.md) and the [full evaluation](docs/evaluation/README.md)

## What was built

- **Mazes.** A 2D generator builds perfect mazes. A 3D generator builds levels joined by vertical shafts; on the 21 × 21 × 3 mazes, the middle level holds little but the shafts. Both are seeded, always solvable and fully connected.
- **Environments.** Gymnasium API. Each exposes its deterministic transition model (`step_from_state`), so planners and tests can use it. The maze stays the same across resets.
- **Agent.** Tabular Q-learning over the agent's cell: ε-greedy (0.1, decaying to 0.01), learning rate 0.1, discount 0.95, and untried actions valued at 0. Rewards are +100 at the goal, −0.1 per move and −1 for bumping into a wall.
- **Viewer.** A WebSocket server trains in a worker thread and streams the maze, the agent's moves, statistics and Q-values to a React/Three.js front end ([guide](docs/demo.md)).
- **Evaluation.** A deterministic suite: 60 mazes per type, with separate maze and exploration seeds. A random walk (computed exactly) and a planner given the maze serve as references. It also runs ablations, follow-up settings and unseen mazes, all with 95% intervals ([report](docs/evaluation/README.md)).
- **Engineering.** 55 Python tests, including the server over a real socket, and 7 frontend tests run in CI, with lint, the production build and a check that both Docker images start.

## What failed, and how it was found

*Archived evidence: an audit on 6 November 2026 of the code as committed at [`3e99fbd`](https://github.com/easensoy/maze-qlearning-rl-agent/tree/3e99fbd) in August 2025.* Every component existed, yet the agent never learned a maze and the viewer never received training data. With the repository's own settings, 0% of training episodes reached the goal, over 25,000 offline episodes and 100 streamed live. Five defects were stacked, so fixing any one alone changed nothing visible:

1. The server's training thread crashed on its first step: it called `asyncio.get_event_loop()` from a worker thread.
2. The viewer's WebSocket hook kept a closed socket as the current one, so its buttons and sliders never reached the server.
3. The 3D generator's dead-end pass walled off the goal in 88% of the mazes the viewer trained on.
4. Every episode built a new maze, which a Q-table indexed by position cannot learn from.
5. When exploiting, the agent considered only actions it had already tried, and 200-step episodes were too short to stumble on the goal.

Each defect was located by measurement, fixed in its own commit and covered by a regression test. The [case study](docs/evaluation/case-study.md) gives the measurement behind each one, and the errors found in the measurements themselves. One example: an early report counted a run as converged after a single optimal check at the very end. Convergence now requires at least 100 consecutive shortest-route checks.

*Rerun at the current code.* The evaluation suite puts the two halves of defect 5 back, one at a time and together, on the same 180 mazes and exploration seeds. Restoring both cuts convergence to 19/60, 15/60 and 3/60 runs, against 54, 60 and 26 now (−38 to −75 points). Each half alone costs less: 18 to 27 points for the short episodes, and 2 to 8 for the action rule.

## What the evidence shows

![Three panels, one per maze type, each with 60 runs. Against moves of experience on a log scale: the share of runs whose exploring training episodes reach the goal reliably rises first; the share whose greedy route is a shortest route rises later and, on three-level mazes, levels off near half.](docs/assets/results.png)

<sub>Each panel: 60 runs on 60 different mazes, 1,000 episodes each. <b>Blue:</b> runs in which at least 45 of the last 50 training episodes reached the goal while exploring. <b>Amber:</b> runs whose greedy route (no exploration) is a shortest route at that point. <b>Dashed:</b> runs that have converged, meaning a shortest route at every later check, for at least 100 checks. Bands are 95% Wilson intervals over the 60 independent runs. A run that has finished its 1,000 episodes keeps its final state. Moves are environment steps, including bumps into walls. Data: <a href="docs/assets/data">docs/assets/data</a>.</sub>

- **It learns a fixed maze.** Within 1,000 episodes the greedy route converges in 60/60 one-level runs, 54/60 2D runs and 26/60 three-level runs. A planner given the maze, breadth-first search or value iteration, finds a shortest route in all 180 mazes without any experience. A uniformly random walk reaches the goal within 1,000 moves in under 1% of attempts (computed exactly, not sampled).
- **Reaching the goal is not learning the route.** On three-level mazes, 99% of the last 100 training episodes reach the goal, helped by exploration. Yet only 26 of the 60 greedy routes have settled on a shortest route.
- **Long routes run out of episodes.** The first shortest greedy route arrives after about 7 to 9 episodes per move of route (correlation 0.92 to 0.98), and no run with a route longer than 120 moves converged. Three times the episodes makes all 180 runs converge. A discount of 0.99 instead of 0.95 rescues 2 of the 40 that failed.

![Left: a converged three-level run, whose greedy route matches a BFS shortest route, but from whose other cells the greedy policy mostly does not reach the goal. Below: an unconverged run after 1,000 episodes, where the cells that lead to the goal cover most of the route but not the start. Right: the share of the shortest route's cells that lead to the goal over training, for both runs.](docs/assets/policy.png)

<sub>Both runs are chosen by rule. Seed 12: of the 26 converged three-level runs, the one at the median convergence episode. Seed 0: the lowest-numbered three-level maze whose run did not converge, continued to 3,000 episodes as in the suite's three-times-the-episodes setting. <b>Blue cells:</b> following the greedy action from there reaches the goal.</sub>

- **It learns a route, not a map.** Value spreads back from the goal one update at a time, and mostly along the route the agent uses. In seed 12 the greedy policy reaches the goal from 19% of the reachable cells. Averaged over all runs, the greedy action is optimal in 54–68% of reachable cells, against the 17–25% a random action would manage.
- **Nothing transfers.** Trained agents reach the goal in 0 of 9,000 tests on unseen mazes. The state is the agent's cell, so a Q-table describes one maze. Training on a new maze every episode, as the original environment did, reaches the goal in almost none of its training episodes and in none of the tests.

These results come from one hyperparameter setting on small, deterministic mazes. They show what this agent learns here. They are not a benchmark result, and nothing here was compared with other learning methods.

## Run and reproduce

The live viewer needs Python 3.13 and Node 22; [docs/demo.md](docs/demo.md) describes the controls. On Windows, `start_backend.bat` and `start_frontend.bat` do the same as the commands below.

```bash
# Terminal 1: the training server, on ws://localhost:8765
python -m venv .venv
.venv/Scripts/python -m pip install -r maze_runner_rl/requirements.txt   # .venv/bin/python on Linux and macOS
cd maze_runner_rl && ../.venv/Scripts/python main.py

# Terminal 2: the viewer, on http://localhost:3001
cd maze-3d-frontend && npm ci && npm start
```

Or run `docker compose up --build`, then open http://localhost:3001.

To run the tests, the evaluation and these visuals, use the virtual environment's Python from `maze_runner_rl/`:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest                             # 55 tests
python -m experiments.evaluate               # the suite behind every number above; 34 min on 20 workers
python -m experiments.plot_evaluation        # the evaluation report's figures
python -m experiments.capture_traces --check ../docs/evaluation/results/runs.csv   # replays the 180 runs
python -m experiments.render_figures         # docs/assets/results.png and policy.png
python -m experiments.render_hero            # docs/assets/hero.gif and hero-poster.png
```

The suite is deterministic. The replay behind the visuals reproduces the suite's record of every one of the 180 runs ([details](docs/assets/README.md)).

## Limitations and remaining work

- **One maze at a time.** The Q-table knows only the agent's position, so it cannot carry anything to a new maze. Generalising would need the maze in the state, for example a small network over the observation, trained and tested on separate maze seeds.
- **Slow value propagation.** One-step updates need about 7 to 9 episodes per move of route, so the default 1,000 episodes are too few for routes over about 120 moves. n-step returns, eligibility traces or Dyna-style planning carry value back faster; comparing them on the same seeds is the natural next experiment.
- **A narrow test bed.** One hyperparameter setting, the viewer's defaults, with nothing tuned. The mazes are small and deterministic, the goal is always in the far corner, and most mazes have a single route.
- **The frontend Docker image runs the development server.** A production build served as static files would be much smaller.

## Repository

| Path | Contents |
|---|---|
| [`maze_runner_rl/`](maze_runner_rl) | Python: generators, environments, agent, server, experiments and tests ([README](maze_runner_rl/README.md)) |
| [`maze-3d-frontend/`](maze-3d-frontend) | React/Three.js viewer ([README](maze-3d-frontend/README.md)) |
| [`docs/evaluation/`](docs/evaluation) | The evaluation report, its figures and raw results, and the [debugging case study](docs/evaluation/case-study.md) |
| [`docs/assets/`](docs/assets) | The visuals on this page, the traces they are drawn from, and how to regenerate them |
| [`docs/demo.md`](docs/demo.md) | The live viewer: how to run it and what it shows |
| `.github/workflows/ci.yml` | Tests, lint, build and Docker checks on every push |

MIT licensed; see [LICENSE](LICENSE).
