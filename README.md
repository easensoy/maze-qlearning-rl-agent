# Maze Q-learning: why it stalls on long routes

[![CI](https://github.com/easensoy/maze-qlearning-rl-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/easensoy/maze-qlearning-rl-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

On the 60 seeded mazes whose shortest route is longer than 120 moves, this tabular Q-learning agent never settled on a shortest route within 1,000 episodes. Dyna-Q, which carries value back through a learned model, settled on all 60 under a protocol fixed before the final runs.

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/hero-poster.png">
  <img src="docs/assets/hero.gif" width="800" alt="A recorded replay on a three-level maze, seed 54, drawn as three plates stacked apart vertically and joined by 9 shaft columns; the shortest route is 146 moves. In episode 1 the agent wanders 1,000 moves without reaching the goal. After 20,000 to 160,000 real moves, Q-learning's greedy policy leads to the goal from at most 15 cells next to the goal, and its greedy walk from the start loops. After the same real moves, Dyna-Q's greedy policy covers most of the maze and from 40,000 moves its greedy route is a shortest route, 146 moves with 6 level changes. End card: on 30 long-route mazes Dyna-Q converged on 30, Watkins's Q(lambda) on 15 and Q-learning on 0.">
</picture>

<sub>A recorded replay of two real training runs on the same maze and exploration seed, drawn by this repository rather than captured from the live viewer. Playback is compressed and episodes between checkpoints are skipped; each frame says which. <a href="docs/assets/README.md#the-animation">Text version, and how it was made</a>.</sub>

1. **Slow value propagation was the limit.** Beyond 120 moves, Q-learning converged on **0 of 60** mazes, Watkins's Q(λ) on **30** and Dyna-Q on **60**. Where both converged, Dyna-Q needed 14–18× fewer real moves, and spent about 3× the Q-value updates.
2. **Faster propagation speeds up both halves of learning.** Dyna-Q turns its first arrival at the goal into a shortest greedy route 22–32× faster than Q-learning does. On three-level mazes it also finds the goal 4.7–6.3× sooner.
3. **Long eligibility traces break exploration.** Q(λ) helped at λ = 0.3. At λ ≥ 0.7 its traces credit the exploring walk's penalties to the very corridors that led to new cells, and it converged on 0–15 of 60 development runs, against 47 for Q-learning.

**Read** the [value-propagation results](docs/value-propagation/README.md) · **Reproduce** [them](#run-and-reproduce) · **Run** the [live 3D viewer](docs/demo.md) · **See** [how the original pipeline failed](docs/evaluation/case-study.md)

## What was built

- **Mazes and environments.** A 2D generator builds perfect mazes. A 3D generator builds levels joined by vertical shafts. Both are seeded, always solvable and fully connected. The environments follow the Gymnasium API and expose their deterministic transition model (`step_from_state`) for planners and tests.
- **Three tabular agents with one difference.** Q-learning (the repository's agent), Watkins's Q(λ) and Dyna-Q share the learning rate, discount, ε schedule, rewards and initial values. Only the update differs. Each new method follows its textbook algorithm (Sutton and Barto, 1998, Section 7.6; 2018, Section 8.2), checked against the authors' own slides and tested by hand.
- **A viewer.** A WebSocket server trains in a worker thread and streams the maze, the agent's moves and its Q-values to a React/Three.js front end ([guide](docs/demo.md)).
- **Two experiment suites.**
  - **[Evaluation](docs/evaluation/README.md):** the repaired agent on 180 mazes, with references, ablations and unseen mazes.
  - **[Value propagation](docs/value-propagation/README.md):** the three agents under a pre-registered protocol.

  Both are deterministic and record their code, settings and seeds.
- **Engineering.**
  - **Tests:** 91 Python tests, including the server over a real socket, the agents' update rules checked by hand, and the experiment loop reproducing the suite run for run. Also 7 frontend tests.
  - **CI:** runs them all, with lint, the production build and a check that both Docker images start.

## Learning-algorithm experiments

### Value propagation: Q-learning, Watkins's Q(λ) and Dyna-Q

The [evaluation suite](docs/evaluation/README.md) found that Q-learning needs about 7 to 9 episodes per move of route, and that no route longer than 120 moves converged. One-step updates move the goal's value back one state at a time, so the explanation was slow value propagation. This experiment tests it.

**Protocol, fixed before the final runs:**

- **Tuning:** each new method's own setting was chosen on separate development mazes, by rules written into the code. The result was Q(λ) with λ = 0.3 and accumulating traces, and Dyna-Q with 50 planning updates per real move.
- **Freezing:** the settings were committed and pushed before any final maze ran.
- **Final runs:** all three methods ran on the same final mazes with the same exploration seeds: 180 suite mazes plus 30 long-route mazes, chosen by route length alone.
- **Measures:**
  - experience in real environment steps;
  - planning and trace updates and agent time, recorded separately;
  - convergence: the frozen greedy route is shortest at every check for at least the last 100 episodes.

![Four panels, one per maze set: the share of runs whose frozen greedy route is a shortest route, against real environment steps on a log scale. Dyna-Q's curve rises first in every panel and reaches 100% on the long-route set, where Q(λ) reaches 50% and Q-learning stays at 0%.](docs/value-propagation/figures/learning_curves.png)

| Final mazes | Q-learning converged | Q(λ) converged | Dyna-Q converged | Dyna-Q: real moves to converge (median) |
|---|---|---|---|---|
| 2D, 21 × 21 (60) | 54 (80–95%) | 60 (94–100%) | 60 (94–100%) | 5.2k, against 98k for Q-learning |
| One level, 21 × 21 (60) | 60 (94–100%) | 60 (94–100%) | 60 (94–100%) | 5.6k, against 75k |
| Three levels, 21 × 21 × 3 (60) | 26 (32–56%) | 45 (63–84%) | 59 (91–100%) | 33k; Q-learning's median not reached |
| Long routes, over 140 moves (30) | **0** (0–11%) | **15** (33–67%) | **30** (89–100%) | 57k; Q-learning never converged |

<sub>Brackets: 95% Wilson intervals. Medians count a run that never converged as never. Q-learning's rows reproduce the evaluation suite exactly. Compute where both converged: Dyna-Q spends 2.8–3.6× Q-learning's Q-value updates and Q(λ) 6.5–8.9×. Over a full 1,000-episode run, Dyna-Q's agent time is 10–16× Q-learning's.</sub>

- **What changed.** Real moves to a shortest greedy route split into finding the goal and turning that into a shortest route. Dyna-Q shortens the second phase 22–32× on every maze type. On 2D and one-level mazes it finds the goal about as fast as Q-learning, which isolates propagation. On three-level mazes, planning also spreads the step penalties through explored corridors, so the agent leaves them sooner and finds the goal 4.7–6.3× faster ([diagnostic](docs/value-propagation/README.md#diagnostic-which-part-of-learning-got-faster)).
- **Two failures, explained.**
  - **Q(λ) with long traces:** it stops exploring; on one development maze it found 26 of 185 cells in 50,000 moves, against 183 for Q-learning ([analysis](docs/value-propagation/README.md#watkinss-qλ-with-long-traces-stops-exploring)).
  - **Dyna-Q on one maze in 60:** fast propagation locked it onto a route 8 moves longer than the shortest, on a maze with more than one route ([analysis](docs/value-propagation/README.md#dyna-q-can-lock-onto-a-longer-route)).

### The repaired Q-learning agent

The [evaluation suite](docs/evaluation/README.md) measured the repository's agent after the repairs. It used 60 seeded mazes of each type and 1,000 episodes, and compared it with exact references:

- a uniformly random walk;
- planners given the maze: breadth-first search and value iteration.

| 60 seeded mazes per type | Late training success | Greedy route converged | Moves to converge | In 3,000 episodes | Unseen mazes solved |
|---|---|---|---|---|---|
| 2D, 21&nbsp;×&nbsp;21 | 100% | **54/60** (80–95%) | 91k | 60/60 | 0 of 3,000 |
| One level, 21&nbsp;×&nbsp;21 | 100% | **60/60** (94–100%) | 75k | 60/60 | 0 of 3,000 |
| Three levels, 21&nbsp;×&nbsp;21&nbsp;×&nbsp;3 | 99% | **26/60** (32–56%) | 148k | 60/60 | 0 of 3,000 |

<sub><b>Late training success:</b> share of the last 100 training episodes that reached the goal while still exploring. <b>Converged:</b> the greedy route was a shortest route at every check for at least the last 100 episodes. <b>Moves to converge:</b> median over converged runs only. <b>Unseen mazes:</b> each trained agent's greedy route on 50 mazes it never saw.</sub>

![Three panels, one per maze type, each with 60 runs. Against moves of experience on a log scale: the share of runs whose exploring training episodes reach the goal reliably rises first; the share whose greedy route is a shortest route rises later and, on three-level mazes, levels off near half.](docs/assets/results.png)

- **Reaching the goal is not learning the route.** On three-level mazes, 99% of late training episodes reach the goal, helped by exploration. Yet only 26 of 60 greedy routes have settled on a shortest route.
- **It learns a route, not a map.** Value spreads back from the goal mostly along the route the agent uses. Averaged over all runs, the greedy action is optimal in 54–68% of reachable cells ([policy figure](docs/assets/policy.png)).
- **Nothing transfers.** The state is the agent's cell, so a Q-table describes one maze. Trained agents reach the goal in 0 of 9,000 tests on unseen mazes.

## Historical debugging evidence

*Archived evidence, from an audit on 6 November 2026 of the code as committed at [`3e99fbd`](https://github.com/easensoy/maze-qlearning-rl-agent/tree/3e99fbd) in August 2025.* Every component existed, yet the agent never learned a maze and the viewer never received training data. With the repository's own settings, 0% of training episodes reached the goal, over 25,000 offline episodes and 100 streamed live. Five stacked defects were behind it:

1. **Server crash:** the training thread crashed on its first step.
2. **Dead controls:** the viewer's controls never reached the server.
3. **Walled-off goals:** the 3D generator walled off the goal in 88% of mazes.
4. **A new maze every episode.**
5. **Wrong action rule and short episodes:** the action rule ignored untried actions, and episodes were too short.

Each defect was located by measurement, fixed in its own commit and covered by a regression test. The [case study](docs/evaluation/case-study.md) gives the measurement behind each one, and the errors found in the measurements themselves. Rerun at the current code, the [evaluation suite](docs/evaluation/README.md) puts the defects back one at a time.

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

The tests and experiments run from `maze_runner_rl/` with the virtual environment's Python:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest                                   # 91 tests
python -m experiments.compare_methods tune         # value propagation: development mazes -> frozen settings (26 min, 20 workers)
python -m experiments.compare_methods final        # the final mazes with the frozen settings (16 min)
python -m experiments.plot_value_propagation       # its figures and analysis.json
python -m experiments.evaluate                     # the evaluation suite (34 min, 20 workers)
python -m experiments.capture_hero && python -m experiments.render_hero   # the animation above
```

Every experiment is deterministic and records the commit, a hash of the code it depends on, its settings and seeds, and the library versions. The raw results (one row per run, plus per-episode data) are kept beside each report.

## Limitations

- **Small, deterministic mazes.** The goal is always in the far corner, the transitions are deterministic and the mazes are at most 21 × 21 × 3. Dyna-Q's table model relies on that determinism.
- **One maze at a time.** The state is the agent's cell, so nothing transfers to a new maze. Generalising would need the maze in the state and training and test sets of mazes.
- **Settings.** Only each new method's own setting was tuned, on development mazes. The shared settings are the repository's defaults, tuned for no method.
- **Compute is not free.** Dyna-Q's sample efficiency costs about 3× the Q-value updates to converge, and 10–16× the agent time per run.
- **The frontend Docker image runs the development server.** A production build served as static files would be much smaller.

## Repository

| Path | Contents |
|---|---|
| [`maze_runner_rl/`](maze_runner_rl) | Python: generators, environments, agents, server, experiments and tests ([README](maze_runner_rl/README.md)) |
| [`maze-3d-frontend/`](maze-3d-frontend) | React/Three.js viewer ([README](maze-3d-frontend/README.md)) |
| [`docs/value-propagation/`](docs/value-propagation) | The Q-learning, Q(λ) and Dyna-Q experiment: report, frozen settings, raw results and figures |
| [`docs/evaluation/`](docs/evaluation) | The evaluation suite's report, figures and raw results, and the [debugging case study](docs/evaluation/case-study.md) |
| [`docs/assets/`](docs/assets) | The visuals on this page, the data they are drawn from, and how to regenerate them |
| [`docs/demo.md`](docs/demo.md) | The live viewer: how to run it and what it shows |
| `.github/workflows/ci.yml` | Tests, lint, build and Docker checks on every push to master and every pull request |

MIT licensed; see [LICENSE](LICENSE).
