# Value propagation on long-horizon mazes: Q-learning, Watkins's Q(λ) and Dyna-Q

[![CI](https://github.com/easensoy/maze-qlearning-rl-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/easensoy/maze-qlearning-rl-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Abstract.** This repository trains tabular Q-learning agents on seeded 2D and multi-level grid mazes. In its evaluation suite, the episode at which the baseline's greedy route first becomes a shortest route grows by 6.7–8.9 episodes per move of shortest route. Within a 1,000-episode budget the baseline converges on no route longer than 120 moves. This study tests whether faster value propagation removes that limit. Two methods replace only the baseline's update rule:

- Watkins's Q(λ), which assigns credit through eligibility traces;
- Dyna-Q, which adds model-based backups drawn from a learned deterministic transition table.

Each method's setting was selected on development mazes and committed before the final runs: λ = 0.3 with accumulating traces, and n = 50 planning updates per real step. The final runs cover 210 mazes, with all three methods trained on the same maze and exploration seed. On the 60 mazes whose shortest route exceeds 120 moves, the frozen greedy route stayed shortest over at least the final 100 of 1,000 episodes in 0 Q-learning runs, 30 Q(λ) runs and 60 Dyna-Q runs. On the 139 suite mazes where both converged, Dyna-Q used 14–18× fewer environment steps than Q-learning and 2.8–3.6× its Q-value updates. Dyna-Q also changed exploration. The results are therefore consistent with a propagation bottleneck, but they do not isolate it.

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/hero-poster.png">
  <img src="docs/assets/hero.gif" width="800" alt="Exploded isometric replay of three-level maze 54 (21 by 21 by 3 cells, shortest route 146 moves), its levels joined by 9 vertical shafts. Episode 1: 1,000 exploratory moves that do not reach the goal. After 20k, 40k, 80k and 160k environment steps, Q-learning's frozen greedy policy reaches the goal from at most 15 cells next to the goal, and its greedy walk from the start loops each time. After the same steps, Dyna-Q's greedy policy reaches the goal from 175 of the 378 non-goal cells at 20k steps and from 292 from 40k steps on, and from 40k steps its greedy route is a shortest route: 146 moves with 6 level changes. End card: on the 30-maze long-route set, Dyna-Q converged on 30, Q(λ) on 15 and Q-learning on 0.">
</picture>

<sub><b>Replay.</b> Two logged training runs, Q-learning and Dyna-Q (n = 50), on three-level suite maze 54 (shortest route 146 moves). <code>render_hero.py</code> renders them offline from the recorded trajectories; this is not the live React/Three.js viewer. A fixed rule chose the maze: the median shortest route among the 34 three-level suite mazes on which Dyna-Q converged and Q-learning did not. Episode 1 plays at 50 moves per frame. Greedy-policy checkpoints follow at 20k, 40k, 80k and 160k environment steps, and the episodes between them are omitted. The final route advances 5 moves per frame, cell by cell, without interpolation. Blue cells are states from which the frozen greedy policy reaches the goal. <a href="docs/assets/README.md#the-animation">Frame-by-frame text version</a>.</sub>

**Findings** (final mazes, paired by maze and exploration seed; 95% intervals in brackets):

1. **Persistent convergence beyond 120 moves.**
   - Converged runs: Q-learning 0/60 [0–6%], Q(λ) 30/60 [38–62%], Dyna-Q 60/60 [94–100%].
   - On the 30-maze long-route set, the exact McNemar test against Q-learning gives p = 1.9 × 10⁻⁹ for Dyna-Q and p = 6.1 × 10⁻⁵ for Q(λ).
   - On the suite, the baseline's failure is budget-bound: with 3,000 episodes it converges on all 30 suite mazes longer than 120 moves (routes up to 206 moves). The long-route set (142–278 moves) was not run with a larger budget.
2. **Interaction–computation trade-off.**
   - On mazes where both it and Q-learning converged, Dyna-Q needs 14.1–18.3× fewer environment steps than Q-learning, at 2.8–3.6× the Q-value updates.
   - On the same basis, Q(λ) needs 1.6–1.8× fewer steps at 6.5–8.9× the updates.
   - Over a full 1,000-episode run, Dyna-Q's median agent wall-clock time is 10–16× Q-learning's.
3. **Both interventions alter exploration.**
   - Dyna-Q shortens the interval from the first goal-reaching training episode to the first shortest greedy route by 22–32× (ratio of medians). On three-level mazes it also reaches the goal 4.7–6.3× sooner.
   - Q(λ) reaches the goal 1.2–3.3× later.
   - With accumulating traces at λ = 0.7 and 0.9, Q(λ)'s median state coverage after 50k development steps is 39% and 16%. Only 5 and 0 of 60 development runs converge (replacing traces: 15 and 0), against 47 for Q-learning.

[Full report](docs/value-propagation/README.md) · [Evaluation protocol](#evaluation-protocol) · [Reproduction](#reproduction) · [Live 3D viewer](docs/demo.md) · [Debugging case study](docs/evaluation/case-study.md)

## System

### Environment

Seeded generators produce three maze types:

- perfect 2D mazes of 21 × 21 cells;
- one-level 3D mazes of 21 × 21 × 1 cells;
- three-level 3D mazes of 21 × 21 × 3 cells, whose levels connect through vertical shafts.

Every maze is solvable and fully connected, with the start and goal in opposite corners.

The environments implement the [Gymnasium](https://gymnasium.farama.org/) API (Farama Foundation, version 1.2.0): `reset()` returns `(obs, info)`, and `step()` returns the five-tuple. The tabular agents index states by cell. The action set is {north, south, west, east}, plus {down, up} on 3D mazes. Transitions are deterministic and exposed as `step_from_state(s, a)` for planners and tests.

| Quantity | Value |
|---|---|
| Rewards | +100 for entering the goal, which is terminal; −0.1 per move; −1 for a move into a wall, which leaves the state unchanged |
| Discount | γ = 0.95 |
| Episode cap | 1,000 steps |
| Training budget | 1,000 episodes per run |

A cell d moves from the goal has optimal value $`Q^*(s,a^*) = 102\,\gamma^{d-1} - 2`$. This is negative for d ≥ 78, so on every route longer than 77 moves, the start's optimal value lies below the zero initial value of an untried action.

### Agents

The three agents share the same settings:

- step size α = 0.1 and discount γ = 0.95;
- zero initialisation, $`Q_0 \equiv 0`$;
- ε-greedy behaviour, with ties broken uniformly at random;
- ε₀ = 0.1, multiplied by 0.995 per episode down to 0.01, which it reaches after 460 episodes.

They differ only in how a transition updates the table. None bootstraps from the terminal state.

```math
Q(s,a) \leftarrow Q(s,a) + \alpha\,\delta,\qquad \delta = r + \gamma\,(1-\mathrm{done})\,\max_{b} Q(s',b) - Q(s,a)
```

| Method | Update | Selected setting |
|---|---|---|
| **Q-learning** ([`q_learning_agent.py`](maze_runner_rl/src/agents/q_learning_agent.py)), the repository's agent, unchanged | One backup of the transition just taken | none |
| **Watkins's Q(λ)** ([`q_lambda_agent.py`](maze_runner_rl/src/agents/q_lambda_agent.py)); Sutton and Barto (1998), §7.6 | δ updates every traced pair, Q ← Q + αδe. The next behaviour action a′ is drawn before the update. Traces then decay by γλ if a′ is greedy in s′ (ties count as greedy) and are cut to zero otherwise. Accumulating traces add 1 to e(s, a); replacing traces set e(s, a) = 1 and clear the state's other actions | λ = 0.3, accumulating |
| **Dyna-Q** ([`dyna_q_agent.py`](maze_runner_rl/src/agents/dyna_q_agent.py)); Sutton and Barto (2018), §8.2 | After each real step, apply the backup to that transition and store M(s, a) = (r, s′, done) in a deterministic model. Then apply n backups to modelled transitions, so n = 50 gives 51 updates per real step. Each modelled transition draws a state uniformly from those observed, then an action uniformly from those taken there. Planning draws from its own generator (seed 700000 + maze seed), so it consumes none of the random numbers that drive exploration | n = 50 planning updates per real step |

Zero initialisation is optimistic under these rewards. An untried action keeps Q = 0, while a tried action whose return excludes the goal turns negative. The greedy choice therefore prefers untried actions, and the first episodes perform a systematic search. Every update rule is checked by hand in [`test_value_agents.py`](maze_runner_rl/tests/test_value_agents.py). The checks include that λ = 0 and n = 0 reduce exactly to Q-learning.

## Evaluation protocol

- **Splits.** Development mazes (seeds 1000–1019 of each type, 60 runs per setting) were used only to choose λ and the trace type for Q(λ), and n for Dyna-Q; the baseline was not tuned. The final mazes are:
  - the evaluation suite's seeds 0–59 of each type, 180 mazes;
  - a long-route set of 30 three-level mazes: the first 30 from seed 2000 upward whose shortest route has at least 141 moves, selected from maze structure before any run.
- **Selection rules** (`select_settings` in [`compare_methods.py`](maze_runner_rl/experiments/compare_methods.py)).
  - **Grid:** λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} × {accumulating, replacing}, and n ∈ {5, 10, 20, 50}.
  - **Q(λ):** the setting with the most converged development runs wins, with ties going to the lower median time to convergence.
  - **Dyna-Q:** take the best setting, then the smallest n within one converged run of it whose median is at most 10% higher. Planning compute is then not spent on a difference the development set cannot resolve.
- **Freezing.** The selected settings and final maze sets were written to [`frozen_settings.json`](docs/value-propagation/frozen_settings.json) and committed in `49c188c` before the final runs started. The final runs' provenance records:
  - commit `2a814c1`, a descendant of `49c188c`;
  - no uncommitted changes to the source files the runs depend on;
  - the tuning runs' source digest;
  - the frozen file's SHA-256.

  This is a commitment recorded in the repository's own history, not an external preregistration.
- **Pairing.** Every method trains on the same maze with the same exploration seed (500000 + maze seed): 3 methods × 210 mazes = 630 runs.

| Term | Definition |
|---|---|
| Frozen greedy-policy evaluation | After every training episode, the greedy policy (argmax, ties to the lowest action index) is followed from the start with no exploration and no learning. The walk ends at the goal or at the first repeated cell |
| Instantaneous greedy optimality | One evaluation reaches the goal in the breadth-first-search shortest-route length |
| Persistent convergence ("converged") | Instantaneous greedy optimality at every evaluation from some episode through episode 1,000, over at least 100 consecutive evaluations. This is a finite-run operational criterion, not a convergence guarantee: a final optimal streak that starts after episode 901 does not count |
| Exploratory training success | An ε-greedy training episode reaches the goal |
| Time to convergence | Environment steps up to and including the first episode of the final optimal streak. Runs that do not converge are right-censored at the budget |
| Environment steps | Every real transition, including moves into walls. Trace and planning updates are counted separately as Q-value updates, and agent wall-clock time is measured inside agent calls |

**Statistics.**

- Proportions carry 95% Wilson intervals.
- Medians of time to convergence include right-censored runs as +∞, and are reported as not reached when at least half the runs are censored. Their intervals are percentile bootstraps over runs.
- Comparisons with Q-learning are paired by maze:
  - an exact McNemar test on the discordant pairs;
  - a paired bootstrap of the difference in converged share;
  - a sign test on time to convergence, counting a censored run as slower;
  - geometric-mean ratios of steps and updates, computed only on mazes where both methods converged.

## Results

![Four panels, one per maze set: the share of runs whose frozen greedy route is a shortest route at that point, against environment steps on a log scale, with 95% Wilson bands. Dyna-Q's curve rises first in every panel and reaches 100% on the long-route set, where Q(λ) reaches 60% and Q-learning stays at 0%.](docs/value-propagation/figures/learning_curves.png)

| Final mazes | Q-learning converged | Q(λ) converged | Dyna-Q converged | Median steps to convergence: Q-learning / Q(λ) / Dyna-Q |
|---|---|---|---|---|
| 2D, 21 × 21 (60) | 54 [80–95%] | 60 [94–100%] | 60 [94–100%] | 98k / 63k / 5.2k |
| One level, 21 × 21 (60) | 60 [94–100%] | 60 [94–100%] | 60 [94–100%] | 75k / 45k / 5.6k |
| Three levels, 21 × 21 × 3 (60) | 26 [32–56%] | 45 [63–84%] | 59 [91–100%] | not reached / 223k / 33k |
| Long routes, ≥ 141 moves (30) | **0** [0–11%] | **15** [33–67%] | **30** [89–100%] | not reached / not reached / 57k |

<sub>Medians include right-censored runs. Updates per environment step: 1 for Q-learning, 18.5–30.8 for Q(λ) and 51 for Dyna-Q. Q-learning's rows reproduce the evaluation suite run for run. Paired ratios with intervals, the McNemar tests and agent times are in the <a href="docs/value-propagation/README.md#results">report</a>.</sub>

**Phase decomposition.** Time to the first shortest greedy route splits at the first exploratory training success:

- **Phase 1:** steps until a training episode first reaches the goal.
- **Phase 2:** steps from there to the first instantaneous greedy optimality.

One-step backups predict a long phase 2. Medians, Q-learning / Q(λ) / Dyna-Q:

| Maze set | Phase 1 | Phase 2 |
|---|---|---|
| 2D | 1.8k / 5.9k / 1.1k | 80k / 51k / 2.5k |
| One level | 1.8k / 5.7k / 1.5k | 65k / 34k / 3.0k |
| Three levels | 33k / 39k / 7.0k | 342k / 173k / 15k |
| Long routes | 103k / 129k / 16k | not reached / 343k / 26k |

On 2D and one-level mazes, Dyna-Q's phase 1 is only 1.2–1.6× shorter than Q-learning's, so its gain there is concentrated in phase 2. On three-level mazes, Dyna-Q also shortens phase 1. In its first 10 episodes it visits a median 69% of reachable cells, against 38–49% for Q-learning. A plausible account is that planning propagates step and wall penalties through explored corridors, which removes their residual optimism sooner. The experiment does not isolate this mechanism.

**Failure cases** ([analysis](docs/value-propagation/README.md#failures)):

- **Q(λ) with long traces.** The cut rule treats a choice among tied untried actions as greedy, so traces persist through exploratory walks. The walk's negative TD errors are then credited to the corridors that led toward unexplored cells. On the median-route 2D development maze, Q(λ) with λ = 0.9 discovered 26 of 185 cells in 50,000 steps; Q-learning discovered 183. The coverage collapse is measured; the credit-assignment account is consistent with it, but was not isolated by an ablation. It applies to this algorithm, initialisation, reward structure and ε schedule.
- **Dyna-Q on 1 of 210 mazes** (three-level maze 58). The greedy route settled at 114 moves against a 106-move optimum. A seeded replay of the run (recorded in [`analysis.json`](docs/value-propagation/figures/analysis.json)) locates the cause: 3 moves from the goal, the goal-ward action was never taken in 1,000 episodes and still has Q = 0. The tried actions in that cell carry propagated estimates of 49–53, so the greedy policy never selects the untried one, and ε = 0.01 exploration did not reach it.

**Interpretation.** Under this protocol, Dyna-Q's planning was sufficient for persistent convergence on all 60 routes longer than 120 moves within 1,000 episodes, and Q(λ)'s traces on half of them. Three observations prevent attributing the effect to goal-value propagation alone:

- Both interventions changed the exploration trajectory.
- Planning also propagates penalties, which deflates zero-initialised estimates.
- Beyond 77 moves from the goal, greedy optimality additionally requires every competing estimate to fall below a negative optimal value.

## Baseline evaluation suite

The [evaluation suite](docs/evaluation/README.md) characterises the repaired Q-learning agent on the same 180 suite mazes. It compares the agent with a uniform random walk and with two planners that have access to the transition model: breadth-first search and value iteration.

| 60 mazes per type | Exploratory success, last 100 episodes | Converged | Median steps to convergence (censored) | Converged with 3,000 episodes | Unseen mazes reached |
|---|---|---|---|---|---|
| 2D, 21&nbsp;×&nbsp;21 | 100% | 54/60 [80–95%] | 98k | 60/60 | 0 of 3,000 |
| One level, 21&nbsp;×&nbsp;21 | 100% | 60/60 [94–100%] | 75k | 60/60 | 0 of 3,000 |
| Three levels, 21&nbsp;×&nbsp;21&nbsp;×&nbsp;3 | 99% | 26/60 [32–56%] | not reached | 60/60 | 0 of 3,000 |

![Three panels, one per maze type, 60 runs each, against environment steps on a log scale: the share of runs whose exploratory training episodes reach the goal rises first; the share whose frozen greedy route is shortest rises later and, on three-level mazes, levels off near half.](docs/assets/results.png)

- **Exploratory success and greedy optimality dissociate.** On three-level mazes, 99% of late ε-greedy episodes reach the goal, while 26 of 60 frozen greedy routes are persistently shortest.
- **The first greedy-optimal episode grows linearly with route length.** The slopes are 7.2, 6.7 and 8.9 episodes per move (r = 0.98, 0.98, 0.92). These lines cross the criterion's episode-901 limit at roughly 125–150 moves.
- **Value is learned along the travelled route, not over the state space.** The greedy action is optimal in 54–68% of reachable cells ([policy figure](docs/assets/policy.png)).
- **The tabular state admits no transfer.** The state is the cell index, and trained agents reach the goal in 0 of 9,000 evaluations on unseen mazes.

## Historical debugging record

The original pipeline (commit [`3e99fbd`](https://github.com/easensoy/maze-qlearning-rl-agent/tree/3e99fbd), August 2025) contained every component but produced no learning. Across 25,000 offline episodes and 100 streamed episodes, 0% of training episodes reached the goal. An audit on 6 November 2026 traced this to five compounding defects. Each was located by measurement, fixed in its own commit and covered by a regression test:

1. The server's training thread raised an exception on its first step.
2. Viewer controls never reached the server.
3. The 3D generator isolated the goal in 88% of mazes.
4. `reset()` generated a new maze every episode, so a position-indexed Q-table could not accumulate value.
5. Action selection ignored untried actions, and the 200-step episode cap was too short.

The [case study](docs/evaluation/case-study.md) gives the measurement behind each defect, including errors found in the measurements themselves. The [evaluation suite](docs/evaluation/README.md) reintroduces the defects one at a time as ablations.

## Reproduction

All experiments are deterministic. Each output records:

- the commit;
- a SHA-256 digest of the source files it depends on;
- its settings and seeds;
- the library versions, the processor and the worker count.

Every command calls the repository-local interpreter explicitly, so no activation is needed. Python 3.13 is required, and Node 22 for the viewer.

**Environment** (from the repository root):

```powershell
# Windows (PowerShell)
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r maze_runner_rl\requirements-dev.txt
```

```bash
# Linux and macOS
python3.13 -m venv .venv
.venv/bin/python -m pip install -r maze_runner_rl/requirements-dev.txt
```

**Experiments** (from `maze_runner_rl/`). There are three separate tasks:

1. Reproduce the published final runs with the committed frozen settings, into a separate directory, and compare them with the published records.
2. Retune on the development mazes.
3. Regenerate tables, figures and the replay from the committed records.

`tune` and `final` refuse to overwrite existing settings or run records unless `--overwrite` is passed.

```powershell
# Windows (PowerShell)
$PY = "..\.venv\Scripts\python.exe"
& $PY -m pytest -q                                                       # 93 tests
# 1. Published final runs with the committed settings (16 min on 20 workers)
& $PY -m experiments.compare_methods final --settings ..\docs\value-propagation\frozen_settings.json --out ..\reproduction\final
& $PY -m experiments.compare_methods verify --out ..\reproduction\final   # every non-timing field against results\runs.csv
# 2. Retuning (26 min on 20 workers), then the final mazes with the new settings
& $PY -m experiments.compare_methods tune --out ..\reproduction\retune
& $PY -m experiments.compare_methods final --out ..\reproduction\retune
# 3. Tables, figures, analysis.json and the replay, from the committed records
& $PY -m experiments.compare_methods tables
& $PY -m experiments.plot_value_propagation
& $PY -m experiments.render_hero
```

```bash
# Linux and macOS
PY=../.venv/bin/python
$PY -m pytest -q                                                         # 93 tests
# 1. Published final runs with the committed settings (16 min on 20 workers)
$PY -m experiments.compare_methods final --settings ../docs/value-propagation/frozen_settings.json --out ../reproduction/final
$PY -m experiments.compare_methods verify --out ../reproduction/final     # every non-timing field against results/runs.csv
# 2. Retuning (26 min on 20 workers), then the final mazes with the new settings
$PY -m experiments.compare_methods tune --out ../reproduction/retune
$PY -m experiments.compare_methods final --out ../reproduction/retune
# 3. Tables, figures, analysis.json and the replay, from the committed records
$PY -m experiments.compare_methods tables
$PY -m experiments.plot_value_propagation
$PY -m experiments.render_hero
```

A complete step 1 should report "630 of 630 reference runs compared … identical in every other field"; only the 12-run subset below has been rerun and verified. Adding `--mazes 1` to `final` restricts it to the first maze of each set: 12 runs, under a minute on 12 workers here. `verify` then checks them against the matching published runs. Timing columns are excluded from the comparison. Wall-clock figures refer to the recorded Intel64 Family 6 Model 170 processor. The evaluation suite (`experiments.evaluate`) and the README figures have their own [regeneration notes](docs/assets/README.md#regenerate).

**Live viewer.** The viewer streams the maze, the agent and its Q-values from a WebSocket training server.

```bash
# Development, from the repository root. Terminal 1: training server on ws://localhost:8765
cd maze_runner_rl && ../.venv/bin/python main.py          # Windows: ..\.venv\Scripts\python.exe main.py, or start_backend.bat
# Terminal 2: viewer development server on http://localhost:3001
cd maze-3d-frontend && npm ci && npm start                # Windows: start_frontend.bat
# Containers instead: the backend image, and the production build served by nginx on port 3001
docker compose up --build
```

The browser connects to `ws://localhost:8765`, the backend's published port. The URL is compiled into the bundle; set it with `REACT_APP_WS_URL=… docker compose up --build`, or `docker build --build-arg REACT_APP_WS_URL=…`. [docs/demo.md](docs/demo.md) describes the controls.

## Limitations

- **Scope.** All results are for tabular agents on deterministic mazes of at most 21 × 21 × 3 cells, with the goal fixed at the far corner and one maze per run. The state is the cell index, so nothing generalises across mazes.
- **Finite-budget criterion.** Persistent convergence means optimality at each of the final ≥ 100 consecutive evaluations of a 1,000-episode run. On the suite, the baseline's failures are budget-bound: 3,000 episodes suffice for all suite mazes, with routes up to 206 moves. The long-route set (142–278 moves) was not run with a larger budget.
- **Mechanism not isolated.** No ablation separates goal-value propagation from penalty propagation or from the changed exploration trajectory.
- **Tuning.** The shared settings (α, γ, the ε schedule, rewards, Q₀) are the repository's defaults and were tuned for no method. Only λ, the trace type and n were selected.
- **One exploration seed per maze.** Intervals reflect variation across mazes; within-maze variation across exploration seeds is not estimated.
- **Model assumption.** Dyna-Q stores the last observed outcome of each state–action pair, which is exact only because the transitions are deterministic.
- **Compute measurement.** Agent time is wall-clock time inside agent calls. The runs used 20 concurrent worker processes on one hybrid-core laptop processor, so the 10–16× time ratio is indicative rather than a benchmark.
- **Protocol.** The settings were frozen by a commit in the repository's own history, which is not an external preregistration.
- **Viewer deployment.** The frontend container serves static files over plain HTTP, and the browser connects directly to the backend's published WebSocket port. The URL is fixed at build time, and there is no TLS or reverse proxy.

## Repository

| Path | Contents |
|---|---|
| [`maze_runner_rl/`](maze_runner_rl) | Python: generators, environments, agents, the WebSocket server, experiments and tests ([README](maze_runner_rl/README.md)) |
| [`maze-3d-frontend/`](maze-3d-frontend) | React/Three.js viewer and its production container ([README](maze-3d-frontend/README.md)) |
| [`docs/value-propagation/`](docs/value-propagation) | The Q-learning, Q(λ) and Dyna-Q study: report, frozen settings, run records, per-episode data and figures |
| [`docs/evaluation/`](docs/evaluation) | The baseline evaluation suite and the [debugging case study](docs/evaluation/case-study.md) |
| [`docs/assets/`](docs/assets) | The replay and figures on this page, their source data and rendering notes |
| [`docs/demo.md`](docs/demo.md) | The live viewer: setup and controls |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CI on every push to master and every pull request. It runs pytest, ESLint, the frontend tests and production build, and builds both Docker images. It also checks that the production bundle opens its WebSocket to the backend in headless Chrome |

## References

- C. J. C. H. Watkins (1989). *Learning from Delayed Rewards.* PhD thesis, University of Cambridge.
- R. S. Sutton (1990). Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. *Proc. 7th International Conference on Machine Learning*, 216–224.
- S. P. Singh and R. S. Sutton (1996). Reinforcement learning with replacing eligibility traces. *Machine Learning* 22, 123–158.
- R. S. Sutton and A. G. Barto (1998, 2018). *Reinforcement Learning: An Introduction*, 1st ed. §7.6 and 2nd ed. §8.2.

MIT licensed; see [LICENSE](LICENSE).
