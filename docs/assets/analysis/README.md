# Visual analyses: data, definitions and text equivalents

[README](../../../README.md) · [Value-propagation report](../../value-propagation/README.md) · [README visuals](../README.md)

Every cell colour, route and number in these visuals comes from recorded snapshots of the published runs. Nothing is interpolated, and every view of a quantity uses the same colour limits.

## Sources

| Visual | Drawn from | New capture? |
|---|---|---|
| [`comparison.gif`](comparison.gif), [`comparison.png`](comparison.png) | [`../data/analysis/comparison.json`](../data/analysis/comparison.json): maze 54, all three methods, snapshots at 2.5k–160k real steps | yes, `capture_analysis.py` |
| [`value.gif`](value.gif), [`value.png`](value.png) | the same snapshots, with V* and the transition model stored beside them | yes, the same capture |
| [`failure-traces.png`](failure-traces.png) | [`../data/analysis/traces.json`](../data/analysis/traces.json): development maze 1004; [`diagnostic_trace_cut.json`](../../value-propagation/figures/diagnostic_trace_cut.json): the intervention | yes, `capture_analysis.py` and `diagnose_trace_cut.py` |
| [`failure-dyna.png`](failure-dyna.png) | [`../data/analysis/dyna.json`](../data/analysis/dyna.json): maze 58 | yes, `capture_analysis.py` |
| [`tradeoff.png`](../../value-propagation/figures/tradeoff.png) | the published `results/` and `tuning/` records only | no |
| numbers in the captions | [`figures.json`](figures.json), written by `render_analysis.py` | — |

**Captures.** [`capture_analysis.py`](../../../maze_runner_rl/experiments/capture_analysis.py) retrains each run exactly as the experiment trained it: same maze, same exploration seed, same random draws. The snapshots are read-only.

- **Check against the records.** At the end of each run, eight fields are compared with the published record: real steps, first goal arrival, first shortest greedy route, convergence episode and steps, final greedy route, its optimality and the total Q-value updates. Any mismatch stops the capture. All seven runs match.
- **Second check.** Maze 54's greedy walks and reach regions also match the independent hero capture at every shared budget (`test_analysis.py`).
- **Provenance.** Each file records the commit, a digest of the code it depends on, the frozen settings' hash, the library versions and the unit of every checkpoint.

**Checkpoints.** A snapshot at a budget is taken at the end of the first episode whose cumulative real steps reach the budget, which is when the experiment's greedy evaluation runs. A snapshot at a step is taken immediately after that step's update. Q-value updates count every real, trace and planning update.

## Encodings

| Mark | Meaning |
|---|---|
| hollow cell | never visited: all actions untried, V = 0. Open cells that are unreachable from the start, behind the absorbing goal, are drawn the same way |
| grey slab | visited |
| blue slab | the frozen greedy policy reaches the goal from this cell |
| pink ramp | \|V(s) − V*(s)\| on visited states, log scale fixed from 0.01 to 100 |
| cyan ramp | visits per cell, log scale fixed from 1 to 100,000 |
| amber line, × | the frozen greedy route from the start; × where it repeats a state |
| dashed off-white line | one shortest route (breadth-first search) |
| method colours (charts only) | Q-learning violet, Q(λ) orange, Dyna-Q green |

## Definitions

- **Frozen greedy route.** The walk from the start that follows argmax_a Q(s, a), with ties going to the lowest action index, no exploration and no learning. It ends at the goal or at the first repeated state.
- **Greedy-reach region.** The non-goal states from which that walk reaches the goal, as a share of the maze's reachable non-goal states.
- **Value error.** e(s) = V(s) − V*(s), where V(s) = max_a Q(s, a) with untried actions at 0, and V* comes from value iteration on the known model. In these mazes V*(s) = 102 γ^(d − 1) − 2 for a state d moves from the goal, which is negative for d ≥ 78.
- **Bellman residual.** ρ(s) = max_a [r(s, a) + γ (1 − done) V(s′)] − V(s) under the exact model. It is 0 where V is locally consistent, positive where a successor's value has not yet been backed up into s, and negative where s keeps an untried action at 0.
- **Frontier cell.** A visited state with an untried action whose successor has never been visited.

The values were rounded to 4 decimals when captured, which bounds the error in e and ρ below 10⁻³. The greedy actions were recorded exactly.

A threshold crossing or a change of greedy action in these snapshots is not an observation of an individual credit-assignment event. The snapshots show where the values stand at a checkpoint, not which update moved them.

## 1. Matched-interaction comparison

**Maze, chosen by rule.** The maze is three-level suite maze 54, the hero's maze: the median shortest route among the 34 three-level suite mazes on which Dyna-Q converged and Q-learning did not. It represents that discordant subset, not all mazes. Q(λ) converged on 19 of those 34 mazes, maze 54 among them (episode 684, 268,644 real steps).

**Animation.** It shows one method per frame, cycling through the three methods at each of 10k, 20k, 40k, 80k and 160k real steps. The camera, scale and maze are fixed. Each frame's table lists all three methods at that budget. Frame by frame:

| Budget | Method | Episode | Greedy route | Greedy policy reaches goal from | Q-value updates |
|---|---|---|---|---|---|
| 10k | Q-learning | 10 | loops | 0 of 378 (0%) | 10,000 |
| 10k | Q(λ), λ = 0.3 | 10 | loops | 0 of 378 (0%) | 91,933 |
| 10k | Dyna-Q, n = 50 | 13 | loops | 46 of 378 (12%) | 538,968 |
| 20k | Q-learning | 20 | loops | 0 of 378 (0%) | 20,000 |
| 20k | Q(λ), λ = 0.3 | 20 | loops | 0 of 378 (0%) | 178,280 |
| 20k | Dyna-Q, n = 50 | 50 | loops | 175 of 378 (46%) | 1,023,315 |
| 40k | Q-learning | 41 | loops | 2 of 378 (1%) | 40,888 |
| 40k | Q(λ), λ = 0.3 | 40 | loops | 0 of 378 (0%) | 372,477 |
| 40k | Dyna-Q, n = 50 | 171 | shortest, 146 moves | 292 of 378 (77%) | 2,046,426 |
| 80k | Q-learning | 83 | loops | 11 of 378 (3%) | 80,410 |
| 80k | Q(λ), λ = 0.3 | 82 | loops | 15 of 378 (4%) | 788,091 |
| 80k | Dyna-Q, n = 50 | 438 | shortest, 146 moves | 292 of 378 (77%) | 4,080,306 |
| 160k | Q-learning | 181 | loops | 15 of 378 (4%) | 160,492 |
| 160k | Q(λ), λ = 0.3 | 182 | loops | 39 of 378 (10%) | 1,929,892 |
| 160k | Dyna-Q, n = 50 | 981 | shortest, 146 moves | 292 of 378 (77%) | 8,162,958 |

**Coverage.** Dyna-Q stops visiting new states after 10k real steps: 290 of the 378 non-goal states are visited, and the other 88 never are. Two never-visited states fall inside its greedy-reach region, because their default greedy action leads into it. Q-learning and Q(λ) visit every state by 160k.

**Static figure.** It shows the three maps at 40k, the first budget at which Dyna-Q's greedy route is shortest, and the reach region at every captured budget.

## 2. Value error and Bellman residual

**Animation.** It shows the |V − V*| map for each method at 2.5k, 5k, 10k, 20k, 40k, 80k and 160k real steps, on the same colour scale throughout.

**Static figure.** It shows:
- the maps at 40k;
- the signed error e(s) and the residual ρ(s) against graph distance from the goal, at 10k, 40k and 160k;
- a dashed reference: the error of a state that still holds V = 0.

| Budget | Method | States visited | Median \|V − V*\| | Share with \|V − V*\| < 0.01 |
|---|---|---|---|---|
| 2,500 | Q-learning | 106 | 1.59 | 0% |
| 2,500 | Q(λ), λ = 0.3 | 63 | 1.42 | 0% |
| 2,500 | Dyna-Q, n = 50 | 207 | 3.46 | 0% |
| 5,000 | Q-learning | 106 | 1.50 | 0% |
| 5,000 | Q(λ), λ = 0.3 | 86 | 1.23 | 0% |
| 5,000 | Dyna-Q, n = 50 | 276 | 2.40 | 1% |
| 10k | Q-learning | 106 | 1.28 | 0% |
| 10k | Q(λ), λ = 0.3 | 92 | 0.87 | 0% |
| 10k | Dyna-Q, n = 50 | 290 | 1.68 | 5% |
| 20k | Q-learning | 146 | 1.02 | 0% |
| 20k | Q(λ), λ = 0.3 | 117 | 0.55 | 1% |
| 20k | Dyna-Q, n = 50 | 290 | 0.70 | 14% |
| 40k | Q-learning | 272 | 2.40 | 0% |
| 40k | Q(λ), λ = 0.3 | 219 | 3.43 | 1% |
| 40k | Dyna-Q, n = 50 | 290 | 0.00 | 72% |
| 80k | Q-learning | 346 | 1.34 | 0% |
| 80k | Q(λ), λ = 0.3 | 337 | 1.36 | 1% |
| 80k | Dyna-Q, n = 50 | 290 | 0.00 | 100% |
| 160k | Q-learning | 378 | 1.14 | 1% |
| 160k | Q(λ), λ = 0.3 | 378 | 1.23 | 0% |
| 160k | Dyna-Q, n = 50 | 290 | 0.00 | 100% |

The median changes partly because the set of visited states changes, not only because values move.

After 160k real steps, the states with ρ(s) > 1 lie within 11 moves of the goal for Q-learning (11 states) and within 14 for Q(λ) (15 states). Dyna-Q has none.

## 3. Interaction–computation trade-off

[`tradeoff.png`](../../value-propagation/figures/tradeoff.png) shows panels A, B, D and E, and [`tradeoff_paired.png`](../../value-propagation/figures/tradeoff_paired.png) shows panel C. `plot_tradeoff.py` draws both from the published `results/` and `tuning/` records only. Before writing, it checks 65 shared numbers against `analysis.json`, `summary.json`, `curves.json` and `frozen_settings.json`. Every number is in [`tradeoff.json`](../../value-propagation/figures/tradeoff.json).

- **A and B.** The share of each method's 210 final runs (all four maze sets pooled) whose frozen greedy route is shortest at that point:
  - A plots it against real environment steps and B against Q-value updates;
  - a run that has finished its 1,000 episodes keeps its final state;
  - the bands are 95% Wilson intervals.

  | Method | Share first reaches 50%: real steps | Same, Q-value updates | Final share |
  |---|---|---|---|
  | Q-learning | 131,276 | 131,276 | 148 of 210 |
  | Q(λ), λ = 0.3 | 79,082 | 1,065,194 | 184 of 210 |
  | Dyna-Q, n = 50 | 8,936 | 455,736 | 209 of 210 |

- **C.** Per-maze comparison on the 180 suite mazes:
  - inside the axes, mazes where both methods converged, with the geometric-mean ratios of `analysis.json`;
  - in the margin strips, mazes only one method converged;
  - in the corner, the count where neither did.

  Dyna-Q against Q-learning: both converged on 139 mazes, only Dyna-Q on 40, only Q-learning on 1 (three-level maze 58). Q(λ) against Q-learning: 140 both, 25 only Q(λ), 15 neither. On all 139 shared mazes Dyna-Q used fewer real steps and more updates. The long-route set is not paired, because Q-learning converged on none of its 30 mazes.
- **D.** The development sweep, 60 runs per setting, separate from the final evaluation:
  - each axis is a separate median over a setting's runs, with runs that did not converge counted as +∞;
  - a setting is dominated if another converged on at least as many runs with no more median steps and no more median updates, strictly better in one;
  - non-dominated among the measured settings: Q-learning and Dyna-Q at n = 5, 10, 20 and 50;
  - every Q(λ) setting is dominated, each by Dyna-Q at n = 5.

  This is a check over the measured settings, not a fitted frontier.
- **E.** Wall-clock time inside agent calls, per 1,000-episode run: median and interquartile range per maze set. The runs used 20 concurrent worker processes on one hybrid-core laptop processor, so the comparison is indicative, not a controlled benchmark. Dyna-Q's median is 10–16× Q-learning's.

## 4. Failure cases

### Long traces on development maze 1004

The maze is chosen by rule, as in the report: the 2D development maze with the median shortest route (100 moves).

**Observed:**
- Q(λ) visited all 26 of its cells within its first 208 real steps.
- After step 209 it never revisited its only frontier cell. That cell's untried actions are valued 0 and would be greedy there, against tried actions of −0.30 and −0.24.
- Over the 1,000,000 steps of the run, it visited no new state.
- Its tried Q-values reach −3.7 by 50,000 steps. Q-learning's reach −0.28 at 1,000 steps.
- Q-learning visited 94 cells in its first 1,000 steps and all 185 by step 53,306.

**Tested** ([`diagnose_trace_cut.py`](../../../maze_runner_rl/experiments/diagnose_trace_cut.py)). Its specification was written before the first run. The commit postdates that run, so git does not timestamp the order. On the 60 development mazes, at λ = 0.9 with accumulating traces, the intervention also cuts the traces whenever the next action has never been tried. Coverage after 50,000 real steps:

| Variant | Median coverage (IQR) | Runs that reached the goal |
|---|---|---|
| Q(λ), unmodified | 16% (10–23%) | 0 of 60 |
| Q(λ), traces also cut at untried next actions | 58% (37–74%) | 16 of 60 |
| Q-learning | 100% (76–100%) | 53 of 60 |

The 60 unmodified runs match the published development records episode by episode.

**Reading.** The cut rule's treatment of untried actions explains part of the collapse, not all of it: with the cut, coverage still falls well short of Q-learning's.

### Dyna-Q on suite maze 58

**Observed:**
- Every greedy route Dyna-Q found was the 114-move route. It first appears at episode 74. Until episode 84 it alternates with a greedy walk that loops before the goal (episodes 75–78, 81–82 and 84). From episode 85 it holds at every episode.
- The route follows a shortest route for 102 moves and leaves it 4 moves from the goal.
- One cell further along, 3 moves from the goal, the goal-ward action was never taken (Q = 0). The tried actions there are valued 49.5, 49.5 and 53.1.
- At the junction, the greedy detour is valued 56.02 and the shortest direction 50.36 at every snapshot, from episode 60 to episode 1,000.
- The shaft up to the goal (2 cells) was never visited by Dyna-Q. Q-learning had visited all three cells of that continuation by episode 60, and found the 106-move route at episode 703.

**Reading, a hypothesis consistent with these values, not tested by intervention.** Planning gave the tried actions near the goal positive values before the shortcut was found. From then on the untried goal-ward action (0) never won the greedy choice, and ε = 0.01 exploration did not take it within the budget.

## Regenerate

From `maze_runner_rl/`, with the repository's virtual environment called explicitly: `..\.venv\Scripts\python.exe` on Windows, `../.venv/bin/python` on Linux and macOS.

```bash
PY=../.venv/bin/python
# Render from committed data (no training)
$PY -m experiments.render_analysis          # the GIFs, PNGs and figures.json in docs/assets/analysis/
$PY -m experiments.plot_tradeoff            # docs/value-propagation/figures/tradeoff.png
# Capture new diagnostic checkpoints (retrains the published runs, checks them against the records)
$PY -m experiments.capture_analysis         # docs/assets/data/analysis/*.json, about 30 s on 7 workers
$PY -m experiments.diagnose_trace_cut       # the intervention, about 50 s on 20 workers
```
