# README visuals

[README](../../README.md) · [Evaluation](../evaluation/README.md) · [Debugging case study](../evaluation/case-study.md)

This page covers the animation and the two figures on the [README](../../README.md): what they show, what they are drawn from, and how to draw them again. Code in this repository generates every image from recorded training runs. None is a mock-up, and none is AI-generated.

## Files

| File | Shows | Drawn by, from |
|---|---|---|
| [`hero.gif`](hero.gif) | One three-level run: its first exploring episode, the greedy policy at checkpoints, and the learned route next to a BFS shortest route. 800 × 560 px, 121 frames, 19.9 s, 1.4 MB | `render_hero.py`, from `data/featured.json`. The end card's 3,000-episode and unseen-maze counts come from the suite's [`runs.csv`](../evaluation/results/runs.csv) |
| [`hero-poster.png`](hero-poster.png) | The animation's learned-route frame at twice the size (1600 × 1120 px). The README shows it instead of the animation when the reader's system asks for reduced motion | `render_hero.py` |
| [`results.png`](results.png) | All 180 runs: exploring success, greedy-route optimality and convergence, against moves of experience | `render_figures.py`, from `data/traces_*.json` |
| [`policy.png`](policy.png) | What the greedy policy knows in a converged run and in one that ran out of episodes | `render_figures.py`, from `data/featured.json`. The episodes-per-move figure comes from the suite's [`summary.json`](../evaluation/results/summary.json) |
| [`data/traces_<kind>.json`](data) | Every run of one maze type, per episode: moves taken, whether the training episode reached the goal, and the greedy check after it | `capture_traces.py` |
| [`data/featured.json`](data/featured.json) | The two runs drawn in detail: the maze, a BFS route, the cells visited in episodes 1 to 3, and the greedy policy every 10 episodes | `capture_traces.py` |
| [`data/provenance.json`](data/provenance.json) | Commit, code digest, settings, seeds, library versions, and the comparison with the suite's records | `capture_traces.py` |
| [`data/render.json`](data/render.json) | Libraries, typeface and playback settings used to draw the images | `render_hero.py`, `render_figures.py` |

The scripts are in [`maze_runner_rl/experiments/`](../../maze_runner_rl/experiments).

## The runs

The visuals use the evaluation suite's current agent, run by run ([evaluation](../evaluation/README.md#setup)):

- **Mazes:** three types (2D 21 × 21, one level 21 × 21, three levels 21 × 21 × 3), maze seeds 0 to 59 of each.
- **Exploration:** each run's exploration seed is 500000 plus its maze seed.
- **Agent:** learning rate 0.1, discount 0.95, ε from 0.1, decaying by 0.995 per episode to 0.01, and untried actions valued at 0.
- **Rewards:** +100 at the goal, −0.1 per move, −1 for bumping into a wall.
- **Budget:** 1,000 episodes of at most 1,000 moves.

After every episode the greedy policy, with no exploration, is followed from the start. The check stops at the goal or at the first cell it repeats. A run has **converged** when that greedy route was a shortest route, by breadth-first search over the maze, at every check for at least the last 100 episodes.

**Replayed and checked.** `capture_traces.py` trains every run again, exactly as `evaluate.py` does, and compares 13 fields of each run with the suite's [`runs.csv`](../evaluation/results/runs.csv). These include the moves of experience, the first success, the final greedy route, the convergence episode and the training success. All 180 runs match ([`provenance.json`](data/provenance.json), `check_against_suite`). Two captures produced byte-identical traces.

**Featured runs, chosen by rule** (`pick_featured`):

- **Seed 12:** of the 26 converged three-level runs, the one at the median convergence episode (episode 504).
- **Seed 0:** the lowest-numbered three-level maze whose run did not converge. It is continued to 3,000 episodes, as in the suite's three-times-the-episodes setting, and converges at episode 1,096, as the suite records. Its first 1,000 episodes are the 1,000-episode run; the capture checks this.

**Checkpoints.** The greedy policy is read from the Q-table after episode 1, every 10 episodes, and after the run's key episodes: first success, first shortest greedy route and convergence. Each checkpoint stores:

- the greedy walk from the start;
- the cells from which the greedy policy reaches the goal (the **solved** cells);
- at a few episodes, each cell's value, greedy action and visit count, and whether the greedy action is optimal under value iteration on the known maze.

**Reachable cells.** An episode ends at the goal, so a cell that can only be reached through the goal is never visited and is not counted. In seed 12 all 398 open cells are reachable; in seed 0, 359 of 397.

## Reading the visuals

The colours are the same in all three:

- blue for exploration and for cells the policy already leads to the goal from;
- amber, the one accent, for the learned greedy route;
- a dashed off-white line for a BFS shortest route;
- a hollow ring for the start, a filled disk for the goal, and a small square for a shaft cell.

**Animation.** Nothing is interpolated: every frame shows recorded cells, and the agent moves cell by cell.

| Part | Shows | Playback |
|---|---|---|
| Setup | The maze, start, goal, seed and dimensions | 2.0 s still |
| Episode 1 | The 1,000 moves of the first training episode, exploring | 25 moves per 70 ms frame, then 1.0 s still |
| Time-lapse | The greedy policy after episodes 10, 20, … 500, with the solved cells and the greedy walk from the start | 1 frame per 10 episodes (80 ms); 1.1 s holds at episode 371 (first shortest route) and 504 (converged) |
| Final walk | The greedy route after episode 1,000 beside the BFS route; episodes 505 to 1,000 are skipped, and the frame says so | 3 moves per 70 ms frame, then 1.5 s still |
| End card | The run's result and the results across all mazes, with their scope | 4.0 s |

Seed 12's level 1 consists of its 13 shaft cells. Each has open cells directly above and below, so level 1 is drawn as the gutter between the panels for level 0 and level 2. The renderer checks this and refuses a maze where it does not hold. A route changes level by passing through a shaft: two moves, up or down, through level 1. The numbered badges ↑1, ↓2, ↑3 mark each change on both panels.

**Results figure.** Each panel shows 60 runs on 60 different mazes. The quantities, against moves of experience on a log scale:

- **Blue:** runs in which at least 45 of the last 50 training episodes reached the goal while exploring.
- **Amber:** runs whose greedy route is a shortest route at that point.
- **Dashed:** runs that have converged.

Bands are 95% Wilson intervals for a share of 60 independent runs, each with its own maze and exploration seed. A run that has finished its 1,000 episodes keeps its final state. Episodes within a run are not treated as independent samples.

**Policy figure.** Blue cells are those from which following the greedy action reaches the goal. The chart on the right follows a BFS shortest route's cells over training, as the share from which the greedy policy reaches the goal. It is not monotone: a Q-value update can change a greedy action, which breaks a chain that worked before.

## Regenerate

From `maze_runner_rl/`, with the development requirements installed (`python -m pip install -r requirements-dev.txt`):

```bash
python -m experiments.capture_traces --check ../docs/evaluation/results/runs.csv   # 180 runs and the 2 featured runs
python -m experiments.render_figures         # results.png and policy.png
python -m experiments.render_hero            # hero.gif and hero-poster.png
```

The capture took 68 s on 8 worker processes here (`--workers` sets the number; the default is up to 8). It writes `docs/assets/data`; the two render scripts read it and write the images. Run the evaluation suite first (`python -m experiments.evaluate`) if `docs/evaluation/results` is missing. Without it the capture still works, but the check, the end card's 3,000-episode and unseen-maze counts, and the episodes-per-move note are left out.

**Dependencies.** numpy, gymnasium and PyYAML from `requirements.txt`, plus matplotlib and its dependency Pillow from `requirements-dev.txt`. The versions used are recorded in `data/provenance.json` and `data/render.json`. Text is set in Segoe UI where it is installed and otherwise in matplotlib's bundled DejaVu Sans. Single-line text shrinks to fit, so the wider fallback still fits the frame.

**Provenance of the current files.**

- **Capture:** run at commit `1793a82`, with no uncommitted changes to the code the results depend on (SHA-256 `d2e0c041…`, as in the suite's own [summary](../evaluation/results/summary.json)).
- **Environment:** Python 3.13.5, numpy 2.3.2, gymnasium 1.2.0.
- **Drawing:** matplotlib 3.10.5 and Pillow 13.0.0, with Segoe UI.

`provenance.json` also records a hash of `capture_traces.py` itself.

## What is not here

Evidence from before the fixes is in the [case study](../evaluation/case-study.md), not here: the audit of commit `3e99fbd`, in which no training episode reached the goal. Everything on this page comes from the current code.
