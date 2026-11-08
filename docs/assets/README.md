# README visuals

[README](../../README.md) · [Value-propagation experiment](../value-propagation/README.md) · [Evaluation](../evaluation/README.md) · [Debugging case study](../evaluation/case-study.md)

This page covers the animation and the two figures on the [README](../../README.md): what they show, what they are drawn from, and how to draw them again. Code in this repository generates every image from recorded training runs. None is a mock-up, and none is AI-generated. The animation is a recorded replay drawn by its own renderer, not a capture of the live viewer.

## Files

| File | Shows | Drawn by, from |
|---|---|---|
| [`hero.gif`](hero.gif) | Q-learning and Dyna-Q on one three-level maze, in an exploded 3D view. 800 × 720 px, 62 frames, 18.1 s, 0.75 MB | `render_hero.py`, from `data/hero.json`. The end card's counts come from [`summary.json`](../value-propagation/results/summary.json) |
| [`hero.webp`](hero.webp) | The same frames without the GIF's 256-colour palette (0.83 MB) | `render_hero.py` |
| [`hero-poster.png`](hero-poster.png) | The final-route frame at twice the size (1600 × 1440 px). The README shows it instead of the animation when the reader's system asks for reduced motion | `render_hero.py` |
| [`results.png`](results.png) | All 180 evaluation-suite runs of Q-learning: exploring success, greedy-route optimality and convergence, against moves of experience | `render_figures.py`, from `data/traces_*.json` |
| [`policy.png`](policy.png) | What Q-learning's greedy policy knows in a converged run and in one that ran out of episodes | `render_figures.py`, from `data/featured.json` |
| [`data/hero.json`](data/hero.json) | The animation's runs: the maze, a BFS route, episode 1's cells, and each method's greedy policy after 10k, 20k, 40k, 80k and 160k real moves | `capture_hero.py` |
| [`data/traces_<kind>.json`](data), [`featured.json`](data/featured.json), [`provenance.json`](data/provenance.json) | The figures' data: every suite run per episode, the two runs drawn in `policy.png`, and the capture's provenance | `capture_traces.py` |
| [`data/render.json`](data/render.json) | Libraries, typeface and playback settings used to draw the images | `render_hero.py`, `render_figures.py` |

The scripts are in [`maze_runner_rl/experiments/`](../../maze_runner_rl/experiments).

## The animation

### Text version

The animation, part by part. Every number is from [`data/hero.json`](data/hero.json).

1. **The maze** (2.4 s). Title: "Three-level maze: 21×21×3 cells, seed 54", three-level maze 54 of the final suite.
   - Level 0, where the agent starts, is at the bottom; level 2, with the goal, at the top.
   - Level 1 consists only of the 9 shaft cells that join them, drawn as vertical columns.
   - The shortest route is 146 moves.
2. **Episode 1** (2.2 s). Title: "Episode 1: exploring from an all-zero Q-table". This is Q-learning's first episode: 1,000 exploratory moves that do not reach the goal, shown at 50 moves per frame with the path in cyan.
3. **Q-learning after 20k, 40k, 80k and 160k environment steps** (3.2 s), at episodes 20, 41, 83 and 181. The episodes between the checkpoints are omitted.
   - **Blue cells:** its frozen greedy policy reaches the goal from 0, 2, 11 and then 15 cells, all next to the goal.
   - **Its greedy walk from the start:** shown in amber, it loops after a few moves at every checkpoint ("greedy route loops before the goal").
4. **Dyna-Q (n = 50) after the same environment steps** (3.2 s), at episodes 50, 171, 438 and 981.
   - **Blue cells:** 175 after 20k steps, then 292 of the maze's 378 non-goal cells.
   - **Its greedy route:** from 40k steps (episode 171) it is a shortest route.
5. **Dyna-Q's greedy route** (3.3 s): 146 moves, the length of the dashed breadth-first-search route. It changes level 6 times through the shafts and plays at 5 moves per frame, cell by cell, with no interpolation.
6. **The result** (3.8 s). End card:
   - "Long-route set: 30 mazes, shortest routes of at least 141 moves";
   - "Converged: Dyna-Q 30 · Q(λ) 15 · Q-learning 0";
   - the criterion, "greedy route shortest at every check for the last ≥ 100 of 1,000 episodes";
   - "Same mazes and exploration seeds · Dyna-Q: 51 Q-value updates per real move".

Every frame's footer reads "Recorded replay, not the live viewer" with the playback rate or the checkpoint budgets.

### How it was made

- **The runs.**
  - **Recapture:** `capture_hero.py` trains the two runs again exactly as the value-propagation experiment trains them: same maze, same exploration seed, same random draws. It records what the animation draws.
  - **Check:** it compares seven fields of each run with the experiment's [`runs.csv`](../value-propagation/results/runs.csv) and stops if any differs. The fields are real steps, first goal, first shortest route, convergence episode and steps, final greedy route, and whether it is shortest.
- **The maze, chosen by rule** (`pick_maze`): among the three-level suite mazes where Dyna-Q converged and Q-learning did not (34 mazes), the one with the median shortest route. The animation shows the typical case of the difference the tables report, not the most dramatic one.
- **The 3D view** ([`iso_maze.py`](../../maze_runner_rl/experiments/iso_maze.py)):
  - **Levels:** each level is an isometric plate of thin slabs, one per open cell. Walls are left empty, so no corridor hides another. The levels are pulled apart vertically and joined by the shaft columns.
  - **Camera:** it looks from behind the start, so a route climbs from the front of the bottom plate to the back of the top one.
  - **Rendering:** the plates for each set of learned cells are drawn once and reused, and only paths, markers and text change between frames.
- **Colours,** as in the other figures:
  - cyan for the path taken while exploring;
  - blue for cells from which the greedy policy reaches the goal;
  - amber, the one accent, for the greedy route;
  - a dashed off-white line for a breadth-first-search route;
  - a hollow ring for the start and a filled disk for the goal.
- **Contrast.** The overlays carry the message, but every structural element still reads against the background:
  - open cells' top faces (`#5b6575`) clear the background (`#101318`) by about 3:1, and stay darker than every overlay;
  - shaft cells (`#a3adbb`) and the columns joining them (`#8f99a7`) are lighter than open cells but darker than the white route dashes;
  - level labels are set in the primary ink;
  - captions and footers are set in the secondary ink, 8.4:1 against the background.
- **Legibility.** Text is sized for the README's 800-pixel column, and it stays readable when GitHub scales the frame to a phone's width (about 0.43×):
  - titles are 28 px and verdicts 20 px;
  - labels, legends and footers are 14–15 px.

  Methodological detail (the maze-selection rule, omitted episodes, playback rates) is in the README caption and this text version rather than in small type inside the frame.
- **Accessibility.** The README shows the poster instead of the animation to readers whose system asks for reduced motion (`prefers-reduced-motion`). Its alt text summarises the run, and this section is the full text version.

## The figures

`results.png` and `policy.png` show the evaluation suite's Q-learning runs ([evaluation](../evaluation/README.md#setup)):

- **Mazes:** three types (2D 21 × 21, one level 21 × 21, three levels 21 × 21 × 3), maze seeds 0 to 59 of each.
- **Exploration:** each run's exploration seed is 500000 plus its maze seed.
- **Agent:** learning rate 0.1, discount 0.95, ε from 0.1, decaying by 0.995 per episode to 0.01, and untried actions valued at 0.
- **Rewards:** +100 at the goal, −0.1 per move, −1 for bumping into a wall.
- **Budget:** 1,000 episodes of at most 1,000 moves.

After every episode the greedy policy, with no exploration, is followed from the start. The check stops at the goal or at the first cell it repeats. A run has **converged** when that greedy route was a shortest route, by breadth-first search over the maze, at every check for at least the last 100 episodes.

**Replayed and checked.** `capture_traces.py` trains every run again, exactly as `evaluate.py` does, and compares 13 fields of each run with the suite's [`runs.csv`](../evaluation/results/runs.csv). All 180 runs match ([`provenance.json`](data/provenance.json), `check_against_suite`). Two captures produced byte-identical traces.

**Featured runs in `policy.png`, chosen by rule** (`pick_featured`):

- **Seed 12:** of the 26 converged three-level runs, the one at the median convergence episode (episode 504).
- **Seed 0:** the lowest-numbered three-level maze whose run did not converge, continued to 3,000 episodes as in the suite's three-times-the-episodes setting. It converges at episode 1,096.

**Results figure.** Each panel shows 60 runs on 60 different mazes, against moves of experience on a log scale:

- **Blue:** runs in which at least 45 of the last 50 training episodes reached the goal while exploring.
- **Amber:** runs whose greedy route is a shortest route at that point.
- **Dashed:** runs that have converged.

Bands are 95% Wilson intervals for a share of 60 independent runs. A run that has finished its 1,000 episodes keeps its final state.

**Policy figure.** Blue cells are those from which following the greedy action reaches the goal. The chart on the right follows a BFS shortest route's cells over training, as the share from which the greedy policy reaches the goal. In seed 12 all 398 open cells are reachable; in seed 0, 359 of 397.

## Regenerate

From `maze_runner_rl/`, with the repository's virtual environment called explicitly and the development requirements installed ([README](../../README.md#reproduction)). Use `..\.venv\Scripts\python.exe` on Windows (PowerShell: `& $PY …`) and `../.venv/bin/python` on Linux and macOS.

```bash
PY=../.venv/bin/python
$PY -m experiments.render_hero            # hero.gif, hero.webp and hero-poster.png, from data/hero.json
$PY -m experiments.render_figures         # results.png and policy.png, from data/traces_*.json and featured.json
# Re-record the source data by training the runs again (each checks itself against the committed results):
$PY -m experiments.capture_hero           # data/hero.json, against docs/value-propagation/results/runs.csv
$PY -m experiments.capture_traces --check ../docs/evaluation/results/runs.csv   # 1-2 min on 8 workers
```

The first two commands only redraw from committed data. The last two retrain runs and stop if any checked field differs from the committed records.

**Dependencies.** numpy, gymnasium and PyYAML from `requirements.txt`, plus matplotlib and its dependency Pillow from `requirements-dev.txt`. The versions used are recorded in `data/provenance.json` and `data/render.json`. Text is set in Segoe UI where it is installed and otherwise in matplotlib's bundled DejaVu Sans. Single-line text shrinks to fit, so the wider fallback still fits the frame.

## What is not here

Evidence from before the fixes is in the [case study](../evaluation/case-study.md), not here: the audit of commit `3e99fbd`, in which no training episode reached the goal. Everything on this page comes from the current code.
