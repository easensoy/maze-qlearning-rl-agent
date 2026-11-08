# Does slow value propagation limit learning on long routes?

[README](../../README.md) · [Evaluation suite](../evaluation/README.md) · [Raw results](results)

**Answer: yes.** Faster value propagation removes the long-route failure:

- **Long routes:** on the 60 final mazes whose shortest route is longer than 120 moves, Q-learning's greedy route converged on none within 1,000 episodes. Dyna-Q's converged on all 60, Watkins's Q(λ)'s on 30.
- **Real moves:** where both converged, Dyna-Q needed 14–18× fewer real moves than Q-learning.
- **Compute:** it paid for that with about 3× the Q-value updates.

The diagnostic shows propagation helping in two places:

- **After the goal is found:** the gap from the first episode that reaches the goal to the first shortest greedy route shrinks 22–32×.
- **While exploring:** on three-level mazes the goal is also found 4.7–6.3× sooner.

**Two failures, both explained:**

- **Q(λ) with long traces:** with long traces, Watkins's Q(λ) stops exploring.
- **Dyna-Q on one maze:** on 1 maze in 60, Dyna-Q locked onto a route 8 moves longer than the shortest.

![Four panels, one per maze set: the share of runs whose frozen greedy route is a shortest route, against real environment steps on a log scale. Dyna-Q's curve rises first in every panel and reaches 100% on the long-route set, where Q(λ) reaches 50% and Q-learning stays at 0%.](figures/learning_curves.png)

## Results

The final runs: 3 methods × 210 mazes, the same mazes and exploration seeds for every method ([`results/runs.csv`](results/runs.csv); all tables in [`results/tables.md`](results/tables.md)). The table gives, for each maze set:

- **Converged:** runs converged, with 95% Wilson intervals.
- **Real moves to converge:** the median over all runs, counting a run that did not converge as never.
- **Updates per real move:** Q-value updates per real move.
- **Agent time:** the median agent time per run of 1,000 episodes, one run per worker process, on the processor the provenance records (Intel64 Family 6 Model 170, 20 workers).

| Maze set | Method | Converged | Real moves to converge | Updates per real move | Agent time per run |
|---|---|---|---|---|---|
| 2D, 21 × 21 (60) | Q-learning | 54/60 (80–95%) | 98k | 1 | 1.8 s |
| | Watkins's Q(λ) | 60/60 (94–100%) | 63k | 23.5 | 5.9 s |
| | Dyna-Q | 60/60 (94–100%) | 5.2k | 51 | 27 s |
| One level, 21 × 21 (60) | Q-learning | 60/60 (94–100%) | 75k | 1 | 1.7 s |
| | Watkins's Q(λ) | 60/60 (94–100%) | 45k | 18.5 | 3.0 s |
| | Dyna-Q | 60/60 (94–100%) | 5.6k | 51 | 27 s |
| Three levels, 21 × 21 × 3 (60) | Q-learning | 26/60 (32–56%) | not reached | 1 | 5.5 s |
| | Watkins's Q(λ) | 45/60 (63–84%) | 223k | 23.0 | 11 s |
| | Dyna-Q | 59/60 (91–100%) | 33k | 51 | 55 s |
| Long routes, > 140 moves (30) | Q-learning | 0/30 (0–11%) | not reached | 1 | 7.8 s |
| | Watkins's Q(λ) | 15/30 (33–67%) | not reached | 30.8 | 24 s |
| | Dyna-Q | 30/30 (89–100%) | 57k | 51 | 101 s |

Q-learning's rows reproduce the evaluation suite exactly: the experiment's training loop is tested to give the suite's runs check for check.

**Paired with Q-learning**, on the mazes where both converged (geometric means, bootstrap 95% intervals). For each method: how many times fewer real moves it needed than Q-learning, and how many times more Q-value updates it spent to converge.

| Maze set | Dyna-Q: fewer real moves | Dyna-Q: more updates | Q(λ): fewer real moves | Q(λ): more updates |
|---|---|---|---|---|
| 2D (54 mazes) | 18.3× (16.8–19.9) | 2.8× (2.6–3.0) | 1.59× (1.57–1.61) | 8.9× (8.4–9.4) |
| One level (60) | 14.1× (13.2–15.1) | 3.6× (3.4–3.9) | 1.68× (1.65–1.71) | 6.5× (6.2–6.8) |
| Three levels (25 for Dyna-Q, 26 for Q(λ)) | 14.2× (12.6–16.1) | 3.6× (3.2–4.1) | 1.82× (1.74–1.92) | 6.5× (5.9–7.0) |

Mazes that only one of the pair converged:

- **Three-level suite:** Dyna-Q converged on 34 where Q-learning did not, and Q-learning on 1 where Dyna-Q did not. Exact McNemar p = 2 × 10⁻⁹; the converged share rose 55 points (42 to 68).
- **Long-route set:** Dyna-Q on 30 and Q-learning on 0 (p = 2 × 10⁻⁹). Q(λ) on 15 and Q-learning on 0 (p = 6 × 10⁻⁵).

### By route length

![Converged share by shortest-route length for the three methods: all converge up to 80 moves; at 81–120 moves Q-learning drops to 81%; beyond 120 moves Q-learning converges on none, Q(λ) on 71% (121–160) and 31% (over 160), Dyna-Q on all.](figures/route_length.png)

| Shortest route | Q-learning | Watkins's Q(λ) | Dyna-Q |
|---|---|---|---|
| up to 80 moves (96 mazes) | 96/96 | 96/96 | 96/96 |
| 81–120 moves (54) | 44/54 | 54/54 | 53/54 |
| more than 120 moves (60) | **0/60** | **30/60** | **60/60** |

The suite's finding, that no route longer than 120 moves converged, holds for Q-learning on 30 more mazes. It does not hold for Dyna-Q.

## Diagnostic: which part of learning got faster

Real moves to a shortest greedy route split into two phases:

1. exploring until an episode first reaches the goal, which measures finding the goal;
2. from there to the first shortest greedy route, which measures how fast that experience reaches the start as value.

The second phase is the one slow propagation predicts.

![Two dot plots of median real steps per maze set and method. Phase 1, exploring until an episode reaches the goal: Dyna-Q is fastest, and Q(λ) is slower than Q-learning. Phase 2, from there to the first shortest greedy route: Dyna-Q is 22 to 32 times faster than Q-learning, Q(λ) about twice as fast, and Q-learning does not get there on the long routes.](figures/diagnostics.png)

| Maze set | Phase 1, median real moves (Q-learning / Q(λ) / Dyna-Q) | Phase 2, median real moves (Q-learning / Q(λ) / Dyna-Q) |
|---|---|---|
| 2D | 1.8k / 5.9k / 1.1k | 80k / 51k / 2.5k |
| One level | 1.8k / 5.7k / 1.5k | 65k / 34k / 3.0k |
| Three levels | 33k / 39k / 7.0k | 342k / 173k / 15k |
| Long routes | 103k / 129k / 16k | not reached / 343k / 26k |

- **Propagation, isolated.** On 2D and one-level mazes, Dyna-Q and Q-learning find the goal at about the same time: medians of 1.1k–1.5k real moves for Dyna-Q and 1.8k for Q-learning. Dyna-Q then turns that first arrival into a shortest greedy route 22–32× faster. This is the propagation the explanation was about.
- **Propagation also speeds up exploration.** On three-level mazes Dyna-Q also finds the goal 4.7–6.3× sooner. In its first 10 episodes it discovers 69% of the reachable cells, against 38–49% for Q-learning. Untried actions are valued at 0 and tried ones go negative, so the agent is drawn to what it has not tried. Planning spreads the step and wall penalties through explored corridors quickly, so they lose their pull sooner and the agent moves on.
- **Q(λ) shortens only phase 2, by a third to a half, and lengthens phase 1.** Its traces carry those same penalties back along the path the agent just walked. That includes corridors that lead toward unexplored cells, so it explores less systematically. This is the mild form of the failure below.

## Failures

### Watkins's Q(λ) with long traces stops exploring

![Top left: the share of development runs converged against λ, peaking at λ 0.3 above Q-learning's 78% and falling to 0 at λ 0.9. Top right: the median share of reachable cells discovered against real steps, levelling off near 40% at λ 0.7 and 16% at λ 0.9. Bottom: one 2D development maze after 50,000 real steps, where Q-learning has discovered 183 of 185 cells and Q(λ) with λ 0.9 only 26, all near the start.](figures/failure.png)

On the development mazes, λ = 0.3 beat Q-learning: 57 of 60 runs converged, against 47. Longer traces failed:

| λ (accumulating traces) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| development runs converged (of 60) | 51 | 57 | 48 | 5 | 0 |
| reachable cells discovered after 50k real moves (median run) | 98% | 95% | 83% | 39% | 16% |

Replacing traces fail the same way: 15 and 0 of 60 runs converged at λ 0.7 and 0.9.

**The mechanism.** All three agents explore through optimism: untried actions are worth 0 and tried ones turn negative.

- **What gets credited:** Watkins's Q(λ) cuts its traces only after a non-greedy action. Choosing among untried actions, which all tie at 0, counts as greedy. So every step and wall penalty of an exploring walk is credited back along the whole path, including the corridors that led toward unexplored cells.
- **What that does:** those corridors turn as negative as the explored ones, the pull toward the unexplored disappears, and the walk becomes local. On development maze 1004, chosen by rule (the 2D development maze with the median route), Q(λ) with λ = 0.9 had found 26 of the maze's 185 cells after 50,000 real moves. Q-learning had found 183.
- **Why the tuning protects against it:** the tuning grid included λ ≥ 0.7 and the rule chose 0.3. That is why the final results show Q(λ) helping rather than collapsing.

### Dyna-Q can lock onto a longer route

Dyna-Q failed to converge on one final maze, three-level maze 58, which has more than one route:

- **What happened:** its greedy route settled on 114 moves and never matched the 106-move shortest route at any of its 1,000 checks. Q-learning converged on the same maze at episode 891, and Q(λ) at episode 332.
- **Why:** once planning has spread the goal's value along the first route found, that route's actions are worth more than 0, so untried actions no longer win the greedy choice. From then on only ε-exploration (1% late in training) can find the shorter branch, and here it did not.
- **The pattern:** fast propagation trades away some of the exploration that slow propagation keeps by accident. This is the one case where it cost Dyna-Q the right answer.

## What this does and does not show

- **Shown:** on these mazes, slow propagation is what stops one-step Q-learning on long routes. Methods that move value faster converge where it does not. The faster the propagation, the larger the gain: Dyna-Q at 51 updates per move beats Q(λ) at λ = 0.3.
- **Not shown:** that Dyna-Q is the better algorithm in general.
  - **Compute:** to converge, Dyna-Q uses about 3× the updates; over a full run it uses 10–16× Q-learning's agent time.
  - **Its model:** it relies on a deterministic, tabular model.
  - **Tuning:** only each method's own setting was tuned. The shared settings (learning rate, discount, ε schedule) are the repository's and were not tuned for either method.
  - **Scale:** the mazes are small and deterministic, the goal is always in the far corner, and nothing transfers between mazes.

## The question

The [evaluation suite](../evaluation/README.md) found two things about the repository's tabular Q-learning agent. Its greedy route needs about 7 to 9 training episodes per move of route before it is a shortest route. And no run with a route longer than 120 moves converged within 1,000 episodes. One-step Q-learning moves the goal's value back by one state per update, so the explanation offered was **slow value propagation**.

That explanation predicts that methods which propagate value faster should converge on long routes where Q-learning does not. If this is the cause, they should do so without exploring any differently. This experiment tests that prediction with two such methods, on the same mazes, from the same exploration seeds.

## Methods

All three agents share everything except how experience updates the Q-table:

- the learning rate 0.1, the discount 0.95 and Q-values starting at 0 (so untried actions look attractive);
- ε-greedy exploration from 0.1, decaying by 0.995 per episode to 0.01, with ties broken at random;
- the rewards: +100 at the goal, −0.1 per move, −1 for a wall bump.

| Method | Update | Extra setting |
|---|---|---|
| **Q-learning** ([`q_learning_agent.py`](../../maze_runner_rl/src/agents/q_learning_agent.py)), the baseline, unchanged | One-step: Q(s, a) moves toward r + γ max Q(s′, ·) | none |
| **Watkins's Q(λ)** ([`q_lambda_agent.py`](../../maze_runner_rl/src/agents/q_lambda_agent.py)) | Eligibility traces spread each update back over the state–action pairs visited since the last exploratory action, decaying by γλ per step | λ, and accumulating or replacing traces |
| **Dyna-Q** ([`dyna_q_agent.py`](../../maze_runner_rl/src/agents/dyna_q_agent.py)) | After each real step, n further Q-learning updates replayed from a learned model | n, the planning updates per real step |

**Watkins's Q(λ)** follows the tabular algorithm in Sutton and Barto (1998), Section 7.6:

- the next action a′ is chosen from s′ before the update;
- a* is the greedy action in s′, and a′ counts as a* if it ties for the best value;
- the TD error δ = r + γ Q(s′, a*) − Q(s, a) is applied to every traced pair;
- after the step the traces decay by γλ if a′ = a*, and are cut to 0 otherwise.

Replacing traces (Singh and Sutton, 1996) set the trace of the pair taken to 1 and those of the state's other actions to 0. The implementation was checked line by line against the authors' Chapter 7 lecture slides. Those slides print the δ line with a′ and a* swapped; the code uses the Q-learning target, which the section describes.

**Dyna-Q** follows the Tabular Dyna-Q box in Sutton and Barto (2018), Section 8.2, checked against Sutton's slides of the same algorithm:

- **Model:** a table of observed transitions, (state, action) → (reward, next state, ended the episode). The maze is deterministic, so the last outcome is the outcome. The model holds only transitions the agent has actually made, never the maze.
- **Planning:** each of the n planning updates picks a random previously observed state, then a random action previously taken there, and applies the one-step Q-learning update to the model's outcome. A planning update into the goal does not bootstrap, exactly as a real one does not.
- **Randomness:** planning draws from its own seeded generator (700000 + maze seed), so it does not shift the random numbers that drive exploration.

Each method's defining update is tested by hand in [`test_value_agents.py`](../../maze_runner_rl/tests/test_value_agents.py):

- Q(λ):
  - traces carry a goal reward back within one episode;
  - an exploratory next action cuts them, and a tie for the best value does not;
  - accumulating and replacing traces differ on revisits;
  - λ = 0 is exactly one-step Q-learning.
- Dyna-Q:
  - the model holds only observed transitions;
  - planning carries the goal's value back without new experience;
  - planning into the goal does not bootstrap;
  - n = 0 is exactly Q-learning.

## Protocol

The protocol was fixed in code ([`compare_methods.py`](../../maze_runner_rl/experiments/compare_methods.py)) before any final run:

1. **Development mazes**, maze seeds 1000–1019 of each type, chose each new method's own setting. Q-learning was not tuned. The grid and the rules:
   - **Watkins's Q(λ):** λ in {0.1, 0.3, 0.5, 0.7, 0.9} × {accumulating, replacing}. Most development runs converged wins; ties go to fewer median real steps to convergence.
   - **Dyna-Q:** n in {5, 10, 20, 50}. Take the best setting, then the smallest n that converged on at most one run fewer and whose median real steps to convergence is at most 10% higher, so planning compute is not spent on a difference the development mazes cannot show.
2. **Freezing:** the chosen settings and the final mazes were written to [`frozen_settings.json`](frozen_settings.json), committed and pushed (`49c188c`) before the final runs started. The final runs record that file's hash.
3. **Final mazes:**
   - the evaluation suite's mazes, seeds 0–59 of each type (2D 21 × 21, one level 21 × 21, three levels 21 × 21 × 3);
   - a **long-route set**: the first 30 three-level mazes from seed 2000 up whose shortest route has at least 141 moves. It was chosen from maze structure alone, before any run.

   Every method trains on the same maze with the same exploration seed (500000 + maze seed).

**Budget and criteria**, the same for every method and fixed in advance:

- **Budget:** 1,000 training episodes of at most 1,000 moves.
- **Greedy check:** after every episode the greedy policy is frozen and followed from the start with no exploration and no learning. It stops at the goal or at the first repeated cell.
- **Reaches:** the greedy walk arrives at the goal.
- **Shortest:** it arrives in the breadth-first-search distance.
- **Converged:** shortest at every check from some episode to the end of the budget, for at least 100 checks. A run that settles too late to show 100 checks counts as not converged; it is not dropped.
- **Experience:** counted in real environment steps, every move including bumps into walls. Planning updates and trace updates are counted separately, and the time spent inside the agent is measured, so extra computation stays visible.

**Statistics.** Shares carry 95% Wilson intervals over independent runs (one maze and one exploration seed each). Medians of real steps count a run that never converged as never converging, rather than dropping it; their intervals are percentile bootstraps over runs. Comparisons with Q-learning are paired by maze:

- an exact McNemar test on the mazes where only one of the two converged;
- a paired bootstrap of the change in converged share;
- a sign test on which method converged with fewer real steps (a run that did not converge counts as slower).

## Reproduce

From `maze_runner_rl/`, with the repository's virtual environment:

```bash
python -m experiments.compare_methods tune      # development mazes -> frozen_settings.json, tuning/ (26 min on 20 workers)
python -m experiments.compare_methods final     # final mazes with the frozen settings -> results/ (16 min)
python -m experiments.compare_methods tables    # summary.json and tables.md from results/runs.csv
python -m experiments.plot_value_propagation    # figures/ and figures/analysis.json
```

The runs are deterministic: the same commands give the same `runs.csv`, apart from the timing columns. Each results folder records its provenance: the commit, a hash of the code the results depend on, the frozen settings' hash, the library versions, the processor and the worker count. The per-episode data behind every run is kept in `episodes.jsonl.gz`, and every number quoted on this page is computed from it in [`figures/analysis.json`](figures/analysis.json).

**Provenance of the current results:**

- **Tuning:** ran at commit `91364a1`.
- **Final runs:** ran at commit `2a814c1`, after the frozen settings were committed and pushed.
- **Same code:** the two runs have the same code digest (`ec8a27ef…`) and no uncommitted changes to the code they depend on. A later commit changed only how `compare_methods.py` formats its tables and console output. A rerun gives the same runs, but a different code digest.
- **Environment:** Python 3.13.5, numpy 2.3.2, gymnasium 1.2.0.

## References

- C. J. C. H. Watkins (1989). *Learning from Delayed Rewards.* PhD thesis, University of Cambridge.
- R. S. Sutton (1990). Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. *Proceedings of the 7th International Conference on Machine Learning*, 216–224.
- S. P. Singh and R. S. Sutton (1996). Reinforcement learning with replacing eligibility traces. *Machine Learning* 22, 123–158.
- R. S. Sutton and A. G. Barto (1998). *Reinforcement Learning: An Introduction*, 1st ed., Section 7.6, Q(λ). Checked against the authors' [Chapter 7 slides](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%207.pdf), slides 23–24 and 31–34.
- R. S. Sutton and A. G. Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed., Section 8.2, Tabular Dyna-Q. Checked against Sutton's [planning and learning slides](https://web.stanford.edu/class/cme241/lecture_slides/rich_sutton_slides/16-17-planning-and-learning.pdf), slide 13.
