"""Does slow value propagation limit learning on long routes?

The evaluation suite found that the Q-learning agent needs about 7 to 9 episodes per move
of route before its greedy route is a shortest one, and that no run with a route longer
than 120 moves converged within 1000 episodes. One-step Q-learning carries the goal's
value back one state per update, so the explanation offered was slow value propagation.
This experiment tests it by comparing the baseline with two methods that propagate value
faster, on the same mazes with the same exploration seeds:

- Q-learning: the repository's agent, unchanged (src/agents/q_learning_agent.py).
- Watkins's Q(lambda): eligibility traces spread each update back over the state-action
  pairs visited since the last exploratory action (src/agents/q_lambda_agent.py).
- Dyna-Q: after every real step, n further updates replayed from a learned table of the
  transitions observed so far (src/agents/dyna_q_agent.py).

All three share the learning rate, discount, epsilon schedule, rewards, initial values
and action selection, so they differ only in how experience updates the Q-table.

Protocol, fixed before the final runs (docs/value-propagation/README.md):
1. Development mazes, seeds 1000-1019 of each type, choose each new method's own setting
   (lambda and the trace type; n) by the rules in `select_settings`.
2. `tune` writes the chosen settings and the final mazes to frozen_settings.json.
3. `final` reads that file and runs every method on the final mazes: seeds 0-59 of each
   type (the evaluation suite's mazes) and a long-route set, the first 30 three-level
   mazes from seed 2000 up whose shortest route is longer than 140 moves.

Every run trains for 1000 episodes of at most 1000 moves. After each episode the greedy
policy is frozen and followed from the start, with no exploration and no learning:
- it *reaches* the goal if it arrives without repeating a cell (a repeat loops forever);
- it is *shortest* if it arrives in the breadth-first-search distance;
- the run has *converged* if the greedy route was shortest at every check from some
  episode to the end of the budget, for at least 100 checks.
Experience is counted in real environment steps (every move, including bumps into walls).
Planning updates and trace updates are counted separately from real steps, and the
time spent in the agent is measured, so extra computation stays visible.

Run from maze_runner_rl/. Reproduce the published final runs with the committed settings,
into a separate folder, then compare them with the published records:
    python -m experiments.compare_methods final --settings ../docs/value-propagation/frozen_settings.json --out ../reproduction
    python -m experiments.compare_methods verify --out ../reproduction
Retune on the development mazes (writes a new frozen_settings.json under --out):
    python -m experiments.compare_methods tune --out ../retune
    python -m experiments.compare_methods final --out ../retune
`tables` rebuilds summary.json and tables.md from <out>/results/runs.csv. `tune` and `final`
refuse to replace existing frozen settings or run records unless --overwrite is given.
"""
import argparse
import contextlib
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from experiments import evaluate as ev
from experiments.stats import bootstrap_interval, paired_bootstrap_difference, wilson_interval
from experiments.training_experiment import convergence_episode, shortest_route_length
from src.agents.dyna_q_agent import DynaQAgent
from src.agents.q_lambda_agent import WatkinsQLambdaAgent
from src.agents.q_learning_agent import QLearningAgent

OUT = os.path.normpath(os.path.join(ev.PACKAGE_ROOT, "..", "docs", "value-propagation"))
KINDS = ("2d", "3d_1", "3d_3")
DEV_SEEDS = tuple(range(1000, 1020))
FINAL_SEEDS = tuple(range(60))
LONG_ROUTES = {"kind": "3d_3", "first_seed": 2000, "min_route": 141, "count": 30}
PLANNING_SEED_BASE = 700_000        # Dyna-Q's planning generator on maze s: 700000 + s
EPISODES = ev.SETTINGS["training"]["max_episodes"]
MAX_STEPS = ev.SETTINGS["training"]["max_steps_per_episode"]
WINDOW = 100
STEP_GRID = [int(x) for x in np.unique(np.round(np.logspace(2, np.log10(EPISODES * MAX_STEPS), 81)))]
ROUTE_BINS = ((0, 80), (81, 120), (121, 10_000))

METHODS = {
    "q_learning": {"label": "Q-learning", "grid": [{}]},
    "q_lambda": {"label": "Watkins's Q(λ)",
                 "grid": [{"trace_decay": lam, "replacing": replacing}
                          for replacing in (True, False) for lam in (0.1, 0.3, 0.5, 0.7, 0.9)]},
    "dyna_q": {"label": "Dyna-Q", "grid": [{"planning_steps": n} for n in (5, 10, 20, 50)]},
}
SELECTION_RULES = {
    "q_lambda": "Most development runs converged; ties go to the lower median real steps to convergence "
                "(a run that did not converge counts as never), then to the earlier setting in the grid.",
    "dyna_q": "Take the setting with the most development runs converged (ties: lower median real steps to "
              "convergence). Then use the smallest n that converged on at most one run fewer and whose median "
              "is at most 10% higher, so planning compute is not spent on a difference the development mazes "
              "cannot show.",
}


# --- agents -----------------------------------------------------------------------------

def params_key(params):
    return ",".join(f"{k}={v}" for k, v in sorted(params.items())) or "default"


def make_agent(method, params, env, maze_seed):
    a = ev.SETTINGS["agent"]
    common = dict(learning_rate=a["learning_rate"], epsilon=a["epsilon"], epsilon_decay=a["epsilon_decay"],
                  discount_factor=a["gamma"], epsilon_min=a["epsilon_min"])
    size = (env.num_states, env.action_space.n)
    if method == "q_learning":
        return QLearningAgent(*size, **common)
    if method == "q_lambda":
        return WatkinsQLambdaAgent(*size, **common, trace_decay=params["trace_decay"], replacing=params["replacing"])
    if method == "dyna_q":
        return DynaQAgent(*size, **common, planning_steps=params["planning_steps"],
                          planning_seed=PLANNING_SEED_BASE + maze_seed)
    raise ValueError(f"unknown method {method!r}")


# --- the maze, from its transition model ---------------------------------------------------

def maze_facts(env):
    """Transition table, start, goal, distances, and the cells on shortest start-goal routes.

    The goal is absorbing, so distances ignore moves out of it: a cell reachable only
    through the goal is never visited and is not counted."""
    nxt, rew, done = ev.transition_model(env)
    start, goal = ev.endpoints(env)
    cells = ev.reachable_states(nxt, start)
    forward = {s: sorted({int(t) for t in nxt[s] if int(t) != s}) for s in cells if s != goal}
    backward = {}
    for s, targets in forward.items():
        for t in targets:
            backward.setdefault(t, []).append(s)

    def bfs(source, edges):
        dist, queue = {source: 0}, deque([source])
        while queue:
            s = queue.popleft()
            for t in edges.get(s, ()):
                if t not in dist:
                    dist[t] = dist[s] + 1
                    queue.append(t)
        return dist

    from_start, to_goal = bfs(start, forward), bfs(goal, backward)
    route = from_start[goal]
    on_route = [s for s in cells if from_start.get(s, -1) + to_goal.get(s, -10 ** 9) == route]
    layers = [[] for _ in range(route + 1)]
    for s in on_route:
        layers[to_goal[s]].append(s)
    return {"nxt": nxt, "rew": rew, "done": done, "start": start, "goal": goal, "cells": cells,
            "to_goal": to_goal, "route": route, "layers": layers}


def greedy_walk(agent, nxt, start, goal, limit):
    """Same check as training_experiment.greedy_route_length, on the transition table."""
    state, visited = start, {start}
    for moves in range(1, limit + 1):
        state = int(nxt[state, agent.greedy_action(state)])
        if state == goal:
            return moves
        if state in visited:
            return None
        visited.add(state)
    return None


def learned_depth(agent, facts):
    """How far back from the goal the greedy policy follows shortest routes.

    A cell on a shortest start-goal route counts if its greedy action moves one step
    closer to the goal, to a cell that counts (the goal counts). Returns the largest
    distance from the goal at which some cell counts: equal to the route length exactly
    when the greedy route from the start is a shortest route. Value spreads back from the
    goal, so this is where the useful part of the propagated value has reached."""
    nxt, to_goal, layers = facts["nxt"], facts["to_goal"], facts["layers"]
    ok, depth = {facts["goal"]}, 0
    for d in range(1, len(layers)):
        found = False
        for s in layers[d]:
            t = int(nxt[s, agent.greedy_action(s)])
            if t in ok and to_goal.get(t) == d - 1:
                ok.add(s)
                found = True
        if not found:
            break
        depth = d
    return depth


def sample_values(cum_steps, values, step_grid, before=0):
    """A per-episode value read at each point of an environment-step grid (the value after
    the last episode that ended within that many steps; a finished run keeps its last)."""
    index = np.searchsorted(cum_steps, step_grid, side="right") - 1
    values = np.asarray(values)
    return np.where(index >= 0, values[np.clip(index, 0, None)], before)


def first_index(flags):
    return next((i for i, f in enumerate(flags) if f), None)


# --- one run --------------------------------------------------------------------------------

def train(kind, method, params, maze_seed, split, episodes=EPISODES, max_steps=MAX_STEPS, window=WINDOW):
    """One training run. Returns its record, its curves on STEP_GRID and its per-episode data."""
    started = time.perf_counter()
    env = ev.make_env(kind, maze_seed)
    facts = maze_facts(env)
    seed = ev.exploration_seed(maze_seed)
    random.seed(seed)
    agent = make_agent(method, params, env, maze_seed)
    begin = getattr(agent, "begin_episode", None)
    end = getattr(agent, "end_episode", None) or agent.decay_epsilon
    nxt, start, goal, limit = facts["nxt"], facts["start"], facts["goal"], env.num_states

    clock = time.perf_counter_ns
    agent_ns = env_ns = 0
    steps, successes, greedy, depth, updates, new_cells = [], [], [], [], [], []
    counted = 0
    seen = {start}
    for _ in range(episodes):
        _, info = env.reset()
        if begin:
            begin()
        state, terminated, n, found = info["state"], False, 0, 0
        for _ in range(max_steps):
            t0 = clock()
            action = agent.choose_action(state)
            t1 = clock()
            _, reward, terminated, truncated, info = env.step(action)
            t2 = clock()
            next_state = info["state"]
            agent.update_q_table(state, action, reward, next_state, terminated)
            t3 = clock()
            agent_ns += t1 - t0 + t3 - t2
            env_ns += t2 - t1
            if next_state not in seen:
                seen.add(next_state)
                found += 1
            state = next_state
            n += 1
            if terminated or truncated:
                break
        t0 = clock()
        end()
        agent_ns += clock() - t0
        steps.append(n)
        successes.append(bool(terminated))
        new_cells.append(found)
        greedy.append(greedy_walk(agent, nxt, start, goal, limit))
        depth.append(learned_depth(agent, facts))
        total = getattr(agent, "updates", None)
        total = sum(steps) if total is None else total     # Q-learning: one update per real step
        updates.append(total - counted)
        counted = total

    route = facts["route"]
    cum = np.cumsum(steps)
    reaches = [g is not None for g in greedy]
    shortest = [g is not None and g == route for g in greedy]
    converged = convergence_episode(shortest, window)
    first_success, first_reach, first_short = first_index(successes), first_index(reaches), first_index(shortest)
    half = first_index(d * 2 >= route for d in depth)

    def at(i):
        return None if i is None else int(cum[i])

    _, q_star = ev.optimal_values(facts["nxt"], facts["rew"], facts["done"], agent.discount_factor)
    analysis = ev.policy_analysis(agent, nxt, q_star, facts["cells"], start, goal)
    record = {
        "split": split, "kind": kind, "method": method, "params": params_key(params), "maze_seed": maze_seed,
        "exploration_seed": seed,
        "planning_seed": PLANNING_SEED_BASE + maze_seed if method == "dyna_q" else None,
        "shortest_route": route, "reachable_cells": len(facts["cells"]),
        "episodes": episodes, "env_steps": int(cum[-1]),
        "training_success_last100": float(np.mean(successes[-window:])),
        "first_success_episode": None if first_success is None else first_success + 1,
        "first_success_env_steps": at(first_success),
        "first_greedy_reach_env_steps": at(first_reach),
        "first_greedy_shortest_episode": None if first_short is None else first_short + 1,
        "first_greedy_shortest_env_steps": at(first_short),
        "half_route_learned_env_steps": at(half),
        "converged_episode": converged,
        "converged_env_steps": None if converged is None else int(cum[converged - 1]),
        "greedy_reaches_final": reaches[-1], "greedy_shortest_final": shortest[-1],
        "final_greedy_route": greedy[-1], "final_learned_depth": depth[-1],
        "updates": int(sum(updates)),
        "planning_updates": int(getattr(agent, "planning_updates", 0)),
        "trace_cuts": int(getattr(agent, "cuts", 0)),
        "agent_seconds": round(agent_ns / 1e9, 3), "env_seconds": round(env_ns / 1e9, 3),
        "run_seconds": round(time.perf_counter() - started, 3),
        "greedy_optimal_share": analysis["greedy_optimal_share"], "visited_share": analysis["visited_share"],
        "cells_found_10_episodes": int(sum(new_cells[:10])) + 1,
    }
    curves = {
        "split": split, "kind": kind, "method": method, "params": record["params"], "maze_seed": maze_seed,
        "shortest": ev.bits(ev.sample_at(cum, shortest, STEP_GRID)),
        "reaches": ev.bits(ev.sample_at(cum, reaches, STEP_GRID)),
        "depth": [round(float(v) / route, 3) for v in sample_values(cum, depth, STEP_GRID)],
    }
    raw = {
        "split": split, "kind": kind, "method": method, "params": record["params"], "maze_seed": maze_seed,
        "steps": steps, "success": ev.bits(successes), "greedy": [-1 if g is None else g for g in greedy],
        "depth": depth, "updates": updates, "new_cells": new_cells,
    }
    return record, curves, raw


def work(task):
    """Entry point for worker processes; the 3D generator's messages are silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        return train(*task)


def run_tasks(tasks, workers, progress=True):
    if workers <= 1:
        return [work(t) for t in tasks]
    outputs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, t) for t in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            outputs.append(future.result())
            if progress and (done % 25 == 0 or done == len(futures)):
                print(f"  {done}/{len(futures)} runs done", flush=True)
    order = {k: i for i, k in enumerate(KINDS)}
    methods = {m: i for i, m in enumerate(METHODS)}
    outputs.sort(key=lambda o: (o[0]["split"], order[o[0]["kind"]], methods[o[0]["method"]], o[0]["params"],
                                o[0]["maze_seed"]))
    return outputs


def task_cost(task):
    """Rough cost, so the slowest runs start first and the pool stays busy."""
    kind, method, params, _, split = task[:5]
    return (params.get("planning_steps", 0) + 1) * (3 if kind == "3d_3" else 1) * (2 if split == "long" else 1)


# --- the long-route set ----------------------------------------------------------------------

def long_route_seeds(spec=LONG_ROUTES):
    """The first `count` mazes from `first_seed` up whose shortest route has at least
    `min_route` moves. Chosen from maze structure alone, before any run."""
    seeds, seed = [], spec["first_seed"]
    with contextlib.redirect_stdout(io.StringIO()):
        while len(seeds) < spec["count"]:
            if shortest_route_length(ev.make_env(spec["kind"], seed)) >= spec["min_route"]:
                seeds.append(seed)
            seed += 1
    return seeds


# --- statistics ------------------------------------------------------------------------------

INF = float("inf")


def proportion(flags):
    k, n = int(sum(flags)), len(flags)
    low, high = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci": [low, high]}


def median_steps(values, seed=0):
    """Median over all runs with None (never reached) counted as infinite, so runs that
    failed are not dropped; the interval is a percentile bootstrap over runs."""
    values = [INF if v is None else float(v) for v in values]
    if not values:
        return {"median": None, "ci": [None, None], "reached": 0, "n": 0}
    low, high = bootstrap_interval(values, np.median, seed=seed)
    return {"median": float(np.median(values)), "ci": [low, high], "reached": sum(v != INF for v in values),
            "n": len(values)}


def sign_test(wins, losses):
    """Two-sided exact binomial test of wins against losses (ties left out)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(base_runs, other_runs):
    """Compare a method with the baseline maze by maze (same maze, same exploration seed)."""
    base = {r["maze_seed"]: r for r in base_runs}
    other = {r["maze_seed"]: r for r in other_runs}
    seeds = sorted(set(base) & set(other))
    b_conv = [base[s]["converged_episode"] is not None for s in seeds]
    o_conv = [other[s]["converged_episode"] is not None for s in seeds]
    diff, low, high = paired_bootstrap_difference(b_conv, o_conv)
    only_other = sum(o and not b for b, o in zip(b_conv, o_conv))
    only_base = sum(b and not o for b, o in zip(b_conv, o_conv))

    def steps(r):
        return INF if r["converged_env_steps"] is None else r["converged_env_steps"]

    faster = sum(steps(other[s]) < steps(base[s]) for s in seeds)
    slower = sum(steps(other[s]) > steps(base[s]) for s in seeds)
    both = [s for s, b, o in zip(seeds, b_conv, o_conv) if b and o]
    ratio = None
    if both:
        logs = np.log([base[s]["converged_env_steps"] / other[s]["converged_env_steps"] for s in both])
        lo, hi = bootstrap_interval(logs, np.mean)
        ratio = {"geometric_mean": float(np.exp(logs.mean())), "ci": [float(np.exp(lo)), float(np.exp(hi))],
                 "mazes": len(both)}
    return {
        "mazes": len(seeds),
        "converged_difference": {"diff": diff, "ci": [low, high]},
        "converged_only_method": only_other, "converged_only_baseline": only_base,
        "converged_mcnemar_p": sign_test(only_other, only_base),
        "faster_to_converge": faster, "slower_to_converge": slower, "neither_converged": len(seeds) - faster - slower,
        "faster_sign_test_p": sign_test(faster, slower),
        "speedup_where_both_converged": ratio,
    }


def summarize_runs(runs):
    lag = [None if r["first_greedy_shortest_env_steps"] is None or r["first_success_env_steps"] is None
           else r["first_greedy_shortest_env_steps"] - r["first_success_env_steps"] for r in runs]
    per_step = [r["updates"] / r["env_steps"] for r in runs]
    return {
        "runs": len(runs),
        "converged": proportion([r["converged_episode"] is not None for r in runs]),
        "greedy_shortest_final": proportion([r["greedy_shortest_final"] for r in runs]),
        "greedy_reaches_final": proportion([r["greedy_reaches_final"] for r in runs]),
        "training_success_last100": float(np.mean([r["training_success_last100"] for r in runs])),
        "converged_env_steps": median_steps([r["converged_env_steps"] for r in runs]),
        "first_greedy_shortest_env_steps": median_steps([r["first_greedy_shortest_env_steps"] for r in runs]),
        "first_success_env_steps": median_steps([r["first_success_env_steps"] for r in runs]),
        "lag_success_to_shortest": median_steps(lag),
        "half_route_learned_env_steps": median_steps([r["half_route_learned_env_steps"] for r in runs]),
        "env_steps": float(np.median([r["env_steps"] for r in runs])),
        "updates": float(np.median([r["updates"] for r in runs])),
        "updates_per_real_step": float(np.median(per_step)),
        "planning_updates": float(np.median([r["planning_updates"] for r in runs])),
        "agent_seconds": float(np.median([r["agent_seconds"] for r in runs])),
        "agent_seconds_total": float(np.sum([r["agent_seconds"] for r in runs])),
        "env_seconds": float(np.median([r["env_seconds"] for r in runs])),
        "greedy_optimal_share": float(np.mean([r["greedy_optimal_share"] for r in runs])),
        "not_converged": [r["maze_seed"] for r in runs if r["converged_episode"] is None],
    }


def route_law(runs):
    points = [(r["shortest_route"], r["first_greedy_shortest_episode"]) for r in runs
              if r["first_greedy_shortest_episode"]]
    if len(points) < 3:
        return None
    x, y = np.array(points, dtype=float).T
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "r": float(np.corrcoef(x, y)[0, 1]),
            "runs": len(points)}


def aggregate(runs):
    """Per set and maze type: each method's summary, its comparison with Q-learning on the
    same mazes, results by route length, and episodes per move of route."""
    out = {}
    sets = [("suite", k) for k in KINDS] + [("long", LONG_ROUTES["kind"])]
    for split, kind in sets:
        group = [r for r in runs if r["split"] == split and r["kind"] == kind]
        if not group:
            continue
        entry = out.setdefault(f"{split}:{kind}", {})
        base = [r for r in group if r["method"] == "q_learning"]
        for method in METHODS:
            mine = [r for r in group if r["method"] == method]
            if not mine:
                continue
            summary = summarize_runs(mine)
            summary["params"] = mine[0]["params"]
            summary["route_law"] = route_law(mine)
            if method != "q_learning" and base:
                summary["vs_q_learning"] = paired(base, mine)
            entry[method] = summary
    pooled = [r for r in runs if r["split"] in ("suite", "long")]
    by_route = {}
    for low, high in ROUTE_BINS:
        label = f"{low}-{high}" if high < 10_000 else f"{low}+"
        for method in METHODS:
            mine = [r for r in pooled if r["method"] == method and low <= r["shortest_route"] <= high]
            if mine:
                by_route.setdefault(label, {})[method] = {
                    "converged": proportion([r["converged_episode"] is not None for r in mine]),
                    "converged_env_steps": median_steps([r["converged_env_steps"] for r in mine]),
                    "lag_success_to_shortest": median_steps(
                        [None if r["first_greedy_shortest_env_steps"] is None or r["first_success_env_steps"] is None
                         else r["first_greedy_shortest_env_steps"] - r["first_success_env_steps"] for r in mine]),
                }
    return {"groups": out, "by_route_length": by_route}


# --- tuning and the frozen settings ------------------------------------------------------------

def dev_scores(runs, method):
    scores = []
    for params in METHODS[method]["grid"]:
        key = params_key(params)
        mine = [r for r in runs if r["method"] == method and r["params"] == key]
        if not mine:
            continue
        conv = sum(r["converged_episode"] is not None for r in mine)
        med = median_steps([r["converged_env_steps"] for r in mine])["median"]
        scores.append({"params": params, "key": key, "runs": len(mine), "converged": conv,
                       "median_converged_env_steps": med,
                       "median_updates": float(np.median([r["updates"] for r in mine])),
                       "median_agent_seconds": float(np.median([r["agent_seconds"] for r in mine]))})
    return scores


def select_settings(runs):
    """The rules in SELECTION_RULES, fixed in code before the final runs, applied to the development runs."""
    chosen, tables = {"q_learning": {}}, {}
    for method in ("q_lambda", "dyna_q"):
        scores = dev_scores(runs, method)
        tables[method] = scores
        if not scores:
            continue
        best = min(scores, key=lambda s: (-s["converged"], s["median_converged_env_steps"]))
        if method == "dyna_q":
            for s in sorted(scores, key=lambda s: s["params"]["planning_steps"]):
                if (s["converged"] >= best["converged"] - 1
                        and s["median_converged_env_steps"] <= 1.10 * best["median_converged_env_steps"]):
                    best = s
                    break
        chosen[method] = best["params"]
    return chosen, tables


# --- provenance ------------------------------------------------------------------------------

def source_paths():
    extra = [os.path.join(ev.PACKAGE_ROOT, "experiments", "compare_methods.py")]
    return sorted(set(ev.source_paths()) | set(extra))


def source_digest():
    digest = hashlib.sha256()
    for path in source_paths():
        digest.update(os.path.relpath(path, ev.PACKAGE_ROOT).replace(os.sep, "/").encode())
        with open(path, "rb") as f:
            digest.update(f.read().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def snapshot():
    status = ev.git("status", "--porcelain", "--untracked-files=all", "--",
                    *(os.path.relpath(p, ev.PACKAGE_ROOT) for p in source_paths()))
    return {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": ev.git("rev-parse", "HEAD"),
            "git_uncommitted_source_changes": None if status is None else bool(status),
            "source_sha256": source_digest()}


def environment(workers, seconds):
    import gymnasium
    return {"python": platform.python_version(), "numpy": np.__version__, "gymnasium": gymnasium.__version__,
            "platform": platform.platform(), "processor": platform.processor(), "workers": workers,
            "wall_seconds": round(seconds, 1)}


def settings_block():
    from config.config import CONFIG
    return {"agent": ev.SETTINGS["agent"], "maze": ev.SETTINGS["maze"],
            "rewards": {"goal": CONFIG.rewards.goal_reward, "step": CONFIG.penalties.step_penalty,
                        "wall": CONFIG.penalties.wall_penalty},
            "episodes": EPISODES, "max_steps": MAX_STEPS, "window": WINDOW,
            "exploration_seed": f"{ev.EXPLORATION_SEED_BASE} + maze seed",
            "planning_seed": f"{PLANNING_SEED_BASE} + maze seed (Dyna-Q)"}


CRITERIA = {
    "greedy_check": "After every episode the greedy policy (no exploration, no learning) is followed from the "
                    "start; it stops at the goal or at the first repeated cell.",
    "reaches": "The greedy walk arrives at the goal.",
    "shortest": "The greedy walk arrives in the breadth-first-search distance.",
    "converged": "Shortest at every check from some episode to the end of the budget, for at least 100 checks.",
    "experience": "Real environment steps, every move counted, including bumps into walls.",
    "compute": "Q-value updates (real, trace and planning) and the time spent inside the agent are counted "
               "separately from real steps.",
}


# --- outputs -----------------------------------------------------------------------------------

RUN_FIELDS = ["split", "kind", "method", "params", "maze_seed", "exploration_seed", "planning_seed",
              "shortest_route", "reachable_cells", "episodes", "env_steps", "training_success_last100",
              "first_success_episode", "first_success_env_steps", "first_greedy_reach_env_steps",
              "first_greedy_shortest_episode", "first_greedy_shortest_env_steps", "half_route_learned_env_steps",
              "converged_episode", "converged_env_steps", "greedy_reaches_final", "greedy_shortest_final",
              "final_greedy_route", "final_learned_depth", "updates", "planning_updates", "trace_cuts",
              "agent_seconds", "env_seconds", "run_seconds", "greedy_optimal_share", "visited_share",
              "cells_found_10_episodes"]
INT_FIELDS = {"cells_found_10_episodes", "maze_seed", "exploration_seed", "planning_seed", "shortest_route",
              "reachable_cells", "episodes",
              "env_steps", "first_success_episode", "first_success_env_steps", "first_greedy_reach_env_steps",
              "first_greedy_shortest_episode", "first_greedy_shortest_env_steps", "half_route_learned_env_steps",
              "converged_episode", "converged_env_steps", "final_greedy_route", "final_learned_depth", "updates",
              "planning_updates", "trace_cuts"}
BOOL_FIELDS = {"greedy_reaches_final", "greedy_shortest_final"}


def write_runs(path, runs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ev.write_csv(path, RUN_FIELDS, runs)


def read_runs(path):
    runs = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            r = {}
            for k, v in row.items():
                if v == "":
                    r[k] = None
                elif k in INT_FIELDS:
                    r[k] = int(v)
                elif k in BOOL_FIELDS:
                    r[k] = v == "True"
                elif k in ("split", "kind", "method", "params"):
                    r[k] = v
                else:
                    r[k] = float(v)
            runs.append(r)
    return runs


def write_raw(path, raws):
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        for raw in raws:
            f.write((json.dumps(raw, separators=(",", ":")) + "\n").encode())


def read_raw(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_curves(path, curves):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write('{"step_grid":' + json.dumps(STEP_GRID, separators=(",", ":")) + ',\n"runs":[\n')
        f.write(",\n".join(json.dumps(c, separators=(",", ":"), ensure_ascii=False) for c in curves))
        f.write("\n]}\n")


def write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ev._finite(data), f, indent=2, ensure_ascii=False)
        f.write("\n")


# --- tables ------------------------------------------------------------------------------------

def fmt_steps(x):
    return ev.fmt_steps(x)


def fmt_ratio(x):
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def fmt_share(p):
    if p is None:
        return "–"
    low, high = p["ci"]
    return f"{p['k']}/{p['n']} ({low * 100:.0f}–{high * 100:.0f}%)"


def fmt_median(m):
    if m is None or m["median"] is None:
        return "–"
    low, high = m["ci"]
    if m["median"] == INF:
        return f"not reached (fewer than half: {m['reached']} of {m['n']})"
    return f"{fmt_steps(m['median'])} ({fmt_steps(low)}–{fmt_steps(high)})"


def render_tables(summary):
    lines = ["# Value propagation: tables", "",
             "Generated by `python -m experiments.compare_methods tables` from `runs.csv`. Medians count a run that "
             "did not converge as never converging; intervals are 95% (Wilson for shares, percentile bootstrap "
             "over runs for medians). Paired comparisons use the same mazes and exploration seeds.", ""]
    for name, entry in summary["groups"].items():
        split, kind = name.split(":")
        title = ("Long-route set, " if split == "long" else "") + ev.KINDS[kind]
        lines += [f"## {title}", "",
                  "| Method | Converged | Real steps to converge (median) | Final greedy route shortest | "
                  "Updates per real step | Agent time per run (median) | Planning updates per run (median) |",
                  "|---|---|---|---|---|---|---|"]
        for method, s in entry.items():
            lines.append(f"| {METHODS[method]['label']} ({s['params']}) | {fmt_share(s['converged'])} | "
                         f"{fmt_median(s['converged_env_steps'])} | {fmt_share(s['greedy_shortest_final'])} | "
                         f"{s['updates_per_real_step']:.1f} | {s['agent_seconds']:.1f} s | "
                         f"{fmt_steps(s['planning_updates']) if s['planning_updates'] else '0'} |")
        lines.append("")
        pairs = [(m, s["vs_q_learning"]) for m, s in entry.items() if "vs_q_learning" in s]
        if pairs:
            lines += ["| Compared with Q-learning, same mazes | Converged: only this method / only Q-learning "
                      "(exact p) | Change in converged share | Converged with fewer real steps / more (p) | "
                      "Speed-up where both converged (geometric mean) |", "|---|---|---|---|---|"]
            for method, p in pairs:
                d = p["converged_difference"]
                speed = p["speedup_where_both_converged"]
                speed_text = "–" if speed is None else (
                    f"{fmt_ratio(speed['geometric_mean'])}× ({fmt_ratio(speed['ci'][0])}–{fmt_ratio(speed['ci'][1])}), "
                    f"{speed['mazes']} mazes")
                lines.append(f"| {METHODS[method]['label']} | {p['converged_only_method']} / "
                             f"{p['converged_only_baseline']} (p = {p['converged_mcnemar_p']:.2g}) | "
                             f"{d['diff'] * 100:+.0f} points ({d['ci'][0] * 100:+.0f} to {d['ci'][1] * 100:+.0f}) | "
                             f"{p['faster_to_converge']} / {p['slower_to_converge']} "
                             f"(p = {p['faster_sign_test_p']:.2g}) | {speed_text} |")
            lines.append("")
        not_conv = {m: s["not_converged"] for m, s in entry.items() if s["not_converged"]}
        if not_conv:
            lines.append("Runs that did not converge (maze seeds): " + "; ".join(
                f"{METHODS[m]['label']}: {', '.join(map(str, seeds))}" for m, seeds in not_conv.items()) + ".")
            lines.append("")
    lines += ["## By shortest-route length (suite and long-route sets together)", "",
              "| Route length | " + " | ".join(METHODS[m]["label"] for m in METHODS) + " |",
              "|---|" + "---|" * len(METHODS)]
    for label, entry in summary["by_route_length"].items():
        lines.append(f"| {label} moves | " + " | ".join(
            fmt_share(entry[m]["converged"]) if m in entry else "–" for m in METHODS) + " |")
    lines.append("")
    return "\n".join(lines)


# --- commands ----------------------------------------------------------------------------------

def make_tasks(kinds, seeds, methods, split, episodes):
    tasks = []
    for kind in kinds:
        for method, grid in methods.items():
            for params in grid:
                for seed in seeds:
                    tasks.append((kind, method, params, seed, split, episodes))
    return sorted(tasks, key=task_cost, reverse=True)


def refuse_to_overwrite(path, args, what):
    """Committed records are replaced only on request: a rerun goes to a separate --out."""
    if os.path.exists(path) and not getattr(args, "overwrite", False):
        raise SystemExit(f"{path} already exists: write {what} to a separate --out, or pass --overwrite to replace it")


def cmd_tune(args):
    refuse_to_overwrite(os.path.join(args.out, "frozen_settings.json"), args, "the new settings")
    out = os.path.join(args.out, "tuning")
    os.makedirs(out, exist_ok=True)
    seeds = DEV_SEEDS[:args.mazes] if args.mazes else DEV_SEEDS
    grids = {m: METHODS[m]["grid"] for m in METHODS}
    if args.quick:
        grids = {"q_learning": [{}], "q_lambda": METHODS["q_lambda"]["grid"][2:3],
                 "dyna_q": METHODS["dyna_q"]["grid"][:2]}
    snap = snapshot()
    t0 = time.time()
    outputs = run_tasks(make_tasks(KINDS, seeds, grids, "dev", args.episodes), args.workers)
    runs = [o[0] for o in outputs]
    write_runs(os.path.join(out, "runs.csv"), runs)
    write_raw(os.path.join(out, "episodes.jsonl.gz"), [o[2] for o in outputs])
    chosen, tables = select_settings(runs)
    long_seeds = long_route_seeds(dict(LONG_ROUTES, count=args.long_count)) if args.long_count else []
    frozen = {
        "frozen": snap,
        "environment": environment(args.workers, time.time() - t0),
        "command": " ".join(["python -m experiments.compare_methods", *sys.argv[1:]]),
        "settings": settings_block(),
        "criteria": CRITERIA,
        "development": {"kinds": list(KINDS), "maze_seeds": list(seeds), "episodes": args.episodes,
                        "rules": SELECTION_RULES, "scores": tables},
        "methods": {m: {"label": METHODS[m]["label"], "params": chosen.get(m, {})} for m in METHODS},
        "final_sets": {
            "suite": {"kinds": list(KINDS), "maze_seeds": list(FINAL_SEEDS)},
            "long": {"kind": LONG_ROUTES["kind"], "maze_seeds": long_seeds,
                     "rule": f"the first {len(long_seeds)} {LONG_ROUTES['kind']} mazes from seed "
                             f"{LONG_ROUTES['first_seed']} up whose shortest route has at least "
                             f"{LONG_ROUTES['min_route']} moves"},
        },
    }
    write_json(os.path.join(args.out, "frozen_settings.json"), frozen)
    print(json.dumps(frozen["methods"], indent=2, ensure_ascii=False))
    for method, scores in tables.items():
        for s in scores:
            print(f"  {method} {s['key']}: converged {s['converged']}/{s['runs']}, "
                  f"median steps {fmt_steps(s['median_converged_env_steps'])}, "
                  f"median updates {ev.fmt_number(s['median_updates'])}, agent {s['median_agent_seconds']:.1f} s")


def cmd_final(args):
    frozen_path = getattr(args, "settings", None) or os.path.join(args.out, "frozen_settings.json")
    if not os.path.exists(frozen_path):
        raise SystemExit(f"{frozen_path} is missing: run `tune` first, so the settings are fixed before the final runs, "
                         "or pass --settings")
    refuse_to_overwrite(os.path.join(args.out, "results", "runs.csv"), args, "the new runs")
    with open(frozen_path, encoding="utf-8") as f:
        frozen = json.load(f)
    methods = {m: [frozen["methods"][m]["params"]] for m in METHODS}
    sets = frozen["final_sets"]
    suite_seeds = sets["suite"]["maze_seeds"][:args.mazes] if args.mazes else sets["suite"]["maze_seeds"]
    long_seeds = sets["long"]["maze_seeds"][:args.mazes] if args.mazes else sets["long"]["maze_seeds"]
    tasks = make_tasks(KINDS, suite_seeds, methods, "suite", args.episodes)
    tasks += make_tasks((sets["long"]["kind"],), long_seeds, methods, "long", args.episodes)
    tasks.sort(key=task_cost, reverse=True)
    snap = snapshot()
    t0 = time.time()
    outputs = run_tasks(tasks, args.workers)
    out = os.path.join(args.out, "results")
    os.makedirs(out, exist_ok=True)
    runs = [o[0] for o in outputs]
    write_runs(os.path.join(out, "runs.csv"), runs)
    write_curves(os.path.join(out, "curves.json"), [o[1] for o in outputs])
    write_raw(os.path.join(out, "episodes.jsonl.gz"), [o[2] for o in outputs])
    provenance = {**snap, "environment": environment(args.workers, time.time() - t0),
                  "command": " ".join(["python -m experiments.compare_methods", *sys.argv[1:]]),
                  "frozen_settings_sha256": file_sha256(frozen_path), "settings": settings_block(),
                  "criteria": CRITERIA, "methods": frozen["methods"], "episodes": args.episodes,
                  "suite_seeds": suite_seeds, "long_seeds": long_seeds}
    write_json(os.path.join(out, "provenance.json"), provenance)
    write_summary(out, runs)


def write_summary(out, runs):
    summary = aggregate(runs)
    write_json(os.path.join(out, "summary.json"), summary)
    with open(os.path.join(out, "tables.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(render_tables(summary))
    return summary


def cmd_tables(args):
    out = os.path.join(args.out, "results")
    summary = write_summary(out, read_runs(os.path.join(out, "runs.csv")))
    print(render_tables(summary))


TIMING_FIELDS = ("agent_seconds", "env_seconds", "run_seconds")
RUN_KEY = ("split", "kind", "method", "params", "maze_seed")


def compare_runs(path, reference):
    """Compare run records field by field, as written, ignoring the timing columns.

    Every run in `path` must have a run with the same key in `reference`, so a smoke run
    over the first few mazes is checked against the matching published runs."""
    def rows(p):
        with open(p, encoding="utf-8", newline="") as f:
            return {tuple(r[k] for k in RUN_KEY): r for r in csv.DictReader(f)}

    mine, ref = rows(path), rows(reference)
    missing = sorted(k for k in mine if k not in ref)
    differences = []
    for key in sorted(k for k in mine if k in ref):
        fields = [f for f in RUN_FIELDS if f not in TIMING_FIELDS and mine[key].get(f) != ref[key].get(f)]
        if fields:
            differences.append((key, fields))
    return {"compared": len(mine) - len(missing), "reference": len(ref), "missing": missing,
            "differences": differences}


def cmd_verify(args):
    path = os.path.join(args.out, "results", "runs.csv")
    reference = getattr(args, "reference", None) or os.path.join(OUT, "results", "runs.csv")
    if os.path.abspath(path) == os.path.abspath(reference):
        raise SystemExit("verify compares a rerun with the published records: pass the rerun's folder as --out")
    report = compare_runs(path, reference)
    print(f"{report['compared']} of {report['reference']} reference runs compared, timing columns "
          f"({', '.join(TIMING_FIELDS)}) excluded")
    for key in report["missing"]:
        print(f"  no reference run for {dict(zip(RUN_KEY, key))}")
    for key, fields in report["differences"][:20]:
        print(f"  {dict(zip(RUN_KEY, key))} differs in {', '.join(fields)}")
    if report["missing"] or report["differences"]:
        raise SystemExit(f"runs that differ: {len(report['differences'])}; "
                         f"runs without a reference run: {len(report['missing'])}")
    print("identical in every other field")
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # labels such as Q(λ) on a cp1252 console
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["tune", "final", "tables", "verify"])
    parser.add_argument("--out", default=OUT,
                        help="folder for frozen_settings.json, tuning/ and results/ (default: the published "
                             "docs/value-propagation)")
    parser.add_argument("--settings", default=None,
                        help="frozen settings for `final` (default: <out>/frozen_settings.json)")
    parser.add_argument("--reference", default=None,
                        help="run records `verify` compares with (default: the published results/runs.csv)")
    parser.add_argument("--overwrite", action="store_true",
                        help="let `tune` and `final` replace existing frozen settings or run records")
    parser.add_argument("--workers", type=int, default=max(1, min(20, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--mazes", type=int, default=0, help="use only the first N mazes of each set (smoke runs)")
    parser.add_argument("--long-count", type=int, default=LONG_ROUTES["count"],
                        help="size of the long-route set (tune)")
    parser.add_argument("--quick", action="store_true", help="a small grid, for smoke runs (tune)")
    args = parser.parse_args()
    {"tune": cmd_tune, "final": cmd_final, "tables": cmd_tables, "verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    main()
