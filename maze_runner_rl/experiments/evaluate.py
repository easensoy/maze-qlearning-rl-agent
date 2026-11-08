"""Evaluation suite for the tabular Q-learning agent.

Each question is answered over independently generated mazes, with 95% intervals:

1. On a fixed maze, how often does the learned (greedy) policy settle on a shortest
   route, and how much experience, counted in environment steps, does that take?
2. Where does that sit between a random walk, which never learns, and breadth-first
   search, a planner that is given the maze?
3. Which repairs matter? Each one is undone in turn (ablation).
4. Does a policy learned on one maze work on mazes it has never seen? It should not:
   the agent's state is only its position.
5. How close are the learned values and actions to the exact optimal ones?
6. Why do runs on long routes fail to converge? Two follow-up settings test two
   explanations: a goal reward that fades with distance, and an episode budget that
   runs out.

Every run gets its own maze (env.reset(seed=maze_seed)) and its own exploration seed
for Python's `random` module, which drives the agent's epsilon-greedy choices. All
settings use the same mazes and exploration seeds, so they can be compared run by run.

Run from maze_runner_rl/:
    python -m experiments.evaluate                       # full suite -> ../docs/evaluation/results
    python -m experiments.evaluate --quick --out <dir>   # small smoke run
"""
import argparse
import contextlib
import csv
import glob
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import yaml

from config.config import CONFIG
from experiments.stats import bootstrap_interval, paired_bootstrap_difference, wilson_interval
from experiments.training_experiment import (TrainingExperiment, convergence_episode, greedy_route_length,
                                             shortest_route_length)
from src.agents.q_learning_agent import QLearningAgent
from src.environments.maze_3d_env import Maze3DEnvironment
from src.environments.maze_env import MazeEnvironment

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.normpath(os.path.join(PACKAGE_ROOT, "..", "docs", "evaluation", "results"))

with open(os.path.join(PACKAGE_ROOT, "websocket", "config.yaml"), encoding="utf-8") as f:
    SETTINGS = yaml.safe_load(f)

KINDS = {
    "2d": "2D maze, 21x21",
    "3d_1": "3D env, 1 level, 21x21 (UI default)",
    "3d_3": "3D env, 3 levels, 21x21x3",
}

# Each setting undoes one repair, so its effect can be measured on its own.
CONFIGS = {
    "current": {"label": "Current agent", "rule": "current", "max_steps": 1000, "new_maze": False},
    "cap_200": {"label": "Step cap back to 200", "rule": "current", "max_steps": 200, "new_maze": False},
    "prefix_rule": {"label": "Pre-fix action rule", "rule": "prefix", "max_steps": 1000, "new_maze": False},
    "original": {"label": "Both undone (original agent)", "rule": "prefix", "max_steps": 200, "new_maze": False},
    "new_maze": {"label": "New maze every episode", "rule": "current", "max_steps": 1000, "new_maze": True},
    # Not repairs: two explanations for why runs on long routes do not converge.
    # 1. The goal fades: with discount 0.95 and a -0.1 step penalty, never arriving is worth
    #    -2 and the goal adds only 102 * 0.95^(d-1) to that at d moves away (0.05 at 150).
    #    Discount 0.99 keeps it visible from much further away.
    # 2. The budget runs out: one-step updates carry the goal's value back a little further
    #    each episode, so a longer route needs proportionally more episodes.
    "gamma_099": {"label": "Discount 0.99 instead of 0.95", "rule": "current", "max_steps": 1000, "new_maze": False,
                  "gamma": 0.99},
    "episodes_x3": {"label": "Three times the episodes", "rule": "current", "max_steps": 1000, "new_maze": False,
                    "episode_factor": 3},
}
REPAIRS = ("current", "cap_200", "prefix_rule", "original")
FOLLOW_UPS = ("gamma_099", "episodes_x3")

HELDOUT_SEED_BASE = 10_000          # unseen test mazes: seeds 10000, 10001, ...
EXPLORATION_SEED_BASE = 500_000     # exploration seed of the run on maze s: 500000 + s


def exploration_seed(maze_seed):
    return EXPLORATION_SEED_BASE + maze_seed


def make_suite(mazes=60, episodes=None, heldout=50, window=100):
    """Everything a worker needs to reproduce a run (passed explicitly, since worker
    processes do not see command-line overrides of module globals)."""
    episodes = episodes or SETTINGS["training"]["max_episodes"]
    max_total = episodes * max(c["max_steps"] for c in CONFIGS.values())
    return {
        "mazes": mazes,
        "episodes": episodes,
        "window": window,
        "maze_seeds": list(range(mazes)),
        "heldout_seeds": list(range(HELDOUT_SEED_BASE, HELDOUT_SEED_BASE + heldout)),
        "step_grid": [int(x) for x in np.unique(np.round(np.logspace(2, np.log10(max_total), 81)))],
    }


# --- tasks and agents -----------------------------------------------------------------

def make_env(kind, maze_seed):
    """The maze for one run; the generators draw only from their own seeded RNG."""
    maze = SETTINGS["maze"]
    if kind == "2d":
        env = MazeEnvironment()
    else:
        depth = {"3d_1": 1, "3d_3": 3}[kind]
        env = Maze3DEnvironment(maze["default_width"], maze["default_height"], depth,
                                wall_density=maze["wall_density"], branching_factor=maze["branching_factor"],
                                dead_end_percentage=maze["dead_end_percentage"])
    env.reset(seed=maze_seed)
    return env


class PreFixQLearningAgent(QLearningAgent):
    """The action rule from commit 3e99fbd, kept for the ablation. When exploiting, and
    when bootstrapping from the next state, only actions already tried in that state
    count; a state with no tried action gets a random action (exploiting) or a value of
    0 (bootstrapping). Everything else is the current agent."""

    def _tried(self, state):
        return self.q_table.get(state) or {}

    def greedy_action(self, state):
        tried = self._tried(state)
        return max(tried, key=tried.get) if tried else 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_space_size)
        tried = self._tried(state)
        if not tried:
            return random.randrange(self.action_space_size)
        return max(tried, key=tried.get)

    def update_q_table(self, state, action, reward, next_state, done=False):
        current_q = self.q_table[state][action]  # adds the action first, as the original did
        tried_next = self.q_table[next_state]
        max_next_q = max(tried_next.values()) if tried_next else 0.0
        target = reward + self.discount_factor * max_next_q
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)


class NewMazeEachEpisode:
    """The pre-fix environment behaviour: every reset builds a new maze."""

    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        return self.env.reset(options={"new_maze": True})

    def step(self, action):
        return self.env.step(action)


def make_agent(env, rule="current", gamma=None):
    a = SETTINGS["agent"]
    cls = PreFixQLearningAgent if rule == "prefix" else QLearningAgent
    return cls(env.num_states, env.action_space.n, learning_rate=a["learning_rate"], epsilon=a["epsilon"],
               epsilon_decay=a["epsilon_decay"], discount_factor=a["gamma"] if gamma is None else gamma,
               epsilon_min=a["epsilon_min"])


# --- exact analysis of a maze from its transition model ----------------------------

def endpoints(env):
    g = env.maze_generator
    return env.state_index(g.get_start()), env.state_index(g.get_end())


def transition_model(env):
    """Next state, reward and done for every (state, action), from env.step_from_state."""
    n, k = env.num_states, env.action_space.n
    nxt = np.empty((n, k), dtype=np.int64)
    rew = np.empty((n, k))
    done = np.empty((n, k), dtype=bool)
    for s in range(n):
        for a in range(k):
            nxt[s, a], rew[s, a], done[s, a] = env.step_from_state(s, a)
    return nxt, rew, done


def reachable_states(nxt, start):
    seen = {start}
    queue = deque([start])
    while queue:
        for t in nxt[queue.popleft()]:
            t = int(t)
            if t not in seen:
                seen.add(t)
                queue.append(t)
    return sorted(seen)


def loop_count(nxt, cells):
    """Independent loops in the corridor graph (edges - cells + 1); 0 means a single route."""
    cells = set(cells)
    edges = {frozenset((s, int(t))) for s in cells for t in nxt[s] if int(t) != s and int(t) in cells}
    return len(edges) - len(cells) + 1


def optimal_values(nxt, rew, done, gamma, tol=1e-12):
    """Exact optimal values V*(s) and Q*(s, a) by value iteration on the known model.
    Walls and the goal are absorbing with value 0."""
    v = np.zeros(nxt.shape[0])
    while True:
        q = rew + gamma * np.where(done, 0.0, v[nxt])
        v_new = q.max(axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, q
        v = v_new


def planner_route_length(nxt, q_star, start, goal, limit):
    """Route length when always taking an optimal action (value iteration as a planner)."""
    state = start
    for moves in range(1, limit + 1):
        state = int(nxt[state, int(np.argmax(q_star[state]))])
        if state == goal:
            return moves
    return None


def random_walk_stats(nxt, start, goal, cells, step_grid):
    """Exact hitting-time statistics of the uniformly random policy from the start cell
    (every action with probability 1/A), computed from the transition model, no sampling.

    Over the cells other than the goal the walk's transition matrix Q is symmetric: every
    move between two open cells has a reverse move, and a move into a wall stays put. So
    Q = V diag(lam) V^T, and the probability that the goal is still unreached after t steps
    is sum_i (V^T e_start)_i (V^T 1)_i lam_i^t. The expected number of steps solves
    (I - Q) h = 1."""
    actions = nxt.shape[1]
    transient = [s for s in cells if s != goal]
    index = {s: i for i, s in enumerate(transient)}
    q = np.zeros((len(transient), len(transient)))
    for s in transient:
        for a in range(actions):
            t = int(nxt[s, a])
            if t != goal:
                q[index[s], index[t]] += 1.0 / actions
    if not np.allclose(q, q.T):
        raise ValueError("random-walk matrix is not symmetric; the closed form does not apply")
    lam, vec = np.linalg.eigh(q)
    weights = vec[index[start]] * vec.sum(axis=0)

    def unreached(t):
        return float(np.clip(np.sum(weights * lam ** int(t)), 0.0, 1.0))

    high = 1
    while unreached(high) > 0.5:
        high *= 2
        if high > 2 ** 40:
            raise ValueError("the goal is not reachable from the start")
    low = high // 2
    while high - low > 1:
        mid = (low + high) // 2
        if unreached(mid) > 0.5:
            low = mid
        else:
            high = mid
    expected = np.linalg.solve(np.eye(len(transient)) - q, np.ones(len(transient)))[index[start]]
    return {
        "reach_200": 1 - unreached(200),
        "reach_1000": 1 - unreached(1000),
        "expected_steps": float(expected),
        "median_steps": high,
        "reached_by_step": [1 - unreached(x) for x in step_grid],
    }


def greedy_route_cells(nxt, agent, start, goal, limit):
    """Cells the greedy policy passes through on its way to the goal (goal excluded), or
    None if it never gets there."""
    state, route = start, [start]
    for _ in range(limit):
        state = int(nxt[state, agent.greedy_action(state)])
        if state == goal:
            return route
        if state in route:
            return None
        route.append(state)
    return None


def policy_analysis(agent, nxt, q_star, cells, start, goal):
    """Compares the learned values and greedy actions with the exact optimal ones."""
    transient = [s for s in cells if s != goal]
    best = q_star.max(axis=1)
    optimal = q_star >= best[:, None] - 1e-9
    greedy_ok = np.array([optimal[s, agent.greedy_action(s)] for s in transient])
    visited = np.array([bool(agent.q_table.get(s)) for s in transient])
    error = np.abs(np.array([max(agent.action_values(s)) for s in transient]) - best[transient])
    route = greedy_route_cells(nxt, agent, start, goal, len(cells))
    on_route = np.array([route is not None and s in route for s in transient])
    return {
        "visited_share": float(visited.mean()),
        "greedy_optimal_share": float(greedy_ok.mean()),
        "greedy_optimal_share_visited": float(greedy_ok[visited].mean()),
        "chance_optimal_share": float(np.mean([optimal[s].mean() for s in transient])),
        "value_error_route": float(error[on_route].mean()) if on_route.any() else None,
        "value_error_visited": float(error[visited].mean()),
    }


# --- work units ---------------------------------------------------------------------

_HELDOUT = {}


def heldout_mazes(kind, seeds):
    key = (kind, tuple(seeds))
    if key not in _HELDOUT:
        _HELDOUT[key] = [(env, shortest_route_length(env)) for env in (make_env(kind, s) for s in seeds)]
    return _HELDOUT[key]


def heldout_scores(kind, agent, seeds):
    """Share of unseen mazes on which the greedy policy reaches the goal, and takes a
    shortest route."""
    reached = shortest = 0
    for env, best in heldout_mazes(kind, seeds):
        route = greedy_route_length(env, agent, env.num_states)
        reached += route is not None
        shortest += route is not None and route == best
    return reached / len(seeds), shortest / len(seeds)


def sample_at(cum_steps, flags, step_grid):
    """A per-episode flag read at each point of an environment-step grid: the check made
    after the last episode that ended within that many steps (False before the first
    episode ends; a run that finished earlier keeps its final value)."""
    flags = np.asarray(flags, dtype=bool)
    index = np.searchsorted(cum_steps, step_grid, side="right") - 1
    return np.where(index >= 0, flags[np.clip(index, 0, None)], False)


def bits(flags):
    return "".join("1" if f else "0" for f in flags)


def maze_baseline(kind, maze_seed, suite):
    """Facts about one training maze, and the two references: the planner and the random walk."""
    env = make_env(kind, maze_seed)
    nxt, rew, done = transition_model(env)
    start, goal = endpoints(env)
    cells = reachable_states(nxt, start)
    _, q_star = optimal_values(nxt, rew, done, SETTINGS["agent"]["gamma"])
    walk = random_walk_stats(nxt, start, goal, cells, suite["step_grid"])
    record = {
        "kind": kind,
        "maze_seed": maze_seed,
        "reachable_cells": len(cells),
        "loops": loop_count(nxt, cells),
        "shortest_route": shortest_route_length(env),
        "planner_route": planner_route_length(nxt, q_star, start, goal, len(cells)),
        "random_walk_reach_200": walk["reach_200"],
        "random_walk_reach_1000": walk["reach_1000"],
        "random_walk_expected_steps": walk["expected_steps"],
        "random_walk_median_steps": walk["median_steps"],
    }
    return record, walk["reached_by_step"]


def train_run(kind, config, maze_seed, suite):
    """One training run. Returns its record and, on a fixed maze, its curves."""
    settings = CONFIGS[config]
    env = make_env(kind, maze_seed)
    seed = exploration_seed(maze_seed)
    random.seed(seed)
    agent = make_agent(env, settings["rule"], settings.get("gamma"))
    new_maze = settings["new_maze"]
    results = TrainingExperiment(NewMazeEachEpisode(env) if new_maze else env, agent).run(
        num_episodes=suite["episodes"] * settings.get("episode_factor", 1), max_steps=settings["max_steps"], log_every=0,
        evaluate_greedy=not new_maze, greedy_limit=env.num_states)

    window = suite["window"]
    cum = np.cumsum(results["steps"])
    successes = results["successes"]
    first = next((i for i, s in enumerate(successes) if s), None)
    record = {
        "kind": kind,
        "config": config,
        "maze_seed": maze_seed,
        "exploration_seed": seed,
        "max_steps": settings["max_steps"],
        "episodes": len(successes),
        "env_steps": int(cum[-1]),
        "training_success_last100": float(np.mean(successes[-window:])),
        "first_success_episode": None if first is None else first + 1,
        "first_success_env_steps": None if first is None else int(cum[first]),
    }
    curves = None
    if not new_maze:
        shortest = results["shortest_route"]
        greedy = results["greedy_steps"]
        reaches = np.array([g is not None for g in greedy])
        is_shortest = np.array([g is not None and g == shortest for g in greedy])
        converged = convergence_episode(list(is_shortest), window)
        first_shortest = next((i + 1 for i, s in enumerate(is_shortest) if s), None)
        record.update({
            "shortest_route": shortest,
            "final_greedy_route": greedy[-1],
            "greedy_reaches_final": bool(reaches[-1]),
            "greedy_shortest_final": bool(is_shortest[-1]),
            "greedy_reaches_last100": float(reaches[-window:].mean()),
            "greedy_shortest_last100": float(is_shortest[-window:].mean()),
            "first_greedy_shortest_episode": first_shortest,
            "converged_episode": converged,
            "converged_env_steps": None if converged is None else int(cum[converged - 1]),
        })
        curves = {
            "kind": kind, "config": config, "maze_seed": maze_seed,
            "shortest": bits(sample_at(cum, is_shortest, suite["step_grid"])),
            "reaches": bits(sample_at(cum, reaches, suite["step_grid"])),
        }
        if config == "current":
            nxt, rew, done = transition_model(env)
            start, goal = endpoints(env)
            _, q_star = optimal_values(nxt, rew, done, agent.discount_factor)
            record.update(policy_analysis(agent, nxt, q_star, reachable_states(nxt, start), start, goal))
    reached, shortest_share = heldout_scores(kind, agent, suite["heldout_seeds"])
    record["heldout_greedy_reaches"] = reached
    record["heldout_greedy_shortest"] = shortest_share
    return record, curves


def work(task):
    """Entry point for worker processes. The 3D generator reports every maze it builds on
    stdout; that is silenced here, since the suite builds thousands."""
    with contextlib.redirect_stdout(io.StringIO()):
        if task[0] == "baseline":
            _, kind, maze_seed, suite = task
            return ("baseline",) + maze_baseline(kind, maze_seed, suite)
        _, kind, config, maze_seed, suite = task
        return ("run",) + train_run(kind, config, maze_seed, suite)


def run_suite(suite, kinds=tuple(KINDS), configs=tuple(CONFIGS), workers=1, progress=False):
    """All baselines and training runs; results sorted so the output does not depend on
    which worker finished first."""
    tasks = [("baseline", k, s, suite) for k in kinds for s in suite["maze_seeds"]]
    # The slowest settings first, so the pool stays busy until the end.
    order = {"prefix_rule": 0, "new_maze": 1, "episodes_x3": 2, "original": 3, "cap_200": 4, "gamma_099": 5, "current": 6}
    tasks += [("run", k, c, s, suite) for c in sorted(configs, key=order.get) for k in kinds
              for s in suite["maze_seeds"]]
    outputs = []
    if workers <= 1:
        outputs = [work(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, t) for t in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                outputs.append(future.result())
                if progress and (done % 50 == 0 or done == len(futures)):
                    print(f"  {done}/{len(futures)} done", flush=True)
    kind_rank = {k: i for i, k in enumerate(KINDS)}
    config_rank = {c: i for i, c in enumerate(CONFIGS)}
    baselines = sorted((o for o in outputs if o[0] == "baseline"),
                       key=lambda o: (kind_rank[o[1]["kind"]], o[1]["maze_seed"]))
    runs = sorted((o for o in outputs if o[0] == "run"),
                  key=lambda o: (kind_rank[o[1]["kind"]], config_rank[o[1]["config"]], o[1]["maze_seed"]))
    return {
        "baselines": [o[1] for o in baselines],
        "walk_curves": {(o[1]["kind"], o[1]["maze_seed"]): o[2] for o in baselines},
        "runs": [o[1] for o in runs],
        "curves": [o[2] for o in runs if o[2] is not None],
    }


# --- aggregation --------------------------------------------------------------------

INF = float("inf")


def _proportion(flags):
    k, n = int(sum(flags)), len(flags)
    low, high = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci": [low, high]}


def _median(values, seed=0):
    values = [INF if v is None else v for v in values]
    low, high = bootstrap_interval(values, np.median, seed=seed)
    return {"median": float(np.median(values)), "ci": [low, high], "reached": sum(v != INF for v in values)}


def _mean(values, seed=0):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "ci": [None, None]}
    low, high = bootstrap_interval(values, np.mean, seed=seed)
    return {"mean": float(np.mean(values)), "ci": [low, high]}


def summarize_group(runs):
    out = {
        "runs": len(runs),
        "training_success_last100": _mean([r["training_success_last100"] for r in runs]),
        "first_success_env_steps": _median([r["first_success_env_steps"] for r in runs]),
        "heldout_greedy_reaches": _mean([r["heldout_greedy_reaches"] for r in runs]),
        "heldout_greedy_shortest": _mean([r["heldout_greedy_shortest"] for r in runs]),
    }
    if "converged_episode" in runs[0]:
        out.update({
            "converged": _proportion([r["converged_episode"] is not None for r in runs]),
            "greedy_shortest_final": _proportion([r["greedy_shortest_final"] for r in runs]),
            "greedy_reaches_final": _proportion([r["greedy_reaches_final"] for r in runs]),
            "greedy_shortest_last100": _mean([r["greedy_shortest_last100"] for r in runs]),
            "converged_env_steps": _median([r["converged_env_steps"] for r in runs]),
            "converged_episode": _median([r["converged_episode"] for r in runs]),
            "env_steps": _median([r["env_steps"] for r in runs]),
        })
    if "greedy_optimal_share" in runs[0]:
        for key in ("visited_share", "greedy_optimal_share", "greedy_optimal_share_visited",
                    "chance_optimal_share", "value_error_route", "value_error_visited"):
            out[key] = _mean([r[key] for r in runs])
    return out


def summarize_baselines(rows):
    shortest = [r["shortest_route"] for r in rows]
    expected = [r["random_walk_expected_steps"] for r in rows]
    return {
        "mazes": len(rows),
        "shortest_route": {"median": float(np.median(shortest)), "min": min(shortest), "max": max(shortest)},
        "planner_matches_bfs": sum(r["planner_route"] == r["shortest_route"] for r in rows),
        "reachable_cells": float(np.median([r["reachable_cells"] for r in rows])),
        "loops": float(np.median([r["loops"] for r in rows])),
        "random_walk_reach_200": _mean([r["random_walk_reach_200"] for r in rows]),
        "random_walk_reach_1000": _mean([r["random_walk_reach_1000"] for r in rows]),
        "random_walk_expected_steps": {"median": float(np.median(expected))},
        "random_walk_expected_over_shortest": {"median": float(np.median(np.divide(expected, shortest)))},
        "random_walk_median_steps": {"median": float(np.median([r["random_walk_median_steps"] for r in rows]))},
    }


def aggregate(result, kinds, configs):
    groups = {}
    for kind in kinds:
        for config in configs:
            runs = [r for r in result["runs"] if r["kind"] == kind and r["config"] == config]
            if runs:
                groups.setdefault(kind, {})[config] = summarize_group(runs)
        if "current" in configs:
            current = {r["maze_seed"]: r for r in result["runs"] if r["kind"] == kind and r["config"] == "current"}
            for config in configs:
                if config in ("current", "new_maze") or config not in groups.get(kind, {}):
                    continue
                other = {r["maze_seed"]: r for r in result["runs"] if r["kind"] == kind and r["config"] == config}
                seeds = sorted(set(current) & set(other))
                diff, low, high = paired_bootstrap_difference(
                    [current[s]["converged_episode"] is not None for s in seeds],
                    [other[s]["converged_episode"] is not None for s in seeds])
                groups[kind][config]["converged_change_vs_current"] = {"diff": diff, "ci": [low, high]}
    baselines = {kind: summarize_baselines([b for b in result["baselines"] if b["kind"] == kind]) for kind in kinds}
    return {"groups": groups, "baselines": baselines, "route_law": route_law(result, kinds)}


def route_law(result, kinds, config="current"):
    """Least-squares line through (shortest route, first episode with a shortest greedy
    route), over the runs where that happened: how learning time grows with route length."""
    law = {}
    for kind in kinds:
        points = [(r["shortest_route"], r["first_greedy_shortest_episode"]) for r in result["runs"]
                  if r["kind"] == kind and r["config"] == config and r.get("first_greedy_shortest_episode")]
        if len(points) >= 3:
            x, y = np.array(points, dtype=float).T
            slope, intercept = np.polyfit(x, y, 1)
            law[kind] = {"slope": float(slope), "intercept": float(intercept),
                         "r": float(np.corrcoef(x, y)[0, 1]), "runs": len(points)}
    return law


# --- output -------------------------------------------------------------------------

def git(*command):
    try:
        return subprocess.run(["git", *command], cwd=PACKAGE_ROOT, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


SOURCE_PATTERNS = ("src/**/*.py", "experiments/evaluate.py", "experiments/stats.py",
                   "experiments/training_experiment.py", "config/config.py", "config/config.yaml", "websocket/config.yaml")


def source_paths():
    return sorted({p for pattern in SOURCE_PATTERNS
                   for p in glob.glob(os.path.join(PACKAGE_ROOT, pattern), recursive=True)})


def source_digest():
    """SHA-256 over the code and settings the results depend on (line endings normalised),
    so a result can be matched to its code even when the tree had uncommitted changes."""
    paths = source_paths()
    digest = hashlib.sha256()
    for path in paths:
        digest.update(os.path.relpath(path, PACKAGE_ROOT).replace(os.sep, "/").encode())
        with open(path, "rb") as f:
            digest.update(f.read().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def code_snapshot():
    """Which code is about to run: taken before the suite starts, so edits made while it
    runs cannot be attributed to its results. Only changes to the code and settings the
    results depend on count as uncommitted; edited docs do not."""
    status = git("status", "--porcelain", "--untracked-files=all", "--",
                 *(os.path.relpath(p, PACKAGE_ROOT) for p in source_paths()))
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_uncommitted_source_changes": None if status is None else bool(status),
        "source_sha256": source_digest(),
    }


def provenance(suite, kinds, configs, workers, seconds, snapshot=None):
    import gymnasium
    return {
        **(snapshot or code_snapshot()),
        "command": " ".join(["python -m experiments.evaluate", *sys.argv[1:]]),
        "suite": {k: v for k, v in suite.items() if k not in ("maze_seeds", "heldout_seeds", "step_grid")},
        "maze_seeds": f"{suite['maze_seeds'][0]}-{suite['maze_seeds'][-1]}",
        "heldout_seeds": f"{suite['heldout_seeds'][0]}-{suite['heldout_seeds'][-1]}",
        "exploration_seed": f"{EXPLORATION_SEED_BASE} + maze seed",
        "kinds": list(kinds),
        "configs": {c: CONFIGS[c] for c in configs},
        "settings": {"agent": SETTINGS["agent"], "maze": SETTINGS["maze"]},
        "rewards": {"goal": CONFIG.rewards.goal_reward, "step": CONFIG.penalties.step_penalty,
                    "wall": CONFIG.penalties.wall_penalty},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "platform": platform.platform(),
        "workers": workers,
        "wall_seconds": round(seconds, 1),
    }


RUN_FIELDS = ["kind", "config", "maze_seed", "exploration_seed", "max_steps", "episodes", "env_steps",
              "shortest_route", "final_greedy_route", "greedy_reaches_final", "greedy_shortest_final",
              "greedy_reaches_last100", "greedy_shortest_last100", "first_greedy_shortest_episode",
              "converged_episode", "converged_env_steps", "training_success_last100", "first_success_episode",
              "first_success_env_steps", "heldout_greedy_reaches", "heldout_greedy_shortest",
              "visited_share", "greedy_optimal_share", "greedy_optimal_share_visited", "chance_optimal_share",
              "value_error_route", "value_error_visited"]
BASELINE_FIELDS = ["kind", "maze_seed", "reachable_cells", "loops", "shortest_route", "planner_route",
                   "random_walk_reach_200", "random_walk_reach_1000", "random_walk_expected_steps",
                   "random_walk_median_steps"]


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})


def _finite(value):
    """inf (an event not reached within the budget) becomes null, which JSON can hold."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return None if isinstance(value, float) and value == INF else value


def write_outputs(out, suite, result, summary, info):
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "runs.csv"), RUN_FIELDS, result["runs"])
    write_csv(os.path.join(out, "baselines.csv"), BASELINE_FIELDS, result["baselines"])
    walk = {}
    for kind in summary["baselines"]:
        rows = [c for (k, _), c in sorted(result["walk_curves"].items()) if k == kind]
        walk[kind] = [round(float(v), 6) for v in np.mean(rows, axis=0)]
    lines = [json.dumps(c, separators=(",", ":")) for c in result["curves"]]
    with open(os.path.join(out, "curves.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write('{"step_grid":' + json.dumps(suite["step_grid"], separators=(",", ":")) + ",\n")
        f.write('"random_walk_reached_mean":' + json.dumps(walk, separators=(",", ":")) + ",\n")
        f.write('"runs":[\n' + ",\n".join(lines) + "\n]}\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(_finite({"provenance": info, **summary}), f, indent=2)
        f.write("\n")
    with open(os.path.join(out, "tables.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(render_tables(summary, result, suite))


# --- Markdown tables ------------------------------------------------------------------

def fmt_steps(x):
    if x is None or x == INF:
        return "not reached"
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e4:
        return f"{x / 1e3:.0f}k"
    if x >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


def fmt_number(x):
    return "n/a" if x is None else f"{x:.2f}"


def _pct(x):
    x *= 100
    if 0 < x < 0.1:
        return "<0.1"
    if 0 < x < 9.95 and round(x) != x:
        return f"{x:.1f}"
    return f"{x:.0f}"


def fmt_pct(rate, ci=None):
    text = f"{_pct(rate)}%"
    if ci and ci[0] is not None:
        text += f" ({_pct(ci[0])}–{_pct(ci[1])})"
    return text


def fmt_median(m, as_steps=True):
    f = fmt_steps if as_steps else (lambda v: "not reached" if v == INF else f"{v:.0f}")
    if m["median"] == INF:
        return f"not reached ({m['reached']} reached)"
    return f"{f(m['median'])} ({f(m['ci'][0])}–{f(m['ci'][1])})"


def render_tables(summary, result, suite):
    groups, baselines = summary["groups"], summary["baselines"]
    kinds = [k for k in KINDS if k in groups]
    out = [f"<!-- generated by experiments/evaluate.py; {suite['mazes']} mazes per row, "
           f"{suite['episodes']} episodes per run, 95% intervals in brackets -->", ""]

    if all("current" in groups[k] for k in kinds):
        out += ["### Current agent on a fixed maze", "",
                "| Maze | Shortest route, moves (median, range) | Greedy route converged | Steps of experience to converge "
                "(median) | Episodes to converge (median) | Greedy route shortest after the last episode | "
                "Training episodes reaching the goal, last 100 |",
                "|---|---|---|---|---|---|---|"]
        for k in kinds:
            g, b = groups[k]["current"], baselines[k]
            sr = b["shortest_route"]
            out.append(f"| {KINDS[k]} | {sr['median']:.0f} ({sr['min']}–{sr['max']}) | "
                       f"{g['converged']['k']}/{g['converged']['n']}, {fmt_pct(g['converged']['rate'], g['converged']['ci'])} | "
                       f"{fmt_median(g['converged_env_steps'])} | {fmt_median(g['converged_episode'], False)} | "
                       f"{fmt_pct(g['greedy_shortest_final']['rate'], g['greedy_shortest_final']['ci'])} | "
                       f"{fmt_pct(g['training_success_last100']['mean'], g['training_success_last100']['ci'])} |")
        out.append("")

        out += ["### References: a random walk and a planner", "",
                "| Maze | Planner given the maze (BFS, value iteration) | Random walk reaches the goal within 1000 steps | "
                "Random walk, expected steps to the goal (median maze) | Q-learning, steps until it first reaches the goal "
                "(median) | Q-learning, steps until its greedy route converges (median) |",
                "|---|---|---|---|---|---|"]
        for k in kinds:
            g, b = groups[k]["current"], baselines[k]
            ratio = b["random_walk_expected_over_shortest"]["median"]
            out.append(f"| {KINDS[k]} | shortest route in {b['planner_matches_bfs']}/{b['mazes']} mazes, no experience | "
                       f"{fmt_pct(b['random_walk_reach_1000']['mean'], b['random_walk_reach_1000']['ci'])} | "
                       f"{fmt_steps(b['random_walk_expected_steps']['median'])} ({ratio:.0f}x the shortest route) | "
                       f"{fmt_median(g['first_success_env_steps'])} | {fmt_median(g['converged_env_steps'])} |")
        out.append("")

    def converged_cell(g):
        text = f"{g['converged']['k']}/{g['converged']['n']} converged"
        change = g.get("converged_change_vs_current")
        if change:
            text += f", {change['diff'] * 100:+.0f} pts ({change['ci'][0] * 100:+.0f} to {change['ci'][1] * 100:+.0f})"
        return text + f"; median {fmt_median(g['converged_env_steps'])} steps"

    ablations = [c for c in REPAIRS if all(c in groups[k] for k in kinds)]
    if ablations:
        out += ["### Ablation: each repair undone on its own (fixed maze)", "",
                "| Setting | " + " | ".join(KINDS[k] for k in kinds) + " |",
                "|---|" + "---|" * len(kinds)]
        for c in ablations:
            label = CONFIGS[c]["label"] + f" ({CONFIGS[c]['max_steps']}-step cap)"
            out.append(f"| {label} | " + " | ".join(converged_cell(groups[k][c]) for k in kinds) + " |")
        out.append("")

    follow_ups = [c for c in FOLLOW_UPS if all(c in groups[k] for k in kinds)]
    if follow_ups and all("current" in groups[k] for k in kinds):
        bins = ((0, 80, "up to 80 moves"), (81, 120, "81 to 120 moves"), (121, 10_000, "over 120 moves"))
        columns = ["current", *follow_ups]
        names = {"current": "Current agent", "gamma_099": "Discount 0.99",
                 "episodes_x3": f"{3 * suite['episodes']} episodes"}
        out += ["### Follow-ups: why long routes do not converge", "",
                "| Maze | Shortest route | Runs | " + " | ".join(f"Converged: {names[c]}" for c in columns) + " |",
                "|---|---|---|" + "---|" * len(columns)]
        for k in kinds:
            for low, high, name in bins:
                rows = {c: [r for r in result["runs"] if r["kind"] == k and r["config"] == c
                            and low <= r["shortest_route"] <= high] for c in columns}
                if rows["current"]:
                    counts = [str(sum(r["converged_episode"] is not None for r in rows[c])) for c in columns]
                    out.append(f"| {KINDS[k]} | {name} | {len(rows['current'])} | " + " | ".join(counts) + " |")
            out.append(f"| {KINDS[k]} | **all** | {groups[k]['current']['converged']['n']} | "
                       + " | ".join(converged_cell(groups[k][c]) for c in columns) + " |")
        # All maze types pooled, each follow-up paired with the current agent run by run.
        key = (lambda r: (list(KINDS).index(r["kind"]), r["maze_seed"]))
        pooled = {c: [r["converged_episode"] is not None
                      for r in sorted((r for r in result["runs"] if r["config"] == c and r["kind"] in kinds), key=key)]
                  for c in columns}
        cells = [f"{sum(pooled['current'])}/{len(pooled['current'])}"]
        for c in follow_ups:
            diff, low, high = paired_bootstrap_difference(pooled["current"], pooled[c])
            cells.append(f"{sum(pooled[c])}/{len(pooled[c])}, {diff * 100:+.0f} pts ({low * 100:+.0f} to {high * 100:+.0f})")
        out.append(f"| All maze types | **all** | {len(pooled['current'])} | " + " | ".join(cells) + " |")
        out.append("")
        law = summary.get("route_law", {})
        if law:
            out += ["Episode at which the greedy route first became a shortest route, against the shortest route's "
                    "length (current agent, least-squares line):", ""]
            out += [f"- {KINDS[k]}: {v['slope']:.1f} episodes per move {v['intercept']:+.0f}, correlation "
                    f"{v['r']:.2f}, over the {v['runs']} runs where it happened" for k, v in law.items()]
            out.append("")

    if all("current" in groups[k] for k in kinds):
        out += ["### Unseen mazes", "",
                "| Maze | Trained on one maze: greedy route reaches the goal on unseen mazes | "
                "Trained on a new maze every episode: training episodes reaching the goal, last 100 | "
                "Trained on a new maze every episode: greedy route reaches the goal on unseen mazes |",
                "|---|---|---|---|"]
        for k in kinds:
            g = groups[k]["current"]
            n = groups[k].get("new_maze")
            new_train = fmt_pct(n["training_success_last100"]["mean"], n["training_success_last100"]["ci"]) if n else "n/a"
            new_held = fmt_pct(n["heldout_greedy_reaches"]["mean"], n["heldout_greedy_reaches"]["ci"]) if n else "n/a"
            out.append(f"| {KINDS[k]} | {fmt_pct(g['heldout_greedy_reaches']['mean'], g['heldout_greedy_reaches']['ci'])} | "
                       f"{new_train} | {new_held} |")
        out.append("")

        out += ["### What the converged policy knows", "",
                "| Maze | Cells visited in training | Greedy action optimal, all reachable cells | "
                "Greedy action optimal, visited cells | A random action would be optimal | "
                "Value error on the greedy route (goal reward 100) |",
                "|---|---|---|---|---|---|"]
        for k in kinds:
            g = groups[k]["current"]
            out.append(f"| {KINDS[k]} | {fmt_pct(g['visited_share']['mean'])} | "
                       f"{fmt_pct(g['greedy_optimal_share']['mean'], g['greedy_optimal_share']['ci'])} | "
                       f"{fmt_pct(g['greedy_optimal_share_visited']['mean'])} | "
                       f"{fmt_pct(g['chance_optimal_share']['mean'])} | "
                       f"{fmt_number(g['value_error_route']['mean'])} |")
        out.append("")

        base = {(b["kind"], b["maze_seed"]): b for b in result["baselines"]}
        failed = [r for r in result["runs"] if r["config"] == "current" and r["converged_episode"] is None]
        out += ["### Runs that did not converge (current agent)", ""]
        if not failed:
            out.append("None.")
        else:
            out += ["| Maze | Maze seed | Shortest route | Loops in the maze | Final greedy route | "
                    "Greedy route shortest, last 100 checks | First episode with a shortest greedy route |",
                    "|---|---|---|---|---|---|---|"]
            for r in sorted(failed, key=lambda r: (list(KINDS).index(r["kind"]), r["maze_seed"])):
                route = r["final_greedy_route"]
                out.append(f"| {KINDS[r['kind']]} | {r['maze_seed']} | {r['shortest_route']} | "
                           f"{base[(r['kind'], r['maze_seed'])]['loops']} | "
                           f"{'does not reach the goal' if route is None else route} | "
                           f"{r['greedy_shortest_last100'] * 100:.0f}% | {r['first_greedy_shortest_episode'] or 'never'} |")
        out.append("")
    return "\n".join(out)


# --- command line ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the tabular Q-learning agent; see the module docstring.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output folder (default: docs/evaluation/results)")
    parser.add_argument("--mazes", type=int, default=60, help="training mazes per maze type and setting")
    parser.add_argument("--episodes", type=int, default=SETTINGS["training"]["max_episodes"])
    parser.add_argument("--heldout", type=int, default=50, help="unseen mazes each trained agent is tested on")
    parser.add_argument("--window", type=int, default=100, help="checks the greedy route must stay shortest")
    parser.add_argument("--kinds", nargs="+", choices=list(KINDS), default=list(KINDS))
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--quick", action="store_true", help="3 mazes, 60 episodes, 5 unseen mazes: a smoke run")
    args = parser.parse_args()
    if args.quick:
        args.mazes, args.episodes, args.heldout, args.window = 3, 60, 5, 20
    sys.stdout.reconfigure(errors="replace")   # a console that cannot show a character gets "?"

    suite = make_suite(args.mazes, args.episodes, args.heldout, args.window)
    runs = len(args.kinds) * len(args.configs) * args.mazes
    print(f"{runs} training runs and {len(args.kinds) * args.mazes} maze baselines on {args.workers} workers")
    snapshot = code_snapshot()
    started = time.perf_counter()
    # Import by module name so worker processes can find the work function.
    from experiments import evaluate as module
    result = module.run_suite(suite, args.kinds, args.configs, args.workers, progress=True)
    summary = aggregate(result, args.kinds, args.configs)
    info = provenance(suite, args.kinds, args.configs, args.workers, time.perf_counter() - started, snapshot)
    write_outputs(args.out, suite, result, summary, info)
    print(render_tables(summary, result, suite))
    print(f"Wrote {args.out} in {info['wall_seconds']:.0f} s")


if __name__ == "__main__":
    main()
