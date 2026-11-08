"""Records the runs the README animation shows, from the value-propagation experiment.

For one three-level maze, Q-learning and Dyna-Q are trained again exactly as
`compare_methods.train` trains them (same maze, same exploration seed, same random draws),
recording what the animation needs:
- the cells visited in episode 1;
- after the first episode that ends at or beyond each real-move budget: the greedy walk
  from the start and the cells from which the greedy policy reaches the goal;
- the final greedy route.
Each run's summary is checked against the experiment's results/runs.csv, so the animation
shows the same runs the tables count.

The maze is chosen by rule (`pick_maze`), from the final results.

Run from maze_runner_rl/:
    python -m experiments.capture_hero
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import random

import numpy as np

from experiments import asset_style as st
from experiments import capture_traces as ct
from experiments import compare_methods as cm
from experiments import evaluate as ev
from experiments.training_experiment import convergence_episode

RESULTS = os.path.join(cm.OUT, "results")
BUDGETS = (10_000, 20_000, 40_000, 80_000, 160_000)
METHODS = ("q_learning", "dyna_q")
CHECKED = ("env_steps", "first_success_env_steps", "first_greedy_shortest_env_steps", "converged_episode",
           "converged_env_steps", "final_greedy_route", "greedy_shortest_final")


def pick_maze(runs):
    """Among three-level mazes of the final suite where Dyna-Q converged and Q-learning did not,
    the one whose shortest route is the median of those mazes (ties: lower seed). The
    animation then shows the typical case of the difference the tables report."""
    by = {(r["method"], r["maze_seed"]): r for r in runs if r["split"] == "suite" and r["kind"] == "3d_3"}
    seeds = sorted(s for (m, s), r in by.items() if m == "dyna_q" and r["converged_episode"] is not None
                   and by.get(("q_learning", s), {}).get("converged_episode", 0) is None)
    if not seeds:
        return None
    ordered = sorted(seeds, key=lambda s: (by[("dyna_q", s)]["shortest_route"], s))
    return ordered[(len(ordered) - 1) // 2], len(seeds)


def trace_run(kind, method, params, seed, budgets=BUDGETS, episodes=cm.EPISODES, window=cm.WINDOW):
    """compare_methods.train's loop, with recording added (no extra random draws)."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = ev.make_env(kind, seed)
    facts = cm.maze_facts(env)
    random.seed(ev.exploration_seed(seed))
    agent = cm.make_agent(method, params, env, seed)
    begin = getattr(agent, "begin_episode", None)
    end = getattr(agent, "end_episode", None) or agent.decay_epsilon
    nxt, start, goal = facts["nxt"], facts["start"], facts["goal"]
    pending = list(budgets)
    steps, shortest, successes, checkpoints, first_path = [], [], [], [], None
    for episode in range(1, episodes + 1):
        _, info = env.reset()
        if begin:
            begin()
        state, terminated, n = info["state"], False, 0
        path = [state] if episode == 1 else None
        for _ in range(cm.MAX_STEPS):
            action = agent.choose_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            agent.update_q_table(state, action, reward, info["state"], terminated)
            state = info["state"]
            n += 1
            if path is not None:
                path.append(state)
            if terminated or truncated:
                break
        end()
        if path is not None:
            first_path = path
        steps.append(n)
        successes.append(bool(terminated))
        g = cm.greedy_walk(agent, nxt, start, goal, env.num_states)
        shortest.append(g is not None and g == facts["route"])
        total = sum(steps)
        while pending and total >= pending[0]:
            budget = pending.pop(0)
            walk, reaches = ct.greedy_walk(env, agent, start, goal)
            solved = ct.solved_region(agent, nxt, facts["cells"], goal)
            checkpoints.append({"budget": budget, "episode": episode, "env_steps": total,
                                "greedy_path": [int(s) for s in walk], "greedy_reaches": reaches,
                                "greedy_shortest": shortest[-1],
                                "learned": sorted(int(s) for s in solved),
                                "learned_depth": cm.learned_depth(agent, facts)})
    cum = np.cumsum(steps)
    converged = convergence_episode(shortest, window)
    first_short = cm.first_index(shortest)
    first_success = cm.first_index(successes)
    final = cm.greedy_walk(agent, nxt, start, goal, env.num_states)
    record = {
        "env_steps": int(cum[-1]),
        "first_success_env_steps": None if first_success is None else int(cum[first_success]),
        "first_greedy_shortest_env_steps": None if first_short is None else int(cum[first_short]),
        "converged_episode": converged,
        "converged_env_steps": None if converged is None else int(cum[converged - 1]),
        "final_greedy_route": final,
        "greedy_shortest_final": final is not None and final == facts["route"],
    }
    final_route, final_reaches = ct.greedy_walk(env, agent, start, goal)
    return {"method": method, "label": cm.METHODS[method]["label"], "params": params, "record": record,
            "episode1": [int(s) for s in first_path], "checkpoints": checkpoints,
            "final_route": [int(s) for s in final_route], "final_reaches": final_reaches}, env, facts


def capture(out=st.DATA, results=RESULTS):
    runs = cm.read_runs(os.path.join(results, "runs.csv"))
    with open(os.path.join(cm.OUT, "frozen_settings.json"), encoding="utf-8") as f:
        frozen = json.load(f)
    seed, candidates = pick_maze(runs)
    traced, env, facts = {}, None, None
    for method in METHODS:
        params = frozen["methods"][method]["params"]
        traced[method], env, facts = trace_run("3d_3", method, params, seed)
        expected = next(r for r in runs if r["split"] == "suite" and r["kind"] == "3d_3"
                        and r["method"] == method and r["maze_seed"] == seed)
        for field in CHECKED:
            got = traced[method]["record"][field]
            if got != expected[field]:
                raise SystemExit(f"{method} on maze {seed}: {field} is {got}, results/runs.csv has {expected[field]}")
    grid = env.maze
    levels = ["\n".join("".join("." if v == 0 else "#" for v in row) for row in level) for level in grid]
    bfs = ct.shortest_route_cells(facts["nxt"], facts["start"], facts["goal"])
    data = {
        "kind": "3d_3", "maze_seed": seed, "rule": " ".join(pick_maze.__doc__.split("\n\n")[0].split()),
        "candidates": candidates, "size": [env.maze_generator.width, env.maze_generator.height,
                                           env.maze_generator.depth],
        "levels": levels, "start": list(ct.xyz(env, facts["start"])), "goal": list(ct.xyz(env, facts["goal"])),
        "start_state": facts["start"], "goal_state": facts["goal"], "shortest_route": facts["route"],
        "bfs_route": bfs, "budgets": list(BUDGETS),
        "runs": traced, "checked_against": os.path.relpath(os.path.join(results, "runs.csv"),
                                                          os.path.dirname(ev.PACKAGE_ROOT)).replace(os.sep, "/"),
        "checked_fields": list(CHECKED),
    }
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "hero.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
    print(f"maze {seed} (route {facts['route']}, {candidates} candidates) -> {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=st.DATA)
    args = parser.parse_args()
    capture(args.out)


if __name__ == "__main__":
    main()
