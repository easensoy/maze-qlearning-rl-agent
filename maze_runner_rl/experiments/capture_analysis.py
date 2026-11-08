"""Checkpointed replays of published runs, for the README's visual analyses.

Each run is trained again exactly as `compare_methods.train` trains it: same maze, same
exploration seed, same random draws. Snapshots and greedy checks are read-only and draw no
random numbers. At the end of training, each replay's summary is compared with the published
record (results/runs.csv or tuning/runs.csv), and any mismatch stops the capture.

The cases are fixed in CASES, with the maze chosen by rule:

- comparison: three-level suite maze 54, the README animation's maze (`capture_hero.pick_maze`:
  the median-route maze among the 34 three-level suite mazes where Dyna-Q converged and Q-learning
  did not). All three methods run with their frozen settings. Snapshots are taken at the first
  episode end at or past 2.5k, 5k, 10k, 20k, 40k, 80k and 160k real environment steps.
- traces: development 2D maze 1004, the failure figure's maze (`plot_value_propagation.failure`:
  the 2D development maze with the median shortest route). Q-learning runs alongside Watkins's
  Q(λ) with λ = 0.9 and accumulating traces. Snapshots are taken after the 25th, 50th, 100th,
  200th and 1,000th real step, and at the first episode end at or past 5k and 50k steps.
- dyna: three-level suite maze 58, the only final maze on which Dyna-Q did not converge. Dyna-Q
  and Q-learning run, with snapshots at the end of episodes 60, 74, 85, 200 and 1,000. These
  bracket the first greedy route to the goal (episode 74) and the start of the final 114-move
  streak (episode 85).

Each snapshot records:
- the episode and the cumulative real environment steps;
- the cumulative Q-value updates: real, trace and planning (Q-learning: one per step);
- for every reachable state: V(s) = max_a Q(s, a) with untried actions at 0, the greedy action
  (ties go to the lowest index, as in the frozen greedy evaluation) and a bitmask of tried
  actions;
- the visited states;
- for the traces and dyna cases, every reachable state's Q row, with null for untried actions,
  and the number of real steps that entered each state.

Each run also records the step at which each state was first and last entered, and the greedy
route length after every episode (-1 where the route does not reach the goal).

The maze's exact solution comes from value iteration on the known transition model: V*(s), the
graph distance to the goal, and r, s' and done for every state and action. It is stored with
the runs so that rendering never trains or plans.

Run from maze_runner_rl/:
    python -m experiments.capture_analysis            # all cases (about 2 minutes on 7 workers)
    python -m experiments.capture_analysis --case dyna
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments import asset_style as st
from experiments import compare_methods as cm
from experiments import evaluate as ev
from experiments.training_experiment import convergence_episode

RESULTS = os.path.join(cm.OUT, "results", "runs.csv")
TUNING = os.path.join(cm.OUT, "tuning", "runs.csv")
OUT = os.path.join(st.DATA, "analysis")
LAMBDA_09 = {"trace_decay": 0.9, "replacing": False}
CASES = {
    "comparison": {"kind": "3d_3", "seed": 54, "split": "suite", "records": RESULTS,
                   "methods": ("q_learning", "q_lambda", "dyna_q"), "q_rows": False,
                   "budgets": (2_500, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000)},
    "traces": {"kind": "2d", "seed": 1004, "split": "dev", "records": TUNING,
               "methods": ("q_learning", "q_lambda"), "params": {"q_lambda": LAMBDA_09}, "q_rows": True,
               "steps": (25, 50, 100, 200, 1_000), "budgets": (5_000, 50_000)},
    "dyna": {"kind": "3d_3", "seed": 58, "split": "suite", "records": RESULTS,
             "methods": ("dyna_q", "q_learning"), "q_rows": True, "episodes": (60, 74, 85, 200, 1_000)},
}
CHECKED = ("env_steps", "first_success_env_steps", "first_greedy_shortest_env_steps", "converged_episode",
           "converged_env_steps", "final_greedy_route", "greedy_shortest_final", "updates")
UNITS = {
    "env_steps": "cumulative real environment steps (every move, including wall bumps), through the snapshot",
    "updates": "cumulative Q-value updates: one per real step for Q-learning, plus trace updates (Q(lambda)) "
               "or planning updates (Dyna-Q)",
    "budget": "a snapshot at a budget is taken at the end of the first episode whose cumulative real steps "
              "reach the budget, when the per-episode greedy evaluation happens",
    "step": "a snapshot at a step is taken immediately after that real step's update, mid-episode",
    "episode": "a snapshot at an episode is taken at that episode's end",
}


def maze(kind, seed):
    """The maze and its exact solution, restricted to states reachable from the start."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = ev.make_env(kind, seed)
    facts = cm.maze_facts(env)
    nxt, rew, done = facts["nxt"], facts["rew"], facts["done"]
    v_star, _ = ev.optimal_values(nxt, rew, done, ev.SETTINGS["agent"]["gamma"])
    cells = sorted(int(s) for s in facts["cells"])
    on_route = {int(s) for layer in facts["layers"] for s in layer}
    grid = np.asarray(env.maze)
    grid = grid if grid.ndim == 3 else grid[None]
    return env, facts, {
        "kind": kind, "maze_seed": seed, "gamma": ev.SETTINGS["agent"]["gamma"],
        "size": [int(grid.shape[2]), int(grid.shape[1]), int(grid.shape[0])],
        "levels": ["\n".join("".join("." if v == 0 else "#" for v in row) for row in level) for level in grid],
        "num_states": int(env.num_states), "actions": int(env.action_space.n),
        "start": int(facts["start"]), "goal": int(facts["goal"]), "shortest_route": int(facts["route"]),
        "states": cells,
        "positions": [[int(c) for c in env.position_from_state(s)] + ([0] if len(env.position_from_state(s)) == 2
                                                                       else []) for s in cells],
        "nxt": [[int(t) for t in nxt[s]] for s in cells],
        "rew": [[float(r) for r in rew[s]] for s in cells],
        "done": [[bool(d) for d in done[s]] for s in cells],
        "v_star": [round(float(v_star[s]), 6) for s in cells],
        "distance": [int(facts["to_goal"][s]) if s in facts["to_goal"] else None for s in cells],
        "on_shortest_route": [s in on_route for s in cells],
    }


def snapshot(agent, states, label, kind, episode, env_steps, updates, seen, q_rows, visits):
    out = {"label": label, "kind": kind, "episode": episode, "env_steps": env_steps, "updates": updates,
           "v": [round(max(agent.action_values(s)), 4) for s in states],
           "greedy": [int(agent.greedy_action(s)) for s in states],
           "tried": [sum(1 << int(a) for a in agent.q_table.get(s, {})) for s in states],
           "visited": sorted(int(s) for s in seen)}
    if q_rows:
        out["q"] = [[round(q[a], 4) if a in q else None for a in range(agent.action_space_size)]
                    for q in (agent.q_table.get(s, {}) for s in states)]
        out["visits"] = [visits.get(s, 0) for s in states]       # real steps that entered the state
    return out


def replay(task):
    """One run of compare_methods.train's loop, with read-only snapshots added. A task is
    (case, method, params), optionally followed by an episode count (default: the experiment's)."""
    case, method, params, *rest = task
    episodes = rest[0] if rest else cm.EPISODES
    spec = CASES[case]
    with contextlib.redirect_stdout(io.StringIO()):
        env = ev.make_env(spec["kind"], spec["seed"])
    facts = cm.maze_facts(env)
    states = sorted(int(s) for s in facts["cells"])
    random.seed(ev.exploration_seed(spec["seed"]))
    agent = cm.make_agent(method, params, env, spec["seed"])
    begin = getattr(agent, "begin_episode", None)
    end = getattr(agent, "end_episode", None) or agent.decay_epsilon
    nxt, start, goal = facts["nxt"], facts["start"], facts["goal"]
    step_marks, budgets = set(spec.get("steps", ())), list(spec.get("budgets", ()))
    episode_marks = set(spec.get("episodes", ()))
    seen, discovered, total = {start}, {start: 0}, 0
    visits, last_visit = {}, {}
    steps, successes, shortest, greedy_lengths, snaps = [], [], [], [], []

    def updates():
        return total if method == "q_learning" else int(agent.updates)

    def take(label, kind, episode):
        snaps.append(snapshot(agent, states, label, kind, episode, total, updates(), seen, spec["q_rows"], visits))


    for episode in range(1, episodes + 1):
        _, info = env.reset()
        if begin:
            begin()
        state, terminated, n = info["state"], False, 0
        for _ in range(cm.MAX_STEPS):
            action = agent.choose_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            agent.update_q_table(state, action, reward, info["state"], terminated)
            state = info["state"]
            n += 1
            total += 1
            visits[state] = visits.get(state, 0) + 1
            last_visit[state] = total
            if state not in seen:
                seen.add(state)
                discovered[state] = total
            if total in step_marks:
                take(f"step {total:,}", "step", episode)
            if terminated or truncated:
                break
        end()
        steps.append(n)
        successes.append(bool(terminated))
        g = cm.greedy_walk(agent, nxt, start, goal, env.num_states)
        greedy_lengths.append(-1 if g is None else g)
        shortest.append(g is not None and g == facts["route"])
        while budgets and total >= budgets[0]:
            budget = budgets.pop(0)
            take(f"{budget:,} steps", "budget", episode)
            snaps[-1]["budget"] = budget
        if episode in episode_marks:
            take(f"episode {episode}", "episode", episode)
    cum = np.cumsum(steps)
    converged = convergence_episode(shortest, cm.WINDOW)
    first_short, first_success = cm.first_index(shortest), cm.first_index(successes)
    final = cm.greedy_walk(agent, nxt, start, goal, env.num_states)
    record = {
        "env_steps": int(cum[-1]),
        "first_success_env_steps": None if first_success is None else int(cum[first_success]),
        "first_greedy_shortest_env_steps": None if first_short is None else int(cum[first_short]),
        "converged_episode": converged,
        "converged_env_steps": None if converged is None else int(cum[converged - 1]),
        "final_greedy_route": final,
        "greedy_shortest_final": final is not None and final == facts["route"],
        "updates": updates(),
    }
    return {"case": case, "method": method, "label": cm.METHODS[method]["label"], "params": params,
            "params_key": cm.params_key(params), "record": record, "snapshots": snaps,
            "discovered_at_step": {str(s): t for s, t in sorted(discovered.items())},
            "last_visit_step": {str(s): t for s, t in sorted(last_visit.items())},
            "greedy_route_length_per_episode": greedy_lengths}


def published(spec, method, params):
    key = cm.params_key(params)
    for r in cm.read_runs(spec["records"]):
        if (r["split"], r["kind"], r["method"], r["params"], r["maze_seed"]) == (
                spec["split"], spec["kind"], method, key, spec["seed"]):
            return r
    raise SystemExit(f"no published run for {method} {key} on {spec['kind']} maze {spec['seed']}")


def source_digest():
    digest = hashlib.sha256(cm.source_digest().encode())
    with open(__file__, "rb") as f:
        digest.update(f.read().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def capture(cases=tuple(CASES), out=OUT, workers=7):
    with open(os.path.join(cm.OUT, "frozen_settings.json"), encoding="utf-8") as f:
        frozen = json.load(f)
    tasks = []
    for case in cases:
        spec = CASES[case]
        for method in spec["methods"]:
            params = spec.get("params", {}).get(method, frozen["methods"][method]["params"])
            tasks.append((case, method, params))
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        runs = list(pool.map(replay, tasks))
    snap = cm.snapshot()
    os.makedirs(out, exist_ok=True)
    paths = []
    for case in cases:
        spec = CASES[case]
        mine = [r for r in runs if r["case"] == case]
        for run in mine:
            expected = published(spec, run["method"], run["params"])
            for field in CHECKED:
                if run["record"][field] != expected[field]:
                    raise SystemExit(f"{case}/{run['method']}: {field} is {run['record'][field]}, the published "
                                     f"record has {expected[field]}")
        _, _, facts = maze(spec["kind"], spec["seed"])
        data = {
            "case": case, "maze": facts,
            "runs": {r["method"]: {k: v for k, v in r.items() if k != "case"} for r in mine},
            "checked_against": os.path.relpath(spec["records"], os.path.dirname(ev.PACKAGE_ROOT)).replace(os.sep, "/"),
            "checked_fields": list(CHECKED), "units": UNITS,
            "provenance": {**snap, "capture_sha256": source_digest(), "command": "python -m experiments.capture_analysis",
                           "frozen_settings_sha256": cm.file_sha256(os.path.join(cm.OUT, "frozen_settings.json")),
                           "environment": cm.environment(workers, 0.0)},
        }
        data["provenance"]["environment"].pop("wall_seconds", None)
        path = os.path.join(out, f"{case}.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, separators=(",", ":"))
            f.write("\n")
        paths.append(path)
        print(f"{case}: {spec['kind']} maze {spec['seed']}, "
              f"{', '.join(r['method'] for r in mine)} checked against {data['checked_against']} -> {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", choices=list(CASES), action="append")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    capture(tuple(args.case or CASES), args.out, args.workers)


if __name__ == "__main__":
    main()
