"""Training traces behind the README visuals in docs/assets.

Replays the evaluation suite's runs of the current agent (experiments/evaluate.py,
setting "current"). The suite is deterministic, so the same maze seed, exploration seed
and settings give the same run, episode for episode. Writes to docs/assets/data:

  traces_<kind>.json  every run, per episode: moves taken, whether the training episode
                      reached the goal, and the greedy check made after it
  featured.json       two three-level runs in detail: the maze, a BFS shortest route,
                      the cells visited in chosen training episodes, and the greedy policy
                      read from the Q-table every 10 episodes
  provenance.json     code version, settings, seeds and, with --check, the comparison
                      with the suite's own records

The featured runs are picked by rule, not by eye (pick_featured): the converged
three-level run at the median convergence episode, and the lowest maze seed whose
three-level run did not converge. The second is continued to three times the episodes, as
in the suite's "episodes_x3" setting; its first 1000 episodes are the 1000-episode run.

Run from maze_runner_rl/:
    python -m experiments.capture_traces                      # 180 runs, a few minutes
    python -m experiments.capture_traces --check <runs.csv>   # also compare with the suite
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments import evaluate as ev
from experiments.training_experiment import TrainingExperiment, convergence_episode

OUT = os.path.normpath(os.path.join(ev.PACKAGE_ROOT, "..", "docs", "assets", "data"))
FEATURED_KIND = "3d_3"
CHECKPOINT_EVERY = 10


def xyz(env, state):
    x, y, *rest = env.position_from_state(state)
    return [int(x), int(y), int(rest[0]) if rest else 0]


def bits(flags):
    return "".join("1" if f else "0" for f in flags)


def run_record(results, window):
    """The suite's per-run fields (evaluate.train_run), from one run's per-episode results."""
    steps, successes = results["steps"], results["successes"]
    shortest, greedy = results["shortest_route"], results["greedy_steps"]
    cum = np.cumsum(steps)
    reaches = [g is not None for g in greedy]
    is_shortest = [g is not None and g == shortest for g in greedy]
    first = next((i for i, s in enumerate(successes) if s), None)
    converged = convergence_episode(is_shortest, window)
    return {
        "env_steps": int(cum[-1]),
        "shortest_route": shortest,
        "final_greedy_route": greedy[-1],
        "greedy_reaches_final": reaches[-1],
        "greedy_shortest_final": is_shortest[-1],
        "greedy_reaches_last100": float(np.mean(reaches[-window:])),
        "greedy_shortest_last100": float(np.mean(is_shortest[-window:])),
        "first_greedy_shortest_episode": next((i + 1 for i, s in enumerate(is_shortest) if s), None),
        "converged_episode": converged,
        "converged_env_steps": None if converged is None else int(cum[converged - 1]),
        "training_success_last100": float(np.mean(successes[-window:])),
        "first_success_episode": None if first is None else first + 1,
        "first_success_env_steps": None if first is None else int(cum[first]),
    }


def episode_trace(results):
    """Per-episode columns, compactly: bit strings for the flags, and the length of any
    greedy route that reached the goal without being a shortest route."""
    shortest, greedy = results["shortest_route"], results["greedy_steps"]
    return {
        "steps": [int(s) for s in results["steps"]],
        "success": bits(results["successes"]),
        "greedy_reaches": bits(g is not None for g in greedy),
        "greedy_shortest": bits(g is not None and g == shortest for g in greedy),
        "greedy_longer": {str(i + 1): g for i, g in enumerate(greedy) if g is not None and g != shortest},
    }


def train(kind, maze_seed, env_wrapper=None):
    """The environment and agent of one run, set up exactly as evaluate.train_run does."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = ev.make_env(kind, maze_seed)
    random.seed(ev.exploration_seed(maze_seed))
    agent = ev.make_agent(env)
    experiment = TrainingExperiment(env if env_wrapper is None else env_wrapper(env), agent)
    return env, agent, experiment


def run_episodes(env, experiment, episodes):
    return experiment.run(num_episodes=episodes, max_steps=ev.CONFIGS["current"]["max_steps"], log_every=0,
                          greedy_limit=env.num_states)


def trace_task(task):
    kind, maze_seed, episodes, window = task
    with contextlib.redirect_stdout(io.StringIO()):
        env, _, experiment = train(kind, maze_seed)
        results = run_episodes(env, experiment, episodes)
    return kind, {"maze_seed": maze_seed, "exploration_seed": ev.exploration_seed(maze_seed),
                  **run_record(results, window), **episode_trace(results)}


# --- featured runs -------------------------------------------------------------------

class Recorder:
    """Passes everything through to the environment, counts visits per cell and keeps the
    cells visited in chosen training episodes (counted from 1)."""

    def __init__(self, env, record_episodes):
        self.env = env
        self.record_episodes = set(record_episodes)
        self.episode = 0
        self.visits = np.zeros(env.num_states, dtype=np.int64)
        self.paths = {}

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.episode += 1
        self.visits[info["state"]] += 1
        if self.episode in self.record_episodes:
            self.paths[self.episode] = [info["state"]]
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.visits[info["state"]] += 1
        if self.episode in self.paths:
            self.paths[self.episode].append(info["state"])
        return observation, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def greedy_walk(env, agent, start, goal):
    """Cells the greedy policy visits from start, ending at the goal or at the first cell it
    visits twice (from there it loops forever: policy and maze are deterministic)."""
    state, cells = start, [start]
    while True:
        state, _, _ = env.step_from_state(state, agent.greedy_action(state))
        cells.append(state)
        if state == goal:
            return cells, True
        if state in cells[:-1]:
            return cells, False


def solved_region(agent, nxt, cells, goal):
    """Cells from which the greedy policy reaches the goal."""
    known = {goal: True}
    for cell in cells:
        path, seen, state = [], set(), cell
        while state not in known and state not in seen:
            seen.add(state)
            path.append(state)
            state = int(nxt[state, agent.greedy_action(state)])
        reached = known.get(state, False)
        for s in path:
            known[s] = reached
    return {s for s, reached in known.items() if reached}


def shortest_route_cells(nxt, start, goal):
    """One shortest route, by breadth-first search over the transition model trying the
    actions in index order."""
    parent = {start: None}
    queue = [start]
    for state in queue:
        if state == goal:
            break
        for t in nxt[state]:
            t = int(t)
            if t not in parent:
                parent[t] = state
                queue.append(t)
    route, state = [], goal
    while state is not None:
        route.append(state)
        state = parent[state]
    return route[::-1]


def count_shortest_routes(nxt, start, goal):
    """Number of distinct shortest routes from start to goal (BFS path counting)."""
    depth, ways, queue = {start: 0}, {start: 1}, [start]
    for state in queue:
        for t in set(int(t) for t in nxt[state]) - {state}:
            if t not in depth:
                depth[t] = depth[state] + 1
                ways[t] = 0
                queue.append(t)
            if depth[t] == depth[state] + 1:
                ways[t] += ways[state]
    return ways.get(goal, 0)


def featured_run(kind, maze_seed, episodes, window, record_episodes, snapshot_episodes, key_episodes=()):
    """A run of the current agent with the detail the hero and policy figures draw. The greedy
    policy is read every 10 episodes, and also after each of key_episodes."""
    recorder = None

    def wrap(env):
        nonlocal recorder
        recorder = Recorder(env, record_episodes)
        return recorder

    env, agent, experiment = train(kind, maze_seed, wrap)
    nxt, rew, done = ev.transition_model(env)
    start, goal = ev.endpoints(env)
    cells = ev.reachable_states(nxt, start)
    states = [s for s in cells if s != goal]
    v_star, q_star = ev.optimal_values(nxt, rew, done, agent.discount_factor)
    optimal = q_star >= q_star.max(axis=1, keepdims=True) - 1e-9

    checkpoints, snapshots, trained = [], [], 0
    stops = sorted(set(range(CHECKPOINT_EVERY, episodes + 1, CHECKPOINT_EVERY)) | set(snapshot_episodes)
                   | {e for e in key_episodes if e} | {1})
    for stop in stops:
        run_episodes(env, experiment, stop - trained)   # training in segments is the same as one call
        trained = stop
        path, reached = greedy_walk(env, agent, start, goal)
        solved = solved_region(agent, nxt, cells, goal)
        checkpoints.append({
            "episode": stop,
            "env_steps": int(np.sum(experiment.results["steps"])),
            "epsilon": agent.epsilon,
            "greedy_path": path,
            "greedy_reaches": reached,
            "solved": bits(s in solved for s in states),
        })
        if stop in snapshot_episodes:
            snapshots.append({
                "episode": stop,
                "value": [round(float(max(agent.action_values(s))), 4) for s in states],
                "greedy_action": [int(agent.greedy_action(s)) for s in states],
                "greedy_optimal": bits(optimal[s, agent.greedy_action(s)] for s in states),
                "tried": bits(bool(agent.q_table.get(s)) for s in states),
                "visits": [int(recorder.visits[s]) for s in states],
            })

    grid = np.asarray(env.maze)
    grid = grid[np.newaxis] if grid.ndim == 2 else grid
    return {
        "kind": kind,
        "label": ev.KINDS[kind],
        "maze_seed": maze_seed,
        "exploration_seed": ev.exploration_seed(maze_seed),
        "episodes": episodes,
        "size": [int(grid.shape[2]), int(grid.shape[1]), int(grid.shape[0])],
        "levels": ["\n".join("".join("#" if c else "." for c in row) for row in level) for level in grid],
        "start": xyz(env, start),
        "goal": xyz(env, goal),
        "start_state": start,
        "goal_state": goal,
        "loops": ev.loop_count(nxt, cells),
        "shortest_routes_count": count_shortest_routes(nxt, start, goal),
        "states": states,
        "state_xyz": [xyz(env, s) for s in states],
        "bfs_route": shortest_route_cells(nxt, start, goal),
        "optimal_value": [round(float(v_star[s]), 4) for s in states],
        "optimal_actions": ["".join(str(a) for a in np.flatnonzero(optimal[s])) for s in states],
        **run_record(experiment.results, window),
        **episode_trace(experiment.results),
        "paths": {str(e): p for e, p in sorted(recorder.paths.items())},
        "checkpoints": checkpoints,
        "snapshots": snapshots,
    }


def featured_task(task):
    name, *args = task
    with contextlib.redirect_stdout(io.StringIO()):
        return name, featured_run(*args)


def pick_featured(runs):
    """A typical success and a typical failure, by rule: the converged run at the median
    convergence episode (the lower of the two middle runs for an even count, ties to the lower
    maze seed), and the lowest maze seed whose run did not converge."""
    converged = sorted((r for r in runs if r["converged_episode"] is not None),
                       key=lambda r: (r["converged_episode"], r["maze_seed"]))
    success = converged[(len(converged) - 1) // 2]["maze_seed"] if converged else None
    failure = next((r["maze_seed"] for r in sorted(runs, key=lambda r: r["maze_seed"])
                    if r["converged_episode"] is None), None)
    return success, failure


# --- checks and output ---------------------------------------------------------------

CHECK_FIELDS = ("env_steps", "shortest_route", "final_greedy_route", "greedy_reaches_final", "greedy_shortest_final",
                "greedy_reaches_last100", "greedy_shortest_last100", "first_greedy_shortest_episode",
                "converged_episode", "converged_env_steps", "training_success_last100", "first_success_episode",
                "first_success_env_steps")


def as_csv_text(value):
    """A value as evaluate.write_csv writes it."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def display_path(path):
    """A path relative to the repository root when it is inside it."""
    path = os.path.abspath(path)
    root = os.path.dirname(ev.PACKAGE_ROOT)
    inside = os.path.commonpath([path, root]) == root
    return os.path.relpath(path, root).replace(os.sep, "/") if inside else path.replace(os.sep, "/")


def compare_with_suite(traces, runs_csv):
    """Every replayed run against the suite's record of it: how many runs were compared, and
    each field that differs. Includes the suite's provenance, when its summary.json is there."""
    with open(runs_csv, encoding="utf-8") as f:
        records = {(r["kind"], int(r["maze_seed"])): r for r in csv.DictReader(f) if r["config"] == "current"}
    mismatches, compared = [], 0
    for kind, runs in traces.items():
        for run in runs:
            record = records.get((kind, run["maze_seed"]))
            if record is None:
                mismatches.append(f"{kind} maze {run['maze_seed']}: not in the suite's records")
                continue
            compared += 1
            for field in CHECK_FIELDS:
                if as_csv_text(run[field]) != record[field]:
                    mismatches.append(f"{kind} maze {run['maze_seed']} {field}: replay {as_csv_text(run[field])!r}, "
                                      f"suite {record[field]!r}")
    comparison = {"runs_csv": display_path(runs_csv), "runs_compared": compared, "fields": list(CHECK_FIELDS),
                  "mismatches": mismatches}
    summary_path = os.path.join(os.path.dirname(os.path.abspath(runs_csv)), "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            suite = json.load(f)["provenance"]
        comparison["suite_provenance"] = {k: suite.get(k) for k in (
            "started_utc", "git_commit", "git_uncommitted_source_changes", "source_sha256")}
    return comparison


def load_traces(out=OUT):
    traces = {}
    for kind in ev.KINDS:
        path = os.path.join(out, f"traces_{kind}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                traces[kind] = json.load(f)["runs"]
    return traces


def recheck(out, runs_csv):
    """Compare traces already on disk with the suite's records, and note it in provenance.json."""
    comparison = compare_with_suite(load_traces(out), runs_csv)
    path = os.path.join(out, "provenance.json")
    with open(path, encoding="utf-8") as f:
        provenance = json.load(f)
    provenance["check_against_suite"] = comparison
    write_json(path, provenance, indent=2)
    return comparison


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, data, indent=None):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=indent, separators=None if indent else (",", ":"))
        f.write("\n")


def capture(out=OUT, kinds=tuple(ev.KINDS), mazes=60, episodes=None, window=100, workers=1, check=None,
            success_seed=None, failure_seed=None, progress=False):
    episodes = episodes or ev.SETTINGS["training"]["max_episodes"]
    os.makedirs(out, exist_ok=True)
    snapshot = ev.code_snapshot()
    started = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    run_map = pool.map if pool else map

    tasks = [(kind, seed, episodes, window) for kind in kinds for seed in range(mazes)]
    traces = {kind: [] for kind in kinds}
    for done, (kind, run) in enumerate(run_map(trace_task, tasks), 1):
        traces[kind].append(run)
        if progress and (done % 30 == 0 or done == len(tasks)):
            print(f"  {done}/{len(tasks)} runs", flush=True)

    settings = {"agent": ev.SETTINGS["agent"], "maze": ev.SETTINGS["maze"],
                "max_steps": ev.CONFIGS["current"]["max_steps"], "episodes": episodes, "window": window,
                "exploration_seed": f"{ev.EXPLORATION_SEED_BASE} + maze seed"}
    for kind, runs in traces.items():
        runs.sort(key=lambda r: r["maze_seed"])
        write_json(os.path.join(out, f"traces_{kind}.json"),
                   {"kind": kind, "label": ev.KINDS[kind], **settings, "runs": runs})

    featured = {}
    if FEATURED_KIND in traces:
        picked = pick_featured(traces[FEATURED_KIND])
        success_seed = picked[0] if success_seed is None else success_seed
        failure_seed = picked[1] if failure_seed is None else failure_seed
        def key_episodes(seed):
            run = next(r for r in traces[FEATURED_KIND] if r["maze_seed"] == seed)
            return (run["first_success_episode"], run["first_greedy_shortest_episode"], run["converged_episode"])

        jobs = []
        if success_seed is not None:
            jobs.append(("success", FEATURED_KIND, success_seed, episodes, window, (1, 2, 3),
                         (1, 100, 300, episodes), key_episodes(success_seed)))
        if failure_seed is not None:
            jobs.append(("failure", FEATURED_KIND, failure_seed, 3 * episodes, window, (1,),
                         (episodes, 3 * episodes), key_episodes(failure_seed)))
        featured = dict(run_map(featured_task, jobs))
        for name, run in featured.items():
            plain = next(r for r in traces[FEATURED_KIND] if r["maze_seed"] == run["maze_seed"])
            if run["steps"][:episodes] != plain["steps"] or run["success"][:episodes] != plain["success"]:
                raise RuntimeError(f"the featured {name} run does not match its trace")
    if pool:
        pool.shutdown()

    comparison = compare_with_suite(traces, check) if check else None

    import gymnasium
    script = os.path.abspath(__file__)
    status = ev.git("status", "--porcelain", "--untracked-files=all", "--", os.path.relpath(script, ev.PACKAGE_ROOT))
    provenance = {
        **snapshot,
        "capture_script_sha256": file_sha256(script),
        "capture_script_uncommitted": None if status is None else bool(status),
        "command": " ".join(["python -m experiments.capture_traces",   # absolute paths shown repo-relative
                             *(display_path(a) if os.path.isabs(a) and os.path.exists(a) else a
                               for a in sys.argv[1:])]),
        "kinds": list(kinds),
        "maze_seeds": f"0-{mazes - 1}",
        **settings,
        "rewards": {"goal": ev.CONFIG.rewards.goal_reward, "step": ev.CONFIG.penalties.step_penalty,
                    "wall": ev.CONFIG.penalties.wall_penalty},
        "featured": {name: {"kind": run["kind"], "maze_seed": run["maze_seed"], "episodes": run["episodes"]}
                     for name, run in featured.items()},
        "featured_rule": "success: the converged three-level run at the median convergence episode (the lower "
                         "middle run for an even count); failure: the lowest maze seed whose three-level run did "
                         "not converge, continued to three times the episodes",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "workers": workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "check_against_suite": comparison,
    }
    write_json(os.path.join(out, "featured.json"), featured)
    write_json(os.path.join(out, "provenance.json"), provenance, indent=2)
    return provenance


def main():
    parser = argparse.ArgumentParser(description="Record the training traces behind docs/assets; see the docstring.")
    parser.add_argument("--out", default=OUT, help="output folder (default: docs/assets/data)")
    parser.add_argument("--kinds", nargs="+", choices=list(ev.KINDS), default=list(ev.KINDS))
    parser.add_argument("--mazes", type=int, default=60, help="maze seeds 0 to mazes-1 for each maze type")
    parser.add_argument("--episodes", type=int, default=None, help="default: websocket/config.yaml")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--check", default=None, help="the suite's runs.csv to compare the replayed runs with")
    parser.add_argument("--success-seed", type=int, default=None, help="override the featured converged run")
    parser.add_argument("--failure-seed", type=int, default=None, help="override the featured unconverged run")
    parser.add_argument("--recheck", metavar="RUNS_CSV", default=None,
                        help="only compare the traces already in --out with the suite's runs.csv")
    args = parser.parse_args()
    if args.recheck:
        check = recheck(args.out, args.recheck)
        print(f"Compared {check['runs_compared']} runs with {check['runs_csv']}: "
              f"{len(check['mismatches'])} mismatching fields")
        for line in check["mismatches"][:20]:
            print("  ", line)
        return
    print(f"{len(args.kinds) * args.mazes} runs on {args.workers} workers")
    # Import by module name so worker processes can find the task functions.
    from experiments import capture_traces as module
    info = module.capture(args.out, args.kinds, args.mazes, args.episodes, 100, args.workers, args.check,
                          args.success_seed, args.failure_seed, progress=True)
    check = info["check_against_suite"]
    if check:
        print(f"Compared {check['runs_compared']} runs with {check['runs_csv']}: "
              f"{len(check['mismatches'])} mismatching fields")
        for line in check["mismatches"][:20]:
            print("  ", line)
    print(f"Wrote {args.out} in {info['wall_seconds']:.0f} s")


if __name__ == "__main__":
    main()
