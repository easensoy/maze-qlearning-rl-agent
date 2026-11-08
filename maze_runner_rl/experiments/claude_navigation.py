"""Claude navigating mazes through MCP tools: a zero-shot tool-use evaluation.

The task. Each episode is one held-out maze (11 x 11, one level, the repository's 3D
generator with the viewer's density settings; seeds 30000-30009, used nowhere else).
The agent starts in one corner and must reach the goal in the opposite corner within 120
moves. Through the MCP server in mcp_maze/ it may call:
    observe(session_id)         its position, the goal's position, the open directions
    step(session_id, direction) one move; a move into a wall costs a move and goes nowhere
    episode_status(session_id)  reached / done / moves used and left / bumps
It never sees the map: what it knows beyond the current cell it has to remember. The
harness starts every session with reset; the agent cannot call reset.

Agents, with what each one knows:
- Claude, zero-shot, through the Messages API with tool use. Its memory is its own context.
- Depth-first explorer (deterministic): the same observations, a perfect memory of the
  cells it has visited, prefers the open move that ends nearest the goal, backtracks from
  dead ends. The planning reference with the same information as Claude.
- Breadth-first search with the whole map: the shortest route. Information Claude does not have.
- Random walk: uniform over north/south/east/west, 1000 seeded walks per maze.
- Per-maze reinforcement learning, for context only: Q-learning and Dyna-Q (the setting the
  value-propagation experiment froze) trained on each maze from scratch. They see their cell index, not coordinates or the goal, and
  learn from rewards over many episodes; the moves they need are a training budget, not
  a zero-shot score, so they are reported beside the table, not ranked in it.

Run from maze_runner_rl/:
    python -m experiments.claude_navigation references                 # no API calls
    python -m experiments.claude_navigation claude --max-usd 20        # needs ANTHROPIC_API_KEY
    python -m experiments.claude_navigation summary
"""
import argparse
import asyncio
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np

from experiments import compare_methods as cm
from experiments import evaluate as ev
from experiments.stats import wilson_interval
from experiments.training_experiment import convergence_episode, shortest_route_length
from mcp_maze.sessions import DIRECTIONS, SessionManager

PACKAGE_ROOT = ev.PACKAGE_ROOT
OUT = os.path.normpath(os.path.join(PACKAGE_ROOT, "..", "docs", "claude-tool-use"))
TASK = {"width": 11, "height": 11, "levels": 1, "max_moves": 120, "maze_seeds": list(range(30000, 30010)),
        "trials": 3}
CLAUDE_TOOLS = ("episode_status", "observe", "step")       # sorted, so the tool list never changes order
MAX_API_CALLS = 200                                         # per episode, a guard against runaway loops
MAX_TOKENS = 16000
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
# US$ per million tokens: input, output, cache write (5-minute TTL, 1.25x input), cache read.
# From the Claude API reference bundled with Claude Code (model table cached 2026-10-06).
# Haiku 5.5's cache-read price is not in that table and is assumed to be 10% of input.
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}
PRICE_SOURCE = ("Claude API reference bundled with Claude Code, model table cached 2026-10-06; cache writes at "
                "1.25x input for the 5-minute TTL; Haiku 5.5 cache reads assumed at 10% of input")

SYSTEM_PROMPT = """You are navigating a maze through tools. Reach the goal using as few moves as possible.

Rules:
- You cannot see the maze. observe returns your position, the goal's position, and the directions that are open from your current cell. Nothing else.
- Positions are (x, y, level). north is y - 1, south is y + 1, west is x - 1, east is x + 1. This maze has one level, so up and down are always blocked.
- step moves one cell and costs one move. A step into a wall costs a move and leaves you where you were. observe and episode_status cost nothing.
- You have {max_moves} moves. The episode ends when you reach the goal or run out of moves.
- The tools do not remember where you have been. Keep track of it yourself.

Use the session_id from the first message in every call."""


def first_message(session_id):
    return f"Your session_id is {session_id}. The episode has started. Reach the goal."


# --- the maze facts the agents are not given ----------------------------------------------------

def maze_facts(seed, task=TASK):
    manager = SessionManager()
    session = manager.reset(seed, task["width"], task["height"], task["levels"], task["max_moves"])
    return {"maze_seed": seed, "shortest_route": shortest_route_length(session.env),
            "open_cells": int((session.env.maze == 0).sum()), "start": list(session.start),
            "goal": list(session.goal)}


# --- reference agents (in-process, the same rules as the MCP server) --------------------------------

def neighbour(position, direction):
    dx, dy, dz = {"north": (0, -1, 0), "south": (0, 1, 0), "west": (-1, 0, 0), "east": (1, 0, 0),
                  "down": (0, 0, -1), "up": (0, 0, 1)}[direction]
    x, y, z = position
    return x + dx, y + dy, z + dz


def run_dfs_explorer(seed, task=TASK):
    """Depth-first exploration from local observations: an open, unvisited neighbour nearest the
    goal (Manhattan distance, ties in a fixed direction order), else one step back."""
    session = SessionManager().reset(seed, task["width"], task["height"], task["levels"], task["max_moves"])
    goal = session.goal
    visited, trail = {session.position}, [session.position]
    while not session.done:
        obs = session.observe()
        here = session.position
        fresh = [d for d in obs["open"] if neighbour(here, d) not in visited]
        if fresh:
            direction = min(fresh, key=lambda d: sum(abs(a - b) for a, b in zip(neighbour(here, d), goal)))
            session.step(direction)
            visited.add(session.position)
            trail.append(session.position)
        elif len(trail) > 1:
            back = trail[-2]
            direction = next(d for d in obs["open"] if neighbour(here, d) == back)
            session.step(direction)
            trail.pop()
        else:
            break
    return {"reached_goal": session.reached_goal, "moves": session.moves, "bumps": session.bumps}


def run_random_walks(seed, walks=1000, task=TASK):
    """Uniformly random moves over north/south/east/west (seed 900000 + maze seed)."""
    rng = random.Random(900_000 + seed)
    session = SessionManager().reset(seed, task["width"], task["height"], task["levels"], task["max_moves"])
    horizontal = ("north", "south", "east", "west")
    reached, moves = 0, []
    for _ in range(walks):
        session.position, session.moves, session.bumps, session.path = session.start, 0, 0, [session.start]
        while not session.done:
            session.step(rng.choice(horizontal))
        if session.reached_goal:
            reached += 1
            moves.append(session.moves)
    return {"walks": walks, "reached_goal": reached, "median_moves_when_reached": float(np.median(moves)) if moves else None}


def run_rl(seed, method, params, task=TASK, episodes=1000, window=100):
    """Per-maze training (context only): real moves until the first goal arrival and until the
    greedy route settled on a shortest route (the value-propagation study's criterion)."""
    session = SessionManager().reset(seed, task["width"], task["height"], task["levels"], task["max_moves"])
    env = session.env
    facts = cm.maze_facts(env)
    random.seed(ev.exploration_seed(seed))
    agent = cm.make_agent(method, params, env, seed)
    begin = getattr(agent, "begin_episode", None)
    end = getattr(agent, "end_episode", None) or agent.decay_epsilon
    steps, successes, shortest = [], [], []
    for _ in range(episodes):
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
            if terminated or truncated:
                break
        end()
        steps.append(n)
        successes.append(bool(terminated))
        g = cm.greedy_walk(agent, facts["nxt"], facts["start"], facts["goal"], env.num_states)
        shortest.append(g is not None and g == facts["route"])
    cum = np.cumsum(steps)
    first = cm.first_index(successes)
    converged = convergence_episode(shortest, window)
    return {"method": method, "params": cm.params_key(params),
            "first_goal_moves": None if first is None else int(cum[first]),
            "converged_moves": None if converged is None else int(cum[converged - 1]),
            "converged_episode": converged}


def dyna_setting():
    """Dyna-Q's setting as frozen by the value-propagation experiment (n = 10 if not frozen yet)."""
    path = os.path.join(cm.OUT, "frozen_settings.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["methods"]["dyna_q"]["params"]
    return {"planning_steps": 10}


def cmd_references(args):
    task = TASK
    rows, rl = [], []
    dyna = dyna_setting()
    for seed in task["maze_seeds"][:args.mazes or None]:
        facts = maze_facts(seed, task)
        dfs = run_dfs_explorer(seed, task)
        walk = run_random_walks(seed, task=task)
        rows.append({**facts, "dfs_reached": dfs["reached_goal"], "dfs_moves": dfs["moves"],
                     "random_reached_share": walk["reached_goal"] / walk["walks"],
                     "random_median_moves_when_reached": walk["median_moves_when_reached"]})
        with contextlib.redirect_stdout(io.StringIO()):
            for method, params in (("q_learning", {}), ("dyna_q", dyna)):
                rl.append({"maze_seed": seed, **run_rl(seed, method, params, task, episodes=args.episodes)})
        print(f"  maze {seed}: shortest {facts['shortest_route']}, DFS {dfs['moves']}, "
              f"random {walk['reached_goal']}/{walk['walks']}", flush=True)
    out = os.path.join(args.out, "results")
    os.makedirs(out, exist_ok=True)
    write_rows(os.path.join(out, "references.csv"), rows)
    write_rows(os.path.join(out, "rl_per_maze.csv"), rl)
    write_task_files(args.out)


# --- Claude through the Messages API ----------------------------------------------------------------

def tool_definitions(listed):
    """Claude's tools, from the MCP server's own listing: name, description, input schema."""
    tools = {t.name: t for t in listed}
    return [{"name": name, "description": tools[name].description, "input_schema": tools[name].input_schema}
            for name in CLAUDE_TOOLS]


def call_cost(model, usage):
    p = PRICES.get(model)
    if p is None:
        return None
    return (usage["input_tokens"] * p["input"] + usage["output_tokens"] * p["output"]
            + usage["cache_creation_input_tokens"] * p["cache_write"]
            + usage["cache_read_input_tokens"] * p["cache_read"]) / 1e6


def usage_dict(usage):
    return {k: int(getattr(usage, k, 0) or 0) for k in
            ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}


def block_dict(block):
    return block.model_dump(mode="json", exclude_none=True) if hasattr(block, "model_dump") else dict(block)


class Budget:
    """Running spend, kept on disk so an interrupted run resumes against the same cap."""

    def __init__(self, path, cap):
        self.path, self.cap = path, cap
        self.spent = 0.0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.spent = json.load(f)["spent_usd"]

    def add(self, cost):
        self.spent += cost or 0.0
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"spent_usd": round(self.spent, 6), "cap_usd": self.cap}, f)

    @property
    def exhausted(self):
        return self.spent >= self.cap


async def claude_episode(mcp, client, tools, model, effort, seed, trial, transcript_path, budget, task=TASK):
    """One episode. The loop ends when the session is done (goal or no moves left), when Claude
    stops calling tools, on a refusal, at MAX_API_CALLS, or when the spending cap is reached."""
    reset = await mcp.call_tool("reset", {"seed": seed, "width": task["width"], "height": task["height"],
                                          "levels": task["levels"], "max_moves": task["max_moves"]})
    session_id = json.loads(reset.content[0].text)["session_id"]
    system = SYSTEM_PROMPT.format(max_moves=task["max_moves"])
    messages = [{"role": "user", "content": first_message(session_id)}]
    totals = {k: 0 for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                             "cache_read_input_tokens")}
    record = {"maze_seed": seed, "trial": trial, "session_id": session_id, "model": model, "effort": effort,
              "api_calls": 0, "tool_calls": 0, "tool_errors": 0, "api_seconds": 0.0, "cost_usd": 0.0,
              "end": None, "served_by": set()}
    started = time.perf_counter()
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8", newline="\n") as log:
        def write(entry):
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")

        write({"type": "episode", "maze_seed": seed, "trial": trial, "session_id": session_id, "model": model,
               "effort": effort, "system": system, "tools": tools, "first_message": messages[0]["content"]})
        while True:
            if budget.exhausted:
                record["end"] = "spending cap reached"
                break
            if record["api_calls"] >= MAX_API_CALLS:
                record["end"] = "API call limit"
                break
            t0 = time.perf_counter()
            response = await client.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=system, tools=tools, messages=messages,
                cache_control={"type": "ephemeral"}, output_config={"effort": effort},
                thinking={"type": "adaptive", "display": "summarized"})
            latency = time.perf_counter() - t0
            usage = usage_dict(response.usage)
            cost = call_cost(model, usage)
            budget.add(cost)
            record["api_calls"] += 1
            record["api_seconds"] += latency
            record["cost_usd"] += cost or 0.0
            record["served_by"].add(response.model)
            for k in totals:
                totals[k] += usage[k]
            content = list(response.content)
            write({"type": "response", "call": record["api_calls"], "latency_s": round(latency, 3),
                   "stop_reason": response.stop_reason, "model": response.model,
                   "request_id": getattr(response, "_request_id", None), "usage": usage,
                   "cost_usd": None if cost is None else round(cost, 6), "content": [block_dict(b) for b in content]})
            messages.append({"role": "assistant", "content": content})
            calls = [b for b in content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                record["end"] = f"stopped calling tools ({response.stop_reason})"
                break
            results = []
            for block in calls:
                record["tool_calls"] += 1
                if block.name not in CLAUDE_TOOLS:
                    result_text, is_error = f"unknown tool {block.name!r}", True
                else:
                    result = await mcp.call_tool(block.name, dict(block.input))
                    result_text, is_error = result.content[0].text, bool(result.is_error)
                record["tool_errors"] += is_error
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text,
                                **({"is_error": True} if is_error else {})})
            write({"type": "tool_results", "results": results})
            messages.append({"role": "user", "content": results})
            status = json.loads((await mcp.call_tool("episode_status", {"session_id": session_id})).content[0].text)
            if status["done"]:
                record["end"] = "goal reached" if status["reached_goal"] else "out of moves"
                break
        status = json.loads((await mcp.call_tool("episode_status", {"session_id": session_id})).content[0].text)
        record.update({"reached_goal": status["reached_goal"], "moves": status["moves_used"],
                       "bumps": status["bumps"], "refused_calls": status["refused_calls"],
                       "episode_seconds": round(time.perf_counter() - started, 3), **totals})
        record["served_by"] = ";".join(sorted(record["served_by"]))
        record["api_seconds"] = round(record["api_seconds"], 3)
        record["cost_usd"] = round(record["cost_usd"], 6)
        write({"type": "end", **record})
    return record


async def run_claude(args, mcp_server=None, client=None):
    """All episodes, in order, against one MCP server process (or an in-process server in tests)."""
    from mcp import Client, StdioServerParameters

    if client is None:
        import anthropic
        client = anthropic.AsyncAnthropic(base_url=args.base_url)
    out = os.path.join(args.out, "results", "claude", args.model)
    os.makedirs(out, exist_ok=True)
    budget = Budget(os.path.join(out, "spend.json"), args.max_usd)
    target = mcp_server or StdioServerParameters(command=sys.executable, args=["-m", "mcp_maze.server"],
                                                 cwd=PACKAGE_ROOT)
    episodes_path = os.path.join(out, "episodes.csv")
    done = {(int(r["maze_seed"]), int(r["trial"])) for r in read_rows(episodes_path)} if os.path.exists(episodes_path) else set()
    facts = {seed: maze_facts(seed) for seed in TASK["maze_seeds"]}
    records = []
    async with Client(target) as mcp:
        tools = tool_definitions((await mcp.list_tools()).tools)
        for trial in range(1, args.trials + 1):
            for seed in TASK["maze_seeds"][:args.mazes or None]:
                if (seed, trial) in done:
                    continue
                if budget.exhausted:
                    print("  spending cap reached; stopping", flush=True)
                    break
                record = await claude_episode(mcp, client, tools, args.model, args.effort, seed, trial,
                                              os.path.join(out, "transcripts", f"{seed}-{trial}.jsonl"), budget)
                record["shortest_route"] = facts[seed]["shortest_route"]
                records.append(record)
                append_row(episodes_path, record)
                print(f"  maze {seed} trial {trial}: {record['end']}, {record['moves']} moves "
                      f"(shortest {record['shortest_route']}), ${record['cost_usd']:.3f}, "
                      f"spent ${budget.spent:.2f}", flush=True)
    write_json(os.path.join(out, "run.json"), {
        "model": args.model, "effort": args.effort, "max_tokens": MAX_TOKENS, "thinking": "adaptive, summarized",
        "prompt_caching": "automatic (top-level cache_control)", "fallbacks": "none: a refusal ends the episode",
        "cap_usd": args.max_usd, "spent_usd": round(budget.spent, 6), "prices": PRICES.get(args.model),
        "price_source": PRICE_SOURCE, **versions(), "finished_utc": now()})
    return records


# --- files ----------------------------------------------------------------------------------------

EPISODE_FIELDS = ["maze_seed", "trial", "shortest_route", "reached_goal", "moves", "bumps", "refused_calls",
                  "tool_calls", "tool_errors", "api_calls", "input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens", "cost_usd", "api_seconds",
                  "episode_seconds", "end", "model", "served_by", "effort", "session_id"]


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS, lineterminator="\n", extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def versions():
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version(), "git_commit": ev.git("rev-parse", "HEAD")}
    for name in ("mcp", "anthropic"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def write_task_files(out_dir):
    """The prompt and the tool schemas exactly as Claude receives them, listed from the server."""
    from mcp import Client

    from mcp_maze.server import build_server

    async def listed():
        server, _ = build_server()
        async with Client(server) as mcp:
            return (await mcp.list_tools()).tools

    tools = tool_definitions(asyncio.run(listed()))
    system = SYSTEM_PROMPT.format(max_moves=TASK["max_moves"])
    write_json(os.path.join(out_dir, "task.json"), {
        "task": TASK, "claude_tools": tools, "system_prompt": system,
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "first_message": first_message("<session_id>"), "max_api_calls_per_episode": MAX_API_CALLS,
        "max_tokens": MAX_TOKENS, "default_model": DEFAULT_MODEL, "default_effort": DEFAULT_EFFORT,
        "prices_usd_per_million_tokens": PRICES, "price_source": PRICE_SOURCE, **versions()})


# --- summary --------------------------------------------------------------------------------------

def summarize_claude(rows):
    rows = [r for r in rows if r.get("end")]
    if not rows:
        return None
    reached = [r for r in rows if r["reached_goal"] == "True"]
    excess = [int(r["moves"]) / int(r["shortest_route"]) - 1 for r in reached]
    low, high = wilson_interval(len(reached), len(rows))
    return {
        "episodes": len(rows), "reached": len(reached), "reached_ci": [low, high],
        "median_excess_when_reached": float(np.median(excess)) if excess else None,
        "bumps_per_episode": float(np.mean([int(r["bumps"]) for r in rows])),
        "tool_errors_per_episode": float(np.mean([int(r["tool_errors"]) for r in rows])),
        "api_calls_per_episode": float(np.mean([int(r["api_calls"]) for r in rows])),
        "median_episode_seconds": float(np.median([float(r["episode_seconds"]) for r in rows])),
        "median_seconds_per_call": float(np.median([float(r["api_seconds"]) / max(1, int(r["api_calls"]))
                                                    for r in rows])),
        "cost_usd_total": float(sum(float(r["cost_usd"]) for r in rows)),
        "cost_usd_per_episode": float(np.mean([float(r["cost_usd"]) for r in rows])),
        "tokens": {k: int(sum(int(r[k]) for r in rows)) for k in
                   ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")},
        "ends": {e: sum(r["end"] == e for r in rows) for e in sorted({r["end"] for r in rows})},
        "served_by": sorted({m for r in rows for m in r["served_by"].split(";") if m}),
    }


def cmd_summary(args):
    results = os.path.join(args.out, "results")
    refs = read_rows(os.path.join(results, "references.csv"))
    rl = read_rows(os.path.join(results, "rl_per_maze.csv"))
    summary = {
        "mazes": len(refs),
        "shortest_route": {"median": float(np.median([int(r["shortest_route"]) for r in refs])),
                           "min": min(int(r["shortest_route"]) for r in refs),
                           "max": max(int(r["shortest_route"]) for r in refs)},
        "dfs_explorer": {"reached": sum(r["dfs_reached"] == "True" for r in refs),
                         "median_excess": float(np.median([int(r["dfs_moves"]) / int(r["shortest_route"]) - 1
                                                           for r in refs if r["dfs_reached"] == "True"]))},
        "random_walk": {"reached_share_mean": float(np.mean([float(r["random_reached_share"]) for r in refs]))},
        "rl_per_maze": {m: {"median_first_goal_moves": float(np.median([int(r["first_goal_moves"]) for r in rl
                                                                       if r["method"] == m and r["first_goal_moves"]])),
                            "converged": sum(bool(r["converged_moves"]) for r in rl if r["method"] == m),
                            "median_converged_moves": float(np.median([int(r["converged_moves"]) for r in rl
                                                                      if r["method"] == m and r["converged_moves"]]
                                                                     or [float("nan")]))}
                        for m in sorted({r["method"] for r in rl})},
        "claude": {},
    }
    claude_dir = os.path.join(results, "claude")
    if os.path.isdir(claude_dir):
        for model in sorted(os.listdir(claude_dir)):
            path = os.path.join(claude_dir, model, "episodes.csv")
            if os.path.exists(path):
                summary["claude"][model] = summarize_claude(read_rows(path))
    write_json(os.path.join(results, "summary.json"), summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["references", "claude", "summary"])
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--mazes", type=int, default=0, help="only the first N mazes")
    parser.add_argument("--episodes", type=int, default=1000, help="training episodes per maze (references)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--trials", type=int, default=TASK["trials"])
    parser.add_argument("--max-usd", type=float, default=20.0, help="hard cap on estimated spend for this model")
    parser.add_argument("--base-url", default="https://api.anthropic.com",
                        help="Claude API endpoint (set explicitly so an inherited ANTHROPIC_BASE_URL is not used)")
    args = parser.parse_args()
    if args.command == "references":
        cmd_references(args)
    elif args.command == "claude":
        asyncio.run(run_claude(args))
    else:
        cmd_summary(args)


if __name__ == "__main__":
    main()
