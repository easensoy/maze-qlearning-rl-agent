"""Figures for the value-propagation experiment, drawn from its raw results.

Inputs (docs/value-propagation/results): runs.csv, curves.json, episodes.jsonl.gz.
Outputs (docs/value-propagation/figures):
- learning_curves.png: share of runs whose frozen greedy route is a shortest route,
  against real environment steps, per maze set and method, with 95% Wilson bands;
- route_length.png: share of runs converged, by shortest-route length;
- diagnostics.png: real steps to a shortest greedy route split in two - exploring until
  an episode first reaches the goal, and from there to the first shortest greedy route;
- failure.png: why Watkins's Q(lambda) fails with long traces (development mazes);
- analysis.json: every derived number the reports quote.

Method colours (validated for colour-vision deficiency on the dark surface with the
dataviz validator: worst pair deltaE 9.4, normal vision 24.6); every series is also
labelled directly. Amber and blue are left to the maze visuals (route, learned cells).

Run from maze_runner_rl/:
    python -m experiments.plot_value_propagation
"""
import argparse
import json
import os

import numpy as np

from experiments import asset_style as st
from experiments import compare_methods as cm
from experiments.stats import wilson_interval

RESULTS = os.path.join(cm.OUT, "results")
FIGURES = os.path.join(cm.OUT, "figures")
COLOURS = {"q_learning": "#9085e9", "q_lambda": "#d95926", "dyna_q": "#199e70"}
SHORT = {"q_learning": "Q-learning", "q_lambda": "Q(λ)", "dyna_q": "Dyna-Q"}
SETS = [("suite", "2d", "2D, 21 × 21"), ("suite", "3d_1", "One level, 21 × 21"),
        ("suite", "3d_3", "Three levels, 21 × 21 × 3"), ("long", "3d_3", "Long routes (> 140 moves)")]
DPI = 200


def figure(width, height):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width, height), dpi=DPI, facecolor=st.BG)
    return fig


def axes(fig, left, bottom, width, height):
    """Axes placed in inches from the bottom-left corner."""
    w, h = fig.get_size_inches()
    ax = fig.add_axes([left / w, bottom / h, width / w, height / h])
    ax.set_facecolor(st.BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(st.RULE)
    ax.tick_params(colors=st.INK3, labelsize=8.5, length=3, width=0.6)
    ax.grid(True, color=st.RULE, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    return ax


def text(fig, x, y, s, size=10, colour=st.INK, weight="normal", ha="left", va="baseline", max_width=None):
    w, h = fig.get_size_inches()
    t = fig.text(x / w, y / h, s, fontsize=size, color=colour, fontweight=weight, ha=ha, va=va,
                 family=st.FONT_FAMILY)
    if max_width:
        fig.canvas.draw()
        width = t.get_window_extent().width / fig.dpi
        if width > max_width:
            t.set_fontsize(size * max_width / width)
    return t


def save(fig, name):
    import matplotlib.pyplot as plt
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=DPI, facecolor=st.BG)
    plt.close(fig)
    return path


def load(results=RESULTS):
    runs = cm.read_runs(os.path.join(results, "runs.csv"))
    with open(os.path.join(results, "curves.json"), encoding="utf-8") as f:
        curves = json.load(f)
    raw = cm.read_raw(os.path.join(results, "episodes.jsonl.gz"))
    return runs, curves, raw


def share_with_band(rows, key):
    flags = np.array([[c == "1" for c in r[key]] for r in rows])
    k, n = flags.sum(axis=0), flags.shape[0]
    bands = np.array([wilson_interval(int(a), n) for a in k])
    return k / n, bands[:, 0], bands[:, 1]


def label_end(ax, x, y, s, colour, dy=0.0):
    ax.annotate(s, (x, y), xytext=(4, dy), textcoords="offset points", color=colour, fontsize=8.5,
                fontweight="bold", va="center", family=st.FONT_FAMILY, annotation_clip=False)


def learning_curves(runs, curves, title):
    grid = np.array(curves["step_grid"], dtype=float)
    keep = grid >= 1000
    fig = figure(8, 4.6)
    text(fig, 0.32, 4.28, title, size=12.5, weight="bold", max_width=7.4)
    text(fig, 0.32, 4.02, "Share of runs whose frozen greedy route (no exploration) is a shortest route, against real "
                          "environment steps. Bands: 95% Wilson intervals over independent runs, from where a share "
                          "first rises above zero.",
         size=8.5, colour=st.INK2, max_width=7.4)
    panel_w, gap, left = 1.45, 0.2, 0.66
    for i, (split, kind, name) in enumerate(SETS):
        ax = axes(fig, left + i * (panel_w + gap), 0.62, panel_w, 2.85)
        rows_by_method = {m: [c for c in curves["runs"] if c["split"] == split and c["kind"] == kind
                              and c["method"] == m] for m in cm.METHODS}
        n = len(rows_by_method["q_learning"])
        ax.set_title(f"{name}\n{n} mazes", fontsize=8.8, color=st.INK, loc="left", family=st.FONT_FAMILY, pad=6)
        ends = []
        for m in cm.METHODS:
            rows = rows_by_method[m]
            if not rows:
                continue
            share, low, high = share_with_band(rows, "shortest")
            start = np.argmax(share > 0) if (share > 0).any() else len(share)
            band = keep & (np.arange(len(share)) >= start)
            ax.fill_between(grid[band], low[band], high[band], color=COLOURS[m], alpha=0.16, linewidth=0)
            ax.plot(grid[keep], share[keep], color=COLOURS[m], linewidth=1.6)
            ends.append((share[keep][-1], m))
        ax.set_xscale("log")
        ax.set_xlim(1e3, 1e6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0%", "50%", "100%"] if i == 0 else [])
        ax.set_xticks([1e3, 1e4, 1e5, 1e6])
        ax.set_xticklabels(["1k", "10k", "100k", "1M"])
        if i == 0:
            ax.set_ylabel("runs with a shortest greedy route", fontsize=8.5, color=st.INK2, family=st.FONT_FAMILY)
        if i == len(SETS) - 1:
            ends.sort()
            last = -1.0
            for value, m in ends:
                y = max(value, last + 0.09)
                label_end(ax, 1e6, y, SHORT[m], COLOURS[m])
                last = y
    text(fig, 0.32 + 4 * (panel_w + gap) / 2, 0.16, "real environment steps (log scale), every move counted",
         size=8.5, colour=st.INK2, ha="center")
    return save(fig, "learning_curves.png")


def route_length(runs, title):
    pooled = [r for r in runs if r["split"] in ("suite", "long")]
    bins = [(0, 80), (81, 120), (121, 160), (161, 10_000)]
    labels = ["≤ 80", "81–120", "121–160", "> 160"]
    fig = figure(8, 3.6)
    text(fig, 0.32, 3.32, title, size=12.5, weight="bold", max_width=7.4)
    text(fig, 0.32, 3.06, "Runs converged within 1,000 episodes, by the maze's shortest-route length (all mazes of the "
                          "three suite types and the long-route set). Bars: 95% Wilson intervals.",
         size=8.5, colour=st.INK2, max_width=7.4)
    ax = axes(fig, 0.75, 0.55, 6.9, 2.2)
    offsets = {"q_learning": -0.22, "q_lambda": 0.0, "dyna_q": 0.22}
    for b, (low, high) in enumerate(bins):
        for m in cm.METHODS:
            mine = [r for r in pooled if r["method"] == m and low <= r["shortest_route"] <= high]
            if not mine:
                continue
            k = sum(r["converged_episode"] is not None for r in mine)
            lo, hi = wilson_interval(k, len(mine))
            x = b + offsets[m]
            ax.plot([x, x], [lo, hi], color=COLOURS[m], linewidth=1.6, solid_capstyle="round")
            ax.plot([x], [k / len(mine)], "o", color=COLOURS[m], markersize=6.5, markeredgecolor=st.BG,
                    markeredgewidth=1.2)
            if b == len(bins) - 1:
                ax.annotate(SHORT[m], (x, k / len(mine)), xytext=(7, 0), textcoords="offset points", ha="left",
                            va="center", color=COLOURS[m], fontsize=8.5, fontweight="bold", family=st.FONT_FAMILY)
        n = len([r for r in pooled if r["method"] == "q_learning" and low <= r["shortest_route"] <= high])
        ax.annotate(f"{n} mazes", (b, -0.13), ha="center", va="top", color=st.INK3, fontsize=8,
                    family=st.FONT_FAMILY, annotation_clip=False)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{s} moves" for s in labels], fontsize=8.8, color=st.INK2)
    ax.set_xlim(-0.6, len(bins) - 0.4)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", length=0, pad=4)
    return save(fig, "route_length.png")


def lag(r):
    if r["first_greedy_shortest_env_steps"] is None or r["first_success_env_steps"] is None:
        return None
    return r["first_greedy_shortest_env_steps"] - r["first_success_env_steps"]


PHASES = (("first_success_env_steps", "1. Exploring until an episode first reaches the goal"),
          (lag, "2. From there to the first shortest greedy route"))


def phase_value(r, phase):
    return phase(r) if callable(phase) else r[phase]


def diagnostics(runs, title):
    """Real steps to a shortest greedy route, split in two: exploring until the goal is
    first reached (how fast the agent finds the goal) and from there to the first shortest
    greedy route (how fast what it found reaches the start). Medians over runs, with a run
    that never got there counted as never; percentile-bootstrap 95% intervals."""
    from experiments.stats import bootstrap_interval

    fig = figure(8, 3.75)
    text(fig, 0.32, 3.47, title, size=12.5, weight="bold", max_width=7.4)
    text(fig, 0.32, 3.21, "Median real steps per run (dots) with 95% bootstrap intervals (bars); a run that never got "
                          "there counts as never, so a median can be 'not reached'.", size=8.5, colour=st.INK2,
         max_width=7.4)
    xmin, xmax = 3e2, 1.2e6
    offsets = {"q_learning": 0.24, "q_lambda": 0.0, "dyna_q": -0.24}
    for p, (phase, name) in enumerate(PHASES):
        ax = axes(fig, 1.75 + p * 3.15, 0.55, 2.75, 2.2)
        ax.set_title(name, fontsize=8.8, color=st.INK, loc="left", family=st.FONT_FAMILY, pad=6)
        for row, (split, kind, label) in enumerate(SETS):
            y = len(SETS) - 1 - row
            for m in cm.METHODS:
                values = [phase_value(r, phase) for r in runs
                          if (r["split"], r["kind"], r["method"]) == (split, kind, m)]
                values = [np.inf if v is None else max(float(v), xmin) for v in values]
                med = float(np.median(values))
                low, high = bootstrap_interval(values, np.median)
                yy = y + offsets[m]
                ax.plot([max(low, xmin), min(high, xmax)], [yy, yy], color=COLOURS[m], linewidth=1.6,
                        solid_capstyle="round", alpha=0.85)
                if np.isfinite(med):
                    ax.plot([med], [yy], "o", color=COLOURS[m], markersize=5.5, markeredgecolor=st.BG,
                            markeredgewidth=1.0)
                else:
                    ax.plot([xmax], [yy], ">", color=COLOURS[m], markersize=6)
                    ax.annotate("not reached", (xmax, yy), xytext=(-9, 0), textcoords="offset points", ha="right",
                                va="center", color=COLOURS[m], fontsize=7.5, family=st.FONT_FAMILY)
                if p == 1 and row == 0:
                    ax.annotate(SHORT[m], (max(low, xmin), yy), xytext=(-7, 0), textcoords="offset points",
                                ha="right", va="center", color=COLOURS[m], fontsize=8.2, fontweight="bold",
                                family=st.FONT_FAMILY)
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.55, len(SETS) - 0.45)
        ax.set_yticks(range(len(SETS)))
        ax.set_yticklabels([label for _, _, label in reversed(SETS)] if p == 0 else [], fontsize=8.5,
                           color=st.INK2)
        ax.tick_params(axis="y", length=0)
        ax.grid(True, axis="x", color=st.RULE, linewidth=0.5, alpha=0.6)
        ax.grid(False, axis="y")
        ax.set_xticks([1e3, 1e4, 1e5, 1e6])
        ax.set_xticklabels(["1k", "10k", "100k", "1M"])
        ax.set_xlabel("real environment steps (log scale)", fontsize=8.5, color=st.INK2, family=st.FONT_FAMILY)
    return save(fig, "diagnostics.png")


TUNING = os.path.join(cm.OUT, "tuning")
LAMBDA_RAMP = {0.1: "#fbd3b6", 0.3: "#f6ad80", 0.5: "#ec8650", 0.7: "#d9652b", 0.9: "#b84d1d"}  # ordinal, validated
MAP_BUDGET = 50_000


def discovered_by(kind, method, params, seed, budget):
    """The cells a recorded run had visited after `budget` real steps, by training it again
    exactly as compare_methods.train does (same seeds, same draws), stopping at the budget.
    Returns the visited cells and the per-episode counts, to check against the recorded run."""
    import contextlib
    import io
    import random

    from experiments import evaluate as ev

    with contextlib.redirect_stdout(io.StringIO()):
        env = ev.make_env(kind, seed)
    facts = cm.maze_facts(env)
    random.seed(ev.exploration_seed(seed))
    agent = cm.make_agent(method, params, env, seed)
    begin = getattr(agent, "begin_episode", None)
    end = getattr(agent, "end_episode", None) or agent.decay_epsilon
    seen, counts, total = {facts["start"]}, [], 0
    while total < budget:
        _, info = env.reset()
        if begin:
            begin()
        state, found = info["state"], 0
        for _ in range(cm.MAX_STEPS):
            action = agent.choose_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            agent.update_q_table(state, action, reward, info["state"], terminated)
            state = info["state"]
            total += 1
            if state not in seen:
                seen.add(state)
                found += 1
            if terminated or truncated:
                break
        end()
        counts.append(found)
    return env, facts, seen, counts


def replay_final_run(r, params):
    """Train a recorded final run again exactly as compare_methods.train does (same seeds, same
    draws; the per-episode greedy checks draw no random numbers) and return the agent, the
    maze facts and the visited cells. Stops if the replay's final greedy route differs."""
    import contextlib
    import io
    import random

    from experiments import evaluate as ev

    with contextlib.redirect_stdout(io.StringIO()):
        env = ev.make_env(r["kind"], r["maze_seed"])
    facts = cm.maze_facts(env)
    random.seed(ev.exploration_seed(r["maze_seed"]))
    agent = cm.make_agent(r["method"], params, env, r["maze_seed"])
    begin = getattr(agent, "begin_episode", None)
    end = getattr(agent, "end_episode", None) or agent.decay_epsilon
    seen = {facts["start"]}
    for _ in range(r["episodes"]):
        _, info = env.reset()
        if begin:
            begin()
        state = info["state"]
        for _ in range(cm.MAX_STEPS):
            action = agent.choose_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            agent.update_q_table(state, action, reward, info["state"], terminated)
            state = info["state"]
            seen.add(state)
            if terminated or truncated:
                break
        end()
    greedy = cm.greedy_walk(agent, facts["nxt"], facts["start"], facts["goal"], env.num_states)
    if greedy != r["final_greedy_route"]:
        raise SystemExit(f"the replay of {r['method']} on {r['kind']} maze {r['maze_seed']} does not match the record")
    return agent, facts, seen


def dyna_failure_trace(r, params):
    """Why a Dyna-Q run's greedy route stayed longer than the shortest route: the cell where
    the final greedy route leaves every shortest route, and the first cell on a shortest
    continuation from there whose goal-ward action was never taken (its Q-value is still 0)."""
    agent, facts, seen = replay_final_run(r, params)
    nxt, goal, to_goal = facts["nxt"], facts["goal"], facts["to_goal"]
    state = facts["start"]
    while to_goal.get(int(nxt[state, agent.greedy_action(state)])) == to_goal[state] - 1:
        state = int(nxt[state, agent.greedy_action(state)])
    leave = {"distance_to_goal": to_goal[state], "moves_from_start": r["shortest_route"] - to_goal[state]}
    cell, untried = state, None
    continuation = []
    while cell != goal:
        step = next(a for a in range(agent.action_space_size) if to_goal.get(int(nxt[cell, a])) == to_goal[cell] - 1)
        if untried is None and (cell, step) not in agent.model:
            tried = [agent.q_table[cell][a] for a in range(agent.action_space_size) if (cell, a) in agent.model]
            untried = {"distance_to_goal": to_goal[cell], "tried_actions_q": sorted(round(q, 3) for q in tried),
                       "untried_goalward_action_q": agent.action_values(cell)[step]}
        cell = int(nxt[cell, step])
        continuation.append(cell)
    return {"leaves_shortest_routes": leave, "first_untried_goalward_action": untried,
            "continuation_cells_never_visited": sum(c not in seen for c in continuation[:-1]),
            "continuation_cells": len(continuation) - 1, "final_epsilon": agent.epsilon}


def draw_map(ax, env, facts, seen, colour):
    """A top-down 2D maze: walls dark, open cells grey, discovered cells in the method's colour."""
    from matplotlib.patches import Rectangle

    grid = np.asarray(env.maze)
    grid = grid if grid.ndim == 2 else grid[0]
    h, w = grid.shape
    cells = {env.position_from_state(s)[:2] for s in seen}
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 0:
                continue
            fill = colour if (x, y) in cells else st.FLOOR
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fill, edgecolor=st.BG, linewidth=0.3))
    for state, marker in ((facts["start"], "o"), (facts["goal"], "o")):
        x, y = env.position_from_state(state)[:2]
        ax.plot([x + 0.5], [y + 0.5], marker, markersize=6, markerfacecolor="none" if state == facts["start"]
                else st.INK, markeredgecolor=st.INK, markeredgewidth=1.4)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def failure(dev_runs, dev_raw, title):
    """Watkins's Q(lambda) on the development mazes: convergence against lambda, cells
    discovered against real steps, and one maze's discovered cells at the same budget."""
    fig = figure(8, 7.1)
    text(fig, 0.32, 6.82, title, size=12.5, weight="bold", max_width=7.4)
    text(fig, 0.32, 6.56, "Development mazes (20 of each type), where λ was chosen. Watkins's Q(λ) cuts its traces only "
                          "after a non-greedy action, and choosing among untried actions (all valued 0) counts as greedy.",
         size=8.5, colour=st.INK2, max_width=7.4)
    # Convergence against lambda.
    ax = axes(fig, 0.75, 4.35, 2.6, 1.85)
    base = [r for r in dev_runs if r["method"] == "q_learning"]
    base_share = sum(r["converged_episode"] is not None for r in base) / len(base)
    ax.axhline(base_share, color=COLOURS["q_learning"], linewidth=1.4, linestyle=(0, (4, 2)))
    ax.annotate(f"Q-learning {base_share:.0%}", (0.98, base_share), xytext=(0, 5), textcoords="offset points",
                ha="right", color=COLOURS["q_learning"], fontsize=8.5, fontweight="bold", family=st.FONT_FAMILY)
    for replacing, marker, name in ((False, "o", "accumulating"), (True, "s", "replacing")):
        xs, ys = [], []
        for lam in sorted(LAMBDA_RAMP):
            key = cm.params_key({"trace_decay": lam, "replacing": replacing})
            mine = [r for r in dev_runs if r["method"] == "q_lambda" and r["params"] == key]
            xs.append(lam)
            ys.append(sum(r["converged_episode"] is not None for r in mine) / len(mine))
        ax.plot(xs, ys, color=COLOURS["q_lambda"], linewidth=1.4, alpha=1.0 if not replacing else 0.6,
                marker=marker, markersize=5, markeredgecolor=st.BG)
        ax.annotate(f"Q(λ), {name}", (xs[3], ys[3]), xytext=(9, 8) if replacing else (-9, -3),
                    textcoords="offset points", ha="left" if replacing else "right", va="center",
                    color=COLOURS["q_lambda"], fontsize=8.5, fontweight="bold", family=st.FONT_FAMILY)
    ax.set_xlim(0, 1)
    ax.set_xticks(sorted(LAMBDA_RAMP))
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xlabel("λ (trace decay)", fontsize=8.5, color=st.INK2, family=st.FONT_FAMILY)
    ax.set_ylabel("runs converged\n(60 development runs)", fontsize=8.5, color=st.INK2, family=st.FONT_FAMILY)
    # Cells discovered against real steps.
    ax2 = axes(fig, 4.55, 4.35, 2.55, 1.85)
    grid = np.unique(np.round(np.logspace(2, 6, 60))).astype(int)
    reach = {(r["kind"], r["maze_seed"]): r["reachable_cells"] for r in dev_runs if r["method"] == "q_learning"}
    series = {("q_learning", None): [], **{("q_lambda", lam): [] for lam in LAMBDA_RAMP}}
    for r in dev_raw:
        if r["method"] == "q_learning":
            key = ("q_learning", None)
        elif r["method"] == "q_lambda" and r["params"].startswith("replacing=False"):
            key = ("q_lambda", float(r["params"].split("trace_decay=")[1]))
        else:
            continue
        cum, found = np.cumsum(r["steps"]), 1 + np.cumsum(r["new_cells"])
        series[key].append(cm.sample_values(cum, found, grid, before=1) / reach[(r["kind"], r["maze_seed"])])
    for (method, lam), rows in series.items():
        colour = COLOURS["q_learning"] if lam is None else LAMBDA_RAMP[lam]
        med = np.median(np.array(rows), axis=0)
        ax2.plot(grid, med, color=colour, linewidth=1.6, label="Q-learning" if lam is None else f"Q(λ), λ {lam}")
        if lam is not None and lam >= 0.7:                       # the two failures, well apart: labelled directly
            label_end(ax2, grid[-1], med[-1], f"λ {lam}", colour)
    legend = ax2.legend(loc="upper left", frameon=False, handlelength=1.4, labelspacing=0.3, borderaxespad=0.2,
                        prop={"family": st.FONT_FAMILY, "size": 7.6})
    for t in legend.get_texts():
        t.set_color(st.INK2)
    ax2.set_xscale("log")
    ax2.set_xlim(1e2, 1e6)
    ax2.set_ylim(0, 1.02)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(["0%", "50%", "100%"])
    ax2.set_xticks([1e2, 1e4, 1e6])
    ax2.set_xticklabels(["100", "10k", "1M"])
    ax2.set_xlabel("real steps (log scale), counted at episode ends", fontsize=8.5, color=st.INK2,
                   family=st.FONT_FAMILY)
    ax2.set_ylabel("reachable cells discovered\n(median run, accumulating)", fontsize=8.5, color=st.INK2,
                   family=st.FONT_FAMILY)
    # One maze, chosen by rule: the 2D development maze with the median shortest route.
    two_d = sorted((r["shortest_route"], r["maze_seed"]) for r in dev_runs
                   if r["method"] == "q_learning" and r["kind"] == "2d")
    route, seed = two_d[(len(two_d) - 1) // 2]
    maps = []
    for i, (method, params, colour, name) in enumerate((
            ("q_learning", {}, COLOURS["q_learning"], "Q-learning"),
            ("q_lambda", {"trace_decay": 0.9, "replacing": False}, LAMBDA_RAMP[0.9], "Q(λ), λ 0.9, accumulating"))):
        env, facts, seen, counts = discovered_by("2d", method, params, seed, MAP_BUDGET)
        recorded = next(r for r in dev_raw if r["kind"] == "2d" and r["maze_seed"] == seed and r["method"] == method
                        and r["params"] == cm.params_key(params))
        if counts != recorded["new_cells"][:len(counts)]:
            raise SystemExit(f"the re-run of {method} on maze {seed} does not match the recorded run")
        ax3 = axes(fig, 0.75 + i * 3.45, 0.3, 2.75, 2.75)
        draw_map(ax3, env, facts, seen, colour)
        reachable = reach[("2d", seed)]
        ax3.set_title(f"{name}: {len(seen)} of {reachable} cells", fontsize=8.8, color=st.INK, loc="left",
                      family=st.FONT_FAMILY, pad=4)
        maps.append(len(seen))
    text(fig, 0.75, 3.55, f"Development 2D maze {seed} (shortest route {route} moves): cells discovered in the first "
                          f"{MAP_BUDGET:,} real steps. Ring: start; disk: goal.", size=8.5, colour=st.INK2, max_width=6.6)
    return save(fig, "failure.png"), {"maze_seed": seed, "route": route, "discovered": maps}


def runs_param(runs, method, name):
    """A method's frozen setting, read back from its runs' params column."""
    key = next(r["params"] for r in runs if r["method"] == method)
    return dict(kv.split("=") for kv in key.split(","))[name]


def geometric_ratio(pairs):
    """Geometric mean of a / b over pairs, with a percentile-bootstrap 95% interval."""
    from experiments.stats import bootstrap_interval

    if not pairs:
        return None
    logs = np.log([a / b for a, b in pairs])
    low, high = bootstrap_interval(logs, np.mean)
    return {"ratio": float(np.exp(logs.mean())), "ci": [float(np.exp(low)), float(np.exp(high))], "mazes": len(pairs)}


def analysis(runs, raw, dev_runs, dev_raw, failure_info):
    """Every derived number the reports quote, from the raw results (figures/analysis.json)."""
    rawd = {(x["split"], x["kind"], x["method"], x["maze_seed"]): x for x in raw}

    def updates_to_converge(r):
        if r["converged_episode"] is None:
            return None
        return int(sum(rawd[(r["split"], r["kind"], r["method"], r["maze_seed"])]["updates"][:r["converged_episode"]]))

    out = {"sets": {}, "route_length": {}, "development": {}, "failure": {}}
    for split, kind, label in SETS:
        entry = out["sets"][f"{split}:{kind}"] = {"label": label}
        group = {m: {r["maze_seed"]: r for r in runs if (r["split"], r["kind"], r["method"]) == (split, kind, m)}
                 for m in cm.METHODS}
        for m, rs in group.items():
            rows = list(rs.values())
            converged = [r for r in rows if r["converged_episode"] is not None]
            upd = [updates_to_converge(r) for r in converged]
            entry[m] = {
                "runs": len(rows), "converged": len(converged),
                "real_steps_to_converge": cm.median_steps([r["converged_env_steps"] for r in rows]),
                "first_goal": cm.median_steps([r["first_success_env_steps"] for r in rows]),
                "first_goal_to_shortest": cm.median_steps([lag(r) for r in rows]),
                "updates_to_converge_median_converged_runs": float(np.median(upd)) if upd else None,
                "updates_per_real_step": float(np.median([r["updates"] / r["env_steps"] for r in rows])),
                "agent_seconds_per_run_median": float(np.median([r["agent_seconds"] for r in rows])),
                "cells_found_10_episodes_median": float(np.median([r["cells_found_10_episodes"] / r["reachable_cells"]
                                                                   for r in rows])),
            }
            if m != "q_learning":
                both = [s for s in rows if s["converged_episode"] is not None
                        and group["q_learning"][s["maze_seed"]]["converged_episode"] is not None]
                base = group["q_learning"]
                entry[m]["vs_q_learning_where_both_converged"] = {
                    "real_steps_ratio": geometric_ratio([(base[s["maze_seed"]]["converged_env_steps"],
                                                          s["converged_env_steps"]) for s in both]),
                    "updates_ratio": geometric_ratio([(updates_to_converge(s), updates_to_converge(base[s["maze_seed"]]))
                                                      for s in both]),
                }
    pooled = [r for r in runs if r["split"] in ("suite", "long")]
    for low, high in ((0, 80), (81, 120), (121, 10_000)):
        key = f"{low}-{high}" if high < 10_000 else f"{low}+"
        out["route_length"][key] = {m: {"converged": sum(r["converged_episode"] is not None for r in pooled
                                                         if r["method"] == m and low <= r["shortest_route"] <= high),
                                        "runs": sum(1 for r in pooled
                                                    if r["method"] == m and low <= r["shortest_route"] <= high)}
                                    for m in cm.METHODS}
    out["failure"]["dyna_q_not_converged"] = [
        {"split": r["split"], "kind": r["kind"], "maze_seed": r["maze_seed"], "shortest_route": r["shortest_route"],
         "final_greedy_route": r["final_greedy_route"], "first_greedy_shortest_episode": r["first_greedy_shortest_episode"],
         "q_learning_converged_episode": next(q["converged_episode"] for q in runs if q["method"] == "q_learning" and
                                              (q["split"], q["kind"], q["maze_seed"]) == (r["split"], r["kind"], r["maze_seed"])),
         "q_lambda_converged_episode": next(q["converged_episode"] for q in runs if q["method"] == "q_lambda" and
                                            (q["split"], q["kind"], q["maze_seed"]) == (r["split"], r["kind"], r["maze_seed"]))}
        for r in runs if r["method"] == "dyna_q" and r["converged_episode"] is None]
    params = {"planning_steps": int(runs_param(runs, "dyna_q", "planning_steps"))}
    for entry in out["failure"]["dyna_q_not_converged"]:
        r = next(x for x in runs if x["method"] == "dyna_q" and (x["split"], x["kind"], x["maze_seed"]) ==
                 (entry["split"], entry["kind"], entry["maze_seed"]))
        entry["replay"] = dyna_failure_trace(r, params)
    out["failure"]["q_lambda_map"] = failure_info
    grid = np.array([MAP_BUDGET])
    for replacing in (False, True):
        for lam in sorted(LAMBDA_RAMP):
            key = cm.params_key({"trace_decay": lam, "replacing": replacing})
            rows = [r for r in dev_runs if r["method"] == "q_lambda" and r["params"] == key]
            found = [float(cm.sample_values(np.cumsum(x["steps"]), 1 + np.cumsum(x["new_cells"]), grid, before=1)[0])
                     / next(r["reachable_cells"] for r in dev_runs if r["method"] == "q_learning"
                            and (r["kind"], r["maze_seed"]) == (x["kind"], x["maze_seed"]))
                     for x in dev_raw if x["method"] == "q_lambda" and x["params"] == key]
            out["development"][key] = {"converged": sum(r["converged_episode"] is not None for r in rows),
                                       "runs": len(rows), "cells_found_by_50k_median": float(np.median(found))}
    base = [r for r in dev_runs if r["method"] == "q_learning"]
    out["development"]["q_learning"] = {"converged": sum(r["converged_episode"] is not None for r in base),
                                        "runs": len(base)}
    path = os.path.join(FIGURES, "analysis.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cm.ev._finite(out), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


TITLES = {
    "learning_curves": "Dyna-Q's greedy route is shortest after far fewer real moves, and on every long route",
    "route_length": "Beyond 120 moves, Q-learning converged on 0 of 60 mazes, Q(λ) on 30, Dyna-Q on all 60",
    "diagnostics": "Dyna-Q shortens both phases; Q(λ) only the second, and it finds the goal more slowly",
    "failure": "With long traces, Q(λ)'s state coverage stalls: at λ ≥ 0.7 it rarely converges",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()
    st.style()
    runs, curves, raw = load()
    dev_runs = cm.read_runs(os.path.join(TUNING, "runs.csv"))
    dev_raw = cm.read_raw(os.path.join(TUNING, "episodes.jsonl.gz"))
    print(learning_curves(runs, curves, TITLES["learning_curves"]))
    print(route_length(runs, TITLES["route_length"]))
    print(diagnostics(runs, TITLES["diagnostics"]))
    path, info = failure(dev_runs, dev_raw, TITLES["failure"])
    print(path)
    print(analysis(runs, raw, dev_runs, dev_raw, info))


if __name__ == "__main__":
    main()
