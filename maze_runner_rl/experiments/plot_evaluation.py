"""Figures for docs/evaluation, drawn from the evaluation suite's results.

Run from maze_runner_rl/ after `python -m experiments.evaluate`:
    python -m experiments.plot_evaluation          # reads ../docs/evaluation/results, writes ../docs/evaluation/figures

The hero and policy figures replay one training run from the suite (same maze and
exploration seed, so the same run) to draw its routes; the script checks that the replay
matches the suite's record.
"""
import argparse
import contextlib
import csv
import io
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from experiments import evaluate as ev  # noqa: E402
from experiments.stats import wilson_interval  # noqa: E402
from experiments.training_experiment import TrainingExperiment  # noqa: E402

# Palette and chrome (light mode) from the reference data-viz palette.
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SOLVED = "#cde2fb"   # step 100 of the blue ramp
KIND_COLORS = {"2d": ORANGE, "3d_1": BLUE, "3d_3": AQUA}   # blue: the viewer's default maze type, shown in the hero
CONFIG_COLORS = {"current": BLUE, "cap_200": ORANGE, "prefix_rule": AQUA, "original": YELLOW}
SHORT_KIND = {"2d": "2D, 21x21", "3d_1": "3D, 1 level", "3d_3": "3D, 3 levels"}
CONFIG_LABELS = {"current": "current agent (1000-step cap, untried actions count as 0)",
                 "cap_200": "step cap back to 200",
                 "prefix_rule": "pre-fix action rule (only tried actions count)",
                 "original": "both undone: the original agent"}

RESULTS = os.path.normpath(os.path.join(ev.PACKAGE_ROOT, "..", "docs", "evaluation", "results"))
FIGURES = os.path.normpath(os.path.join(ev.PACKAGE_ROOT, "..", "docs", "evaluation", "figures"))
DASHED = (0, (2.2, 1.6))


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": BASE, "ytick.color": BASE,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9.5, "lines.solid_capstyle": "round",
    })


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def title_block(fig, title, subtitle, top=0.965, gap=0.07):
    fig.text(0.012, top, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.012, top - gap, subtitle, ha="left", va="top", fontsize=10, color=INK2)


# --- data ---------------------------------------------------------------------------

def load(results):
    with open(os.path.join(results, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    with open(os.path.join(results, "curves.json"), encoding="utf-8") as f:
        curves = json.load(f)
    with open(os.path.join(results, "runs.csv"), encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    return summary, curves, runs


def share_curve(curves, kind, config, key):
    """Share of runs whose flag is set at each point of the step grid, with 95% Wilson bands."""
    rows = [c[key] for c in curves["runs"] if c["kind"] == kind and c["config"] == config]
    flags = np.array([[ch == "1" for ch in row] for row in rows])
    n = flags.shape[0]
    k = flags.sum(axis=0)
    low, high = np.array([wilson_interval(int(ki), n) for ki in k]).T
    return np.array(curves["step_grid"]), k / n, low, high, n


def number(value):
    return None if value in ("", None) else float(value)


def tick_label(x):
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def log_steps_axis(ax, low=1e3, high=1e6):
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    ticks = [t for t in (100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6) if low <= t <= high]
    ax.set_xticks(ticks)
    ax.set_xticklabels([tick_label(t) for t in ticks])
    ax.minorticks_off()


def percent_axis(ax):
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def end_labels(ax, items, x, min_gap=7.0):
    """Direct labels just right of the line ends, nudged apart so they do not collide."""
    items = sorted(items, key=lambda item: item[0])
    placed = []
    for y, text in items:
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, text))
    shift = max(0.0, placed[-1][0] - 102) if placed else 0.0
    for y, text in placed:
        ax.text(x, y - shift, text, fontsize=9, color=INK2, va="center", ha="left", clip_on=False)


# --- replaying one run for the route drawings -------------------------------------

def greedy_path(env, agent):
    """Cells the greedy policy visits from the start; ends at the goal or where it starts
    to loop. Returns (cells, reached_goal)."""
    start, goal = ev.endpoints(env)
    state, cells = start, [start]
    while True:
        state, _, _ = env.step_from_state(state, agent.greedy_action(state))
        if state == goal:
            return cells + [goal], True
        if state in cells:
            return cells + [state], False
        cells.append(state)


def solved_region(env, agent, nxt):
    """Cells from which the greedy policy reaches the goal. The greedy policy maps each
    cell to one next cell, so following it from a cell either ends at the goal or loops."""
    start, goal = ev.endpoints(env)
    known = {goal: True}
    for cell in ev.reachable_states(nxt, start):
        path, seen, state = [], set(), cell
        while state not in known and state not in seen:
            seen.add(state)
            path.append(state)
            state = int(nxt[state, agent.greedy_action(state)])
        reached = known.get(state, False)
        for s in path:
            known[s] = reached
    return {s for s, reached in known.items() if reached}


def replay(kind, maze_seed, episodes, every=10):
    """Repeat the suite's 'current' run on one maze, keeping the greedy path from the start
    and the solved region every few episodes. Training in segments is identical to one long
    run: the agent, epsilon and the exploration RNG simply carry on."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = ev.make_env(kind, maze_seed)
    random.seed(ev.exploration_seed(maze_seed))
    agent = ev.make_agent(env)
    experiment = TrainingExperiment(env, agent)
    if episodes % every:
        raise ValueError(f"episodes ({episodes}) must be a multiple of {every}")
    nxt, _, _ = ev.transition_model(env)
    paths, solved = {}, {}
    for done in range(every, episodes + 1, every):
        experiment.run(num_episodes=every, max_steps=ev.CONFIGS["current"]["max_steps"], log_every=0,
                       greedy_limit=env.num_states)
        paths[done] = greedy_path(env, agent)
        solved[done] = solved_region(env, agent, nxt)
    return env, agent, experiment.results, paths, solved


def check_replay(results, runs, kind, maze_seed):
    record = next(r for r in runs if r["kind"] == kind and r["config"] == "current" and int(r["maze_seed"]) == maze_seed)
    if int(record["env_steps"]) != int(np.sum(results["steps"])):
        raise RuntimeError(f"replay of {kind} maze {maze_seed} does not match the suite's run")
    return record


def maze_grid(env):
    grid = np.asarray(env.maze)
    return grid[np.newaxis] if grid.ndim == 2 else grid


def xy(env, state):
    x, y, *rest = env.position_from_state(state)
    return x, y, (rest[0] if rest else 0)


def draw_maze(ax, env, level=0):
    grid = maze_grid(env)[level]
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if grid[y, x] != 0:
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=BASE, edgecolor="none", zorder=1))
    ax.set_xlim(-0.6, width - 0.4)
    ax.set_ylim(height - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_path(ax, env, cells, color, lw=2.2, dashed=False, zorder=3):
    points = np.array([xy(env, s)[:2] for s in cells], dtype=float)
    ax.plot(points[:, 0], points[:, 1], color=color, lw=lw, ls=DASHED if dashed else "-",
            solid_joinstyle="round", dash_capstyle="round", zorder=zorder)


def draw_endpoints(ax, env):
    start, goal = ev.endpoints(env)
    sx, sy, _ = xy(env, start)
    gx, gy, _ = xy(env, goal)
    ax.plot(sx, sy, "o", ms=8.5, mfc=SURFACE, mec=INK, mew=2, zorder=6)
    ax.plot(gx, gy, "o", ms=8.5, mfc=INK, mec=SURFACE, mew=1.5, zorder=6)


ENDPOINT_HANDLES = [Line2D([], [], marker="o", ms=8, mfc=SURFACE, mec=INK, mew=2, lw=0, label="start"),
                    Line2D([], [], marker="o", ms=8, mfc=INK, mec=SURFACE, lw=0, label="goal")]


def shortest_cells(env):
    """One shortest route, by breadth-first search over the transition model."""
    start, goal = ev.endpoints(env)
    parent = {start: None}
    queue = [start]
    for state in queue:
        if state == goal:
            break
        for action in range(env.action_space.n):
            nxt, _, _ = env.step_from_state(state, action)
            if nxt not in parent:
                parent[nxt] = state
                queue.append(nxt)
    route, state = [], goal
    while state is not None:
        route.append(state)
        state = parent[state]
    return route[::-1]


def pick_checkpoints(paths):
    """Checkpoints about 35% and 70% of the way to the first one at which the greedy route
    from the start reaches the goal, and that one itself (the last checkpoint if it never
    does). With very few checkpoints the same one can appear more than once."""
    episodes = sorted(paths)
    first = next((e for e in episodes if paths[e][1]), episodes[-1])
    before = [e for e in episodes if e <= first]
    return [before[max(0, round(len(before) * share) - 1)] for share in (0.35, 0.7)] + [first]


# --- figures --------------------------------------------------------------------------

def hero(curves, summary, runs, out, kind="3d_1", maze_seed=0):
    episodes = summary["provenance"]["suite"]["episodes"]
    env, agent, results, paths, solved = replay(kind, maze_seed, episodes)
    check_replay(results, runs, kind, maze_seed)
    route = shortest_cells(env)
    nxt, _, _ = ev.transition_model(env)
    start, goal = ev.endpoints(env)
    cells = ev.reachable_states(nxt, start)

    fig = plt.figure(figsize=(14, 5.1))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5], left=0.012, right=0.925, top=0.78, bottom=0.17,
                            wspace=0.10)
    for i, episode in enumerate(pick_checkpoints(paths)):
        ax = fig.add_subplot(grid[0, i])
        draw_maze(ax, env)
        for s in solved[episode]:
            x, y, _ = xy(env, s)
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=SOLVED, edgecolor="none", zorder=1.5))
        draw_path(ax, env, route, INK, lw=1.3, dashed=True, zorder=2)
        reached = paths[episode][1]
        if reached:
            draw_path(ax, env, paths[episode][0], BLUE, zorder=3)
        draw_endpoints(ax, env)
        ax.set_title(f"After {episode} episodes", pad=4)
        share = len(solved[episode]) / len(cells)
        note = (f"from the start too: a shortest route, {len(paths[episode][0]) - 1} moves" if reached
                else "not yet from the start")
        ax.text(0.0, -0.03, f"reaches the goal from {share:.0%} of cells,\n{note}", transform=ax.transAxes,
                fontsize=9, color=INK2, va="top", linespacing=1.3)

    ax = fig.add_subplot(grid[0, 3])
    ends = []
    for k, color in KIND_COLORS.items():
        x, share, low, high, n = share_curve(curves, k, "current", "shortest")
        ax.fill_between(x, low * 100, high * 100, color=color, alpha=0.14, lw=0)
        ax.plot(x, share * 100, color=color, lw=2)
        ends.append((share[-1] * 100, SHORT_KIND[k]))
    log_steps_axis(ax)
    percent_axis(ax)
    end_labels(ax, ends, 1.12e6)
    ax.set_xlabel("Environment steps of experience")
    ax.set_title(f"All runs: share whose greedy route is a shortest route\n({n} mazes per type, bands: 95% intervals)",
                 pad=4)

    handles = [Patch(color=SOLVED, label="greedy policy reaches the goal from here"),
               Line2D([], [], color=BLUE, lw=2.2, label="greedy route from the start"),
               Line2D([], [], color=INK, lw=1.3, ls=DASHED, label="a shortest route (BFS)"),
               *ENDPOINT_HANDLES]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.008, 0.0), ncol=5, fontsize=9.5)

    rates = {k: summary["groups"][k]["current"]["converged"]["rate"] for k in ev.KINDS}
    title = (f"Tabular Q-learning settles on a shortest route through a fixed maze in {rates['2d']:.0%} of 2D, "
             f"{rates['3d_1']:.0%} of one-level and {rates['3d_3']:.0%} of three-level runs")
    subtitle = (f"Left: one run on the viewer's default maze type ({SHORT_KIND[kind]}, maze seed {maze_seed}). What the "
                f"agent knows spreads backwards from the goal, one update at a time, until it covers the start.")
    title_block(fig, title, subtitle)
    save(fig, out, "hero.png")
    return env, agent


def first_success(curves, summary, runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.74, bottom=0.21, wspace=0.12)
    grid = np.array(curves["step_grid"])
    ratios = {}
    for ax, kind in zip(axes, ev.KINDS):
        steps = np.array([number(r["first_success_env_steps"]) or np.inf for r in runs
                          if r["kind"] == kind and r["config"] == "current"])
        share = np.array([(steps <= x).mean() for x in grid]) * 100
        walk = np.array(curves["random_walk_reached_mean"][kind]) * 100
        ax.plot(grid, share, color=BLUE, lw=2)
        ax.plot(grid, walk, color=INK2, lw=1.6, ls=(0, (3, 2)))
        b = summary["baselines"][kind]
        q_median = summary["groups"][kind]["current"]["first_success_env_steps"]["median"]
        w_median = b["random_walk_median_steps"]["median"]
        if q_median:
            ratios[kind] = w_median / q_median
        ax.axvline(b["shortest_route"]["median"], color=MUTED, lw=1, ls=(0, (1, 2)))
        ax.text(b["shortest_route"]["median"] * 0.9, 96, "shortest\nroute", fontsize=8.5, color=INK2, va="top",
                ha="right")
        ax.text(0.98, 0.04, f"median steps to first\nreach the goal\nQ-learning: {ev.fmt_steps(q_median)}\n"
                f"random walk: {ev.fmt_steps(w_median)}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=INK2, linespacing=1.3)
        log_steps_axis(ax, 30, 1e6)
        percent_axis(ax)
        ax.set_title(ev.KINDS[kind])
        ax.set_xlabel("Environment steps")
    axes[0].set_ylabel("Has reached the goal at least once")
    handles = [Line2D([], [], color=BLUE, lw=2.2, label="Q-learning, current agent: share of runs"),
               Line2D([], [], color=INK2, lw=1.6, ls=(0, (3, 2)), label="random walk: exact probability, mean over mazes")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=2)
    one_level = [ratios[k] for k in ("2d", "3d_1") if k in ratios]
    if one_level and "3d_3" in ratios:
        speedup = (f"{min(one_level):.0f} to {max(one_level):.0f} times sooner than a random walk on one level, "
                   f"{ratios['3d_3']:.1f} times on three")
    else:
        speedup = "sooner than a random walk"
    title = f"Optimistic initial values make exploration systematic: the agent first reaches the goal {speedup}"
    subtitle = ("Untried actions are valued at 0, tried ones pick up step penalties, so the agent keeps taking moves "
                "it has not made yet. Random walk: every action with probability 1/A.")
    title_block(fig, title, subtitle)
    save(fig, out, "first_success.png")


def ablations(curves, summary, out):
    configs = ("current", "cap_200", "prefix_rule", "original")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.73, bottom=0.21, wspace=0.22)
    for ax, kind in zip(axes, ev.KINDS):
        ends = []
        for config in configs:
            x, share, low, high, n = share_curve(curves, kind, config, "shortest")
            color = CONFIG_COLORS[config]
            ax.fill_between(x, low * 100, high * 100, color=color, alpha=0.12, lw=0)
            ax.plot(x, share * 100, color=color, lw=2)
            ends.append((share[-1] * 100, f"{share[-1] * 100:.0f}%"))
        end_labels(ax, ends, 1.12e6)
        log_steps_axis(ax)
        percent_axis(ax)
        ax.set_title(ev.KINDS[kind])
        ax.set_xlabel("Environment steps of experience")
    axes[0].set_ylabel(f"Runs whose greedy route is\na shortest route ({n} mazes)")
    handles = [Line2D([], [], color=CONFIG_COLORS[c], lw=2.2, label=CONFIG_LABELS[c]) for c in configs]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=4, fontsize=9.2)
    title = "Ablation: what each repair contributes, measured in steps of experience"
    subtitle = (f"Each line undoes one repair; the same {n} mazes and exploration seeds per type, one fixed maze per run. "
                f"Bands: 95% Wilson intervals.")
    title_block(fig, title, subtitle)
    save(fig, out, "ablations.png")


def policy_map(summary, runs, out, kind="3d_1", maze_seed=0, env=None, agent=None):
    episodes = summary["provenance"]["suite"]["episodes"]
    if env is None:
        env, agent, results, _, _ = replay(kind, maze_seed, episodes)
        check_replay(results, runs, kind, maze_seed)
    nxt, rew, done = ev.transition_model(env)
    start, goal = ev.endpoints(env)
    _, q_star = ev.optimal_values(nxt, rew, done, agent.discount_factor)
    optimal = q_star >= q_star.max(axis=1, keepdims=True) - 1e-9
    cells = [s for s in ev.reachable_states(nxt, start) if s != goal]

    fig = plt.figure(figsize=(12.5, 6.4))
    gridspec = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.012, right=0.975, top=0.80, bottom=0.12,
                                wspace=0.12)
    ax = fig.add_subplot(gridspec[0, 0])
    draw_maze(ax, env)
    good = 0
    for s in cells:
        x, y, _ = xy(env, s)
        ok = optimal[s, agent.greedy_action(s)]
        good += ok
        ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, facecolor=BLUE if ok else ORANGE,
                               edgecolor="none", zorder=2))
    path, _ = greedy_path(env, agent)
    draw_path(ax, env, path, INK, lw=1.8, zorder=4)
    draw_endpoints(ax, env)
    ax.legend(handles=[Patch(color=BLUE, label="greedy action optimal"),
                       Patch(color=ORANGE, label="not optimal"),
                       Line2D([], [], color=INK, lw=1.8, label="final greedy route"), *ENDPOINT_HANDLES],
              loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=9, handlelength=1.4)
    ax.set_title(f"One run after {episodes} episodes ({SHORT_KIND[kind]}, maze seed {maze_seed}):\n"
                 f"greedy action optimal in {good / len(cells):.0%} of reachable cells")

    ax = fig.add_subplot(gridspec[0, 1])
    kinds = list(ev.KINDS)
    bars = [("greedy_optimal_share", BLUE, "greedy action is optimal (line: 95% interval)"),
            ("chance_optimal_share", BASE, "a uniformly random action would be optimal")]
    y = np.arange(len(kinds))
    for i, (key, color, label) in enumerate(bars):
        offsets = y + (i - 0.5) * 0.32
        for off, k in zip(offsets, kinds):
            stat = summary["groups"][k]["current"][key]
            value = stat["mean"] * 100
            ax.barh(off, value, height=0.26, color=color, label=label if k == kinds[0] else None)
            right = value
            if key == "greedy_optimal_share" and stat["ci"][0] is not None:
                ax.plot([stat["ci"][0] * 100, stat["ci"][1] * 100], [off, off], color=INK, lw=1.2,
                        solid_capstyle="butt")
                right = max(right, stat["ci"][1] * 100)
            ax.text(right + 1.5, off, f"{value:.0f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_KIND[k] for k in kinds])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.07), ncol=1)
    ax.set_title("Share of reachable cells, mean over all runs of the current agent")

    share = {k: summary["groups"][k]["current"]["greedy_optimal_share"]["mean"] for k in kinds}
    title = (f"It learns a route, not a map: after {episodes} episodes the greedy action is optimal in "
             f"{min(share.values()):.0%} to {max(share.values()):.0%} of the maze")
    subtitle = ("Optimal actions come from value iteration on the known maze (the exact Q*). Away from its route the "
                "agent rarely returns, so its values there stay rough.")
    title_block(fig, title, subtitle, gap=0.06)
    save(fig, out, "policy_map.png")


ROUTE_BINS = ((0, 80, "up to 80"), (81, 120, "81 to 120"), (121, 10_000, "over 120"))
FOLLOW_UP_STYLE = {"current": (BLUE, "current agent: discount 0.95, {episodes} episodes"),
                   "gamma_099": (ORANGE, "discount 0.99"),
                   "episodes_x3": (AQUA, "{x3} episodes")}


def route_length(summary, runs, out):
    """Why long routes fail: learning time against route length, and the two follow-ups."""
    episodes = summary["provenance"]["suite"]["episodes"]
    window = summary["provenance"]["suite"]["window"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.065, right=0.985, top=0.75, bottom=0.21, wspace=0.22)

    never = episodes * 1.12
    for kind, color in KIND_COLORS.items():
        rows = [r for r in runs if r["kind"] == kind and r["config"] == "current"]
        x = np.array([float(r["shortest_route"]) for r in rows])
        first = np.array([number(r["first_greedy_shortest_episode"]) or np.nan for r in rows])
        hit = ~np.isnan(first)
        left.scatter(x[hit], first[hit], s=34, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        left.scatter(x[~hit], np.full((~hit).sum(), never), s=34, facecolor=SURFACE, edgecolor=color,
                     linewidth=1.4, zorder=3)
        law = summary.get("route_law", {}).get(kind)
        if law and hit.any():
            xs = np.array([x[hit].min(), x[hit].max()])   # only over the runs the line was fitted to
            left.plot(xs, law["slope"] * xs + law["intercept"], color=color, lw=1.6, zorder=2)
    left.axhline(episodes - window, color=INK2, lw=1, ls=(0, (3, 2)))
    left.text(0.99, (episodes - window + 14) / (episodes * 1.18), f"last chance to converge: episode "
              f"{episodes - window}\n({episodes} episodes, {window}-check window)", transform=left.transAxes,
              fontsize=8.5, color=INK2, va="bottom", ha="right")
    left.axhspan(episodes * 1.06, episodes * 1.18, color=GRID, lw=0, zorder=0)
    left.text(42, never, f"never within {episodes} episodes", fontsize=8.5, color=INK2, va="center")
    left.set_ylim(0, episodes * 1.18)
    left.set_xlabel("Shortest route through the maze (moves)")
    left.set_ylabel("First episode whose greedy route\nis a shortest route")
    left.grid(True, axis="y")
    left.set_axisbelow(True)
    law = summary.get("route_law", {})
    if law:
        slopes = [v["slope"] for v in law.values()]
        rs = [v["r"] for v in law.values()]
        left.set_title(f"Current agent: about {min(slopes):.0f} to {max(slopes):.0f} episodes per move of route "
                       f"(correlation {min(rs):.2f} to {max(rs):.2f})")

    configs = [c for c in FOLLOW_UP_STYLE if any(r["config"] == c for r in runs)]
    width = 0.8 / len(configs)
    for i, config in enumerate(configs):
        color, label = FOLLOW_UP_STYLE[config]
        label = label.format(episodes=episodes, x3=3 * episodes)
        for j, (low, high, _) in enumerate(ROUTE_BINS):
            rows = [r for r in runs if r["config"] == config and low <= float(r["shortest_route"]) <= high]
            k, n = sum(r["converged_episode"] != "" for r in rows), len(rows)
            if n == 0:
                continue
            lo, hi = wilson_interval(k, n)
            xpos = j + (i - (len(configs) - 1) / 2) * width
            right.bar(xpos, k / n * 100, width=width * 0.86, color=color, label=label)
            label = None   # one legend entry per setting
            right.plot([xpos, xpos], [lo * 100, hi * 100], color=INK, lw=1.1, solid_capstyle="butt")
            right.text(xpos, hi * 100 + 2, f"{k}/{n}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    right.set_xticks(range(len(ROUTE_BINS)))
    right.set_xticklabels([f"{name} moves" for _, _, name in ROUTE_BINS])
    right.set_xlabel("Shortest route (all three maze types pooled)")
    percent_axis(right)
    right.set_ylim(0, 112)
    right.set_ylabel("Runs converged")
    right.set_title("Converged runs by route length (line: 95% interval)")
    fig.legend(loc="lower left", bbox_to_anchor=(0.55, 0.0), ncol=3, fontsize=9.2,
               handles=right.get_legend_handles_labels()[0])
    kind_handles = [Line2D([], [], marker="o", lw=0, ms=7, color=c, mec=SURFACE, label=SHORT_KIND[k])
                    for k, c in KIND_COLORS.items()]
    kind_handles.append(Line2D([], [], marker="o", lw=0, ms=7, mfc=SURFACE, mec=INK2, label="never reached"))
    fig.legend(handles=kind_handles, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=4, fontsize=9.2)

    title = "Long routes need more episodes, not a longer horizon"
    subtitle = ("One-step Q-learning carries the goal's value back a little further each episode, so learning time "
                "grows in step with the route. A discount of 0.99 does not change that; more episodes do.")
    title_block(fig, title, subtitle, gap=0.065)
    save(fig, out, "route_length.png")


def draw_all(results=RESULTS, out=FIGURES):
    style()
    summary, curves, runs = load(results)
    env, agent = hero(curves, summary, runs, out)
    first_success(curves, summary, runs, out)
    ablations(curves, summary, out)
    policy_map(summary, runs, out, env=env, agent=agent)
    route_length(summary, runs, out)


def main():
    parser = argparse.ArgumentParser(description="Draw the evaluation figures.")
    parser.add_argument("--results", default=RESULTS)
    parser.add_argument("--out", default=FIGURES)
    args = parser.parse_args()
    draw_all(args.results, args.out)


if __name__ == "__main__":
    main()
