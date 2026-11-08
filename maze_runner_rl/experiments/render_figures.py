"""The README's results figure and policy figure, drawn from docs/assets/data
(capture_traces.py), in the hero animation's palette (asset_style.py).

  results.png  all 180 runs of the current agent: how many have exploring training
               episodes that reliably reach the goal, how many have a greedy route that is
               a shortest route, and how many have converged, against moves of experience
  policy.png   what the greedy policy knows in a converged run and in one that ran out of
               episodes, and how knowledge of the route spreads back from the goal

Run from maze_runner_rl/ after capture_traces:
    python -m experiments.render_figures
"""
import argparse
import os

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from experiments import asset_style as st
from experiments.stats import wilson_interval

KINDS = {"2d": "2D maze · 21 × 21", "3d_1": "One level · 21 × 21", "3d_3": "Three levels · 21 × 21 × 3"}
RELIABLE = (45, 50)   # training is "reliable" once 45 of the last 50 episodes reached the goal
DPI = 200


def figure(width, height):
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(width, height), facecolor=st.BG)


def close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def axes_in(fig, left, bottom, width, height):
    """Axes placed in inches from the figure's bottom-left corner."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def text_in(fig, x, y, s, max_width=None, **kwargs):
    """Figure text placed in inches; with max_width (inches) it is made smaller to fit on one
    line, for typefaces wider than Segoe UI."""
    w, h = fig.get_size_inches()
    t = fig.text(x / w, y / h, s, **kwargs)
    if max_width:
        width = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
        if width > max_width:
            t.set_fontsize(t.get_fontsize() * max_width / width)
    return t


def frame_axes(ax, labelsize=8.4):
    ax.tick_params(labelsize=labelsize, length=3, color=st.RULE)
    ax.grid(True, axis="y", color=st.RULE, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(st.RULE)


def percent_axis(ax, labels=True):
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"] if labels else [])


# --- results figure ------------------------------------------------------------------

def at_grid(cum_steps, flags, grid):
    """A per-episode flag read at each point of a step grid: its value after the last episode
    that ended within that many moves (False before the first one ends). A run that has
    finished its episodes keeps its final value."""
    index = np.searchsorted(cum_steps, grid, side="right") - 1
    flags = np.asarray(flags, dtype=bool)
    return np.where(index >= 0, flags[np.clip(index, 0, None)], False)


def reliable_training(success, k=RELIABLE[0], window=RELIABLE[1]):
    """Per episode: did at least k of the last `window` training episodes reach the goal?"""
    counts = np.convolve(np.asarray(success, dtype=float), np.ones(window))[:len(success)]
    return counts >= k


def share_with_interval(flags):
    flags = np.asarray(flags)
    n = flags.shape[0]
    k = flags.sum(axis=0)
    low, high = np.array([wilson_interval(int(x), n) for x in k]).T
    return k / n, low, high, n


def results_figure(data, out):
    traces = {k: st.load_json(f"traces_{k}.json", data) for k in KINDS}
    grid = np.unique(np.round(np.logspace(3, 6, 300)))
    fig = figure(8.0, 4.85)
    numbers = {}
    for i, (kind, label) in enumerate(KINDS.items()):
        ax = axes_in(fig, 0.62 + i * 2.5, 1.12, 2.28, 2.5)
        runs = traces[kind]["runs"]
        reliable, shortest, settled = [], [], []
        for run in runs:
            cum = np.cumsum(run["steps"])
            success = [c == "1" for c in run["success"]]
            reliable.append(at_grid(cum, reliable_training(success), grid))
            shortest.append(at_grid(cum, [c == "1" for c in run["greedy_shortest"]], grid))
            done = run["converged_env_steps"]
            settled.append(grid >= (done if done is not None else np.inf))
        for flags, color in ((reliable, st.BLUE), (shortest, st.AMBER)):
            share, low, high, n = share_with_interval(flags)
            shown = grid >= grid[np.argmax(share > 0)]   # no band while every run is still at zero
            ax.fill_between(grid[shown], low[shown] * 100, high[shown] * 100, color=color, alpha=0.17, lw=0,
                            step="post", zorder=2)
            ax.step(grid, share * 100, where="post", color=color, lw=2.0, zorder=4)
        share, _, _, n = share_with_interval(settled)
        ax.step(grid, share * 100, where="post", color=st.AMBER, lw=1.1, ls=(0, (2.4, 1.8)), zorder=5)

        converged = sum(r["converged_episode"] is not None for r in runs)
        final_short = sum(r["greedy_shortest_final"] for r in runs)
        late = np.mean([r["training_success_last100"] for r in runs])
        numbers[kind] = dict(n=n, converged=converged, final_short=final_short, late=late)
        ax.annotate(label, (0.0, 1.0), xycoords="axes fraction", xytext=(0, 21), textcoords="offset points",
                    ha="left", va="bottom", fontsize=9.4, fontweight=600, color=st.INK)
        ax.annotate(f"{converged} of {n} runs converged", (0.0, 1.0), xycoords="axes fraction", xytext=(0, 7),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8.4, color=st.AMBER, fontweight=600)
        ax.set_xscale("log")
        ax.set_xlim(1e3, 1e6)
        ax.set_xticks([1e3, 1e4, 1e5, 1e6])
        ax.set_xticklabels(["1k", "10k", "100k", "1M"])
        ax.minorticks_off()
        percent_axis(ax, labels=i == 0)
        frame_axes(ax)
        if i == 0:
            ax.set_ylabel("share of the 60 runs", fontsize=8.6, color=st.INK2)
        if i == 1:
            ax.set_xlabel("moves of experience (environment steps, log scale)", fontsize=8.6, color=st.INK2,
                          labelpad=5)

    three = numbers["3d_3"]
    text_in(fig, 0.1, 4.68, "Reaching the goal while exploring is not the same as learning the route", max_width=7.8,
            fontsize=13, fontweight=600, color=st.INK, va="top")
    text_in(fig, 0.1, 4.34, f"Three levels: {three['late']:.0%} of late training episodes reach the goal, yet only "
                            f"{three['converged']} of {three['n']} greedy routes settle on a shortest route "
                            f"within 1,000 episodes.", max_width=7.8, fontsize=9.2, color=st.INK2, va="top")
    handles = [Line2D([], [], color=st.BLUE, lw=2.0,
                      label=f"exploring: {RELIABLE[0]} of the last {RELIABLE[1]} training episodes reach the goal"),
               Line2D([], [], color=st.AMBER, lw=2.0, label="greedy route is a shortest route"),
               Line2D([], [], color=st.AMBER, lw=1.1, ls=(0, (2.4, 1.8)),
                      label="converged: shortest at every later check, at least 100"),
               Patch(color=st.INK2, alpha=0.3, label="band: 95% Wilson interval over 60 independent runs")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.1 / 8.0, 0.04), ncol=2, fontsize=7.9,
               frameon=False, handlelength=2.0, columnspacing=1.5, labelcolor=st.INK2)
    path = os.path.join(out, "results.png")
    fig.savefig(path, dpi=DPI)
    close(fig)
    return path, numbers


# --- policy figure -------------------------------------------------------------------

def draw_level(ax, maze, z, solved=(), x0=0.0):
    rgb = np.empty((maze.height, maze.width, 3))
    rgb[:] = to_rgb(st.PANEL)
    rgb[~maze.grid[z]] = to_rgb(st.FLOOR)
    for s in solved:
        x, y, level = maze.xyz[s]
        if level == z:
            rgb[y, x] = to_rgb(st.BLUE_SOFT)
    ax.imshow(rgb, extent=(x0 - 0.5, x0 + maze.width - 0.5, maze.height - 0.5, -0.5), interpolation="nearest",
              zorder=1)
    ax.add_patch(Rectangle((x0 - 0.5, -0.5), maze.width, maze.height, facecolor="none", edgecolor=st.RULE, lw=0.8,
                           zorder=2))
    for x, y in maze.shafts(z):
        ax.add_patch(Rectangle((x0 + x - 0.27, y - 0.27), 0.54, 0.54, facecolor="none", edgecolor=st.INK2, lw=0.6,
                               alpha=0.8, zorder=3))


def draw_states(ax, maze, states, offsets, color, lw, dashed=False, zorder=6):
    runs, _ = maze.segments(states)
    for z, cells in runs:
        if z == 1 or len(cells) < 2:
            continue
        pts = np.array(cells, dtype=float)
        ax.plot(pts[:, 0] + offsets[z], pts[:, 1], color=color, lw=lw, ls=(0, (1.2, 1.6)) if dashed else "-",
                solid_joinstyle="round", solid_capstyle="round", dash_capstyle="round", zorder=zorder)


def draw_marks(ax, maze, offsets, stuck=None):
    sx, sy, sz = maze.xyz[maze.start]
    gx, gy, gz = maze.xyz[maze.goal]
    ax.add_patch(Circle((sx + offsets[sz], sy), 0.38, facecolor=st.BG, edgecolor=st.INK, lw=1.5, zorder=9))
    ax.add_patch(Circle((gx + offsets[gz], gy), 0.4, facecolor=st.INK, edgecolor=st.BG, lw=1.0, zorder=9))
    if stuck is not None:
        x, y, z = maze.xyz[stuck]
        for level in ((0, 2) if z == 1 else (z,)):
            cx, r = x + offsets[level], 0.34
            ax.plot([cx - r, cx + r], [y - r, y + r], color=st.AMBER, lw=1.9, zorder=10)
            ax.plot([cx - r, cx + r], [y + r, y - r], color=st.AMBER, lw=1.9, zorder=10)


def route_coverage(run):
    """For each checkpoint: the share of a BFS shortest route's cells (goal excluded) from which
    the greedy policy reaches the goal."""
    position = {s: i for i, s in enumerate(run["states"])}
    route = run["bfs_route"][:-1]
    episodes, share = [], []
    for c in run["checkpoints"]:
        episodes.append(c["episode"])
        share.append(np.mean([c["solved"][position[s]] == "1" for s in route]))
    return np.array(episodes), np.array(share)


def checkpoint_facts(run, episode):
    """The greedy policy at one checkpoint: its walk from the start, and the cells from which
    it reaches the goal."""
    checkpoint = next(c for c in run["checkpoints"] if c["episode"] == episode)
    solved = {s for s, b in zip(run["states"], checkpoint["solved"]) if b == "1"}
    reachable = len(run["states"]) + 1
    return {"solved": solved, "walk": checkpoint["greedy_path"], "reaches": checkpoint["greedy_reaches"],
            "share": (len(solved) + 1) / reachable, "reachable": reachable}


def maze_row(fig, run, facts, bottom, head, body):
    """One run at one checkpoint: level 0 and level 2 side by side, with a heading and notes."""
    maze = st.Maze(run)
    solved = facts["solved"]
    ax = axes_in(fig, 0.1, bottom, 4.62, 2.32)
    gap = 2.4
    offsets = {0: 0.0, 2: maze.width + gap}
    for z, x0 in offsets.items():
        draw_level(ax, maze, z, solved, x0)
        ax.text(x0 - 0.5, -1.0, f"LEVEL {z}", fontsize=7.8, fontweight=700, color=st.INK, va="bottom")
        ax.text(x0 + maze.width - 0.5, -1.0, "start" if z == 0 else "goal", fontsize=7.8, color=st.INK2, va="bottom",
                ha="right")
    ax.text(maze.width + gap / 2 - 0.5, maze.height / 2 - 0.5, "LEVEL 1 · shafts", rotation=90, ha="center",
            va="center", fontsize=6.6, color=st.INK3)
    draw_states(ax, maze, run["bfs_route"], offsets, st.INK, 1.0, dashed=True, zorder=7)
    walk, reaches = facts["walk"], facts["reaches"]
    draw_states(ax, maze, walk if reaches else walk[:-1], offsets, st.AMBER, 2.4, zorder=6)
    draw_marks(ax, maze, offsets, stuck=None if reaches else walk[-2])
    ax.set_xlim(-0.8, offsets[2] + maze.width - 0.2)
    ax.set_ylim(maze.height - 0.3, -2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    text_in(fig, 0.12, bottom + 2.46, head, fontsize=9.6, fontweight=600, color=st.INK, va="bottom")
    text_in(fig, 0.12, bottom - 0.06, body, fontsize=8.3, color=st.INK2, va="top", linespacing=1.45)


def policy_figure(data, out, route_law=None):
    featured = st.load_json("featured.json", data)
    success, failure = featured["success"], featured["failure"]
    budget = success["episodes"]
    fig = figure(8.0, 7.95)

    s = checkpoint_facts(success, budget)
    if not s["reaches"]:
        raise ValueError("the featured converged run should reach the goal at the end of its budget")
    same = "the same cells as" if s["walk"] == success["bfs_route"] else "as long as"
    maze_row(fig, success, s, 4.55, f"A converged run · maze seed {success['maze_seed']} · after {budget:,} episodes",
             f"Its greedy route takes {len(s['walk']) - 1} moves, {same} a BFS shortest route.\n"
             f"Yet the greedy policy reaches the goal from only {s['share']:.0%} of the {s['reachable']} reachable "
             f"cells:\nit has learned a route, not a map.")
    f = checkpoint_facts(failure, budget)
    if f["reaches"]:
        raise ValueError("the featured unconverged run should not reach the goal at the end of the budget")
    maze_row(fig, failure, f, 1.3, f"Not converged · maze seed {failure['maze_seed']} · after {budget:,} episodes",
             f"Value has spread back from the goal along most of the {failure['shortest_route']}-move route, but "
             f"not to the start:\nthe greedy walk from the start is stuck after {len(f['walk']) - 2} moves. "
             f"Continued to {failure['episodes']:,} episodes,\nthe same run converges at episode "
             f"{failure['converged_episode']:,}.")

    ax = axes_in(fig, 5.42, 4.95, 2.42, 1.95)
    for run, color, style in ((success, st.BLUE, "-"), (failure, st.INK, "-")):
        episodes, share = route_coverage(run)
        ax.step(episodes, share * 100, where="post", color=color, lw=1.5, ls=style,
                label=f"seed {run['maze_seed']}, route {run['shortest_route']} moves")
    ax.axvline(budget, color=st.INK3, lw=1.0, ls=(0, (3, 2)))
    ax.text(budget + 70, 2, f"{budget:,}-episode budget", fontsize=7.2, color=st.INK3, va="bottom")
    ax.set_xlim(0, failure["episodes"])
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xticklabels(["0", "1,000", "2,000", "3,000"])
    percent_axis(ax)
    frame_axes(ax, labelsize=7.8)
    ax.set_xlabel("training episode", fontsize=8.0, color=st.INK2)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=7.2, frameon=False, labelcolor=st.INK2,
              handlelength=1.5)
    text_in(fig, 5.0, 7.36, "Shortest-route cells from which\nthe greedy policy reaches the goal", fontsize=9.0,
            fontweight=600, color=st.INK, va="top", linespacing=1.3)
    law = ""
    if route_law:
        slopes = [v["slope"] for v in route_law.values()]
        law = (f"\nAcross all runs: about {min(slopes):.0f} to {max(slopes):.0f} episodes\nper move of route "
               f"(evaluation suite).")
    text_in(fig, 5.0, 4.38, "One-step updates carry the goal's value\nback along the route a little each episode,\n"
                            "so a longer route needs more episodes." + law, fontsize=8.0, color=st.INK2, va="top",
            linespacing=1.45)

    text_in(fig, 0.1, 7.82, "What the greedy policy knows: a route, learned backwards from the goal", fontsize=13,
            fontweight=600, color=st.INK, va="top")
    handles = [Patch(color=st.BLUE_SOFT, label="greedy policy reaches the goal from this cell"),
               Line2D([], [], color=st.AMBER, lw=2.4, label="greedy walk from the start"),
               Line2D([], [], color=st.AMBER, marker="x", lw=0, ms=6, mew=1.8, label="walk stuck in a loop"),
               Line2D([], [], color=st.INK, lw=1.0, ls=(0, (1.2, 1.6)), label="a BFS shortest route"),
               Line2D([], [], marker="o", lw=0, ms=6.5, mfc=st.BG, mec=st.INK, mew=1.4, label="start"),
               Line2D([], [], marker="o", lw=0, ms=6.5, mfc=st.INK, mec=st.BG, label="goal"),
               Patch(facecolor="none", edgecolor=st.INK2, lw=0.6, label="shaft to another level")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.1 / 8.0, 0.01), ncol=4, fontsize=7.6,
               frameon=False, handlelength=1.8, columnspacing=1.3, labelcolor=st.INK2)
    path = os.path.join(out, "policy.png")
    fig.savefig(path, dpi=DPI)
    close(fig)
    return path


def draw_all(data=st.DATA, out=st.ASSETS, suite=None):
    """Both figures; `suite` is the evaluation suite's summary.json (optional)."""
    st.style()
    results, numbers = results_figure(data, out)
    law = None
    if suite and os.path.exists(suite):
        import json
        with open(suite, encoding="utf-8") as f:
            law = json.load(f).get("route_law")
    policy = policy_figure(data, out, law)
    featured = st.load_json("featured.json", data)
    st.record_render(out, {
        "results.png": {"dpi": DPI, "runs": numbers, "reliable_training": f"{RELIABLE[0]} of the last {RELIABLE[1]} "
                                                                           f"training episodes reach the goal"},
        "policy.png": {"dpi": DPI, "maze_seeds": {name: run["maze_seed"] for name, run in featured.items()},
                       "route_law_from_suite": law is not None},
    })
    return results, policy


def main():
    parser = argparse.ArgumentParser(description="Draw the README figures; see the module docstring.")
    parser.add_argument("--data", default=st.DATA)
    parser.add_argument("--out", default=st.ASSETS)
    parser.add_argument("--suite", default=os.path.join(st.ASSETS, "..", "evaluation", "results", "summary.json"),
                        help="the evaluation suite's summary.json, for its episodes-per-move estimate (optional)")
    args = parser.parse_args()
    for path in draw_all(args.data, args.out, args.suite):
        print("wrote", path)


if __name__ == "__main__":
    main()
