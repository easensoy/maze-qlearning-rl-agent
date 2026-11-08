"""The README's visual analyses, drawn from checkpointed replays (docs/assets/data/analysis/).

Nothing here trains, plans or replays. Every cell, route and number comes from the records
capture_analysis.py wrote, which hold snapshots of published runs and the exact solution of
each maze. experiments.analysis_data derives the plotted quantities from them.

Encodings, the same in every view:
- hollow cell: never visited (all actions untried, V = 0 by initialisation);
- neutral slab: visited;
- blue slab: the frozen greedy policy reaches the goal from this cell;
- pink ramp: |V - V*| on a fixed log scale from 0.01 to 100;
- cyan ramp: visits on a fixed log scale;
- amber: the frozen greedy route from the start, ending in x where it loops;
- dashed off-white: a shortest route (breadth-first search).

Outputs in docs/assets/analysis/:
- comparison.gif and comparison.png;
- value.gif and value.png;
- failure-traces.png and failure-dyna.png;
- figures.json, with every number the visuals annotate.

Run from maze_runner_rl/:
    python -m experiments.render_analysis
"""
import json
import math
import os

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from PIL import Image

from experiments import analysis_data as ad
from experiments import asset_style as st
from experiments import render_hero as rh
from experiments.iso_maze import OUTLINE, TOP, TOP_LEARNED, IsoMaze, slab_fill

OUT = os.path.join(st.ASSETS, "analysis")
W, H = 800, 800
ORIGIN = (352.0, 372.0)
METHODS = ("q_learning", "q_lambda", "dyna_q")
NAMES = {"q_learning": "Q-learning", "q_lambda": "Watkins's Q(λ)", "dyna_q": "Dyna-Q"}
SETTINGS = {"q_learning": "", "q_lambda": "λ = 0.3", "dyna_q": "n = 50"}
COMPARISON_BUDGETS = (10_000, 20_000, 40_000, 80_000, 160_000)
VALUE_BUDGETS = (2_500, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000)
ERROR_RAMP = ("#4a2541", "#73295e", "#a8317f", "#d9489e", "#ff8fcb")     # |V - V*|, log10 from -2 to 2
ERROR_LIMITS = (-2.0, 2.0)
VISIT_RAMP = ("#1d3f47", "#1f6470", "#25899a", "#43b4c4", "#9be8f2")     # visits, log10 from 0 to 5
VISIT_LIMITS = (0.0, 5.0)
FRAME_MS, END_MS = 1100, 4200
REPLAY = "Recorded snapshots of real runs, not the live viewer"


def pt(px, scale=1):
    return px * 0.72 * scale


def ramp(stops, t):
    t = min(1.0, max(0.0, t)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    a = [int(stops[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(stops[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#{:02x}{:02x}{:02x}".format(*(round(x + (y - x) * f) for x, y in zip(a, b)))


def error_colour(err):
    low, high = ERROR_LIMITS
    return ramp(ERROR_RAMP, (math.log10(min(max(abs(err), 10 ** low), 10 ** high)) - low) / (high - low))


def visit_colour(n):
    low, high = VISIT_LIMITS
    return ramp(VISIT_RAMP, (math.log10(max(n, 1)) - low) / (high - low))


def compact(n):
    if n >= 999_500:
        return f"{n / 1e6:.1f}M"
    if n >= 10_000:
        return f"{n / 1e3:.0f}k"
    return f"{n:,}"


# --- a maze drawn from a case -------------------------------------------------------------------

class CaseScene:
    """A case's maze in the hero's exploded isometric view, drawable on any matplotlib axes
    whose units are pixels (y down)."""

    def __init__(self, case, origin=ORIGIN, unit=rh.UNIT, gap=rh.LEVEL_GAP):
        self.case = case
        self.grid = np.array([[[0 if ch == "." else 1 for ch in row] for row in level.split("\n")]
                              for level in case.maze["levels"]])
        self.levels = self.grid.shape[0]
        self.iso = IsoMaze(self.grid, origin, unit, gap, rh.SLAB)
        self.shafts = int((self.grid[1] == 0).sum()) if self.levels == 3 else 0

    def fills(self, snap=None, colour=None):
        """Per-cell fills: hollow for never-visited states and for open cells unreachable from the
        start (behind the absorbing goal), else `colour(state)` or neutral."""
        reachable = {self.case.pos[s] for s in self.case.states}
        out = {(int(x), int(y), int(z)): "outline" for z, y, x in zip(*np.nonzero(self.grid == 0))
               if (int(x), int(y), int(z)) not in reachable}
        visited = set(snap["visited"]) if snap is not None else set(self.case.states)
        for s in self.case.states:
            if s not in visited:
                out[self.case.pos[s]] = "outline"
            elif colour is not None and colour(s) is not None:
                out[self.case.pos[s]] = slab_fill(colour(s))
        return out

    def plates(self, ax, scale=1.0, fills=None, learned=(), level_labels=None):
        iso = self.iso
        iso.scale = scale
        learned = frozenset(self.case.pos[s] for s in learned)
        if self.levels == 3:
            iso.draw_level(ax, 0, learned, zorder=2, fills=fills)
            iso.draw_columns(ax, 0, zorder=10)
            ax.add_patch(Polygon(iso.plate_outline(1), closed=True, facecolor="none", edgecolor=rh.GHOST,
                                 linewidth=pt(1.2, scale), linestyle=(0, (3, 3)), zorder=9))
            iso.draw_level(ax, 1, learned, zorder=12, fills=fills)
            iso.draw_columns(ax, 1, zorder=20)
            iso.draw_level(ax, 2, learned, zorder=22, fills=fills)
        else:
            iso.draw_level(ax, 0, learned, zorder=2, fills=fills)
        positions = []
        for level, label in enumerate(level_labels or ()):
            _, right, _, _ = iso.plate_outline(level)
            positions.append((right[0] + 12, right[1], label))
        return positions

    def path(self, ax, states, colour, width, alpha=1.0, dashes=None, boost=0.0):
        cells = [self.case.pos[s] for s in states]
        self.iso.draw_path(ax, cells, colour, width=width, alpha=alpha, dashes=dashes,
                           level_zorder={k: v + boost for k, v in rh.LEVEL_Z.items()})

    def point(self, state, lift=1.5):
        return self.iso.centre(*self.case.pos[state], lift=lift)

    def markers(self, ax, scale=1.0, radius=7.5):
        for state, fill in ((self.case.start, "none"), (self.case.goal, st.INK)):
            cx, cy = self.point(state)
            ax.add_patch(Circle((cx, cy), radius, facecolor=fill, edgecolor=st.INK, linewidth=pt(2.6, scale),
                                zorder=45))


def level_names(scene):
    if scene.levels == 3:
        return ["Level 0 · start", f"Level 1 · {scene.shafts} shafts", "Level 2 · goal"]
    return None


# --- animation frames ---------------------------------------------------------------------------

class Frames:
    """Draws animation frames on an 800 x 800 canvas, caching each background (the plates)."""

    def __init__(self, scene):
        self.scene = scene
        self.cache = {}

    def background(self, cv, key, fills, learned):
        if (cv.scale, key) not in self.cache:
            cv.clear()
            texts = list(self.scene.plates(cv.ax, cv.scale, fills, learned, level_names(self.scene)) or [])
            for x, y, label in texts:
                cv.fit(x, y, label, W - 12 - x, size=rh.LABEL_SIZE, color=st.INK, weight="bold", va="center")
            self.cache[(cv.scale, key)] = cv.image()
        return self.cache[(cv.scale, key)]

    def draw(self, cv, f):
        image = self.background(cv, f["key"], f.get("fills"), f.get("learned", ()))
        cv.clear()
        cv.ax.imshow(image, extent=(0, W, H, 0), interpolation="none", zorder=1)
        scene = self.scene
        if f.get("bfs", True):
            scene.path(cv.ax, scene.case.bfs, st.INK, pt(2.2, cv.scale), alpha=0.95, dashes=(3, 2.5), boost=1)
        if f.get("route"):
            scene.path(cv.ax, f["route"], st.AMBER, pt(4.0, cv.scale), boost=3)
            if f.get("loop"):
                x, y = scene.point(f["route"][-1], lift=2.5)
                cv.text(x, y, "×", size=26, color=st.AMBER, weight="bold", ha="center", va="center", zorder=48)
        scene.markers(cv.ax, cv.scale)
        for state, word in ((scene.case.start, "start"), (scene.case.goal, "goal")):
            x, y = scene.point(state)
            cv.text(x + 14, y + 4, word, size=rh.LABEL_SIZE, color=st.INK, weight="bold", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=st.BG, edgecolor="none", alpha=0.85))
        cv.fit(24, 42, f["title"], W - 48, size=28, weight="bold")
        if f.get("subtitle"):
            cv.fit(24, 76, f["subtitle"], W - 48, size=20, color=st.INK2)
        if f.get("panel"):
            f["panel"](cv)
        legend(cv, f.get("legend", ()), y=748)
        cv.ax.plot([24, W - 24], [766, 766], color=st.RULE, linewidth=pt(1, cv.scale), zorder=50)
        cv.fit(24, 790, f["footer"], W - 48, size=rh.NOTE_SIZE, color=st.INK2)
        if f.get("endcard"):
            cv.ax.add_patch(Rectangle((0, 0), W, H, facecolor=st.BG, edgecolor="none", alpha=0.9, zorder=60))
            y = 300
            for i, (text, size, colour, weight) in enumerate(f["endcard"]):
                cv.fit(W / 2, y, text, W - 64, size=size, color=colour, weight=weight, ha="center", zorder=61)
                if i + 1 < len(f["endcard"]):
                    y += 0.5 * size + 0.9 * f["endcard"][i + 1][1] + 14


def legend(cv, items, y):
    x = 24
    for kind, colour, label in items:
        if kind == "line":
            cv.ax.plot([x, x + 22], [y, y], color=colour, linewidth=pt(3.5, cv.scale), solid_capstyle="round",
                       zorder=50)
        elif kind == "dash":
            line, = cv.ax.plot([x, x + 22], [y, y], color=colour, linewidth=pt(2.4, cv.scale), zorder=50)
            line.set_dashes((3, 2.5))
        elif kind == "hollow":
            cv.ax.add_patch(Polygon([(x + 11, y - 7), (x + 22, y), (x + 11, y + 7), (x, y)], closed=True,
                                    facecolor="none", edgecolor=OUTLINE, linewidth=pt(1.4, cv.scale), zorder=50))
        else:
            cv.ax.add_patch(Polygon([(x + 11, y - 7), (x + 22, y), (x + 11, y + 7), (x, y)], closed=True,
                                    facecolor=colour, edgecolor="none", zorder=50))
        t = cv.text(x + 30, y, label, size=rh.LABEL_SIZE, color=st.INK2, va="center")
        x += 30 + t.get_window_extent(renderer=cv.fig.canvas.get_renderer()).width / cv.scale + 22


def colourbar(cv, x, y, width, stops, ticks, label, height=12):
    """A horizontal ramp with tick labels; `ticks` are (position 0-1, text)."""
    n = 60
    for i in range(n):
        cv.ax.add_patch(Rectangle((x + width * i / n, y), width / n + 0.5, height, facecolor=ramp(stops, i / (n - 1)),
                                  edgecolor="none", zorder=50))
    for p, text in ticks:
        cv.ax.plot([x + width * p] * 2, [y + height, y + height + 4], color=st.INK2, linewidth=pt(1, cv.scale),
                   zorder=50)
        cv.text(x + width * p, y + height + 18, text, size=13, color=st.INK2, ha="center", va="baseline")
    cv.text(x, y - 8, label, size=14, color=st.INK, va="baseline")


def save_animation(frames_images, durations, name, poster_index=-1):
    def quantize(images):
        picks = np.linspace(0, len(images) - 1, min(len(images), 16)).astype(int)
        sample = Image.fromarray(np.concatenate([images[i] for i in picks], axis=0))
        palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        return [Image.fromarray(im).quantize(palette=palette, dither=Image.Dither.NONE) for im in images]

    gif = os.path.join(OUT, f"{name}.gif")
    pal = quantize(frames_images)
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=False)
    return gif


# --- 1. matched-interaction comparison ------------------------------------------------------------

def comparison_rows(case, budget):
    rows = {}
    for m in METHODS:
        snap = next(s for s in case.runs[m]["snapshots"] if s.get("budget") == budget)
        walk, ok = ad.greedy_walk(case, snap)
        region = ad.reach_region(case, snap)
        rows[m] = {"snap": snap, "walk": walk, "reaches": ok, "verdict": ad.route_verdict(case, walk, ok),
                   "region": region, "visited": len(set(snap["visited"]) - {case.goal}),
                   "unvisited_in_region": len(region - set(snap["visited"])),
                   "episode": snap["episode"], "env_steps": snap["env_steps"], "updates": snap["updates"]}
    return rows


def verdict_text(row, case):
    if row["verdict"] == "loops":
        return "loops before the goal"
    moves = len(row["walk"]) - 1
    return f"shortest, {moves} moves" if row["verdict"] == "shortest" else f"reaches the goal, {moves} moves"


def method_label(m):
    return NAMES[m] + (f", {SETTINGS[m]}" if SETTINGS[m] else "")


def comparison(case, summary):
    scene = CaseScene(case)
    frames = Frames(scene)
    cells = len(case.states) - 1
    specs = []
    for budget in COMPARISON_BUDGETS:
        rows = comparison_rows(case, budget)
        summary.setdefault("comparison", {})[str(budget)] = {
            m: {"episode": r["episode"], "env_steps": r["env_steps"], "updates": r["updates"], "verdict": r["verdict"],
                "route_moves": len(r["walk"]) - 1 if r["reaches"] else None, "region": len(r["region"]),
                "region_share": len(r["region"]) / cells, "visited": r["visited"],
                "unvisited_in_region": r["unvisited_in_region"]} for m, r in rows.items()}
        for m in METHODS:
            r = rows[m]

            def panel(cv, rows=rows, current=m, budget=budget):
                xs = (24, 214, 448, 664)
                heads = (f"at {compact(budget)} real steps", "greedy route from start", "greedy policy reaches goal from",
                         "Q-value updates")
                for x, h in zip(xs, heads):
                    cv.text(x, 646, h, size=13, color=st.INK2)
                for i, mm in enumerate(METHODS):
                    rr, y = rows[mm], 672 + 24 * i
                    style = dict(color=st.INK if mm == current else st.INK2, weight="bold" if mm == current else "normal")
                    cv.fit(xs[0], y, method_label(mm), xs[1] - xs[0] - 10, size=15, **style)
                    cv.fit(xs[1], y, verdict_text(rr, case), xs[2] - xs[1] - 10, size=15, **style)
                    cv.text(xs[2], y, f"{len(rr['region'])} of {cells} cells ({len(rr['region']) / cells:.0%})", size=15,
                            **style)
                    cv.text(xs[3], y, compact(rr["updates"]), size=15, **style)

            specs.append(dict(
                key=("cmp", m, budget), fills=scene.fills(r["snap"]), learned=r["region"],
                route=r["walk"], loop=not r["reaches"],
                title=f"{NAMES[m]} after {compact(budget)} real steps",
                subtitle=f"Episode {r['episode']} · greedy route: {verdict_text(r, case)}",
                panel=panel, duration=FRAME_MS,
                legend=(("tile", TOP_LEARNED, "greedy policy reaches goal"), ("tile", TOP, "visited"),
                        ("hollow", None, "never visited"), ("line", st.AMBER, "greedy route"),
                        ("dash", st.INK, "a shortest route")),
                footer=f"{REPLAY} · same maze and exploration seed for all three · one frame per method and budget"))
        specs[-1]["duration"] = 1800
    last = summary["comparison"][str(COMPARISON_BUDGETS[-1])]
    specs.append(dict(specs[-1], duration=END_MS, endcard=[
        (f"After {compact(COMPARISON_BUDGETS[-1])} real steps on this maze", 18, st.INK2, "normal"),
        (f"greedy policy reaches the goal from {last['q_learning']['region_share']:.0%} · "
         f"{last['q_lambda']['region_share']:.0%} · {last['dyna_q']['region_share']:.0%} of cells", 22, st.INK, "bold"),
        ("Q-learning · Q(λ) · Dyna-Q", 16, st.INK2, "normal"),
        (f"Q-value updates: {compact(last['q_learning']['updates'])} · {compact(last['q_lambda']['updates'])} · "
         f"{compact(last['dyna_q']['updates'])}", 16, st.INK2, "normal"),
    ]))
    cv = rh.Canvas(width=W, height=H)
    images = []
    for spec in specs:
        frames.draw(cv, spec)
        images.append(cv.image())
    cv.close()
    gif = save_animation(images, [s["duration"] for s in specs], "comparison")
    return gif, len(images), sum(s["duration"] for s in specs)


def comparison_static(case, summary, budget=40_000):
    """Three learners side by side at one matched budget, and the reach region over all budgets."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10.2), dpi=100)
    fig.patch.set_facecolor(st.BG)
    rows = comparison_rows(case, budget)
    cells = len(case.states) - 1
    fig.text(0.02, 0.962, f"Matched real interactions: the same maze after {compact(budget)} real environment steps",
             color=st.INK, fontsize=21, fontweight="bold", family=st.FONT_FAMILY)
    fig.text(0.02, 0.93, f"Three-level suite maze {case.maze['maze_seed']} (shortest route {case.route_length} moves); "
                         "same exploration seed; snapshot at the first episode end at or past the budget",
             color=st.INK2, fontsize=13.5, family=st.FONT_FAMILY)
    for i, m in enumerate(METHODS):
        r = rows[m]
        ax = fig.add_axes([0.005 + i * 0.333, 0.345, 0.323, 0.53])
        pw, ph = 540, 590
        ax.set_xlim(0, pw)
        ax.set_ylim(ph, 0)
        ax.axis("off")
        scene = CaseScene(case, origin=(pw / 2, 300), unit=11.0, gap=118.0)
        list(scene.plates(ax, 1.3, scene.fills(r["snap"]), r["region"]) or [])
        scene.path(ax, case.bfs, st.INK, pt(2.4), alpha=0.95, dashes=(3, 2.5), boost=1)
        scene.path(ax, r["walk"], st.AMBER, pt(4.4), boost=3)
        if not r["reaches"]:
            x, y = scene.point(r["walk"][-1], lift=2.5)
            ax.text(x, y, "×", fontsize=22, color=st.AMBER, fontweight="bold", ha="center", va="center", zorder=48)
        scene.markers(ax, 1.3, radius=8)
        fig.text(0.02 + i * 0.333, 0.888, method_label(m), color=st.INK, fontsize=17, fontweight="bold",
                 family=st.FONT_FAMILY)
        fig.text(0.02 + i * 0.333, 0.863, f"episode {r['episode']} · greedy route: {verdict_text(r, case)}",
                 color=st.INK2, fontsize=12.5, family=st.FONT_FAMILY)
        fig.text(0.02 + i * 0.333, 0.84, f"reaches goal from {len(r['region'])} of {cells} cells · "
                                          f"{compact(r['updates'])} Q-value updates", color=st.INK2, fontsize=12.5,
                 family=st.FONT_FAMILY)
    # Reach region against real steps, every captured budget.
    ax = fig.add_axes([0.08, 0.085, 0.5, 0.24])
    colours = {"q_learning": "#9085e9", "q_lambda": "#d95926", "dyna_q": "#199e70"}
    for m in METHODS:
        xs, ys, kinds = [], [], []
        for snap in case.runs[m]["snapshots"]:
            walk, ok = ad.greedy_walk(case, snap)
            xs.append(snap["env_steps"])
            ys.append(len(ad.reach_region(case, snap)) / cells)
            kinds.append(ad.route_verdict(case, walk, ok))
        ax.plot(xs, ys, color=colours[m], linewidth=2.2, zorder=3)
        for x, y, k in zip(xs, ys, kinds):
            if k == "shortest":
                ax.plot([x], [y], "o", color=colours[m], markersize=9, markeredgecolor=st.BG, zorder=4)
            elif k == "reaches":
                ax.plot([x], [y], "o", color=st.BG, markersize=9, markeredgecolor=colours[m], markeredgewidth=2, zorder=4)
            else:
                ax.plot([x], [y], "x", color=colours[m], markersize=8, markeredgewidth=2, zorder=4)
        dy = {"q_learning": -0.045, "q_lambda": 0.045, "dyna_q": 0.0}[m]
        ax.text(xs[-1] * 1.15, ys[-1] + dy, NAMES[m], color=colours[m], fontsize=13, fontweight="bold", va="center",
                family=st.FONT_FAMILY)
    ax.axvline(budget, color=st.INK3, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(2000, 500_000)
    ax.set_ylim(-0.04, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks([2500, 10_000, 40_000, 160_000])
    ax.set_xticklabels(["2.5k", "10k", "40k", "160k"])
    ax.tick_params(labelsize=12.5, colors=st.INK2)
    ax.set_xlabel("real environment steps (log scale)", color=st.INK2, fontsize=13)
    ax.set_ylabel("non-goal cells from which the\ngreedy policy reaches the goal", color=st.INK2, fontsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(0.66, 0.3, "Route at each snapshot", color=st.INK, fontsize=13.5, fontweight="bold", family=st.FONT_FAMILY)
    for j, (marker, text) in enumerate((("x", "loops before the goal"), ("o-hollow", "reaches the goal"),
                                         ("o", "shortest route"))):
        y = 0.265 - j * 0.032
        mk = fig.add_axes([0.66, y - 0.008, 0.012, 0.016])
        mk.axis("off")
        if marker == "x":
            mk.plot([0], [0], "x", color=st.INK, markersize=8, markeredgewidth=2)
        elif marker == "o":
            mk.plot([0], [0], "o", color=st.INK, markersize=9)
        else:
            mk.plot([0], [0], "o", color=st.BG, markersize=9, markeredgecolor=st.INK, markeredgewidth=2)
        fig.text(0.68, y, text, color=st.INK2, fontsize=13, va="center", family=st.FONT_FAMILY)
    fig.text(0.66, 0.16, "Maps: blue, greedy policy reaches the goal;\ngrey, visited; hollow, never visited;\n"
                         "amber, greedy route (× where it loops);\ndashed, a shortest route", color=st.INK2,
             fontsize=13, va="top", family=st.FONT_FAMILY, linespacing=1.5)
    path = os.path.join(OUT, "comparison.png")
    fig.savefig(path, dpi=100, facecolor=st.BG)
    plt.close(fig)
    return path


# --- 2. value error and Bellman residual ----------------------------------------------------------

METHOD_COLOURS = {"q_learning": "#9085e9", "q_lambda": "#d95926", "dyna_q": "#199e70"}
PROFILE_BUDGETS = ((10_000, 0.38), (40_000, 0.68), (160_000, 1.0))


def budget_snapshot(case, method, budget):
    return next(s for s in case.runs[method]["snapshots"] if s.get("budget") == budget)


def value_stats(case, snap):
    err, rho = ad.value_error(case, snap), ad.bellman_residual(case, snap)
    visited = sorted(set(snap["visited"]) - {case.goal})
    e = np.array([abs(err[s]) for s in visited])
    r = np.array([rho[s] for s in visited])
    return {"visited": len(visited), "median_abs_error": float(np.median(e)), "max_abs_error": float(e.max()),
            "share_below_0.01": float(np.mean(e < 0.01)), "max_residual": float(r.max()),
            "min_residual": float(r.min())}


def error_panel(cv):
    colourbar(cv, 24, 662, 400, ERROR_RAMP, [(0, "0.01"), (0.25, "0.1"), (0.5, "1"), (0.75, "10"), (1, "100")],
              "|V(s) − V*(s)| on visited states (log scale)")
    x = 470
    cv.ax.add_patch(Polygon([(x + 11, 666), (x + 22, 673), (x + 11, 680), (x, 673)], closed=True, facecolor="none",
                            edgecolor=OUTLINE, linewidth=pt(1.4, cv.scale), zorder=50))
    cv.text(x + 30, 673, "never visited (V = 0, untried)", size=rh.LABEL_SIZE, color=st.INK2, va="center")


def value_animation(case, summary):
    scene = CaseScene(case)
    frames = Frames(scene)
    specs = []
    for budget in VALUE_BUDGETS:
        for m in METHODS:
            snap = budget_snapshot(case, m, budget)
            err = ad.value_error(case, snap)
            stats = value_stats(case, snap)
            summary.setdefault("value", {}).setdefault(str(budget), {})[m] = {
                **stats, "episode": snap["episode"], "env_steps": snap["env_steps"]}
            specs.append(dict(
                key=("val", m, budget),
                fills=scene.fills(snap, colour=lambda s, err=err: error_colour(err[s]) if s in err else None),
                title=f"{NAMES[m]}: value error after {compact(budget)} real steps",
                subtitle=f"{stats['visited']} states visited · median |V − V*| over them {stats['median_abs_error']:.2f}",
                panel=error_panel, duration=900, legend=(("dash", st.INK, "a shortest route"),),
                footer=f"{REPLAY} · one colour scale for every method and budget"))
        specs[-1]["duration"] = 1500
    at = summary["value"]["40000"]
    specs.append(dict(specs[-1], duration=END_MS, endcard=[
        ("Median |V − V*| over visited states after 40k real steps", 18, st.INK2, "normal"),
        (f"Q-learning {at['q_learning']['median_abs_error']:.2f} · Q(λ) {at['q_lambda']['median_abs_error']:.2f} · "
         f"Dyna-Q {at['dyna_q']['median_abs_error']:.2f}", 26, st.INK, "bold"),
        (f"Dyna-Q: |V − V*| < 0.01 on {at['dyna_q']['share_below_0.01']:.0%} of its visited states", 16, st.INK2,
         "normal"),
        ("V* by value iteration on the known model · V = max over actions, untried actions at 0", 16, st.INK2,
         "normal"),
    ]))
    cv = rh.Canvas(width=W, height=H)
    images = []
    for spec in specs:
        frames.draw(cv, spec)
        images.append(cv.image())
    cv.close()
    gif = save_animation(images, [s["duration"] for s in specs], "value")
    return gif, len(images), sum(s["duration"] for s in specs)


def mix(colour, weight):
    """`colour` blended with the page background; weight 1 is the colour itself."""
    a = [int(colour[k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(st.BG[k:k + 2], 16) for k in (1, 3, 5)]
    return "#{:02x}{:02x}{:02x}".format(*(round(y + (x - y) * weight) for x, y in zip(a, b)))


def value_static(case, summary, budget=40_000):
    """Where error remains (maps at one budget), and how error and Bellman residual are distributed
    over graph distance from the goal (profiles at three budgets)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig = plt.figure(figsize=(16, 15.4), dpi=100)
    fig.patch.set_facecolor(st.BG)
    fig.text(0.02, 0.973, "Value error and Bellman residual: where the learned values still differ from the optimum",
             color=st.INK, fontsize=21, fontweight="bold", family=st.FONT_FAMILY)
    fig.text(0.02, 0.953, f"Three-level suite maze {case.maze['maze_seed']}; maps after {compact(budget)} real steps; "
                          "profiles over visited states after", color=st.INK2, fontsize=13.5, family=st.FONT_FAMILY)
    for j, (b, weight) in enumerate(PROFILE_BUDGETS):
        x = 0.545 + j * 0.075
        fig.text(x, 0.953, "●", color=mix(st.INK, weight), fontsize=15, family=st.FONT_FAMILY)
        fig.text(x + 0.013, 0.953, f"{compact(b)} steps", color=st.INK2, fontsize=13.5, family=st.FONT_FAMILY)
    dmax = max(d for d in case.distance.values() if d is not None)
    for i, m in enumerate(METHODS):
        snap = budget_snapshot(case, m, budget)
        err = ad.value_error(case, snap)
        stats = value_stats(case, snap)
        ax = fig.add_axes([0.005 + i * 0.333, 0.62, 0.323, 0.29])
        pw, ph = 540, 520
        ax.set_xlim(0, pw)
        ax.set_ylim(ph, 0)
        ax.axis("off")
        scene = CaseScene(case, origin=(pw / 2, 262), unit=10.5, gap=106.0)
        list(scene.plates(ax, 1.3, scene.fills(snap, colour=lambda s, err=err: error_colour(err[s])
                                               if s in err else None)) or [])
        scene.path(ax, case.bfs, st.INK, pt(2.2), alpha=0.9, dashes=(3, 2.5), boost=1)
        scene.markers(ax, 1.3, radius=7)
        fig.text(0.02 + i * 0.333, 0.927, method_label(m), color=st.INK, fontsize=17, fontweight="bold",
                 family=st.FONT_FAMILY)
        fig.text(0.02 + i * 0.333, 0.909, f"{stats['visited']} states visited · median |V − V*| "
                                           f"{stats['median_abs_error']:.2f}", color=st.INK2, fontsize=12.5,
                 family=st.FONT_FAMILY)
    cax = fig.add_axes([0.08, 0.6, 0.34, 0.011])
    cmap = ListedColormap([ramp(ERROR_RAMP, t) for t in np.linspace(0, 1, 256)])
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap, extent=(-2, 2, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-2, -1, 0, 1, 2])
    cax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    cax.tick_params(labelsize=12, colors=st.INK2)
    for spine in cax.spines.values():
        spine.set_visible(False)
    fig.text(0.08, 0.617, "|V(s) − V*(s)| on visited states (fixed log scale)", color=st.INK, fontsize=13,
             family=st.FONT_FAMILY)
    fig.text(0.47, 0.602, "hollow: never visited (V = 0, all actions untried)", color=st.INK2, fontsize=13,
             family=st.FONT_FAMILY)
    v_star_by_d = {}
    for s, d in case.distance.items():
        if d:
            v_star_by_d.setdefault(d, case.v_star[s])
    ds = sorted(v_star_by_d)
    rows = (("signed error  e(s) = V(s) − V*(s)", lambda snap: ad.value_error(case, snap), 0.345, True),
            ("Bellman residual  ρ(s)", lambda snap: ad.bellman_residual(case, snap), 0.07, False))
    for title, quantity, bottom, reference in rows:
        for i, m in enumerate(METHODS):
            ax = fig.add_axes([0.07 + i * 0.315, bottom, 0.27, 0.18])
            if reference:
                ax.plot(ds, [-v_star_by_d[d] for d in ds], color=st.INK3, linewidth=1.3, linestyle=(0, (4, 3)), zorder=1)
            for b, weight in PROFILE_BUDGETS:
                snap = budget_snapshot(case, m, b)
                q = quantity(snap)
                visited = set(snap["visited"]) - {case.goal}
                ax.scatter([case.distance[s] for s in visited], [q[s] for s in visited], s=8,
                           color=mix(METHOD_COLOURS[m], weight), linewidths=0, zorder=2 + weight)
            ax.axhline(0, color=st.INK3, linewidth=1, zorder=1)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_ylim(-150, 150)
            ax.set_yticks([-100, -1, -0.1, 0, 0.1, 1, 100])
            ax.set_yticklabels(["−100", "−1", "−0.1", "0", "0.1", "1", "100"])
            ax.set_xlim(0, dmax + 2)
            ax.tick_params(labelsize=11.5, colors=st.INK2)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if i == 0:
                ax.set_ylabel(title, color=st.INK, fontsize=13)
            ax.set_xlabel("graph distance to the goal, d(s)", color=st.INK2, fontsize=12)
            ax.set_title(NAMES[m], color=METHOD_COLOURS[m], fontsize=13.5, fontweight="bold", loc="left")
    fig.text(0.07, 0.56, "Below 0: the goal's value has not reached s. Above 0: the zero initialisation is not yet "
                         "deflated (V* < 0 beyond 77 moves). Dashed: the error of a state with V = 0.", color=st.INK2,
             fontsize=12.5, family=st.FONT_FAMILY)
    fig.text(0.07, 0.285, "ρ(s) = max_a [r + γ(1 − done) V(s′)] − V(s): 0 where V is locally consistent; > 0 where a "
                          "successor's value is not backed up into s; < 0 where an untried action holds V(s) at 0.",
             color=st.INK2,
             fontsize=12.5, family=st.FONT_FAMILY)
    path = os.path.join(OUT, "value.png")
    fig.savefig(path, dpi=100, facecolor=st.BG)
    plt.close(fig)
    return path


# --- 4. failure cases ---------------------------------------------------------------------------------

def frontier_cells(case, snap):
    """Visited non-goal states with an untried action whose successor has never been visited."""
    q, visited = case.q_rows(snap), set(snap["visited"])
    return [s for s in visited if s != case.goal and q[s] is not None
            and any(v is None and case.nxt[s][a] not in visited for a, v in enumerate(q[s]))]


def map_axes(fig, rect, pw, ph):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, pw)
    ax.set_ylim(ph, 0)
    ax.axis("off")
    return ax


def ring(ax, scene, state, colour=st.INK, radius=10, width=2.2):
    x, y = scene.point(state, lift=2.0)
    ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor=colour, linewidth=width, zorder=49))
    return x, y


def failure_traces(case, diagnostic, summary):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10.4), dpi=100)
    fig.patch.set_facecolor(st.BG)
    fig.text(0.02, 0.963, "Failure A · Watkins's Q(λ), λ = 0.9: state coverage stops inside the first episode",
             color=st.INK, fontsize=21, fontweight="bold", family=st.FONT_FAMILY)
    fig.text(0.02, 0.935, f"Development 2D maze {case.maze['maze_seed']} (shortest route {case.route_length} moves), "
                          "same exploration seed; cells coloured by visits; hollow, never visited; amber, greedy route; "
                          "ring, frontier cell", color=st.INK2, fontsize=13, family=st.FONT_FAMILY)
    reachable = len(case.states)
    info = {}
    labels = {"q_learning": "Q-learning", "q_lambda": "Q(λ), λ = 0.9"}
    for row, m in enumerate(("q_learning", "q_lambda")):
        run = case.runs[m]
        for col, label in enumerate(("step 1,000", "50,000 steps")):
            snap = case.snapshot(m, label)
            visits = dict(zip(case.states, snap["visits"]))
            visited = set(snap["visited"])
            ax = map_axes(fig, [0.01 + col * 0.285, 0.5 - row * 0.4, 0.28, 0.34], 440, 300)
            scene = CaseScene(case, origin=(220, 52), unit=10.0)
            fills = scene.fills(snap, colour=lambda s, v=visits: visit_colour(v[s]) if v.get(s) else TOP)
            list(scene.plates(ax, 1.2, fills) or [])
            walk, ok = ad.greedy_walk(case, snap)
            scene.path(ax, case.bfs, st.INK, pt(2.0), alpha=0.9, dashes=(3, 2.5), boost=1)
            scene.path(ax, walk, st.AMBER, pt(3.6), boost=3)
            if not ok:
                x, y = scene.point(walk[-1], lift=2.5)
                ax.text(x, y, "×", fontsize=18, color=st.AMBER, fontweight="bold", ha="center", va="center", zorder=48)
            frontier = frontier_cells(case, snap)
            for s in frontier:
                ring(ax, scene, s, radius=9)
            scene.markers(ax, 1.2, radius=6.5)
            steps = snap["env_steps"]
            fig.text(0.02 + col * 0.285, 0.878 - row * 0.4, f"{labels[m]} · {steps:,} real steps", color=st.INK,
                     fontsize=14, fontweight="bold", family=st.FONT_FAMILY)
            note = f"{len(visited)} of {reachable} cells visited · {len(frontier)} frontier cell" + \
                   ("s" if len(frontier) != 1 else "")
            if m == "q_lambda" and col == 1 and len(frontier) == 1:
                last_visit = run["last_visit_step"].get(str(frontier[0]))
                note += f", last visited at step {last_visit:,} of {run['record']['env_steps']:,}"
            fig.text(0.02 + col * 0.285, 0.856 - row * 0.4, note, color=st.INK2, fontsize=12.5, family=st.FONT_FAMILY)
            info.setdefault(m, {})[label] = {"env_steps": steps, "visited": len(visited), "frontier": frontier,
                                             "greedy_reaches": ok}
        if m == "q_lambda":
            last = case.snapshot(m, "50,000 steps")
            only = frontier_cells(case, last)
            info[m]["frontier_last_visit_step"] = {str(s): run["last_visit_step"].get(str(s)) for s in only}
            info[m]["frontier_q"] = {str(s): case.q_rows(last)[s] for s in only}
    # Coverage against real steps, exact (every discovery step is recorded).
    ax = fig.add_axes([0.62, 0.56, 0.35, 0.3])
    for m, colour in (("q_learning", METHOD_COLOURS["q_learning"]), ("q_lambda", METHOD_COLOURS["q_lambda"])):
        steps = sorted(case.runs[m]["discovered_at_step"].values())
        xs = [max(t, 1) for t in steps] + [case.runs[m]["record"]["env_steps"]]
        ys = list(range(1, len(steps) + 1)) + [len(steps)]
        ax.step(xs, [y / reachable for y in ys], where="post", color=colour, linewidth=2.2)
        ax.text(xs[-1] * 0.8, ys[-1] / reachable + 0.05, labels[m], color=colour, fontsize=12.5, fontweight="bold",
                ha="right", family=st.FONT_FAMILY)
        info[m]["discovered_total"] = len(steps)
        info[m]["last_discovery_step"] = steps[-1]
    ax.set_xscale("log")
    ax.set_xlim(10, 1.2e6)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.tick_params(labelsize=12, colors=st.INK2)
    ax.set_xlabel("real environment steps (log scale), the whole 1,000-episode run", color=st.INK2, fontsize=12.5)
    ax.set_ylabel("reachable cells visited", color=st.INK2, fontsize=12.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Observed: coverage against real steps", color=st.INK, fontsize=14, fontweight="bold", loc="left")
    # The intervention, on the 60 development mazes.
    ax = fig.add_axes([0.62, 0.1, 0.27, 0.3])
    rows = (("q_lambda", "Q(λ), λ = 0.9 (accumulating), unmodified", METHOD_COLOURS["q_lambda"]),
            ("q_lambda_cut_untried", "traces also cut at untried next actions", st.INK),
            ("q_learning", "Q-learning", METHOD_COLOURS["q_learning"]))
    for k, (variant, name, colour) in enumerate(rows):
        shares = [r["share"] for r in diagnostic["runs"] if r["variant"] == variant]
        y = 2 - k
        jitter = np.linspace(-0.18, 0.18, len(shares))
        ax.scatter(sorted(shares), y + jitter, s=14, color=mix(colour, 0.75), linewidths=0, zorder=2)
        s = diagnostic["summary"][variant]
        ax.plot(s["quartiles"], [y, y], color=colour, linewidth=5, solid_capstyle="butt", zorder=3, alpha=0.9)
        ax.plot([s["median_share"]] * 2, [y - 0.3, y + 0.3], color=colour, linewidth=3, zorder=4)
        ax.text(1.04, y, f"median {s['median_share']:.0%}\ngoal reached {s['reached_goal']}/{s['runs']}",
                color=st.INK2, fontsize=12, va="center", transform=ax.get_yaxis_transform(), family=st.FONT_FAMILY,
                linespacing=1.3)
        ax.text(0.0, y + 0.42, name, color=colour if colour != st.INK else st.INK, fontsize=12.5, fontweight="bold",
                family=st.FONT_FAMILY)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.75)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(labelsize=12, colors=st.INK2)
    ax.set_xlabel("cells visited after 50k real steps, 60 development runs\n(dots: runs; bar: IQR; tick: median)",
                  color=st.INK2, fontsize=12)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Tested: one-rule intervention, specified before it was run", color=st.INK, fontsize=14,
                 fontweight="bold", loc="left")
    # Colour scale for visits.
    cax = fig.add_axes([0.06, 0.035, 0.25, 0.011])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([ramp(VISIT_RAMP, t) for t in np.linspace(0, 1, 256)])
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap, extent=(0, 5, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([0, 1, 2, 3, 4, 5])
    cax.set_xticklabels(["1", "10", "100", "1k", "10k", "100k"])
    cax.tick_params(labelsize=11.5, colors=st.INK2)
    for spine in cax.spines.values():
        spine.set_visible(False)
    fig.text(0.06, 0.052, "visits per cell (log scale)", color=st.INK2, fontsize=12.5, family=st.FONT_FAMILY)
    path = os.path.join(OUT, "failure-traces.png")
    fig.savefig(path, dpi=100, facecolor=st.BG)
    plt.close(fig)
    summary["failure_traces"] = info
    summary["failure_traces"]["intervention"] = diagnostic["summary"]
    return path


def junction(case, snap):
    """Where the final greedy route leaves every shortest route, and the first state on a shortest
    continuation from there whose goal-ward action is untried (as in plot_value_propagation)."""
    greedy, q = case.greedy(snap), case.q_rows(snap)
    state = case.start
    while case.distance.get(case.nxt[state][greedy[state]]) == case.distance[state] - 1:
        state = case.nxt[state][greedy[state]]
    leave, cell, untried, continuation = state, state, None, []
    while cell != case.goal:
        step = next(a for a, t in enumerate(case.nxt[cell]) if case.distance.get(t) == case.distance[cell] - 1)
        if untried is None and q[cell][step] is None:
            untried = (cell, step)
        cell = case.nxt[cell][step]
        continuation.append(cell)
    return leave, untried, continuation


def failure_dyna(case, summary):
    import matplotlib.pyplot as plt

    run = case.runs["dyna_q"]
    final = case.snapshot("dyna_q", "episode 1000")
    leave, untried, continuation = junction(case, final)
    fig = plt.figure(figsize=(16, 10.4), dpi=100)
    fig.patch.set_facecolor(st.BG)
    walk, ok = ad.greedy_walk(case, final)
    fig.text(0.02, 0.963, f"Failure B · Dyna-Q stabilises on a {len(walk) - 1}-move route on three-level maze "
                          f"{case.maze['maze_seed']} (optimum {case.route_length})", color=st.INK, fontsize=21,
             fontweight="bold", family=st.FONT_FAMILY)
    fig.text(0.02, 0.935, "Dyna-Q after 1,000 episodes: hollow, never visited; amber, greedy route; dashed, a shortest "
                          "route; rings mark where the route diverges", color=st.INK2, fontsize=13, family=st.FONT_FAMILY)
    ax = map_axes(fig, [0.0, 0.03, 0.56, 0.88], 760, 820)
    scene = CaseScene(case, origin=(370, 405), unit=13.0, gap=150.0)
    texts = list(scene.plates(ax, 1.4, scene.fills(final), (), level_names(scene)) or [])
    for x, y, label in texts:
        ax.text(x, y, label, color=st.INK, fontsize=13, fontweight="bold", va="center", family=st.FONT_FAMILY)
    scene.path(ax, case.bfs, st.INK, pt(2.4), alpha=0.95, dashes=(3, 2.5), boost=1)
    scene.path(ax, walk, st.AMBER, pt(4.4), boost=3)
    scene.markers(ax, 1.4, radius=8)
    x1, y1 = ring(ax, scene, leave, radius=13, width=2.6)
    cell, action = untried
    x2, y2 = ring(ax, scene, cell, colour=st.AMBER, radius=13, width=2.6)
    q_final = case.q_rows(final)
    tried_q = sorted(v for v in q_final[cell] if v is not None)
    visited = set(final["visited"])
    unvisited_cont = [s for s in continuation[:-1] if s not in visited]
    q_visited = set(case.snapshot("q_learning", "episode 60")["visited"])
    for s in unvisited_cont:
        x3, y3 = scene.point(s, lift=2.0)
        ax.add_patch(Circle((x3, y3), 11, facecolor="none", edgecolor=st.INK2, linewidth=1.8, linestyle=(0, (2, 2)),
                            zorder=49))
    box = dict(boxstyle="round,pad=0.35", facecolor=st.BG, edgecolor=st.RULE, alpha=0.95)
    arrow = dict(arrowstyle="-", color=st.INK2, linewidth=1.2)
    ax.annotate(f"① route leaves the shortest routes,\n{case.distance[leave]} moves from the goal", (x1, y1),
                xytext=(14, 120), color=st.INK, fontsize=12.5, family=st.FONT_FAMILY, arrowprops=arrow, bbox=box,
                zorder=60)
    ax.annotate(f"② {case.distance[cell]} moves from the goal, the goal-ward action\nwas never taken (Q = 0); its tried "
                f"actions: {tried_q[0]:.1f}–{tried_q[-1]:.1f}", (x2, y2), xytext=(14, 205), color=st.AMBER,
                fontsize=12.5, family=st.FONT_FAMILY, arrowprops=arrow, bbox=box, zorder=60)
    if unvisited_cont:
        x3, y3 = scene.point(unvisited_cont[0], lift=2.0)
        ax.annotate("③ dashed rings: the shaft up to the goal,\nnever visited by Dyna-Q", (x3, y3), xytext=(14, 290),
                    color=st.INK2, fontsize=12.5, family=st.FONT_FAMILY, arrowprops=arrow, bbox=box, zorder=60)
    # Route length per episode.
    ax2 = fig.add_axes([0.62, 0.58, 0.35, 0.28])
    for m, name in (("dyna_q", "Dyna-Q"), ("q_learning", "Q-learning")):
        lengths = case.runs[m]["greedy_route_length_per_episode"]
        eps = np.arange(1, len(lengths) + 1)
        ys = np.array([v if v > 0 else np.nan for v in lengths], dtype=float)
        miss = np.array([v <= 0 for v in lengths])
        ax2.plot(eps, ys, color=METHOD_COLOURS[m], linewidth=2.0, drawstyle="steps-post", zorder=3)
        ax2.scatter(eps[miss], np.full(miss.sum(), 92 if m == "dyna_q" else 90), s=4,
                    color=mix(METHOD_COLOURS[m], 0.6), linewidths=0, zorder=2)
    ax2.axhline(case.route_length, color=st.INK3, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax2.text(20, case.route_length - 1.5, f"shortest route, {case.route_length} moves", color=st.INK2, fontsize=11.5,
             va="top", family=st.FONT_FAMILY)
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(86, 134)
    ax2.set_yticks([90, 100, 106, 114, 120, 130])
    ax2.set_yticklabels(["no route", "100", "106", "114", "120", "130"])
    ax2.tick_params(labelsize=12, colors=st.INK2)
    ax2.set_xlabel("episode", color=st.INK2, fontsize=12.5)
    ax2.set_ylabel("greedy route length (moves)", color=st.INK2, fontsize=12.5)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.set_title("Observed: greedy route after every episode", color=st.INK, fontsize=14, fontweight="bold",
                  loc="left")
    first = next(i + 1 for i, v in enumerate(run["greedy_route_length_per_episode"]) if v > 0)
    q_first = next(i + 1 for i, v in enumerate(case.runs["q_learning"]["greedy_route_length_per_episode"])
                   if v == case.route_length)
    ax2.text(20, 130.5, f"Dyna-Q: first route to the goal at episode {first} ({len(walk) - 1} moves), unchanged after",
             color=METHOD_COLOURS["dyna_q"], fontsize=11.5, family=st.FONT_FAMILY)
    ax2.text(20, 126.5, f"Q-learning: shortest route from episode {q_first}", color=METHOD_COLOURS["q_learning"],
             fontsize=11.5, family=st.FONT_FAMILY)
    # Action values at the two cells, at each snapshot.
    ax3 = fig.add_axes([0.62, 0.11, 0.35, 0.3])
    labels = [s["label"].replace("episode ", "ep ") for s in run["snapshots"]]
    detour = case.greedy(final)[leave]
    toward = next(a for a, t in enumerate(case.nxt[leave]) if case.distance.get(t) == case.distance[leave] - 1)
    series = ((leave, detour, "① greedy action (detour)", st.AMBER),
              (leave, toward, "① shortest direction", st.INK),
              (cell, action, "② goal-ward action", METHOD_COLOURS["dyna_q"]))
    rows = []
    for state, a, name, colour in series:
        vals = [case.q_rows(s)[state][a] for s in run["snapshots"]]
        rows.append({"state": state, "action": a, "name": name, "values": vals})
        xs = np.arange(len(vals))
        ys = [0.0 if v is None else v for v in vals]
        ax3.plot(xs, ys, color=colour, linewidth=2.0, marker="o", markersize=7, zorder=3,
                 markerfacecolor=colour, markeredgecolor=colour)
        for x, v in zip(xs, vals):
            if v is None:
                ax3.plot([x], [0.0], "o", markersize=9, markerfacecolor=st.BG, markeredgecolor=colour,
                         markeredgewidth=2, zorder=4)
        ax3.text(len(vals) - 0.85, ys[-1], name, color=colour, fontsize=11.5, va="center", family=st.FONT_FAMILY)
    ax3.set_xticks(np.arange(len(labels)))
    ax3.set_xticklabels(labels)
    ax3.set_xlim(-0.3, len(labels) + 2.6)
    ax3.tick_params(labelsize=12, colors=st.INK2)
    ax3.set_ylabel("Q(s, a)", color=st.INK2, fontsize=12.5)
    for spine in ("top", "right"):
        ax3.spines[spine].set_visible(False)
    ax3.set_title("Observed: action values at the two ringed cells (hollow: untried, 0)", color=st.INK, fontsize=14,
                  fontweight="bold", loc="left")
    seen_by_q = sum(s in q_visited for s in continuation[:-1])
    fig.text(0.62, 0.05, f"Shortest continuation from ①, {len(continuation) - 1} cells before the goal: Dyna-Q never "
                         f"visited {len(unvisited_cont)};\nQ-learning had visited {seen_by_q} by episode 60.",
             color=st.INK2, fontsize=12.5, family=st.FONT_FAMILY, linespacing=1.4)
    path = os.path.join(OUT, "failure-dyna.png")
    fig.savefig(path, dpi=100, facecolor=st.BG)
    plt.close(fig)
    summary["failure_dyna"] = {"route_moves": len(walk) - 1, "shortest": case.route_length,
                               "leave_state": leave, "leave_distance": case.distance[leave],
                               "untried_state": cell, "untried_distance": case.distance[cell],
                               "tried_q_at_untried_state": tried_q, "continuation_unvisited_by_dyna": len(unvisited_cont),
                               "continuation_visited_by_q_learning_ep60": sum(s in q_visited for s in continuation[:-1]),
                               "first_route_episode": first, "q_learning_first_shortest_episode": q_first,
                               "action_values": rows}
    return path


# --- entry point ----------------------------------------------------------------------------------------

def main():
    st.style()
    os.makedirs(OUT, exist_ok=True)
    summary = {"definitions": {
        "region": "non-goal states from which the frozen greedy policy reaches the goal (share of the maze's "
                  "reachable non-goal states)",
        "value_error": "e(s) = V(s) - V*(s), V(s) = max_a Q(s, a) with untried actions at 0, V* by value iteration",
        "bellman_residual": "rho(s) = max_a [r(s, a) + gamma (1 - done) V(s')] - V(s) under the exact model",
        "frontier_cell": "a visited non-goal state with an untried action whose successor has never been visited",
        "budget_snapshot": "the end of the first episode whose cumulative real steps reach the budget"}}
    comparison_case, traces_case, dyna_case = ad.Case("comparison"), ad.Case("traces"), ad.Case("dyna")
    with open(os.path.join(st.ASSETS, "..", "value-propagation", "figures", "diagnostic_trace_cut.json"),
              encoding="utf-8") as f:
        diagnostic = json.load(f)
    record = {}
    gif, n, ms = comparison(comparison_case, summary)
    record["analysis/comparison.gif"] = {"size": [W, H], "frames": n, "seconds": round(ms / 1000, 2),
                                         "megabytes": round(os.path.getsize(gif) / 1e6, 2)}
    gif, n, ms = value_animation(comparison_case, summary)
    record["analysis/value.gif"] = {"size": [W, H], "frames": n, "seconds": round(ms / 1000, 2),
                                    "megabytes": round(os.path.getsize(gif) / 1e6, 2)}
    for path in (comparison_static(comparison_case, summary), value_static(comparison_case, summary),
                 failure_traces(traces_case, diagnostic, summary), failure_dyna(dyna_case, summary)):
        with Image.open(path) as im:
            record["analysis/" + os.path.basename(path)] = {"size": list(im.size)}
        print(path)
    with open(os.path.join(OUT, "figures.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o, set) else o)
        f.write("\n")
    st.record_render(st.ASSETS, record)
    for name, entry in record.items():
        print(name, entry)


if __name__ == "__main__":
    main()
