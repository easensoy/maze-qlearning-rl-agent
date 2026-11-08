"""The README animation: recorded runs on one three-level maze, in an exploded 3D view.

Everything drawn comes from docs/assets/data/hero.json (recorded by capture_hero.py, checked
against the value-propagation results) and docs/value-propagation/results/summary.json.
It is a replay of recorded training runs drawn by this script, not the live viewer.

One message per part:
1. the maze: three levels, the shafts between them, start and goal;
2. episode 1: the agent explores (Q-learning's first episode, compressed);
3. Q-learning's greedy policy after fixed numbers of real moves;
4. Dyna-Q's greedy policy after the same numbers of real moves;
5. Dyna-Q's learned route, replayed move by move through the shafts;
6. the measured result across the long-route mazes, with its scope.
Episodes between checkpoints are skipped, and every frame says what it shows.

Run from maze_runner_rl/:
    python -m experiments.render_hero
"""
import json
import os

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from PIL import Image

from experiments import asset_style as st
from experiments import compare_methods as cm
from experiments.iso_maze import TOP_LEARNED, TRAIL, IsoMaze

# Text is sized for the README's 800-pixel column and stays readable when GitHub scales the
# frame to a phone's width (about 0.43x): titles and verdicts carry the message in-frame;
# methodological detail lives in the README text beside the animation.
W, H = 800, 720
UNIT, LEVEL_GAP, SLAB = 12.0, 132.0, 6.0
ORIGIN = (352.0, 372.0)        # screen point of the back corner of the bottom plate
GHOST = "#5f6977"              # dashed outline of the shaft level's plane
LABEL_SIZE, NOTE_SIZE = 15, 14
FRAME_MS = 70                  # GIF delays are stored in units of 10 ms
CHECKPOINT_MS = 700
EXPLORE_MOVES_PER_FRAME = 50
ROUTE_FRAMES = 36              # the learned route is replayed in about this many frames
SHOWN_BUDGETS = 4             # checkpoints shown per method: the last four budgets
LEVEL_Z = {0: 4, 1: 14, 2: 24}  # overlay zorder for paths on each level (plates sit just below)
SUMMARY = os.path.join(cm.OUT, "results", "summary.json")


def pt(px, scale=1):
    return px * 0.72 * scale


class Canvas:
    """A W x H pixel canvas; scale 2 draws the same layout at twice the resolution."""

    def __init__(self, scale=1):
        import matplotlib.pyplot as plt
        self.scale = scale
        self.fig = plt.figure(figsize=(W * scale / 100, H * scale / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])

    def clear(self):
        self.ax.cla()
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")
        self.ax.add_patch(Rectangle((0, 0), W, H, facecolor=st.BG, edgecolor="none", zorder=0))

    def text(self, x, y, s, size=13, color=st.INK, weight="normal", ha="left", va="baseline", zorder=40, **kw):
        return self.ax.text(x, y, s, fontsize=pt(size, self.scale), color=color, fontweight=weight, ha=ha, va=va,
                            family=st.FONT_FAMILY, zorder=zorder, **kw)

    def fit(self, x, y, s, max_width, size=13, **kwargs):
        """One line of text, made smaller if it would be wider than max_width."""
        t = self.text(x, y, s, size=size, **kwargs)
        width = t.get_window_extent(renderer=self.fig.canvas.get_renderer()).width / self.scale
        if width > max_width:
            t.set_fontsize(pt(size * max_width / width, self.scale))
        return t

    def image(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        import matplotlib.pyplot as plt
        plt.close(self.fig)


class Scene:
    def __init__(self, data):
        self.data = data
        self.backgrounds = {}
        grid = np.array([[[0 if ch == "." else 1 for ch in row] for row in level.split("\n")]
                         for level in data["levels"]])
        self.grid = grid
        self.levels, self.h, self.w = grid.shape
        self.iso = IsoMaze(grid, ORIGIN, UNIT, LEVEL_GAP, SLAB)
        self.start, self.goal = tuple(data["start"]), tuple(data["goal"])
        self.shafts = int(sum((grid[1] == 0).ravel())) if self.levels == 3 else 0

    def xyz(self, state):
        z, rest = divmod(int(state), self.w * self.h)
        y, x = divmod(rest, self.w)
        return x, y, z

    def cells(self, states):
        return [self.xyz(s) for s in states]

    # --- drawing ---------------------------------------------------------------------------
    def plates(self, cv, learned=frozenset()):
        iso, ax = self.iso, cv.ax
        iso.draw_level(ax, 0, learned, zorder=2)
        iso.draw_columns(ax, 0, zorder=10)
        ghost = iso.plate_outline(1)
        ax.add_patch(Polygon(ghost, closed=True, facecolor="none", edgecolor=GHOST, linewidth=pt(1.2, cv.scale),
                             linestyle=(0, (3, 3)), zorder=9))
        iso.draw_level(ax, 1, learned, zorder=12)
        iso.draw_columns(ax, 1, zorder=20)
        iso.draw_level(ax, 2, learned, zorder=22)
        labels = {0: "Level 0 · start", 1: f"Level 1 · {self.shafts} shafts", 2: "Level 2 · goal"}
        for level, label in labels.items():
            _, right, _, _ = iso.plate_outline(level)
            cv.fit(right[0] + 12, right[1], label, W - 12 - (right[0] + 12), size=LABEL_SIZE, color=st.INK,
                   weight="bold", va="center")

    def markers(self, cv):
        for pos, fill in ((self.start, "none"), (self.goal, st.INK)):
            cx, cy = self.iso.centre(*pos, lift=1.5)
            cv.ax.add_patch(Circle((cx, cy), 7.5, facecolor=fill, edgecolor=st.INK, linewidth=pt(2.6, cv.scale),
                                   zorder=45))
        for pos, word in ((self.start, "start"), (self.goal, "goal")):
            cx, cy = self.iso.centre(*pos, lift=1.5)
            cv.text(cx + 14, cy + 4, word, size=LABEL_SIZE, color=st.INK, weight="bold", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=st.BG, edgecolor="none", alpha=0.85))

    def path(self, cv, states, colour, width, alpha=1.0, dashes=None, boost=0.0):
        cells = self.cells(states)
        self.iso.draw_path(cv.ax, cells, colour, width=pt(width, cv.scale), alpha=alpha,
                           dashes=dashes, level_zorder={k: v + boost for k, v in LEVEL_Z.items()})

    def agent(self, cv, state, colour=st.INK):
        cx, cy = self.iso.centre(*self.xyz(state), lift=2.5)
        cv.ax.add_patch(Circle((cx, cy), 11, facecolor=colour, edgecolor="none", alpha=0.25, zorder=46))
        cv.ax.add_patch(Circle((cx, cy), 6, facecolor=colour, edgecolor=st.BG, linewidth=pt(1.2, cv.scale),
                               zorder=47))

    def loop_mark(self, cv, state):
        cx, cy = self.iso.centre(*self.xyz(state), lift=2.5)
        cv.text(cx, cy, "×", size=26, color=st.AMBER, weight="bold", ha="center", va="center", zorder=48)

    def header(self, cv, title, subtitle=None):
        cv.fit(24, 42, title, W - 48, size=28, weight="bold")
        if subtitle:
            cv.fit(24, 76, subtitle, W - 48, size=20, color=st.INK2)

    def legend(self, cv, items, y=656):
        x = 24
        for kind, colour, label in items:
            if kind == "line":
                cv.ax.plot([x, x + 22], [y, y], color=colour, linewidth=pt(3.5, cv.scale), solid_capstyle="round",
                           zorder=50)
            elif kind == "dash":
                line, = cv.ax.plot([x, x + 22], [y, y], color=colour, linewidth=pt(2.4, cv.scale), zorder=50)
                line.set_dashes((3, 2.5))
            else:
                cv.ax.add_patch(Polygon([(x + 11, y - 7), (x + 22, y), (x + 11, y + 7), (x, y)], closed=True,
                                        facecolor=colour, edgecolor="none", zorder=50))
            t = cv.text(x + 30, y, label, size=LABEL_SIZE, color=st.INK2, va="center")
            x += 30 + t.get_window_extent(renderer=cv.fig.canvas.get_renderer()).width / cv.scale + 24

    def footer(self, cv, note):
        cv.ax.plot([24, W - 24], [681, 681], color=st.RULE, linewidth=pt(1, cv.scale), zorder=50)
        cv.fit(24, 704, note, W - 48, size=NOTE_SIZE, color=st.INK2)

    def endcard(self, cv, lines):
        cv.ax.add_patch(Rectangle((0, 0), W, H, facecolor=st.BG, edgecolor="none", alpha=0.88, zorder=60))
        y = 300
        for i, (text, size, colour, weight) in enumerate(lines):
            cv.fit(W / 2, y, text, W - 64, size=size, color=colour, weight=weight, ha="center", zorder=61)
            if i + 1 < len(lines):
                y += 0.5 * size + 0.9 * lines[i + 1][1] + 14     # baseline to baseline

    def background(self, cv, learned):
        """The plates, shafts and level labels for one set of learned cells, drawn once and
        reused: only the paths, markers and text change from frame to frame."""
        key = (cv.scale, learned)
        if key not in self.backgrounds:
            cv.clear()
            self.iso.scale = cv.scale
            self.plates(cv, learned=learned)
            self.backgrounds[key] = cv.image()
        return self.backgrounds[key]

    def draw(self, cv, f):
        image = self.background(cv, frozenset(self.cells(f.get("learned", ()))))
        cv.clear()
        cv.ax.imshow(image, extent=(0, W, H, 0), interpolation="none", zorder=1)
        if f.get("bfs"):
            self.path(cv, self.data["bfs_route"], st.INK, 2.2, alpha=0.95, dashes=(3, 2.5), boost=1)
        if f.get("trail"):
            self.path(cv, f["trail"], TRAIL, 2.0, alpha=0.95, boost=2)
        if f.get("route"):
            self.path(cv, f["route"], st.AMBER, 4.0, boost=3)
            if f.get("loop"):
                self.loop_mark(cv, f["route"][-1])
        self.markers(cv)
        if f.get("agent") is not None:
            self.agent(cv, f["agent"], f.get("agent_colour", st.INK))
        self.header(cv, f["title"], f.get("subtitle"))
        if f.get("legend"):
            self.legend(cv, f["legend"])
        self.footer(cv, f["footer"])
        if f.get("endcard"):
            self.endcard(cv, f["endcard"])


# --- the storyboard -----------------------------------------------------------------------

LEGEND_EXPLORE = [("line", TRAIL, "path in episode 1")]
LEGEND_POLICY = [("tile", TOP_LEARNED, "greedy policy reaches the goal"),
                 ("line", st.AMBER, "greedy route from the start"), ("dash", st.INK, "shortest route")]
LEGEND_ROUTE = [("line", st.AMBER, "greedy route"), ("dash", st.INK, "shortest route (BFS)")]
REPLAY = "Recorded replay, not the live viewer"


def thousands(n):
    return f"{n / 1000:.0f}k" if n >= 10_000 else f"{n:,}"


def frames(data, summary):
    q, dq = data["runs"]["q_learning"], data["runs"]["dyna_q"]
    n = dq["params"]["planning_steps"]
    size = "×".join(str(v) for v in data["size"])
    route = data["shortest_route"]
    out = [dict(title=f"Three-level maze: {size} cells, seed {data['maze_seed']}",
                subtitle=f"Start on level 0, goal on level 2 · shortest route: {route} moves",
                footer=f"{REPLAY} · rendered from logged training runs", duration=2400)]

    # 2. Episode 1 of Q-learning, compressed.
    path = q["episode1"]
    reached = path[-1] == data["goal_state"]
    moves = len(path) - 1
    for i in range(1, moves + EXPLORE_MOVES_PER_FRAME, EXPLORE_MOVES_PER_FRAME):
        k = min(i, moves)
        out.append(dict(title="Episode 1: exploring from an all-zero Q-table",
                        subtitle=f"Move {k:,} of {moves:,}" + ("" if reached else " · goal not reached"),
                        trail=path[:k + 1], agent=path[k], legend=LEGEND_EXPLORE,
                        footer=f"{REPLAY} · {EXPLORE_MOVES_PER_FRAME} moves per frame", duration=FRAME_MS))
    out[-1]["duration"] = 800

    # 3-4. Each method's greedy policy after the same numbers of real moves.
    budgets = ", ".join(thousands(b) for b in data["budgets"][-SHOWN_BUDGETS:])
    for run, name in ((q, "Q-learning"), (dq, f"Dyna-Q (n = {n})")):
        for cp in run["checkpoints"][-SHOWN_BUDGETS:]:
            walk = cp["greedy_path"]
            verdict = ("greedy route is a shortest route" if cp["greedy_shortest"]
                       else "greedy route reaches the goal" if cp["greedy_reaches"]
                       else "greedy route loops before the goal")
            out.append(dict(title=f"{name} after {thousands(cp['budget'])} real moves",
                            subtitle=f"Episode {cp['episode']} · {verdict}",
                            learned=cp["learned"], route=walk, loop=not cp["greedy_reaches"], bfs=True,
                            legend=LEGEND_POLICY, footer=f"{REPLAY} · checkpoints at {budgets} real moves",
                            duration=CHECKPOINT_MS))
        out[-1]["duration"] = 1100

    # 5. Dyna-Q's learned route, move by move.
    final = dq["final_route"]
    changes = sum(a // (data["size"][0] * data["size"][1]) != b // (data["size"][0] * data["size"][1])
                  for a, b in zip(final, final[1:]))
    per_frame = max(2, -(-(len(final) - 1) // ROUTE_FRAMES))
    for k in range(0, len(final), per_frame):
        out.append(dict(title=f"Dyna-Q's greedy route: {len(final) - 1} moves, a shortest route",
                        subtitle=f"{changes} level changes through the shafts",
                        route=final[:k + 1], agent=final[k], agent_colour=st.AMBER, bfs=True, legend=LEGEND_ROUTE,
                        footer=f"{REPLAY} · {per_frame} moves per frame", duration=FRAME_MS))
    out.append(dict(out[-1], route=final, agent=final[-1], duration=1200))

    # 6. The measured result, with its definition and scope.
    long = summary["groups"]["long:3d_3"]
    lq, ll, ld = (long[m]["converged"] for m in ("q_learning", "q_lambda", "dyna_q"))
    upd = long["dyna_q"]["updates_per_real_step"]
    out.append(dict(out[-1], duration=3800, endcard=[
        (f"Long-route set: {ld['n']} mazes, shortest routes of at least {cm.LONG_ROUTES['min_route']} moves", 18,
         st.INK2, "normal"),
        (f"Converged: Dyna-Q {ld['k']} · Q(λ) {ll['k']} · Q-learning {lq['k']}", 28, st.INK, "bold"),
        ("Converged: greedy route shortest at every check for the last ≥ 100 of 1,000 episodes", 16, st.INK2,
         "normal"),
        (f"Same mazes and exploration seeds · Dyna-Q: {upd:.0f} Q-value updates per real move", 16, st.INK2,
         "normal"),
    ]))
    return out


def quantize(images, colors=256):
    """One shared palette for every frame (no dithering), so flat areas stay flat and the
    GIF only stores what changes between frames."""
    picks = np.linspace(0, len(images) - 1, min(len(images), 16)).astype(int)
    sample = Image.fromarray(np.concatenate([images[i] for i in picks], axis=0))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return [Image.fromarray(im).quantize(palette=palette, dither=Image.Dither.NONE) for im in images]


def poster_frame(specs):
    """The frame the poster shows: the full learned route, before the end card."""
    return next(f for f in reversed(specs) if f.get("route") and not f.get("endcard"))


def render(data, summary, out=st.ASSETS, frames_dir=None):
    scene = Scene(data)
    specs = frames(data, summary)
    cv = Canvas()
    images = []
    for i, spec in enumerate(specs):
        scene.draw(cv, spec)
        images.append(cv.image())
        if frames_dir:
            os.makedirs(frames_dir, exist_ok=True)
            Image.fromarray(images[-1]).save(os.path.join(frames_dir, f"frame_{i:03d}.png"))
    cv.close()
    durations = [spec["duration"] for spec in specs]
    gif = os.path.join(out, "hero.gif")
    pal = quantize(images)
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=False)
    # The same frames without the GIF's 256-colour palette, for viewing at full quality.
    webp = os.path.join(out, "hero.webp")
    full = [Image.fromarray(im) for im in images]
    full[0].save(webp, save_all=True, append_images=full[1:], duration=durations, loop=0, quality=88, method=6)
    poster = Canvas(scale=2)
    scene.draw(poster, dict(poster_frame(specs), footer="Still from a recorded replay, not the live viewer"))
    Image.fromarray(poster.image()).save(os.path.join(out, "hero-poster.png"), optimize=True)
    poster.close()
    st.record_render(out, {
        "hero.gif": {"size": [W, H], "frames": len(images), "seconds": round(sum(durations) / 1000, 2),
                     "megabytes": round(os.path.getsize(gif) / 1e6, 2), "maze_seed": data["maze_seed"],
                     "explore_moves_per_frame": EXPLORE_MOVES_PER_FRAME, "route_frames": ROUTE_FRAMES,
                     "checkpoint_budgets": data["budgets"][-SHOWN_BUDGETS:],
                     "checkpoint_ms": CHECKPOINT_MS},
        "hero-poster.png": {"size": [2 * W, 2 * H]},
        "hero.webp": {"size": [W, H], "megabytes": round(os.path.getsize(webp) / 1e6, 2), "quality": 88},
    })
    return gif, len(images), sum(durations)


def main():
    st.style()
    data = st.load_json("hero.json")
    with open(SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)
    gif, count, ms = render(data, summary)
    print(f"{gif}: {count} frames, {ms / 1000:.1f} s, {os.path.getsize(gif) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
