"""The animated hero for the README: one three-level run, from its first exploring episode
to the learned route. Drawn from docs/assets/data/featured.json (capture_traces.py).

Every frame shows recorded data: the cells the agent visited in its first training
episode, the greedy policy read from the Q-table at checkpoints, and the final greedy route
next to a BFS shortest route. Nothing is interpolated; the agent moves cell by cell. The
animation compresses time and skips between checkpoints, and says so on screen.

Run from maze_runner_rl/ after capture_traces:
    python -m experiments.render_hero      # docs/assets/hero.gif and docs/assets/hero-poster.png
"""
import argparse
import os

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Rectangle
from PIL import Image

from experiments import asset_style as st
from experiments import evaluate as ev

W, H = 800, 560
CELL = 17
MARGIN = 22
SIDE = 21 * CELL
GUTTER = W - 2 * MARGIN - 2 * SIDE
PANEL_TOP = 104
FRAME_MS = 70       # GIF delays are stored in units of 10 ms
TIMELAPSE_MS = 80   # GIF delays are stored in units of 10 ms
EXPLORE_MOVES_PER_FRAME = 25
ROUTE_MOVES_PER_FRAME = 3
TIMELAPSE_EVERY = 10


def pt(px, scale=1):
    """Pixels on the 100-dpi canvas as points (line widths and font sizes)."""
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

    def text(self, x, y, s, size=13, color=st.INK, weight="normal", ha="left", va="baseline", rotation=0,
             alpha=1.0, zorder=20):
        return self.ax.text(x, y, s, fontsize=pt(size, self.scale), color=color, fontweight=weight, ha=ha, va=va,
                            rotation=rotation, family=st.FONT_FAMILY, alpha=alpha, zorder=zorder)

    def width(self, artist):
        """Width of a text artist in canvas pixels."""
        box = artist.get_window_extent(renderer=self.fig.canvas.get_renderer())
        return box.width / self.scale

    def fit(self, x, y, s, max_width, size=13, **kwargs):
        """Text on one line, made smaller if it would be wider than max_width (a wider fallback
        typeface than Segoe UI would otherwise run past the frame)."""
        t = self.text(x, y, s, size=size, **kwargs)
        width = self.width(t)
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
    """One featured three-level run, drawn as level 0 and level 2 side by side. Level 1 holds
    only the shafts between them, so it is shown as the gutter between the two panels."""

    def __init__(self, run):
        self.run = run
        self.maze = maze = st.Maze(run)
        if maze.depth != 3:
            raise ValueError("the hero shows a three-level maze")
        middle = {(x, y) for y in range(maze.height) for x in range(maze.width) if maze.open(x, y, 1)}
        if not middle <= set(maze.shafts(0)) or not middle <= set(maze.shafts(2)):
            raise ValueError("level 1 has cells that are not shafts; it would need a panel of its own")
        self.shafts = sorted(middle)
        self.panels = {0: (MARGIN, PANEL_TOP), 2: (MARGIN + SIDE + GUTTER, PANEL_TOP)}
        self.position = {s: i for i, s in enumerate(run["states"])}

    def centre(self, x, y, z):
        ox, oy = self.panels[z]
        return ox + (x + 0.5) * CELL, oy + (y + 0.5) * CELL

    def places(self, state):
        """Where a state is drawn: one panel, or both for a shaft cell on level 1."""
        x, y, z = self.maze.xyz[state]
        return [self.centre(x, y, level) for level in ((0, 2) if z == 1 else (z,))]

    # --- layers ------------------------------------------------------------------------

    def levels(self, cv, solved=()):
        maze = self.maze
        for z, (ox, oy) in self.panels.items():
            cv.ax.add_patch(Rectangle((ox - 3, oy - 3), SIDE + 6, SIDE + 6, facecolor=st.PANEL, edgecolor=st.RULE,
                                      lw=pt(1, cv.scale), zorder=1))
            rgb = np.empty((maze.height, maze.width, 3))
            rgb[:] = to_rgb(st.PANEL)
            rgb[~maze.grid[z]] = to_rgb(st.FLOOR)
            for s in solved:
                x, y, level = maze.xyz[s]
                if level == z:
                    rgb[y, x] = to_rgb(st.BLUE_SOFT)
            cv.ax.imshow(rgb, extent=(ox, ox + SIDE, oy + SIDE, oy), interpolation="nearest", zorder=2)
            for x, y in self.shafts:
                cx, cy = self.centre(x, y, z)
                cv.ax.add_patch(Rectangle((cx - 4.5, cy - 4.5), 9, 9, facecolor="none", edgecolor=st.INK2,
                                          lw=pt(1.2, cv.scale), alpha=0.8, zorder=3))
        for z, (ox, oy) in self.panels.items():
            name = cv.text(ox - 2, oy - 11, f"LEVEL {z}", size=12.5, weight=700)
            role = "start" if z == 0 else "goal"
            cv.text(ox - 2 + cv.width(name) + 8, oy - 11, f"{role} · {maze.level_cells(z)} open cells", size=12.5,
                    color=st.INK2)
        gx = MARGIN + SIDE + GUTTER / 2
        cv.text(gx, PANEL_TOP + SIDE / 2, f"LEVEL 1  ·  {len(self.shafts)} shaft cells  ⇅", size=11.5,
                color=st.INK2, ha="center", va="center", rotation=90)

    def endpoints(self, cv, labels=False):
        s = cv.scale
        (sx, sy), = self.places(self.maze.start)
        (gx, gy), = self.places(self.maze.goal)
        cv.ax.add_patch(Circle((sx, sy), 5.8, facecolor=st.BG, edgecolor=st.INK, lw=pt(2.3, s), zorder=15))
        cv.ax.add_patch(Circle((gx, gy), 6.2, facecolor=st.INK, edgecolor=st.BG, lw=pt(1.6, s), zorder=15))
        if labels:
            box = dict(boxstyle="round,pad=0.25", facecolor=st.BG, edgecolor="none", alpha=0.92)
            cv.text(sx + 10, sy + 4.5, "start", size=12.5, weight=700, zorder=16).set_bbox(box)
            cv.text(gx - 10, gy + 4.5, "goal", size=12.5, weight=700, ha="right", zorder=16).set_bbox(box)

    def polylines(self, states):
        """Pixel polylines, one per stretch the states spend on level 0 or level 2."""
        runs, _ = self.maze.segments(states)
        return [np.array([self.centre(x, y, z) for x, y in cells]) for z, cells in runs if z != 1]

    def trail(self, cv, states, tail=40):
        """A training trajectory: faint where the agent has been, bright for its latest moves."""
        s = cv.scale
        old = [p for p in self.polylines(states) if len(p) > 1]
        if old:
            cv.ax.add_collection(LineCollection(old, colors=st.BLUE, linewidths=pt(2.4, s), alpha=0.3,
                                                capstyle="round", joinstyle="round", zorder=6))
        new = [p for p in self.polylines(states[-tail:]) if len(p) > 1]
        if new:
            cv.ax.add_collection(LineCollection(new, colors=st.BLUE, linewidths=pt(2.6, s), alpha=0.95,
                                                capstyle="round", joinstyle="round", zorder=7))
        for cx, cy in self.places(states[-1]):
            cv.ax.add_patch(Circle((cx, cy), 4.8, facecolor=st.BLUE, edgecolor=st.INK, lw=pt(1.4, s), zorder=14))

    def route(self, cv, states, color=st.AMBER, width=4.8, dashed=False, zorder=9):
        for p in self.polylines(states):
            if len(p) > 1:
                cv.ax.plot(p[:, 0], p[:, 1], color=color, lw=pt(width, cv.scale), ls=(0, (1.4, 2.0)) if dashed else "-",
                           solid_joinstyle="round", solid_capstyle="round", dash_capstyle="round", zorder=zorder)

    def loop_mark(self, cv, state):
        """Where a greedy walk that never reaches the goal starts repeating itself."""
        for cx, cy in self.places(state):
            r = 4.4
            for a, b in (((cx - r, cy - r), (cx + r, cy + r)), ((cx - r, cy + r), (cx + r, cy - r))):
                cv.ax.plot([a[0], b[0]], [a[1], b[1]], color=st.AMBER, lw=pt(2.6, cv.scale), zorder=17)

    def agent(self, cv, state, color=st.AMBER):
        for cx, cy in self.places(state):
            cv.ax.add_patch(Circle((cx, cy), 5.2, facecolor=color, edgecolor=st.BG, lw=pt(1.6, cv.scale), zorder=14))

    def badges(self, cv, transits):
        for number, ((x, y), a, b) in enumerate(transits, 1):
            label = f"{'↑' if b > a else '↓'}{number}"
            for level in (0, 2):
                cx, cy = self.centre(x, y, level)
                cv.ax.add_patch(Circle((cx, cy), 7.6, facecolor="none", edgecolor=st.AMBER, lw=pt(1.6, cv.scale),
                                       zorder=13))
                cv.text(cx + 9.5, cy - 7.5, label, size=12, weight=700, color=st.AMBER, zorder=16)

    # --- frame ---------------------------------------------------------------------------

    def header(self, cv):
        run = self.run
        w, h, d = run["size"]
        cv.fit(MARGIN - 1, 34, "A tabular Q-learning agent learns a three-level maze", W - 2 * MARGIN, size=21,
               weight=600)
        cv.fit(MARGIN, 58, f"{w} × {h} × {d} cells  ·  maze seed {run['maze_seed']}  ·  start on level 0, goal on "
                           f"level 2  ·  moves: N, S, E, W, up, down", W - 2 * MARGIN, size=13, color=st.INK2)

    def legend(self, cv, items, y=512):
        """Small glyph-and-label pairs laid out from the left; returns where they end."""
        x, s = MARGIN, cv.scale
        for kind, label in items:
            if kind == "trail":
                cv.ax.plot([x, x + 18], [y - 4, y - 4], color=st.BLUE, lw=pt(2.6, s), zorder=20)
            elif kind == "solved":
                cv.ax.add_patch(Rectangle((x + 2, y - 11), 14, 14, facecolor=st.BLUE_SOFT, edgecolor="none", zorder=20))
            elif kind == "route":
                cv.ax.plot([x, x + 18], [y - 4, y - 4], color=st.AMBER, lw=pt(4.4, s), zorder=20)
            elif kind == "bfs":
                cv.ax.plot([x, x + 18], [y - 4, y - 4], color=st.INK, lw=pt(1.9, s), ls=(0, (1.4, 2.0)), zorder=20)
            elif kind == "loop":
                cx, cy, r = x + 9, y - 4, 4.4
                cv.ax.plot([cx - r, cx + r], [cy - r, cy + r], color=st.AMBER, lw=pt(2.4, s), zorder=20)
                cv.ax.plot([cx - r, cx + r], [cy + r, cy - r], color=st.AMBER, lw=pt(2.4, s), zorder=20)
            elif kind == "shaft":
                cv.ax.add_patch(Rectangle((x + 4.5, y - 8.5), 9, 9, facecolor="none", edgecolor=st.INK2,
                                          lw=pt(1.2, s), zorder=20))
            elif kind == "badge":
                cv.text(x + 9, y, "↑1", size=12, weight=700, color=st.AMBER, ha="center")
            t = cv.text(x + 24, y, label, size=11.5, color=st.INK2)
            x += 24 + cv.width(t) + 18
        return x - 18

    def footer(self, cv, caption, note=None, episode=None, events=(), note_left=MARGIN):
        episodes = self.run["episodes"]
        cv.fit(MARGIN, 490, caption, W - 2 * MARGIN, size=14, color=st.INK)
        if note:
            cv.fit(W - MARGIN, 512, note, W - MARGIN - note_left - 16, size=11.5, color=st.INK3, ha="right")
        x0, x1, y = MARGIN, W - MARGIN, 532
        s = cv.scale
        cv.ax.plot([x0, x1], [y, y], color=st.RULE, lw=pt(3, s), solid_capstyle="butt", zorder=4)
        xe = x0 + (x1 - x0) * min(episode or 0, episodes) / episodes
        if episode:
            cv.ax.plot([x0, xe], [y, y], color=st.INK3, lw=pt(3, s), solid_capstyle="butt", zorder=5)
        for e, label in events:   # an event appears once the playhead has passed it
            if episode is None or episode < e:
                continue
            xv = x0 + (x1 - x0) * e / episodes
            cv.ax.plot([xv, xv], [y - 5, y + 5], color=st.INK2, lw=pt(1.5, s), zorder=6)
            if label:
                cv.text(xv, y + 18, label, size=10.5, color=st.INK2, ha="center")
        if episode is not None:
            cv.ax.add_patch(Circle((xe, y), 4.4, facecolor=st.INK, edgecolor=st.BG, lw=pt(1.2, s), zorder=7))
        cv.text(x0, y + 18, "0", size=10.5, color=st.INK3)
        cv.text(x1, y + 18, f"{episodes:,} training episodes", size=10.5, color=st.INK3, ha="right")

    def endcard(self, cv, lines):
        cv.ax.add_patch(Rectangle((MARGIN - 6, PANEL_TOP - 30), W - 2 * MARGIN + 12, SIDE + 36, facecolor=st.BG,
                                  edgecolor="none", alpha=0.9, zorder=30))
        y = PANEL_TOP + 40
        for size, weight, color, text in lines:
            if text:
                cv.fit(MARGIN + 18, y, text, W - 2 * MARGIN - 36, size=size, weight=weight, color=color, zorder=31)
            y += size * 1.9 if text else size * 0.9

    def draw(self, cv, f):
        cv.clear()
        self.header(cv)
        self.levels(cv, f.get("solved", ()))
        if f.get("bfs"):
            self.route(cv, self.run["bfs_route"], color=st.INK, width=1.9, dashed=True, zorder=10)
        if f.get("trail"):
            self.trail(cv, f["trail"])
        if f.get("walk"):
            walk = f["walk"]
            if f.get("reaches"):
                self.route(cv, walk)
            else:   # the last cell repeats an earlier one: draw up to the last new cell and mark it
                self.route(cv, walk[:-1])
                self.loop_mark(cv, walk[-2])
            if f.get("agent"):
                self.agent(cv, walk[-1])
        if f.get("transits"):
            self.badges(cv, f["transits"])
        self.endpoints(cv, labels=f.get("endpoint_labels", False))
        legend_end = self.legend(cv, f["legend"]) if f.get("legend") else MARGIN
        self.footer(cv, f.get("caption", ""), f.get("note"), f.get("episode"), f.get("events", ()), legend_end)
        if f.get("endcard"):
            self.endcard(cv, f["endcard"])


def transits(maze, states):
    """Level changes along a route, one per shaft: [((x, y), from_level, to_level)]."""
    _, moves = maze.segments(states)
    out = []
    for x, y, a, b in moves:
        if out and out[-1][0] == (x, y) and out[-1][2] == a:
            out[-1] = ((x, y), out[-1][1], b)
        else:
            out.append(((x, y), a, b))
    return [t for t in out if t[1] != t[2]]


def solved_states(run, checkpoint):
    """Cells from which the greedy policy reached the goal at a checkpoint (goal excluded)."""
    return {s for s, bit in zip(run["states"], checkpoint["solved"]) if bit == "1"}


# --- the sequence of frames ---------------------------------------------------------

def frames(run, aggregate):
    """Frame specifications (dicts read by Scene.draw), each with a duration in ms."""
    episodes = run["episodes"]
    maze = st.Maze(run)
    checkpoints = {c["episode"]: c for c in run["checkpoints"]}
    final = checkpoints[episodes]
    first_short, converged = run["first_greedy_shortest_episode"], run["converged_episode"]
    route_len = run["shortest_route"]
    events = [(e, label) for e, label in ((run["first_success_episode"], ""),
                                          (first_short, f"{first_short}: first shortest"),
                                          (converged, f"{converged}: settled")) if e]
    selection = (f"seed {run['maze_seed']}: the median of the {aggregate['converged']} converged runs, "
                 f"by convergence episode")
    out = []

    out.append(dict(caption="The task: reach the goal. The agent observes only its cell; its Q-table starts at zero.",
                    legend=[("shaft", "shaft: up or down a level, via level 1")], note=selection,
                    endpoint_labels=True, events=events, episode=0, duration=2000))

    path = run["paths"]["1"]
    moves = len(path) - 1
    reached = run["success"][0] == "1"
    visited = len(set(path))
    epsilon = ev.SETTINGS["agent"]["epsilon"]   # the agent's exploration rate during its first episode
    for n in list(range(EXPLORE_MOVES_PER_FRAME, moves, EXPLORE_MOVES_PER_FRAME)) + [moves]:
        out.append(dict(trail=path[:n + 1],
                        caption=f"Training episode 1, exploring (ε = {epsilon:.2f}): move {n:,} of {moves:,}",
                        legend=[("trail", "training trajectory")],
                        note=f"playback: {EXPLORE_MOVES_PER_FRAME} moves per frame", events=events, episode=1,
                        duration=FRAME_MS))
    ending = "reaches the goal" if reached else "ends at the 1,000-move cap without reaching the goal"
    out.append(dict(trail=path, caption=f"Episode 1 {ending}; it visited {visited} of {len(run['states']) + 1} "
                                        f"open cells.", legend=[("trail", "training trajectory")],
                    note=f"playback: {EXPLORE_MOVES_PER_FRAME} moves per frame", events=events, episode=1,
                    duration=1000))

    legend = [("solved", "greedy policy reaches the goal from here"), ("route", "greedy walk from the start"),
              ("loop", "stuck in a loop")]
    shown = sorted(e for e in checkpoints if (e % TIMELAPSE_EVERY == 0 and e <= converged + TIMELAPSE_EVERY // 2)
                   or e in (first_short, converged))
    for e in shown:
        c = checkpoints[e]
        solved = solved_states(run, c)
        share = (len(solved) + 1) / (len(run["states"]) + 1)
        walk, reaches = c["greedy_path"], c["greedy_reaches"]
        if e == first_short:
            caption = f"Episode {e}: from the start, the greedy walk now reaches the goal by a shortest route."
            hold = 1100
        elif e == converged:
            caption = f"Episode {e}: from here on, the greedy route is a shortest route at every check."
            hold = 1100
        else:
            start = "from the start too" if reaches else "not yet from the start"
            caption = (f"After {e} episodes ({c['env_steps']:,} moves): the greedy policy reaches the goal from "
                       f"{share:.0%} of cells, {start}")
            hold = TIMELAPSE_MS
        out.append(dict(solved=solved, walk=walk, reaches=reaches, caption=caption, legend=legend,
                        note=f"time-lapse: 1 frame = {TIMELAPSE_EVERY} episodes", events=events, episode=e,
                        duration=hold))

    route = final["greedy_path"]
    same = route == run["bfs_route"]
    changes = transits(maze, route)
    legend = [("route", "learned greedy route"), ("bfs", "a BFS shortest route"), ("badge", "level change")]
    out.append(dict(caption=f"Jump to episode {episodes:,}. The greedy policy, no exploration: always the best action.",
                    bfs=True, legend=legend, note=f"episodes {shown[-1] + 1:,}–{episodes:,} not shown", events=events,
                    episode=episodes, duration=700))
    steps = list(range(ROUTE_MOVES_PER_FRAME, len(route) - 1, ROUTE_MOVES_PER_FRAME)) + [len(route) - 1]
    for n in steps:
        passed = transits(maze, route[:n + 1])
        out.append(dict(walk=route[:n + 1], reaches=True, agent=True, bfs=True, transits=passed,
                        caption=f"Greedy walk after {episodes:,} episodes: move {n} of {len(route) - 1}",
                        legend=legend, note=f"playback: {ROUTE_MOVES_PER_FRAME} moves per frame", events=events,
                        episode=episodes, duration=FRAME_MS))
    match = "the same cells as" if same else "a different route of the same length as"
    out.append(dict(walk=route, reaches=True, bfs=True, transits=changes,
                    caption=f"Learned route: {len(route) - 1} moves, {match} the BFS shortest route.",
                    legend=legend, note=f"{len(changes)} level changes through shafts", events=events,
                    episode=episodes, duration=1500))
    out.append(dict(walk=route, reaches=True, bfs=True, transits=changes, caption="", legend=legend,
                    events=events, episode=episodes, endcard=endcard_lines(run, aggregate), duration=4000))
    return out


def endcard_lines(run, agg):
    big, mid, small = 19, 14.5, 13
    episodes = run["episodes"]
    x3 = f"; with {3 * episodes:,}, all {agg['converged_x3']} did." if "converged_x3" in agg else "."
    unseen = f": {agg['heldout']} tests on unseen mazes reached the goal." if "heldout" in agg else "."
    return [
        (big, 600, st.INK, f"This run converged at episode {run['converged_episode']}: its greedy route"),
        (big, 600, st.INK, f"stayed a shortest route at every check through episode {episodes:,}."),
        (small, "normal", st.INK, ""),
        (mid, "normal", st.INK, f"Across {agg['n']} three-level mazes, {agg['converged']} runs converged within "
                                f"{episodes:,} episodes{x3}"),
        (mid, "normal", st.INK, f"One-level mazes: {agg['one_level']} converged. 2D mazes: {agg['two_d']}."),
        (small, "normal", st.INK, ""),
        (small, "normal", st.INK2, "Scope: one fixed maze per run. The Q-table is indexed by the agent's cell only,"),
        (small, "normal", st.INK2, f"so nothing it learns transfers to another maze{unseen}"),
        (small, "normal", st.INK, ""),
        (small, "normal", st.INK3, "Converged: a shortest greedy route at every check for at least the last 100 "
                                   "episodes."),
    ]


# --- output -------------------------------------------------------------------------

def quantize(images, colors=256):
    """One shared palette for every frame (no dithering), so flat areas stay flat and the
    GIF only stores what changes between frames."""
    picks = np.linspace(0, len(images) - 1, min(len(images), 12)).astype(int)
    sample = Image.fromarray(np.concatenate([images[i] for i in picks], axis=0))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return [Image.fromarray(im).quantize(palette=palette, dither=Image.Dither.NONE) for im in images]


def render(run, aggregate, out, frames_dir=None):
    scene = Scene(run)
    specs = frames(run, aggregate)
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

    poster = Canvas(scale=2)
    scene.draw(poster, specs[-2])   # the learned route next to the BFS route, before the end card
    Image.fromarray(poster.image()).save(os.path.join(out, "hero-poster.png"), optimize=True)
    poster.close()
    st.record_render(out, {
        "hero.gif": {"size": [W, H], "frames": len(images), "seconds": round(sum(durations) / 1000, 2),
                     "megabytes": round(os.path.getsize(gif) / 1e6, 2), "maze_seed": run["maze_seed"],
                     "explore_moves_per_frame": EXPLORE_MOVES_PER_FRAME,
                     "timelapse_episodes_per_frame": TIMELAPSE_EVERY, "route_moves_per_frame": ROUTE_MOVES_PER_FRAME,
                     "end_card": aggregate},
        "hero-poster.png": {"size": [2 * W, 2 * H]},
    })
    return gif, len(images), sum(durations)


def aggregate_from_traces(data=st.DATA, suite_runs=None):
    """The numbers on the end card, from the replayed traces and, for the settings the replay
    does not cover (3000 episodes, unseen mazes), the suite's records."""
    import csv
    traces = {k: st.load_json(f"traces_{k}.json", data)["runs"] for k in ("2d", "3d_1", "3d_3")}

    def count(kind):
        runs = traces[kind]
        return sum(r["converged_episode"] is not None for r in runs), len(runs)

    agg = {"n": len(traces["3d_3"]), "converged": count("3d_3")[0],
           "one_level": "{}/{}".format(*count("3d_1")), "two_d": "{}/{}".format(*count("2d"))}
    if suite_runs and os.path.exists(suite_runs):
        with open(suite_runs, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        x3 = [r for r in rows if r["kind"] == "3d_3" and r["config"] == "episodes_x3"]
        agg["converged_x3"] = sum(r["converged_episode"] != "" for r in x3)
        current = [r for r in rows if r["kind"] == "3d_3" and r["config"] == "current"]
        tests = 50 * len(current)
        reached = round(sum(float(r["heldout_greedy_reaches"]) * 50 for r in current))
        agg["heldout"] = f"{reached:,} of {tests:,}"
    return agg


def main():
    parser = argparse.ArgumentParser(description="Render the README hero; see the module docstring.")
    parser.add_argument("--data", default=st.DATA)
    parser.add_argument("--out", default=st.ASSETS)
    parser.add_argument("--suite", default=os.path.join(st.ASSETS, "..", "evaluation", "results", "runs.csv"),
                        help="the evaluation suite's runs.csv, for the 3000-episode and unseen-maze numbers")
    parser.add_argument("--frames", default=None, help="also write every frame as a PNG to this folder")
    args = parser.parse_args()
    st.style()
    run = st.load_json("featured.json", args.data)["success"]
    agg = aggregate_from_traces(args.data, args.suite)
    gif, count, ms = render(run, agg, args.out, args.frames)
    print(f"wrote {gif}: {count} frames, {ms / 1000:.1f} s, {os.path.getsize(gif) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
