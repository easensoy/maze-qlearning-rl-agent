"""Colours, type and maze drawing shared by the README visuals in docs/assets.

One palette for the animation, the results figure and the policy figure: charcoal and
off-white, blue for exploration and what the agent has learned so far, and a single amber
accent for the learned greedy route. A BFS shortest route is always a dashed off-white
line, the start a hollow ring and the goal a filled disk.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from experiments import evaluate as ev  # noqa: E402

ASSETS = os.path.normpath(os.path.join(ev.PACKAGE_ROOT, "..", "docs", "assets"))
DATA = os.path.join(ASSETS, "data")

BG = "#101318"          # page
PANEL = "#171b22"       # a level: its walls
FLOOR = "#2a313c"       # open cells
RULE = "#2c333d"        # hairlines, grid
INK = "#ecebe7"         # off-white: text, BFS route, markers
INK2 = "#a7afba"        # secondary text
INK3 = "#6f7782"        # tertiary text, ticks
BLUE = "#3fb6f2"        # exploration and training episodes
BLUE_SOFT = "#1d4e6e"   # cells from which the greedy policy already reaches the goal
AMBER = "#f5a524"       # the learned greedy route (the one accent)

FONT_FAMILY = ["Segoe UI", "DejaVu Sans"]
MONO_FAMILY = ["Cascadia Mono", "DejaVu Sans Mono"]


def font_in_use():
    """The typeface actually used (Segoe UI where installed, otherwise DejaVu Sans)."""
    for name in FONT_FAMILY:
        try:
            # A list, not a bare string: a string would be parsed as a fontconfig pattern.
            font_manager.findfont(font_manager.FontProperties(family=[name]), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Sans"


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "font.family": "sans-serif", "font.sans-serif": FONT_FAMILY, "font.size": 10,
        "text.color": INK, "axes.edgecolor": RULE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK3, "ytick.color": INK3, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "grid.color": RULE, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
        "svg.fonttype": "none",
    })


def load_json(name, data=DATA):
    with open(os.path.join(data, name), encoding="utf-8") as f:
        return json.load(f)


def record_render(out, entries):
    """Note how the assets in `out` were drawn (data/render.json): the libraries, the typeface
    and each file's size. No timestamps, so drawing again from the same data changes nothing."""
    import PIL
    path = os.path.join(out, "data", "render.json")
    record = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    record.update({"matplotlib": matplotlib.__version__, "pillow": PIL.__version__, "numpy": np.__version__,
                   "typeface": font_in_use()})
    record.setdefault("assets", {}).update(entries)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")


class Maze:
    """A featured run's maze and the coordinates of its states (from capture_traces)."""

    def __init__(self, run):
        self.run = run
        self.width, self.height, self.depth = run["size"]
        self.grid = np.array([[[c == "#" for c in row] for row in level.split("\n")] for level in run["levels"]])
        self.states = run["states"]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.xyz = {s: tuple(p) for s, p in zip(self.states, run["state_xyz"])}
        self.xyz[run["goal_state"]] = tuple(run["goal"])
        self.start, self.goal = run["start_state"], run["goal_state"]

    def open(self, x, y, z):
        return 0 <= z < self.depth and 0 <= y < self.height and 0 <= x < self.width and not self.grid[z, y, x]

    def shafts(self, level):
        """(x, y) cells of `level` with an open cell directly above or below."""
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if self.open(x, y, level) and (self.open(x, y, level + 1) or self.open(x, y, level - 1))]

    def level_cells(self, z):
        return int((~self.grid[z]).sum())

    def segments(self, states):
        """Split a sequence of states into runs that stay on one level, and the vertical moves
        between them: ([(level, [(x, y), ...]), ...], [(x, y, from_level, to_level), ...])."""
        runs, moves = [], []
        for s in states:
            x, y, z = self.xyz[s]
            if runs and runs[-1][0] == z:
                runs[-1][1].append((x, y))
            else:
                if runs:
                    moves.append((x, y, runs[-1][0], z))
                runs.append((z, [(x, y)]))
        return runs, moves
