"""Exploded isometric drawing of a multi-level maze, for the README animation.

Geometry only; the data comes from recorded runs. Each level is drawn as a plate of thin
slabs, one per open cell, with walls left empty so no corridor hides another. Levels are
separated vertically; a shaft cell is drawn as a column joining the levels it connects.

The camera looks from behind the start corner, so the start (x, y small) is at the front
of the bottom plate and the goal (x, y large) at the back of the top plate: a route climbs
from bottom-front to top-back. Screen coordinates are pixels with y pointing down.
"""
import numpy as np
from matplotlib.patches import Polygon

from experiments import asset_style as st

# Slab colours: the top face is lit, the two side faces are progressively darker. Corridors
# stay quiet so the overlays carry the message; cells the greedy policy already leads to the
# goal from turn deep blue; shaft cells are a lighter grey, so the columns' ends are visible.
TOP, LEFT, RIGHT = "#343c48", "#232931", "#1a1f26"
TOP_LEARNED, LEFT_LEARNED, RIGHT_LEARNED = "#1f6c98", "#17506f", "#123d55"
TOP_SHAFT = "#525c6a"
COLUMN = "#4d5663"
TRAIL = "#67e8f9"          # exploration: cyan, distinct from the blue of learned cells


class IsoMaze:
    """Projects maze cells (x, y, level) to the screen and draws plates, shafts and paths."""

    def __init__(self, grid, origin, unit=12.0, level_gap=135.0, slab=6.0):
        self.grid = np.asarray(grid)              # [level, y, x], 0 = open
        self.levels, self.height, self.width = self.grid.shape
        self.ox, self.oy = origin                 # screen point of the back corner of the bottom plate
        self.u, self.gap, self.slab = unit, level_gap, slab
        self.scale = 1.0                          # line widths are multiplied by this (2 for the poster)

    # --- projection ---------------------------------------------------------------------------
    def depth(self, x, y):
        return (self.width - 1 - x) + (self.height - 1 - y)

    def centre(self, x, y, level, lift=0.0):
        """Screen centre of the top face of cell (x, y) on `level`."""
        sx = self.ox + ((self.height - 1 - y) - (self.width - 1 - x)) * self.u
        sy = self.oy + self.depth(x, y) * self.u / 2 - level * self.gap - lift
        return sx, sy

    def top_face(self, x, y, level):
        cx, cy = self.centre(x, y, level)
        u, h = self.u, self.u / 2
        return [(cx, cy - h), (cx + u, cy), (cx, cy + h), (cx - u, cy)]

    def plate_outline(self, level):
        """The diamond around a whole level: top, right, bottom and left extreme points."""
        points = []
        for x, y in ((0, 0), (self.width - 1, 0), (0, self.height - 1), (self.width - 1, self.height - 1)):
            points += self.top_face(x, y, level)
        return [min(points, key=lambda p: p[1]), max(points, key=lambda p: p[0]),
                max(points, key=lambda p: p[1]), min(points, key=lambda p: p[0])]

    def extent(self):
        xs, ys = [], []
        for level in range(self.levels):
            for x, y in ((0, 0), (self.width - 1, 0), (0, self.height - 1), (self.width - 1, self.height - 1)):
                cx, cy = self.centre(x, y, level)
                xs += [cx - self.u, cx + self.u]
                ys += [cy - self.u / 2, cy + self.u / 2 + self.slab]
        return min(xs), max(xs), min(ys), max(ys)

    # --- cells ----------------------------------------------------------------------------------
    def open_cells(self, level):
        ys, xs = np.nonzero(self.grid[level] == 0)
        return sorted(zip(xs.tolist(), ys.tolist()), key=lambda c: -self.depth(*c))   # back to front

    def is_shaft(self, x, y, level):
        up = level + 1 < self.levels and self.grid[level + 1, y, x] == 0
        down = level > 0 and self.grid[level - 1, y, x] == 0
        return up or down

    def draw_slab(self, ax, x, y, level, learned=False, zorder=2):
        cx, cy = self.centre(x, y, level)
        u, h, t = self.u, self.u / 2, self.slab
        if learned:
            top, left, right = TOP_LEARNED, LEFT_LEARNED, RIGHT_LEARNED
        else:
            top = TOP_SHAFT if self.is_shaft(x, y, level) else TOP
            left, right = LEFT, RIGHT
        faces = [
            ([(cx - u, cy), (cx, cy + h), (cx, cy + h + t), (cx - u, cy + t)], left),
            ([(cx, cy + h), (cx + u, cy), (cx + u, cy + t), (cx, cy + h + t)], right),
            (self.top_face(x, y, level), top),
        ]
        for points, colour in faces:
            ax.add_patch(Polygon(points, closed=True, facecolor=colour, edgecolor=colour, linewidth=0.4 * self.scale,
                                 zorder=zorder))

    def draw_level(self, ax, level, learned=frozenset(), zorder=2):
        for x, y in self.open_cells(level):
            self.draw_slab(ax, x, y, level, (x, y, level) in learned, zorder)

    def draw_columns(self, ax, lower, zorder=2, colour=COLUMN, width=2.2, alpha=0.9):
        """Vertical shaft columns between `lower` and lower + 1."""
        for x, y in self.open_cells(lower):
            if lower + 1 < self.levels and self.grid[lower + 1, y, x] == 0:
                x0, y0 = self.centre(x, y, lower)
                x1, y1 = self.centre(x, y, lower + 1)
                ax.plot([x0, x1], [y0 + self.slab, y1 + self.slab], color=colour, linewidth=width * self.scale, alpha=alpha,
                        solid_capstyle="round", zorder=zorder)

    # --- overlays -------------------------------------------------------------------------------
    def points(self, cells, lift=1.5):
        return [self.centre(x, y, z, lift) for x, y, z in cells]

    def segments_by_level(self, cells):
        """A path split into runs on one level and the vertical moves between levels."""
        runs, verticals, current = [], [], [cells[0]]
        for a, b in zip(cells, cells[1:]):
            if a[2] != b[2]:
                runs.append(current)
                verticals.append((a, b))
                current = [b]
            else:
                current.append(b)
        runs.append(current)
        return runs, verticals

    def draw_path(self, ax, cells, colour, width=2.4, alpha=1.0, dashes=None, zorder=6, level_zorder=None):
        """A path of cells; runs on a level get that level's zorder band, so paths on the
        lower plate stay under the upper plate."""
        if len(cells) < 2:
            return
        runs, verticals = self.segments_by_level(cells)
        for run in runs:
            if len(run) < 2:
                continue
            z = run[0][2]
            xs, ys = zip(*self.points(run))
            line, = ax.plot(xs, ys, color=colour, linewidth=width, alpha=alpha, solid_capstyle="round",
                            solid_joinstyle="round", zorder=(level_zorder or {}).get(z, zorder))
            if dashes:
                line.set_dashes(dashes)
        for a, b in verticals:
            (x0, y0), (x1, y1) = self.points([a, b])
            z = max(a[2], b[2])
            line, = ax.plot([x0, x1], [y0, y1], color=colour, linewidth=width, alpha=alpha, solid_capstyle="round",
                            zorder=(level_zorder or {}).get(z, zorder) - 0.5)
            if dashes:
                line.set_dashes(dashes)
