import random
from collections import deque

import numpy as np


class Maze3DGenerator:
    """Multi-level maze on a (depth, height, width) grid; 0 = open cell, 1 = wall.

    Generation steps:
      1. depth-first (recursive backtracker) carving from the start, with occasional
         vertical shafts between levels
      2. extra short side corridors; their number scales with `branching_factor`
         (0.4 -> 10 branches). One that meets another corridor creates a loop; most
         do not, and step 4 trims them again as dead ends
      3. more dead ends by closing junction cells, towards `dead_end_percentage`
      4. dead-end tips pruned towards `wall_density`, the target share of wall cells;
         pruning stops early once no dead ends are left
    Steps 3 and 4 only ever close a cell if every open cell stays reachable from the
    start, so every generated maze is solvable and has no unreachable pockets.
    """

    def __init__(self, width=None, height=None, depth=None, wall_density=0.65, branching_factor=0.4,
                 dead_end_percentage=0.3, rng=None):
        self.width = max(width or 21, 15)
        self.height = max(height or 21, 15)
        self.depth = max(depth or 3, 1)

        if self.width % 2 == 0:
            self.width += 1
        if self.height % 2 == 0:
            self.height += 1
        if self.depth % 2 == 0:
            self.depth += 1

        self.wall_density = wall_density
        self.branching_factor = branching_factor
        self.dead_end_percentage = dead_end_percentage
        self.rng = rng or random.Random()

        self.maze = None
        self.start = (1, 1, 0)
        self.end = (self.width - 2, self.height - 2, min(self.depth - 1, 2))

    def generate_maze(self):
        self.maze = np.ones((self.depth, self.height, self.width), dtype=int)

        self._carve_maze_from(self.start)

        end_x, end_y, end_z = self.end
        self.maze[end_z, end_y, end_x] = 0

        if not self._is_reachable(self.start, self.end):
            self._carve_direct_path(self.start, self.end)

        self._add_branching_corridors()

        self._enhance_dead_end_visibility()

        self._adjust_wall_density()

        if not self._is_reachable(self.start, self.end):
            self._carve_direct_path(self.start, self.end)

        print(f"Generated maze: {self._calculate_wall_percentage():.1f}% walls, "
              f"{len(self._find_dead_ends())} dead ends")

    def _directions_for(self, x, y, z):
        """Moves of two cells (skipping the wall in between), shuffled once per visit."""
        directions = [
            (0, -2, 0),   # North
            (0, 2, 0),    # South
            (-2, 0, 0),   # West
            (2, 0, 0),    # East
        ]
        if self.depth > 1 and self.rng.random() < 0.3:
            if z > 1:
                directions.append((0, 0, -2))  # Down
            if z < self.depth - 2:
                directions.append((0, 0, 2))   # Up
        self.rng.shuffle(directions)
        return iter(directions)

    def _carve_maze_from(self, start):
        """Recursive backtracking with an explicit stack (no recursion-depth limit)."""
        x, y, z = start
        self.maze[z, y, x] = 0
        stack = [((x, y, z), self._directions_for(x, y, z))]
        while stack:
            (x, y, z), directions = stack[-1]
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if self._is_valid_maze_cell(nx, ny, nz) and self.maze[nz, ny, nx] == 1:
                    self.maze[z + dz // 2, y + dy // 2, x + dx // 2] = 0
                    self.maze[nz, ny, nx] = 0
                    stack.append(((nx, ny, nz), self._directions_for(nx, ny, nz)))
                    break
            else:
                stack.pop()

    def _is_valid_maze_cell(self, x, y, z):
        """Check if coordinates are within maze bounds for carving"""
        return (1 <= x < self.width - 1 and
                1 <= y < self.height - 1 and
                0 <= z < self.depth)

    def _carve_direct_path(self, start, end):
        """Carve an L-shaped path from start to end (x first, then y, then z)."""
        x1, y1, z1 = start
        x2, y2, z2 = end
        current_x, current_y, current_z = x1, y1, z1

        while current_x != x2:
            current_x += 1 if x2 > current_x else -1
            self.maze[current_z, current_y, current_x] = 0

        while current_y != y2:
            current_y += 1 if y2 > current_y else -1
            self.maze[current_z, current_y, current_x] = 0

        while current_z != z2:
            current_z += 1 if z2 > current_z else -1
            self.maze[current_z, current_y, current_x] = 0

    def _add_branching_corridors(self):
        """Carve short corridors off existing ones; a branch that meets another corridor
        creates a loop (an alternative route). branching_factor 0.4 gives 10 branches,
        0.2 gives 5, 0.6 gives 15."""
        corridors = [(x, y, z)
                     for z in range(self.depth)
                     for y in range(1, self.height - 1)
                     for x in range(1, self.width - 1)
                     if self.maze[z, y, x] == 0]

        num_branches = min(len(corridors) // 4, int(round(25 * self.branching_factor)))
        for _ in range(num_branches):
            if not corridors:
                break
            x, y, z = self.rng.choice(corridors)
            dx, dy, dz = self.rng.choice([(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)])
            for step in range(1, 4):
                nx, ny, nz = x + dx * step, y + dy * step, z + dz * step
                if 1 <= nx < self.width - 1 and 1 <= ny < self.height - 1 and 0 <= nz < self.depth:
                    self.maze[nz, ny, nx] = 0
                else:
                    break

    def _enhance_dead_end_visibility(self):
        """Move the number of dead ends towards dead_end_percentage of the open cells."""
        dead_ends = self._find_dead_ends()
        target_dead_ends = int(self._count_path_cells() * self.dead_end_percentage)

        print(f"Found {len(dead_ends)} dead ends, target: {target_dead_ends}")

        if len(dead_ends) < target_dead_ends:
            self._create_additional_dead_ends(target_dead_ends - len(dead_ends))
        elif len(dead_ends) > target_dead_ends * 1.5:
            self._reduce_excessive_dead_ends(len(dead_ends) - target_dead_ends)

    def _neighbours(self, x, y, z, vertical=True):
        directions = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)]
        if vertical:
            directions += [(0, 0, 1), (0, 0, -1)]
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth:
                yield nx, ny, nz

    def _find_dead_ends(self):
        """Open cells (inside the border) with exactly one open neighbour."""
        dead_ends = []
        for z in range(self.depth):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.maze[z, y, x] != 0:
                        continue
                    open_neighbours = sum(1 for nx, ny, nz in self._neighbours(x, y, z, vertical=self.depth > 1)
                                          if self.maze[nz, ny, nx] == 0)
                    if open_neighbours == 1:
                        dead_ends.append((x, y, z))
        return dead_ends

    def _count_path_cells(self):
        """Count total number of path cells"""
        return int(np.sum(self.maze == 0))

    def _all_open_cells_connected(self):
        return len(self._reachable_from(self.start)) == self._count_path_cells()

    def _create_additional_dead_ends(self, needed):
        """Close a neighbour of a junction to create a dead end, but only if every open
        cell (including the goal) stays reachable from the start."""
        for _ in range(needed):
            for _attempt in range(100):
                x = self.rng.randint(2, self.width - 3)
                y = self.rng.randint(2, self.height - 3)
                z = self.rng.randint(0, min(self.depth - 1, 2))
                if self.maze[z, y, x] != 0:
                    continue
                neighbours = [(nx, ny, nz) for nx, ny, nz in self._neighbours(x, y, z, vertical=False)
                              if self.maze[nz, ny, nx] == 0]
                if len(neighbours) < 3:
                    continue
                nx, ny, nz = self.rng.choice(neighbours)
                if (nx, ny, nz) in (self.start, self.end):
                    continue
                self.maze[nz, ny, nx] = 1
                if self._all_open_cells_connected():
                    break
                self.maze[nz, ny, nx] = 0

    def _reduce_excessive_dead_ends(self, excess):
        """Reduce excessive dead ends by extending some of them into a wall (opens cells only)."""
        dead_ends = self._find_dead_ends()

        for _ in range(min(excess, len(dead_ends) // 2)):
            if len(dead_ends) < 2:
                break
            dead_end = self.rng.choice(dead_ends)
            dead_ends.remove(dead_end)

            x, y, z = dead_end
            directions = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)]
            self.rng.shuffle(directions)

            for dx, dy, dz in directions:
                nx, ny = x + dx * 2, y + dy * 2
                if (1 <= nx < self.width - 1 and
                        1 <= ny < self.height - 1 and
                        self.maze[z, ny, nx] == 1):
                    self.maze[z, y + dy, x + dx] = 0
                    self.maze[z, ny, nx] = 0
                    break

    def _adjust_wall_density(self):
        """Trim dead-end tips until walls make up wall_density of the maze.
        Removing a tip never disconnects anything; start and goal are never removed.
        Stops early once no dead ends are left (only loops and the start-to-goal route
        remain), so a high wall_density may not be reached. A maze that already has
        more walls than wall_density is left as is."""
        total = self.maze.size
        while True:
            missing = int(np.ceil(self.wall_density * total)) - int(np.sum(self.maze == 1))
            if missing <= 0:
                return
            tips = [c for c in self._find_dead_ends() if c not in (self.start, self.end)]
            if not tips:
                return
            self.rng.shuffle(tips)
            for x, y, z in tips[:missing]:
                self.maze[z, y, x] = 1

    def _calculate_wall_percentage(self):
        """Calculate current wall density percentage"""
        total_cells = self.maze.size
        wall_cells = np.sum(self.maze == 1)
        return (wall_cells / total_cells) * 100

    def _reachable_from(self, start):
        """All open cells reachable from start (6-neighbour moves, like the environment)."""
        start = tuple(start)
        if self.maze[start[2], start[1], start[0]] != 0:
            return set()
        seen = {start}
        queue = deque([start])
        while queue:
            x, y, z = queue.popleft()
            for nx, ny, nz in self._neighbours(x, y, z):
                if (nx, ny, nz) not in seen and self.maze[nz, ny, nx] == 0:
                    seen.add((nx, ny, nz))
                    queue.append((nx, ny, nz))
        return seen

    def _is_reachable(self, start, end):
        """BFS check that end is reachable from start"""
        return tuple(end) in self._reachable_from(start)

    def get_maze(self):
        return self.maze

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def is_valid_position(self, x, y, z):
        return (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth and
                self.maze[z, y, x] == 0)
