"""Shared helpers for the tests. Run from maze_runner_rl/:  python -m pytest"""
import os
import random
from collections import deque

import numpy as np
import yaml

from src.agents.q_learning_agent import QLearningAgent
from src.environments.maze_3d_env import Maze3DEnvironment
from src.environments.maze_env import MazeEnvironment

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(PACKAGE_ROOT, "websocket", "config.yaml"), encoding="utf-8") as f:
    SETTINGS = yaml.safe_load(f)

KINDS = ("2d", "3d_1", "3d_3")


def make_env(kind, seed):
    """Build and seed an environment the way the experiment CLI does."""
    random.seed(seed)
    maze = SETTINGS["maze"]
    if kind == "2d":
        env = MazeEnvironment()
    else:
        depth = {"3d_1": 1, "3d_3": 3}[kind]
        env = Maze3DEnvironment(maze["default_width"], maze["default_height"], depth,
                                wall_density=maze["wall_density"], branching_factor=maze["branching_factor"],
                                dead_end_percentage=maze["dead_end_percentage"])
    env.reset(seed=seed)
    return env


def make_agent(env):
    a = SETTINGS["agent"]
    return QLearningAgent(env.num_states, env.action_space.n, learning_rate=a["learning_rate"],
                          epsilon=a["epsilon"], epsilon_decay=a["epsilon_decay"],
                          discount_factor=a["gamma"], epsilon_min=a["epsilon_min"])


def as_xyz(position):
    """(x, y) or (x, y, z) as an (x, y, z) tuple."""
    position = tuple(int(v) for v in position)
    return position + (0,) * (3 - len(position))


def as_grid(maze):
    """Maze as a [z, y, x] array; a 2D maze becomes a single level."""
    grid = np.asarray(maze)
    return grid[np.newaxis] if grid.ndim == 2 else grid


def open_cells(maze):
    grid = as_grid(maze)
    return {(int(x), int(y), int(z)) for z, y, x in zip(*np.nonzero(grid == 0))}


def grid_bfs(maze, start):
    """Distance from start to every reachable open cell, moving one cell along one axis
    at a time. Written independently of the environments' transition code."""
    grid = as_grid(maze)
    depth, height, width = grid.shape
    start = as_xyz(start)
    distance = {start: 0}
    queue = deque([start])
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nxt = (x + dx, y + dy, z + dz)
            nx, ny, nz = nxt
            if (0 <= nx < width and 0 <= ny < height and 0 <= nz < depth
                    and grid[nz, ny, nx] == 0 and nxt not in distance):
                distance[nxt] = distance[(x, y, z)] + 1
                queue.append(nxt)
    return distance
