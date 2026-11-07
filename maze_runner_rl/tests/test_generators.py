"""Maze generators: always solvable, fully connected, seedable; density settings take effect."""
import random

import numpy as np
import pytest
from helpers import as_xyz, grid_bfs, open_cells

from src.environments.maze_3d_generator import Maze3DGenerator
from src.environments.maze_generator import MazeGenerator

SEEDS = range(25)


def generate_3d(size, seed, **params):
    generator = Maze3DGenerator(*size, rng=random.Random(seed), **params)
    generator.generate_maze()
    return generator


def wall_share(generator):
    return float(np.mean(np.asarray(generator.get_maze()) == 1))


def test_2d_mazes_are_solvable_and_fully_connected():
    for seed in SEEDS:
        generator = MazeGenerator(rng=random.Random(seed))
        generator.generate_maze()
        reachable = grid_bfs(generator.get_maze(), generator.get_start())
        assert as_xyz(generator.get_end()) in reachable, f"seed {seed}: goal unreachable"
        assert set(reachable) == open_cells(generator.get_maze()), f"seed {seed}: cut-off cells"


@pytest.mark.parametrize("size", [(21, 21, 1), (21, 21, 3), (31, 31, 1), (15, 15, 5)])
def test_3d_mazes_are_solvable_and_fully_connected(size):
    # Before the fix the dead-end pass walled off the goal in 88% of 21x21x1 mazes.
    for seed in SEEDS:
        generator = generate_3d(size, seed)
        reachable = grid_bfs(generator.get_maze(), generator.get_start())
        assert as_xyz(generator.get_end()) in reachable, f"seed {seed}: goal unreachable"
        assert set(reachable) == open_cells(generator.get_maze()), f"seed {seed}: cut-off cells"


def test_generation_is_reproducible_per_seed():
    a, b, c = generate_3d((21, 21, 3), 7), generate_3d((21, 21, 3), 7), generate_3d((21, 21, 3), 8)
    assert np.array_equal(a.get_maze(), b.get_maze())
    assert not np.array_equal(a.get_maze(), c.get_maze())


def test_wall_density_is_a_target_that_never_disconnects_the_maze():
    low, high = [], []
    for seed in range(20):
        for density, shares in ((0.60, low), (0.85, high)):
            generator = generate_3d((21, 21, 1), seed, wall_density=density)
            shares.append(wall_share(generator))
            assert set(grid_bfs(generator.get_maze(), generator.get_start())) == open_cells(generator.get_maze())
    assert min(low) >= 0.60  # a reachable target is met
    assert np.mean(high) > np.mean(low) + 0.15  # a higher target means more walls (it may stop short)


def test_branching_factor_changes_the_maze():
    for seed in range(10):
        a = generate_3d((21, 21, 1), seed, branching_factor=0.2)
        b = generate_3d((21, 21, 1), seed, branching_factor=0.6)
        assert not np.array_equal(a.get_maze(), b.get_maze()), f"seed {seed}"
