"""Both environments: Gymnasium API, maze kept across resets, and a transition model
(step_from_state) that matches the real dynamics, since planning and the greedy-route
checks rely on it."""
import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from helpers import KINDS, as_xyz, grid_bfs, make_env

from experiments.training_experiment import shortest_route_length


@pytest.mark.parametrize("kind", KINDS)
def test_check_env_passes_without_warnings(kind):
    env = make_env(kind, seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, skip_render_check=True)
    assert [str(w.message) for w in caught] == []


@pytest.mark.parametrize("kind", KINDS)
def test_reset_keeps_the_maze_and_regenerates_on_request(kind):
    env = make_env(kind, seed=0)
    first = np.array(env.maze, copy=True)
    env.reset()
    assert np.array_equal(env.maze, first), "maze must be kept between episodes"
    env.reset(options={"new_maze": True})
    assert not np.array_equal(env.maze, first), "new_maze must generate a new maze"
    env.reset(seed=0)
    assert np.array_equal(env.maze, first), "the same seed must give the same maze"


@pytest.mark.parametrize("kind", KINDS)
def test_step_matches_the_transition_model(kind):
    env = make_env(kind, seed=1)
    _, info = env.reset()
    state = info["state"]
    rng = np.random.default_rng(0)
    for _ in range(500):
        action = int(rng.integers(env.action_space.n))
        expected = env.step_from_state(state, action)
        observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation)
        assert truncated is False
        assert (info["state"], reward, terminated) == expected
        assert info["state"] == env.state_index(info["agent_position"])
        state = info["state"]
        if terminated:
            _, info = env.reset()
            state = info["state"]


@pytest.mark.parametrize("kind", KINDS)
def test_state_index_round_trips(kind):
    env = make_env(kind, seed=0)
    assert all(env.state_index(env.position_from_state(s)) == s for s in range(env.num_states))


@pytest.mark.parametrize("kind", KINDS)
def test_shortest_route_matches_an_independent_grid_search(kind):
    for seed in range(3):
        env = make_env(kind, seed)
        generator = env.maze_generator
        distance = grid_bfs(env.maze, generator.get_start())
        assert shortest_route_length(env) == distance[as_xyz(generator.get_end())]
