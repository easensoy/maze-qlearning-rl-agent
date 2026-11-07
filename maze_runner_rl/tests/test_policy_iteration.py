"""PolicyIteration runs on the maze environments and finds a shortest route."""
import numpy as np
from helpers import make_env

from experiments.training_experiment import shortest_route_length
from src.algorithms.policy_iteration import PolicyIteration


def test_policy_iteration_finds_a_shortest_route():
    env = make_env("3d_1", seed=0)
    np.random.seed(0)
    policy, _ = PolicyIteration(env, gamma=0.99).iterate()
    generator = env.maze_generator
    state, goal = env.state_index(generator.get_start()), env.state_index(generator.get_end())
    moves = 0
    while state != goal and moves < 1000:
        state, _, _ = env.step_from_state(state, policy[state])
        moves += 1
    assert moves == shortest_route_length(env)
