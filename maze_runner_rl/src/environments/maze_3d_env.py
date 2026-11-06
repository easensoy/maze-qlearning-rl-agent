import random

import gymnasium as gym
import numpy as np

from .maze_3d_generator import Maze3DGenerator
from config.config import CONFIG


class Maze3DEnvironment(gym.Env):
    """Multi-level maze following the Gymnasium API (same contract as MazeEnvironment).

    Actions: 0 -y, 1 +y, 2 -x, 3 +x, 4 down a level (-z), 5 up a level (+z).
    The maze is kept across resets; pass a seed or options={"new_maze": True} for a new one.
    info["state"] is z * width * height + y * width + x.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, width=11, height=11, depth=5, render_mode=None, **maze_params):
        super().__init__()
        self.maze_generator = Maze3DGenerator(width, height, depth, **maze_params)
        self.render_mode = render_mode
        self.maze = None
        self.agent_position = None

        self.actions = [
            [0, -1, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, 0, -1],
            [0, 0, 1]
        ]

        g = self.maze_generator
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=CONFIG.maze.agent_value,
            shape=(g.depth, g.height, g.width),
            dtype=np.uint8
        )

    @property
    def num_states(self):
        g = self.maze_generator
        return g.width * g.height * g.depth

    def state_index(self, position=None):
        x, y, z = self.agent_position if position is None else position
        g = self.maze_generator
        return z * (g.width * g.height) + y * g.width + x

    def position_from_state(self, state):
        g = self.maze_generator
        z, rest = divmod(state, g.width * g.height)
        y, x = divmod(rest, g.width)
        return x, y, z

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None:
            self.maze_generator.rng = random.Random(seed)
        if self.maze is None or seed is not None or options.get("new_maze", False):
            self.maze_generator.generate_maze()
            self.maze = self.maze_generator.get_maze()
        self.agent_position = list(self.maze_generator.get_start())
        return self._get_observation(), self._get_info()

    def _transition(self, position, action):
        dx, dy, dz = self.actions[action]
        x, y, z = position
        new_position = (x + dx, y + dy, z + dz)

        valid_move = self.maze_generator.is_valid_position(*new_position)
        if not valid_move:
            new_position = (x, y, z)

        terminated = new_position == tuple(self.maze_generator.get_end())
        if terminated:
            reward = CONFIG.rewards.goal_reward
        else:
            reward = CONFIG.penalties.step_penalty if valid_move else CONFIG.penalties.wall_penalty
        return new_position, float(reward), terminated

    def step(self, action):
        new_position, reward, terminated = self._transition(tuple(self.agent_position), action)
        self.agent_position = list(new_position)
        return self._get_observation(), reward, terminated, False, self._get_info()

    def step_from_state(self, state, action):
        """Deterministic model of the dynamics for planning (e.g. PolicyIteration):
        returns (next_state, reward, done). Walls and the goal are absorbing."""
        x, y, z = self.position_from_state(state)
        if self.maze[z, y, x] != 0 or (x, y, z) == tuple(self.maze_generator.get_end()):
            return state, 0.0, True
        new_position, reward, terminated = self._transition((x, y, z), action)
        return self.state_index(new_position), reward, terminated

    def render(self):
        return f"Agent at: {self.agent_position}, Goal at: {self.maze_generator.get_end()}"

    def _get_observation(self):
        observation = self.maze.copy().astype(np.uint8)

        if self.agent_position:
            x, y, z = self.agent_position
            observation[z, y, x] = CONFIG.maze.agent_value

        return observation

    def _get_info(self):
        return {"state": self.state_index(), "agent_position": tuple(self.agent_position)}
