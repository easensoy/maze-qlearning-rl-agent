import random

import gymnasium as gym
import numpy as np

from .maze_generator import MazeGenerator
from config.config import CONFIG


class MazeEnvironment(gym.Env):
    """2D maze following the Gymnasium API.

    reset(seed=None, options=None) -> (observation, info)
        The maze is generated on the first reset and then kept, so an agent can learn
        it across episodes. A new maze is generated when a seed is passed or when
        options={"new_maze": True}.
    step(action) -> (observation, reward, terminated, truncated, info)
        truncated is always False: episode length limits belong to the training loop.
    info["state"] is the agent's cell index (y * width + x), a hashable state for
    tabular agents.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, width=None, height=None, render_mode=None):
        super().__init__()
        self.maze_generator = MazeGenerator(width, height)
        self.render_mode = render_mode
        self.maze = None
        self.agent_position = None
        self.actions = [CONFIG.actions.up, CONFIG.actions.down, CONFIG.actions.left, CONFIG.actions.right]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(
            low=CONFIG.environment.observation_low,
            high=CONFIG.maze.agent_value,
            shape=(self.maze_generator.height, self.maze_generator.width),
            dtype=np.uint8
        )

    @property
    def num_states(self):
        return self.maze_generator.width * self.maze_generator.height

    def state_index(self, position=None):
        x, y = self.agent_position if position is None else position
        return y * self.maze_generator.width + x

    def position_from_state(self, state):
        return state % self.maze_generator.width, state // self.maze_generator.width

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if seed is not None:
            self.maze_generator.rng = random.Random(seed)
        if self.maze is None or seed is not None or options.get("new_maze", False):
            self.maze_generator.generate_maze()
            self.maze = self.maze_generator.get_maze()
        self.agent_position = self.maze_generator.get_start()
        return self._get_observation(), self._get_info()

    def _transition(self, position, action):
        dx, dy = self.actions[action]
        x, y = position
        new_x, new_y = x + dx, y + dy

        valid_move = (CONFIG.maze.boundary_min <= new_x < self.maze_generator.width and
                      CONFIG.maze.boundary_min <= new_y < self.maze_generator.height and
                      self.maze[new_y][new_x] == CONFIG.maze.empty_cell)

        new_position = (new_x, new_y) if valid_move else (x, y)
        terminated = new_position == self.maze_generator.get_end()
        if terminated:
            reward = CONFIG.rewards.goal_reward
        else:
            reward = CONFIG.penalties.step_penalty if valid_move else CONFIG.penalties.wall_penalty
        return new_position, float(reward), terminated

    def step(self, action):
        self.agent_position, reward, terminated = self._transition(self.agent_position, action)
        return self._get_observation(), reward, terminated, False, self._get_info()

    def step_from_state(self, state, action):
        """Deterministic model of the dynamics for planning (e.g. PolicyIteration):
        returns (next_state, reward, done). Walls and the goal are absorbing."""
        x, y = self.position_from_state(state)
        if self.maze[y][x] != CONFIG.maze.empty_cell or (x, y) == self.maze_generator.get_end():
            return state, 0.0, True
        new_position, reward, terminated = self._transition((x, y), action)
        return self.state_index(new_position), reward, terminated

    def render(self):
        symbols = {
            self.agent_position: CONFIG.rendering.agent_symbol,
            self.maze_generator.get_start(): CONFIG.rendering.start_symbol,
            self.maze_generator.get_end(): CONFIG.rendering.end_symbol,
        }

        maze_str = CONFIG.rendering.empty_string
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                pos = (x, y)
                char = symbols.get(pos, CONFIG.rendering.wall_symbol if cell else CONFIG.rendering.empty_symbol)
                maze_str += char
            maze_str += CONFIG.rendering.line_separator
        return maze_str

    def _get_observation(self):
        observation = np.array(self.maze, dtype=np.uint8)

        agent_x, agent_y = self.agent_position
        observation[agent_y][agent_x] = CONFIG.maze.agent_value

        return observation

    def _get_info(self):
        return {"state": self.state_index(), "agent_position": tuple(self.agent_position)}
