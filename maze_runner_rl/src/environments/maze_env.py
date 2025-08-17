import gymnasium as gym
from maze_generator import MazeGenerator
import numpy as np
from config.config import CONFIG

class MazeEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.maze_generator = MazeGenerator()
        self.maze = None 
        self.agent_position = None
        self.actions = [CONFIG.actions.up, CONFIG.actions.down, CONFIG.actions.left, CONFIG.actions.right]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(
            low=CONFIG.environment.observation_low, 
            high=CONFIG.environment.observation_high, 
            shape=(CONFIG.environment.observation_width, CONFIG.environment.observation_height), 
            dtype=np.uint8
        )

    def reset(self):
        self.maze_generator.generate_maze()
        self.maze = self.maze_generator.get_maze()
        self.agent_position = self.maze_generator.get_start()
        return self._get_observation()
    
    def step(self, action):
        dx, dy = self.actions[action]
        new_x, new_y = self.agent_position[CONFIG.maze.x_index] + dx, self.agent_position[CONFIG.maze.y_index] + dy

        valid_move = (CONFIG.maze.boundary_min <= new_x < self.maze_generator.width and
                      CONFIG.maze.boundary_min <= new_y < self.maze_generator.height and
                      self.maze[new_y][new_x] == CONFIG.maze.empty_cell)
        
        reward = CONFIG.penalties.step_penalty if valid_move else CONFIG.penalties.wall_penalty

        if valid_move:
            self.agent_position = (new_x, new_y)

        done = self.agent_position == self.maze_generator.get_end()
        reward = CONFIG.rewards.goal_reward if done else reward

        return self._get_observation(), reward, done, {}
    
    def render(self):
        symbols = {
            self.agent_position: CONFIG.rendering.agent_symbol,
            self.maze_generator.get_start(): CONFIG.rendering.start_symbol,
            self.maze_generator.get_end(): CONFIG.rendering.end_symbol,
        }

        maze_str = CONFIG.rendering.empty_string
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                pos = (x,y)
                char = symbols.get(pos, CONFIG.rendering.wall_symbol if cell else CONFIG.rendering.empty_symbol)
                maze_str += char
            maze_str += CONFIG.rendering.line_separator
        return maze_str

    def _get_observation(self):
        observation = np.array(self.maze, dtype=np.uint8)

        agent_x, agent_y = self.agent_position
        observation[agent_y][agent_x] = CONFIG.maze.agent_value

        return observation

