import gymnasium as gym
from .maze_3d_generator import Maze3DGenerator
import numpy as np
from config.config import CONFIG

class Maze3DEnvironment(gym.Env):
    def __init__(self, width=11, height=11, depth=5, **maze_params):
        super().__init__()
        self.maze_generator = Maze3DGenerator(width, height, depth, **maze_params)
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
        
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(depth, height, width), 
            dtype=np.uint8
        )

    def reset(self):
        self.maze_generator.generate_maze()
        self.maze = self.maze_generator.get_maze()
        self.agent_position = list(self.maze_generator.get_start())
        return self._get_observation()
    
    def step(self, action):
        dx, dy, dz = self.actions[action]
        new_x = self.agent_position[0] + dx
        new_y = self.agent_position[1] + dy
        new_z = self.agent_position[2] + dz

        valid_move = self.maze_generator.is_valid_position(new_x, new_y, new_z)
        
        reward = -0.1 if valid_move else -1.0

        if valid_move:
            self.agent_position = [new_x, new_y, new_z]

        goal_pos = self.maze_generator.get_end()
        done = tuple(self.agent_position) == goal_pos
        reward = 100 if done else reward

        return self._get_observation(), reward, done, {}
    
    def render(self, mode='text'):
        return f"Agent at: {self.agent_position}, Goal at: {self.maze_generator.get_end()}"

    def _get_observation(self):
        observation = self.maze.copy().astype(np.uint8)
        
        if self.agent_position:
            x, y, z = self.agent_position
            observation[z, y, x] = 2

        return observation

