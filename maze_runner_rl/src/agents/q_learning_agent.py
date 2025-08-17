from environments.maze_env import MazeEnvironment
import numpy as np
from config.config import CONFIG


class QLearningAgent:
    def __init__(self, maze_width, maze_height, learning_rate=None, epsilon=None):
        self.actions = [CONFIG.actions.up, CONFIG.actions.down, CONFIG.actions.left, CONFIG.actions.right]
        self.learning_rate = learning_rate or CONFIG.agent.learning_rate
        self.epsilon = epsilon or CONFIG.agent.epsilon
        self.q_table = {}

    def choose_action(self, observation):
        agent_positions = np.where(observation == CONFIG.maze.agent_value)
        if len(agent_positions[CONFIG.agent.min_positions]) > CONFIG.agent.min_positions:
            return np.random.choice(len(self.actions))
        return CONFIG.agent.default_action

