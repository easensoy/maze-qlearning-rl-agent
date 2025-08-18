import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, epsilon=0.1, epsilon_decay=0.995, discount_factor=0.95):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.discount_factor = discount_factor
        
        self.q_table = defaultdict(lambda: defaultdict(float))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            q_values = self.q_table[state]
            if not q_values:
                return random.randint(0, self.action_space_size - 1)
            return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        
        next_q_values = self.q_table[next_state]
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_value(self, state, action):
        return self.q_table[state][action]

    def get_state_values(self, state):
        return dict(self.q_table[state])

