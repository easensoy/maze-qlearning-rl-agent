import random
from collections import defaultdict


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    Q-values start at 0 for every action, including actions never tried in a state.
    With negative step rewards this makes untried actions look attractive, which is
    what drives systematic exploration of the maze.
    """

    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, epsilon=0.1,
                 epsilon_decay=0.995, discount_factor=0.95, epsilon_min=0.01):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor

        self.q_table = defaultdict(lambda: defaultdict(float))

    def action_values(self, state):
        """Q(state, a) for every action; actions never tried count as 0.0."""
        q = self.q_table.get(state)
        if not q:
            return [0.0] * self.action_space_size
        return [q.get(a, 0.0) for a in range(self.action_space_size)]

    def greedy_action(self, state):
        """Best action without exploration; ties go to the lowest action index."""
        values = self.action_values(state)
        return max(range(self.action_space_size), key=values.__getitem__)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_space_size)
        values = self.action_values(state)
        best = max(values)
        return random.choice([a for a, v in enumerate(values) if v == best])

    def update_q_table(self, state, action, reward, next_state, done=False):
        current_q = self.q_table[state][action]
        max_next_q = 0.0 if done else max(self.action_values(next_state))
        target = reward + self.discount_factor * max_next_q
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_value(self, state, action):
        return self.action_values(state)[action]

    def get_state_values(self, state):
        return dict(self.q_table.get(state, {}))
