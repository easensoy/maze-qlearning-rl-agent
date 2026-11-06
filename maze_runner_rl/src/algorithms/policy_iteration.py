import numpy as np


class PolicyIteration:
    """Policy iteration for an environment with known, deterministic dynamics.

    The environment must provide `step_from_state(state, action) -> (next_state, reward, done)`
    and either `num_states` or a Discrete `observation_space` (both maze envs provide
    `num_states` and `step_from_state`).
    """

    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.num_states = getattr(env, "num_states", None) or env.observation_space.n
        self.num_actions = env.action_space.n
        self.policy = np.random.choice(self.num_actions, size=self.num_states)
        self.value_function = np.zeros(self.num_states)

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.num_states):
                v = self.value_function[state]
                action = self.policy[state]
                next_state, reward, done = self.env.step_from_state(state, action)
                self.value_function[state] = reward + self.gamma * self.value_function[next_state] * (1 - done)
                delta = max(delta, abs(v - self.value_function[state]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in range(self.num_states):
            old_action = self.policy[state]
            action_values = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                next_state, reward, done = self.env.step_from_state(state, action)
                action_values[action] = reward + self.gamma * self.value_function[next_state] * (1 - done)
            self.policy[state] = np.argmax(action_values)
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def iterate(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
        return self.policy, self.value_function
