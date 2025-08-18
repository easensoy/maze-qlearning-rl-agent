import numpy as np

class PolicyIteration:
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = np.random.choice(env.action_space.n, size=env.observation_space.n)
        self.value_function = np.zeros(env.observation_space.n)
    
    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.env.observation_space.n):
                v = self.value_function[state]
                action = self.policy[state]
                next_state, reward, done = self.env.step_from_state(state, action)
                self.value_function[state] = reward + self.gamma * self.value_function[next_state] * (1 - done)
                delta = max(delta, abs(v - self.value_function[state]))
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        policy_stable = True
        for state in range(self.env.observation_space.n):
            old_action = self.policy[state]
            action_values = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
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