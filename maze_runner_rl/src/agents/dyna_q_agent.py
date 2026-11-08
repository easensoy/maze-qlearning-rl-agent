"""Tabular Dyna-Q: Q-learning plus planning updates replayed from a learned model.

References:
- R. S. Sutton (1990). Integrated architectures for learning, planning, and reacting
  based on approximating dynamic programming. Proceedings of the 7th International
  Conference on Machine Learning, 216-224.
- R. S. Sutton and A. G. Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.,
  Section 8.2, "Dyna: Integrated Planning, Acting, and Learning", the Tabular Dyna-Q
  box. Checked against Sutton's lecture slides on planning and learning ("The Dyna-Q
  Algorithm", reproducing the book's Figure 8.4).

After every real step from S with action A to reward R and state S':

    (d) direct RL:       Q(S, A) += alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
    (e) model learning:  Model(S, A) = R, S'     (the environment is deterministic)
    (f) planning, n times:
            S = a random previously observed state
            A = a random action previously taken in S
            R, S' = Model(S, A)
            Q(S, A) += alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]

Without (e) and (f) this is one-step Q-learning, which is the baseline. The model here
also records whether S' ended the episode, so a planning update into the goal does not
bootstrap, exactly as the real update does not.

The learned model knows only transitions the agent has made: it is a table of observed
(state, action) -> (reward, next state, terminal) outcomes, never the maze itself.
Planning draws from its own random generator, so planning does not shift the stream of
random numbers that drives exploration.
"""
import random

from .q_learning_agent import QLearningAgent


class DynaQAgent(QLearningAgent):
    """Tabular Dyna-Q with `planning_steps` (n) planning updates per real step.

    Counters: `updates` counts every Q-value update (real and planning) and
    `planning_updates` the planning ones alone, so the extra computation stays visible.
    """

    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, epsilon=0.1,
                 epsilon_decay=0.995, discount_factor=0.95, epsilon_min=0.01, planning_steps=10, planning_seed=None):
        super().__init__(state_space_size, action_space_size, learning_rate=learning_rate, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, discount_factor=discount_factor, epsilon_min=epsilon_min)
        if planning_steps < 0:
            raise ValueError("planning_steps must be >= 0")
        self.planning_steps = planning_steps
        self.model = {}               # (state, action) -> (reward, next_state, done), last outcome seen
        self.observed_states = []     # states with at least one action taken, in order of first visit
        self.taken_actions = {}       # state -> actions taken there, in order of first use
        self.planning_rng = random.Random(planning_seed)
        self.updates = 0
        self.planning_updates = 0

    def begin_episode(self):
        pass

    def end_episode(self):
        self.decay_epsilon()

    def update_q_table(self, state, action, reward, next_state, done=False):
        super().update_q_table(state, action, reward, next_state, done)          # (d)
        self.updates += 1

        actions = self.taken_actions.get(state)                                   # (e)
        if actions is None:
            actions = self.taken_actions[state] = []
            self.observed_states.append(state)
        if (state, action) not in self.model:
            actions.append(action)
        self.model[(state, action)] = (reward, next_state, done)

        rng = self.planning_rng                                                   # (f)
        for _ in range(self.planning_steps):
            s = rng.choice(self.observed_states)
            a = rng.choice(self.taken_actions[s])
            r, s_next, d = self.model[(s, a)]
            QLearningAgent.update_q_table(self, s, a, r, s_next, d)
        self.updates += self.planning_steps
        self.planning_updates += self.planning_steps
