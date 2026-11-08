"""Watkins's Q(lambda): Q-learning with eligibility traces that are cut after exploration.

References:
- C. J. C. H. Watkins (1989). Learning from Delayed Rewards. PhD thesis, University of Cambridge.
- R. S. Sutton and A. G. Barto (1998). Reinforcement Learning: An Introduction, 1st ed.,
  Section 7.6, "Q(lambda)": the tabular algorithm implemented here. Checked against the
  authors' Chapter 7 lecture slides, slide 24 ("Watkins's Q(lambda)").
- S. P. Singh and R. S. Sutton (1996). Reinforcement learning with replacing eligibility
  traces. Machine Learning 22, 123-158: replacing traces, where revisiting a state sets
  the trace of the action taken to 1 and those of its other actions to 0.

One step of the algorithm, from state s after taking action a:

    observe r, s'
    choose a' from s' (epsilon-greedy on Q, before this step's update)
    a* = argmax_b Q(s', b); if a' ties for the max, a* = a'
    delta = r + gamma * Q(s', a*) - Q(s, a)        (delta = r - Q(s, a) when s' is terminal)
    e(s, a) = e(s, a) + 1   (accumulating)   or   e(s, .) = 0, e(s, a) = 1   (replacing)
    Q = Q + alpha * delta * e                      for every (state, action)
    e = gamma * lambda * e  if a' = a*,  otherwise e = 0

The slide prints the delta line as r + gamma Q(s', a') - Q(s, a*); the target above,
r + gamma max_b Q(s', b) - Q(s, a), is the Q-learning target the section describes.

Everything else (Q-values start at 0, epsilon-greedy with random tie-breaking, the
epsilon schedule) is QLearningAgent's, so only the update differs from the baseline.
"""
from .q_learning_agent import QLearningAgent


class WatkinsQLambdaAgent(QLearningAgent):
    """Tabular Watkins's Q(lambda).

    Call begin_episode() before and end_episode() after every episode: traces belong to
    one episode. Within an episode, choose_action(s') returns the action a' that the
    update already chose for s' (the algorithm picks a' before it updates), so the
    agent acts with the same a' that decided whether the traces survive.

    Counters: `updates` is the number of Q-value changes made (one per non-zero trace per
    step), `cuts` the number of times exploration cut the traces.
    """

    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, epsilon=0.1,
                 epsilon_decay=0.995, discount_factor=0.95, epsilon_min=0.01, trace_decay=0.9, replacing=True):
        super().__init__(state_space_size, action_space_size, learning_rate=learning_rate, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, discount_factor=discount_factor, epsilon_min=epsilon_min)
        if not 0.0 <= trace_decay <= 1.0:
            raise ValueError("trace_decay (lambda) must be in [0, 1]")
        self.trace_decay = trace_decay
        self.replacing = replacing
        self.traces = {}          # (state, action) -> eligibility; only pairs with a non-zero trace
        self._next = None         # (state, action): a' chosen for the next step
        self.updates = 0
        self.cuts = 0

    def begin_episode(self):
        self.traces.clear()
        self._next = None

    def end_episode(self):
        self.traces.clear()
        self._next = None
        self.decay_epsilon()

    def behaviour_action(self, state):
        """The epsilon-greedy choice of QLearningAgent (a separate method so tests can fix it)."""
        return QLearningAgent.choose_action(self, state)

    def choose_action(self, state):
        pending, self._next = self._next, None
        if pending is not None and pending[0] == state:
            return pending[1]
        return self.behaviour_action(state)

    def update_q_table(self, state, action, reward, next_state, done=False):
        current = self.q_table[state][action]
        if done:
            target, keep_traces = reward, False
        else:
            values = self.action_values(next_state)
            best = max(values)
            next_action = self.behaviour_action(next_state)                 # a', from Q before the update
            self._next = (next_state, next_action)
            keep_traces = values[next_action] == best                       # a' = a*, ties included
            target = reward + self.discount_factor * best                   # Q(s', a*) = max_b Q(s', b)
        delta = target - current

        if self.replacing:
            for other in range(self.action_space_size):
                self.traces.pop((state, other), None)
            self.traces[(state, action)] = 1.0
        else:
            self.traces[(state, action)] = self.traces.get((state, action), 0.0) + 1.0

        step = self.learning_rate * delta
        for (s, a), trace in self.traces.items():
            self.q_table[s][a] += step * trace
        self.updates += len(self.traces)

        decay = self.discount_factor * self.trace_decay
        if keep_traces and decay > 0.0:
            for key in self.traces:
                self.traces[key] *= decay
        else:
            if not done and not keep_traces:
                self.cuts += 1
            self.traces.clear()
