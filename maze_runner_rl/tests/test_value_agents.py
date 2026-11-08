"""Watkins's Q(lambda) and Dyna-Q: the update rules that define them, checked by hand.

Small hand-built transitions (state 0 -> 1 -> goal 2, and wall bumps that stay put) are
fed straight to update_q_table, so every expected value can be worked out on paper.
"""
import pytest

from src.agents.dyna_q_agent import DynaQAgent
from src.agents.q_lambda_agent import WatkinsQLambdaAgent
from src.agents.q_learning_agent import QLearningAgent

ALPHA, GAMMA, LAM = 0.5, 0.9, 0.8


class FixedNext(WatkinsQLambdaAgent):
    """Q(lambda) whose next action a' is set by the test instead of drawn epsilon-greedily."""

    next_action = 0

    def behaviour_action(self, state):
        return self.next_action


def q_lambda(replacing=False, lam=LAM, next_action=0):
    agent = FixedNext(3, 4, learning_rate=ALPHA, discount_factor=GAMMA, epsilon=0.0, trace_decay=lam,
                      replacing=replacing)
    agent.next_action = next_action
    agent.begin_episode()
    return agent


# --- Watkins's Q(lambda) -----------------------------------------------------------------------

def test_the_goal_reward_reaches_earlier_states_within_one_episode():
    agent = q_lambda()
    agent.update_q_table(0, 0, -0.1, 1)               # delta = -0.1; e(0,0) = 1, then decays to 0.72
    assert agent.get_q_value(0, 0) == pytest.approx(ALPHA * -0.1)
    agent.update_q_table(1, 3, 100.0, 2, done=True)   # delta = 100, applied through e(0,0) = 0.72 too
    assert agent.get_q_value(1, 3) == pytest.approx(ALPHA * 100.0)
    assert agent.get_q_value(0, 0) == pytest.approx(ALPHA * -0.1 + ALPHA * 100.0 * GAMMA * LAM)
    assert agent.traces == {}                          # traces end with the episode

    one_step = QLearningAgent(3, 4, learning_rate=ALPHA, discount_factor=GAMMA)
    one_step.update_q_table(0, 0, -0.1, 1)
    one_step.update_q_table(1, 3, 100.0, 2, done=True)
    assert one_step.get_q_value(0, 0) == pytest.approx(ALPHA * -0.1)  # one-step: no credit yet


def test_a_non_greedy_next_action_cuts_the_traces():
    agent = q_lambda(next_action=2)
    agent.q_table[1][0] = 1.0                          # greedy action in state 1 is 0; a' = 2 is exploratory
    agent.update_q_table(0, 0, -0.1, 1)
    first = ALPHA * (-0.1 + GAMMA * 1.0)               # the target still uses max_b Q(1, b)
    assert agent.get_q_value(0, 0) == pytest.approx(first)
    assert agent.traces == {} and agent.cuts == 1
    agent.update_q_table(1, 2, 100.0, 2, done=True)
    assert agent.get_q_value(0, 0) == pytest.approx(first)  # no credit through a cut trace


def test_a_next_action_that_ties_for_the_best_keeps_the_traces():
    agent = q_lambda(next_action=3)                    # all of Q(1, .) is 0, so a' = 3 ties for the max
    agent.update_q_table(0, 0, -0.1, 1)
    assert agent.cuts == 0
    assert agent.traces == {(0, 0): pytest.approx(GAMMA * LAM)}


def test_accumulating_and_replacing_traces_differ_on_revisits():
    accumulating, replacing = q_lambda(replacing=False), q_lambda(replacing=True)
    for agent in (accumulating, replacing):
        agent.update_q_table(0, 1, -1.0, 0)            # bump a wall: Q(0,1) = -0.5, e(0,1) -> 0.72
        agent.update_q_table(0, 1, -1.0, 0)            # bump again: delta = -1 + 0.9 * 0 + 0.5 = -0.5
    # accumulating: e(0,1) = 0.72 + 1 = 1.72; replacing: e(0,1) = 1
    assert accumulating.get_q_value(0, 1) == pytest.approx(-0.5 + ALPHA * -0.5 * (1 + GAMMA * LAM))
    assert replacing.get_q_value(0, 1) == pytest.approx(-0.5 + ALPHA * -0.5 * 1.0)


def test_replacing_traces_clear_the_other_actions_of_the_revisited_state():
    accumulating, replacing = q_lambda(replacing=False), q_lambda(replacing=True)
    for agent in (accumulating, replacing):
        agent.update_q_table(0, 1, -1.0, 0)            # Q(0,1) = -0.5, e(0,1) -> 0.72
        agent.update_q_table(0, 0, -1.0, 0)            # another action in the same state: delta = -1
    assert accumulating.get_q_value(0, 1) == pytest.approx(-0.5 + ALPHA * -1.0 * GAMMA * LAM)
    assert replacing.get_q_value(0, 1) == pytest.approx(-0.5)   # its trace was set to 0


def test_lambda_zero_is_one_step_q_learning():
    transitions = [(0, 1, -1.0, 0, False), (0, 3, -0.1, 1, False), (1, 0, -0.1, 0, False),
                   (0, 3, -0.1, 1, False), (1, 2, 100.0, 2, True)]
    traces = q_lambda(lam=0.0)
    one_step = QLearningAgent(3, 4, learning_rate=ALPHA, discount_factor=GAMMA)
    for s, a, r, s2, done in transitions:
        traces.update_q_table(s, a, r, s2, done)
        one_step.update_q_table(s, a, r, s2, done)
    for s in range(3):
        assert traces.action_values(s) == pytest.approx(one_step.action_values(s))


def test_the_agent_acts_with_the_next_action_its_update_chose():
    agent = q_lambda(next_action=3)
    agent.update_q_table(0, 0, -0.1, 1)
    agent.next_action = 1                              # a later draw would give 1 ...
    assert agent.choose_action(1) == 3                 # ... but the step uses the a' the update chose
    assert agent.choose_action(1) == 1                 # used once
    agent.update_q_table(0, 0, -0.1, 1)
    agent.begin_episode()                              # a new episode forgets it
    assert agent._next is None and agent.traces == {}


# --- Dyna-Q -----------------------------------------------------------------------------------

def test_dyna_without_planning_is_q_learning():
    transitions = [(0, 1, -1.0, 0, False), (0, 3, -0.1, 1, False), (1, 2, 100.0, 2, True)]
    dyna = DynaQAgent(3, 4, learning_rate=ALPHA, discount_factor=GAMMA, planning_steps=0)
    one_step = QLearningAgent(3, 4, learning_rate=ALPHA, discount_factor=GAMMA)
    for s, a, r, s2, done in transitions:
        dyna.update_q_table(s, a, r, s2, done)
        one_step.update_q_table(s, a, r, s2, done)
    for s in range(3):
        assert dyna.action_values(s) == one_step.action_values(s)
    assert dyna.updates == 3 and dyna.planning_updates == 0


def test_the_model_holds_only_observed_transitions():
    dyna = DynaQAgent(3, 4, planning_steps=5, planning_seed=1)
    dyna.update_q_table(0, 1, -1.0, 0)
    dyna.update_q_table(0, 3, -0.1, 1)
    dyna.update_q_table(0, 1, -1.0, 0)
    dyna.update_q_table(1, 2, 100.0, 2, done=True)
    assert dyna.model == {(0, 1): (-1.0, 0, False), (0, 3): (-0.1, 1, False), (1, 2): (100.0, 2, True)}
    assert dyna.observed_states == [0, 1]
    assert dyna.taken_actions == {0: [1, 3], 1: [2]}
    assert dyna.updates == 4 * (1 + 5) and dyna.planning_updates == 4 * 5


def test_planning_carries_the_goal_value_back_without_new_experience():
    def chain(agent):
        agent.update_q_table(0, 0, -0.1, 1)
        agent.update_q_table(1, 0, 100.0, 2, done=True)
        return agent

    one_step = chain(QLearningAgent(3, 4, learning_rate=0.1, discount_factor=0.95))
    dyna = chain(DynaQAgent(3, 4, learning_rate=0.1, discount_factor=0.95, planning_steps=200, planning_seed=0))
    assert one_step.get_q_value(0, 0) == pytest.approx(-0.01)
    v1 = dyna.get_q_value(1, 0)
    assert v1 == pytest.approx(100.0, abs=0.01)        # replayed toward its target, 100
    assert dyna.get_q_value(0, 0) > 0.8 * (-0.1 + 0.95 * v1)


def test_planning_into_the_goal_does_not_bootstrap():
    dyna = DynaQAgent(3, 4, learning_rate=0.1, discount_factor=0.9, planning_steps=300, planning_seed=0)
    dyna.q_table[2][0] = 50.0                          # a value at the goal must never be used
    dyna.update_q_table(1, 0, 100.0, 2, done=True)
    assert dyna.get_q_value(1, 0) == pytest.approx(100.0, abs=1e-6)


def test_planning_is_reproducible_and_separate_from_exploration():
    import random

    def run(seed):
        random.seed(123)
        agent = DynaQAgent(3, 4, planning_steps=10, planning_seed=seed)
        agent.update_q_table(0, 1, -1.0, 0)
        agent.update_q_table(0, 3, -0.1, 1)
        agent.update_q_table(1, 2, 100.0, 2, done=True)
        return [agent.action_values(s) for s in range(3)], random.random()

    first, draw = run(7)
    again, draw_again = run(7)
    assert first == again
    assert draw == draw_again                          # planning drew nothing from `random`
