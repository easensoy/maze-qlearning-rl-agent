"""QLearningAgent: the update and action-selection rules fixed on 6 Nov 2026."""
import random

import pytest

from src.agents.q_learning_agent import QLearningAgent


def test_untried_actions_count_as_zero():
    agent = QLearningAgent(10, 4, learning_rate=0.5, epsilon=0.0)
    agent.update_q_table(0, 2, -1.0, 0)  # one wall bump with action 2
    assert agent.get_q_value(0, 2) < 0
    # Before the fix only tried actions were candidates, so the agent kept bumping.
    assert agent.greedy_action(0) == 0
    random.seed(0)
    assert {agent.choose_action(0) for _ in range(200)} == {0, 1, 3}  # ties broken at random


def test_terminal_update_does_not_bootstrap():
    agent = QLearningAgent(10, 4, learning_rate=1.0, discount_factor=0.9)
    agent.q_table[5][0] = 50.0
    agent.update_q_table(4, 1, 100.0, 5, done=True)
    assert agent.get_q_value(4, 1) == 100.0


def test_update_bootstraps_from_the_best_next_action():
    agent = QLearningAgent(10, 4, learning_rate=1.0, discount_factor=0.9)
    agent.q_table[5][3] = 50.0
    agent.update_q_table(4, 1, -0.1, 5)
    assert agent.get_q_value(4, 1) == pytest.approx(-0.1 + 0.9 * 50.0)


def test_bootstrap_counts_untried_next_actions_as_zero():
    agent = QLearningAgent(10, 4, learning_rate=1.0, discount_factor=0.9)
    agent.q_table[5][0] = -5.0  # the only tried action in the next state is bad
    agent.update_q_table(4, 1, -0.1, 5)
    assert agent.get_q_value(4, 1) == pytest.approx(-0.1)


def test_epsilon_decays_to_its_floor():
    agent = QLearningAgent(10, 4, epsilon=0.1, epsilon_decay=0.5, epsilon_min=0.01)
    for _ in range(20):
        agent.decay_epsilon()
    assert agent.epsilon == 0.01
