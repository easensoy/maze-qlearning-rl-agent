"""Training loop, the convergence rule and learning regressions."""
from helpers import make_agent, make_env

from experiments.training_experiment import TrainingExperiment, convergence_episode, summarize


def train(kind, seed, episodes, max_steps=1000):
    env = make_env(kind, seed)  # seeds `random` before building the env, like the CLI
    return TrainingExperiment(env, make_agent(env)).run(episodes, max_steps, log_every=0)


def test_convergence_needs_a_full_window_at_the_end():
    assert convergence_episode([False] * 900 + [True] * 100) == 901
    assert convergence_episode([False] * 901 + [True] * 99) is None  # settled too late to tell
    assert convergence_episode([True] * 999 + [False]) is None  # broke at the last check
    assert convergence_episode([True] * 1000) == 1
    assert convergence_episode([False] * 10 + [True] * 5, window=5) == 11
    assert convergence_episode([]) is None


def test_training_is_reproducible_per_seed():
    a, b = train("3d_1", seed=3, episodes=60), train("3d_1", seed=3, episodes=60)
    assert a["successes"] == b["successes"]
    assert a["steps"] == b["steps"]
    assert a["greedy_steps"] == b["greedy_steps"]


def test_agent_learns_the_ui_default_maze():
    """Regression for the 6 Nov fixes: with the config settings the greedy route settles
    on a shortest route well within 500 episodes on the UI's default maze (seed 0)."""
    summary = summarize(train("3d_1", seed=0, episodes=500), window=100)
    assert summary["training_success_last_window"] == 1.0
    assert summary["greedy_shortest_last_window"] == 1.0
    assert summary["converged_episode"] is not None and summary["converged_episode"] <= 400


def test_summary_keeps_training_success_apart_from_the_greedy_route():
    results = {"successes": [True] * 100, "greedy_steps": [None] * 50 + [12] * 50, "shortest_route": 10}
    summary = summarize(results, window=100)
    assert summary["training_success_last_window"] == 1.0
    assert summary["greedy_reaches_goal_last_window"] == 0.5
    assert summary["greedy_shortest_last_window"] == 0.0
    assert summary["converged_episode"] is None
