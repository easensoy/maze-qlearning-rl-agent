"""The evaluation suite: its statistics, its exact references and a small end-to-end run."""
import contextlib
import io
import os

import numpy as np
import pytest

from experiments import evaluate as ev
from experiments.stats import bootstrap_interval, paired_bootstrap_difference, wilson_interval
from experiments.training_experiment import greedy_route_length, shortest_route_length
from src.algorithms.policy_iteration import PolicyIteration
from src.environments.maze_3d_env import Maze3DEnvironment


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def small_maze(seed=0, depth=1):
    """The smallest maze the 3D generator builds (it enforces at least 15x15)."""
    env = quiet(Maze3DEnvironment, 15, 15, depth)
    quiet(env.reset, seed=seed)
    return env


# --- statistics -------------------------------------------------------------------------

def test_wilson_interval_matches_known_values():
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_interval(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert wilson_interval(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(0, 0) == (None, None)


def test_bootstrap_interval_brackets_the_median_and_handles_unreached_runs():
    values = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    low, high = bootstrap_interval(values, np.median, seed=1)
    assert low <= np.median(values) <= high
    assert (low, high) == bootstrap_interval(values, np.median, seed=1)
    # Most runs never reached the event: the median, and the top of its interval, are inf.
    low, high = bootstrap_interval([1, 2, np.inf, np.inf, np.inf], np.median, seed=1)
    assert high == np.inf


def test_paired_difference():
    same = [1, 0, 1, 1, 0]
    assert paired_bootstrap_difference(same, same) == (0.0, 0.0, 0.0)
    diff, low, high = paired_bootstrap_difference([1] * 20, [0] * 20)
    assert diff == low == high == -1.0
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1, 0], [1])


# --- exact references -------------------------------------------------------------------

def test_random_walk_on_a_corridor_matches_the_textbook_result():
    # Cells 0..L in a row, action 0 steps left (blocked at 0), action 1 steps right; the
    # goal is L. Starting at 0, the expected number of steps to reach L is L(L + 1).
    length = 12
    nxt = np.array([[max(s - 1, 0), min(s + 1, length)] for s in range(length + 1)])
    walk = ev.random_walk_stats(nxt, 0, length, list(range(length + 1)), [length])
    assert walk["expected_steps"] == pytest.approx(length * (length + 1))
    assert walk["reached_by_step"][0] == pytest.approx(0.5 ** length)   # only "right" L times in a row


def test_random_walk_closed_form_matches_step_by_step_propagation():
    env = small_maze(3)
    nxt, _, _ = ev.transition_model(env)
    start, goal = ev.endpoints(env)
    cells = ev.reachable_states(nxt, start)
    checkpoints = [10, 100, 1000, 5000]
    exact = ev.random_walk_stats(nxt, start, goal, cells, checkpoints)

    # Push the probability distribution forward one step at a time; the goal absorbs.
    index = {s: i for i, s in enumerate(cells)}
    move = np.zeros((len(cells), len(cells)))
    for s in cells:
        for t in nxt[s]:
            move[index[s], index[goal if s == goal else int(t)]] += 1 / nxt.shape[1]
    p = np.zeros(len(cells))
    p[index[start]] = 1.0
    reached, median, step = {}, None, 0
    while median is None or step < checkpoints[-1]:
        step += 1
        p = p @ move
        if step in checkpoints:
            reached[step] = p[index[goal]]
        if median is None and p[index[goal]] >= 0.5:
            median = step

    for limit, probability in zip(checkpoints, exact["reached_by_step"]):
        assert probability == pytest.approx(reached[limit], abs=1e-9)
    assert exact["median_steps"] == median
    assert exact["expected_steps"] >= shortest_route_length(env)


def test_value_iteration_matches_policy_iteration_and_bfs():
    env = small_maze(1, depth=3)
    nxt, rew, done = ev.transition_model(env)
    start, goal = ev.endpoints(env)
    v_star, q_star = ev.optimal_values(nxt, rew, done, 0.95)
    np.random.seed(0)
    _, v_pi = PolicyIteration(env, gamma=0.95, theta=1e-10).iterate()
    cells = ev.reachable_states(nxt, start)
    np.testing.assert_allclose(v_star[cells], v_pi[cells], atol=1e-6)
    assert ev.planner_route_length(nxt, q_star, start, goal, len(cells)) == shortest_route_length(env)


def test_loop_count():
    # A perfect 2D maze is a tree: one route between any two cells.
    env = quiet(ev.make_env, "2d", 4)
    nxt, _, _ = ev.transition_model(env)
    assert ev.loop_count(nxt, ev.reachable_states(nxt, ev.endpoints(env)[0])) == 0
    # Four cells in a ring, actions "next" and "previous": one loop.
    ring = np.array([[1, 3], [2, 0], [3, 1], [0, 2]])
    assert ev.loop_count(ring, [0, 1, 2, 3]) == 1


# --- agents and curves --------------------------------------------------------------------

def test_prefix_rule_only_counts_tried_actions():
    env = small_maze(2)
    current, prefix = ev.make_agent(env, "current"), ev.make_agent(env, "prefix")
    for agent in (current, prefix):
        agent.update_q_table(5, 3, -1.0, 6)
    # Only action 3 was tried in state 5, and it cost something.
    assert current.greedy_action(5) == 0      # untried actions count as 0, which beats -0.1
    assert prefix.greedy_action(5) == 3       # the pre-fix rule only looks at tried actions
    # Bootstrapping from state 5: max over all actions is 0, max over tried actions is negative.
    for agent in (current, prefix):
        agent.update_q_table(4, 1, 0.0, 5)
    assert current.q_table[4][1] == 0.0
    assert prefix.q_table[4][1] < 0.0


def test_greedy_route_stops_when_the_policy_loops():
    env = small_maze(2)

    class Bump:
        """Always moves down a level from the bottom level: blocked, so it never moves."""
        def greedy_action(self, state):
            return 4

    assert greedy_route_length(env, Bump(), 10_000) is None


def test_sample_at_reads_the_last_finished_episode():
    cumulative = np.array([100, 150, 400])
    flags = [False, True, False]
    grid = [50, 100, 120, 150, 399, 400, 10_000]
    assert ev.sample_at(cumulative, flags, grid).tolist() == [False, False, False, True, True, False, False]


# --- end to end -------------------------------------------------------------------------

def test_suite_is_deterministic(tmp_path):
    suite = ev.make_suite(mazes=2, episodes=30, heldout=2, window=10)
    first = ev.run_suite(suite, ["3d_1"], ["current", "cap_200"])
    second = ev.run_suite(suite, ["3d_1"], ["current", "cap_200"])
    assert first["runs"] == second["runs"]
    assert first["curves"] == second["curves"]
    assert [r["maze_seed"] for r in first["runs"]] == [0, 1, 0, 1]
    assert all(r["exploration_seed"] == ev.exploration_seed(r["maze_seed"]) for r in first["runs"])


def test_suite_outputs_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments import plot_evaluation

    suite = ev.make_suite(mazes=2, episodes=10, heldout=1, window=5)
    configs = ["current", "cap_200", "prefix_rule", "original"]
    result = ev.run_suite(suite, list(ev.KINDS), configs)
    summary = ev.aggregate(result, list(ev.KINDS), configs)
    info = ev.provenance(suite, list(ev.KINDS), configs, 1, 0.0)
    ev.write_outputs(str(tmp_path), suite, result, summary, info)
    for name in ("runs.csv", "baselines.csv", "curves.json", "summary.json", "tables.md"):
        assert (tmp_path / name).stat().st_size > 0
    with open(tmp_path / "runs.csv", encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 1 + 3 * len(configs) * 2
    assert all(b["planner_route"] == b["shortest_route"] for b in result["baselines"])

    figures = tmp_path / "figures"
    quiet(plot_evaluation.draw_all, str(tmp_path), str(figures))
    assert sorted(os.listdir(figures)) == ["ablations.png", "first_success.png", "hero.png", "policy_map.png",
                                           "route_length.png"]
