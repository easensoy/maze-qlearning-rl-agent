"""The visual analyses: the checkpointed replays and the quantities drawn from them."""
import contextlib
import io
import json
import os

import pytest

from experiments import analysis_data as ad
from experiments import capture_analysis as ca
from experiments import compare_methods as cm


def corridor(greedy, values, visited):
    """A hand-checkable case: states 0-1-2-3 in a row, 3 the goal; action 0 moves left,
    action 1 right; rewards -0.1 per move, +100 into the goal; gamma 0.95."""
    nxt = {0: [0, 1], 1: [0, 2], 2: [1, 3], 3: [3, 3]}
    rew = {s: [-0.1 if t != 3 or s == 3 else 100.0 for t in nxt[s]] for s in nxt}
    rew[0][0] = -1.0                                         # a wall bump
    done = {s: [t == 3 and s != 3 for t in nxt[s]] for s in nxt}
    gamma = 0.95
    v_star = {3: 0.0, 2: 100.0, 1: -0.1 + gamma * 100.0, 0: -0.1 + gamma * (-0.1 + gamma * 100.0)}
    maze = {"levels": ["...."], "states": [0, 1, 2, 3], "positions": [[s, 0, 0] for s in range(4)],
            "nxt": [nxt[s] for s in range(4)], "rew": [rew[s] for s in range(4)],
            "done": [done[s] for s in range(4)], "v_star": [v_star[s] for s in range(4)],
            "distance": [3, 2, 1, 0], "on_shortest_route": [True] * 4, "start": 0, "goal": 3,
            "gamma": gamma, "shortest_route": 3}
    snap = {"label": "test", "visited": visited, "v": [values[s] for s in range(4)],
            "greedy": [greedy[s] for s in range(4)], "tried": [0] * 4}
    return ad.Case("corridor", {"maze": maze, "runs": {"m": {"snapshots": [snap]}}}), snap


def test_bellman_residual_vanishes_at_the_optimum_and_value_error_is_v_minus_v_star():
    case, _ = corridor({0: 1, 1: 1, 2: 1, 3: 0}, {s: 0.0 for s in range(4)}, [0])
    optimal = {"visited": [0, 1, 2, 3], "v": [case.v_star[s] for s in range(4)], "greedy": [1, 1, 1, 0]}
    assert all(abs(r) < 1e-9 for r in ad.bellman_residual(case, optimal).values())
    assert all(abs(e) < 1e-9 for e in ad.value_error(case, optimal).values())
    # All values 0: state 2 lacks the goal's +100; state 0's best backup is -0.1 + 0.95 * 0.
    zero = {"visited": [0], "v": [0.0] * 4, "greedy": [1, 1, 1, 0]}
    rho = ad.bellman_residual(case, zero)
    assert rho[2] == pytest.approx(100.0) and rho[0] == pytest.approx(-0.1) and rho[1] == pytest.approx(-0.1)
    assert ad.value_error(case, zero)[2] == pytest.approx(-100.0)


def test_greedy_walk_reach_region_and_frontier_distance():
    case, snap = corridor({0: 1, 1: 0, 2: 1, 3: 0}, {s: 0.0 for s in range(4)}, [0, 1])
    walk, reaches = ad.greedy_walk(case, snap)
    assert walk == [0, 1, 0] and not reaches                  # 0 -> 1 -> 0 repeats a state
    assert ad.reach_region(case, snap) == {2}                 # only 2 steps straight into the goal
    assert ad.route_verdict(case, walk, reaches) == "loops"
    dist = ad.frontier_distance(case, {0, 1})                 # state 2 is the nearest never-visited state
    assert dist[2] == 0 and dist[1] == 1 and dist[0] == 2
    good = dict(snap, greedy=[1, 1, 1, 0])
    walk, reaches = ad.greedy_walk(case, good)
    assert reaches and ad.route_verdict(case, walk, reaches) == "shortest"


def test_the_capture_loop_is_the_experiment_loop():
    """capture_analysis replays a run with its own copy of compare_methods.train's loop; on a
    short run both must give the same record, so the snapshots belong to the published runs."""
    params = {"planning_steps": 5}
    got = ca.replay(("comparison", "dyna_q", params, 12))
    with contextlib.redirect_stdout(io.StringIO()):
        expected, _, raw = cm.train("3d_3", "dyna_q", params, 54, "suite", episodes=12)
    for field in ("env_steps", "first_success_env_steps", "final_greedy_route", "updates"):
        assert got["record"][field] == expected[field], field
    assert got["greedy_route_length_per_episode"] == raw["greedy"]
    assert sum(len(s["visited"]) > 0 for s in got["snapshots"]) == len(got["snapshots"])


@pytest.mark.skipif(not os.path.exists(os.path.join(ad.DATA, "comparison.json")), reason="no committed capture")
def test_the_comparison_capture_agrees_with_the_hero_capture():
    """Two capture scripts recorded maze 54 independently; at the shared budgets the greedy walks
    and the cells from which the greedy policy reaches the goal must coincide."""
    case = ad.Case("comparison")
    with open(os.path.join(os.path.dirname(ad.DATA), "hero.json"), encoding="utf-8") as f:
        hero = json.load(f)
    for method in ("q_learning", "dyna_q"):
        for cp in hero["runs"][method]["checkpoints"]:
            snap = next(s for s in case.runs[method]["snapshots"] if s.get("budget") == cp["budget"])
            walk, reaches = ad.greedy_walk(case, snap)
            assert walk == cp["greedy_path"] and reaches == cp["greedy_reaches"]
            assert ad.reach_region(case, snap) == set(cp["learned"]) - {case.goal}
