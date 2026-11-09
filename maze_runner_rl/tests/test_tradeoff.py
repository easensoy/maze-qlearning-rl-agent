"""The trade-off figures' own rules, on small inputs whose answers are worked out by hand."""
import numpy as np
import pytest

from experiments import plot_tradeoff as pt

INF = float("inf")


def test_a_run_counts_from_its_first_episode_end_and_keeps_its_final_state():
    # Run 1 ends its episodes at 10, 20 and 30 and is shortest after the 2nd and 3rd; run 2 ends
    # them at 5 and 100 and is shortest after the 1st only.
    cums = [np.array([10, 20, 30]), np.array([5, 100])]
    flags = [np.array([False, True, True]), np.array([True, False])]
    grid = np.array([4, 5, 10, 19, 20, 50, 100, 10_000])
    # run 1: 0 0 0 0 1 1 1 1   (finished at 30: its last state is carried forward)
    # run 2: 0 1 1 1 1 1 0 0   (not counted before its first episode ends at 5)
    assert pt.pooled_count(cums, flags, grid).tolist() == [0, 1, 1, 1, 2, 2, 1, 1]


def test_the_first_point_at_a_share_is_exact_and_none_if_never_reached():
    cums = [np.array([10, 20, 30]), np.array([5, 100])]
    flags = [np.array([False, True, True]), np.array([True, False])]
    assert pt.first_reaching(cums, flags, 0.5) == 5       # one run of two, from run 2's first episode end
    assert pt.first_reaching(cums, flags, 1.0) == 20      # both, from 20 (until 100)
    never = (cums + [np.array([7])], flags + [np.array([False])])
    assert pt.first_reaching(*never, 0.5) == 20           # two of three are needed for half
    assert pt.first_reaching(*never, 1.0) is None


def test_the_updates_axis_shifts_a_method_by_its_updates_per_step():
    """Dyna-Q with n = 50 makes 51 updates per real step, so the same greedy checks come 51 times
    further along the updates axis; a greedy walk that misses the goal (-1) or is longer than the
    shortest route does not count."""
    runs, raw = [], []
    for seed, route, steps, greedy in ((0, 4, [10, 6, 4], [-1, 5, 4]), (1, 6, [20, 6], [6, 7])):
        key = {"split": "suite", "kind": "2d", "method": "dyna_q", "params": "planning_steps=50", "maze_seed": seed}
        runs.append({**key, "shortest_route": route})
        raw.append({**key, "steps": steps, "updates": [51 * s for s in steps], "greedy": greedy})
    steps_axis = pt.method_series(runs, raw, "dyna_q", "steps")
    updates_axis = pt.method_series(runs, raw, "dyna_q", "updates")
    assert [f.tolist() for f in updates_axis[1]] == [[False, False, True], [True, False]]
    assert pt.first_reaching(*steps_axis, 1.0) == 20      # run 1 shortest from 20, run 2 until 26
    assert pt.first_reaching(*updates_axis, 1.0) == 51 * 20
    grid = np.array([1019, 1020, 1325, 1326, 10 ** 6])
    assert pt.pooled_count(*updates_axis, grid).tolist() == [0, 2, 2, 1, 1]


@pytest.mark.parametrize("values,expected", [
    ([1, 2, 3, None], 2.5),        # three of four converged: both middle values are finite
    ([1, 2, None, None], INF),     # exactly half converged: the upper middle value is +inf, so not reached
    ([4, None, 6], 6.0),           # the middle of 4, 6 and +inf
    ([None, None], INF),
])
def test_a_run_that_did_not_converge_counts_as_never_in_a_median(values, expected):
    assert pt.censored_median(values) == expected


def setting(converged, steps, updates):
    return {"converged": converged, "median_real_steps": steps, "median_updates": updates}


def test_dominance_among_the_measured_settings():
    fast = setting(60, 10_000, 500_000)                # few real steps, many updates
    cheap = setting(58, 50_000, 300_000)               # more steps, fewer updates: neither beats the other
    worse = setting(57, 90_000, 1_300_000)             # beaten on all three by both of them
    twin = setting(60, 10_000, 500_000)                # equal to `fast`: neither is strictly better
    stalled = setting(0, INF, INF)                     # medians not reached
    stalled_twin = setting(0, INF, INF)
    more_converged = setting(61, 200_000, 2_000_000)   # worse medians, but more runs converged
    configs = [fast, cheap, worse, twin, stalled, stalled_twin, more_converged]
    assert pt.non_dominated(configs) == [True, True, False, True, False, False, True]
    assert pt.dominates(fast, worse) and pt.dominates(cheap, worse)
    assert not pt.dominates(fast, cheap) and not pt.dominates(cheap, fast)
    assert not pt.dominates(fast, twin) and not pt.dominates(stalled, stalled_twin)
    assert pt.dominates(worse, stalled)                # finite medians beat ones not reached


def test_updates_to_convergence_count_through_the_converged_episode():
    raw = {"updates": [5, 7, 11, 13]}
    assert pt.updates_to_converge({"converged_episode": 3}, raw) == 23
    assert pt.updates_to_converge({"converged_episode": None}, raw) is None


def test_pairs_ratios_and_the_mazes_only_one_method_converged():
    # maze: (Q-learning, Dyna-Q), each (converged episode, real steps to it, updates per episode)
    mazes = {0: ((2, 1000, [600, 400, 9]), (1, 100, [2000, 9])),           # 10x fewer steps, 2x the updates
             1: ((1, 4000, [4000, 9]), (2, 100, [30_000, 2000, 9])),       # 40x fewer steps, 8x the updates
             2: ((None, None, [9]), (1, 300, [900])),                      # only Dyna-Q converged
             3: ((None, None, [9]), (None, None, [9])),                    # neither
             4: ((1, 500, [500]), (None, None, [9]))}                      # only Q-learning
    runs, raw = [], []
    for seed, sides in mazes.items():
        for method, (episode, steps, updates) in zip(("q_learning", "dyna_q"), sides):
            key = {"split": "suite", "kind": "2d", "method": method, "params": "p", "maze_seed": seed}
            runs.append({**key, "converged_episode": episode, "converged_env_steps": steps})
            raw.append({**key, "updates": updates})
    p = pt.paired(runs, raw, "suite", "2d", "dyna_q")
    assert [q["maze_seed"] for q in p["both"]] == [0, 1]
    assert p["both"][1] == {"maze_seed": 1, "q_learning": (4000, 4000), "method": (100, 32_000)}
    assert [q["maze_seed"] for q in p["only_method"]] == [2]
    assert [q["maze_seed"] for q in p["only_q_learning"]] == [4]
    assert [q["maze_seed"] for q in p["neither"]] == [3]
    assert p["real_steps_ratio"]["ratio"] == pytest.approx(20)     # geometric mean of 10 and 40
    assert p["updates_ratio"]["ratio"] == pytest.approx(4)         # geometric mean of 2 and 8
    assert p["real_steps_ratio"]["mazes"] == p["updates_ratio"]["mazes"] == 2
    assert (p["fewer_real_steps"], p["more_updates"]) == (2, 2)


@pytest.mark.parametrize("ratio,shown", [(14.69, "15"), (18.29, "18"), (3.47, "3.5"), (2.79, "2.8"), (9.96, "10")])
def test_title_ratios_round_to_whole_numbers_from_ten(ratio, shown):
    assert pt.times(ratio) == shown
