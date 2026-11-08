"""The README visuals: the replay they are drawn from, and drawing them from the recorded data."""
import contextlib
import io
import json
import os

import pytest

from experiments import capture_traces as ct
from experiments import evaluate as ev


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def test_replayed_runs_match_the_suite(tmp_path):
    suite = ev.make_suite(mazes=2, episodes=30, heldout=1, window=10)
    quiet(ct.capture, str(tmp_path), kinds=("3d_1", "3d_3"), mazes=2, episodes=30, window=10, workers=1,
          success_seed=0, failure_seed=1)
    traces = ct.load_traces(str(tmp_path))
    for kind, runs in traces.items():
        for run in runs:
            record, _ = quiet(ev.train_run, kind, "current", run["maze_seed"], suite)
            for field in ct.CHECK_FIELDS:
                assert run[field] == record[field], (kind, run["maze_seed"], field)

    with open(tmp_path / "featured.json", encoding="utf-8") as f:
        featured = json.load(f)
    success, failure = featured["success"], featured["failure"]
    assert failure["episodes"] == 3 * success["episodes"]
    # Training in segments (for the checkpoints) is the same run as training in one call.
    plain = {r["maze_seed"]: r for r in traces["3d_3"]}
    assert failure["steps"][:30] == plain[1]["steps"] and failure["success"][:30] == plain[1]["success"]
    # A BFS route runs from the start to the goal, one move at a time.
    route = success["bfs_route"]
    assert route[0] == success["start_state"] and route[-1] == success["goal_state"]
    assert len(route) - 1 == success["shortest_route"]
    # A greedy walk that reaches the goal is never shorter than a shortest route.
    for c in success["checkpoints"]:
        if c["greedy_reaches"]:
            assert len(c["greedy_path"]) - 1 >= success["shortest_route"]


def test_figures_draw_from_the_recorded_data(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments import asset_style as st
    from experiments import render_figures

    if not os.path.exists(os.path.join(st.DATA, "featured.json")):
        pytest.skip("docs/assets/data has not been recorded")
    render_figures.draw_all(st.DATA, str(tmp_path))
    assert sorted(f for f in os.listdir(tmp_path) if f.endswith(".png")) == ["policy.png", "results.png"]


def test_the_animation_draws_from_the_recorded_runs():
    pytest.importorskip("matplotlib")
    from experiments import asset_style as st
    from experiments import render_hero

    if not os.path.exists(os.path.join(st.DATA, "hero.json")) or not os.path.exists(render_hero.SUMMARY):
        pytest.skip("the animation's runs have not been recorded")
    data = st.load_json("hero.json")
    with open(render_hero.SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)
    dyna = data["runs"]["dyna_q"]
    # The route the animation replays is a recorded greedy route, a shortest one, start to goal.
    route = dyna["final_route"]
    assert route[0] == data["start_state"] and route[-1] == data["goal_state"]
    assert len(route) - 1 == data["shortest_route"] == len(data["bfs_route"]) - 1
    # Checkpoints are the policy after the first episode at or past each real-move budget.
    for run in data["runs"].values():
        assert [c["budget"] for c in run["checkpoints"]] == data["budgets"][:len(run["checkpoints"])]
        assert all(c["env_steps"] >= c["budget"] for c in run["checkpoints"])
    specs = render_hero.frames(data, summary)
    assert 15_000 <= sum(s["duration"] for s in specs) <= 22_000   # about 15 to 20 seconds
    scene, canvas = render_hero.Scene(data), render_hero.Canvas()
    for spec in (specs[0], specs[len(specs) // 2], specs[-1]):
        scene.draw(canvas, spec)
        assert canvas.image().shape == (render_hero.H, render_hero.W, 3)
    canvas.close()
