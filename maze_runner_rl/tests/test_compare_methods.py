"""The value-propagation experiment: its training loop, diagnostic, selection rules and protocol."""
import argparse
import contextlib
import io
import json
import os

import pytest

from experiments import compare_methods as cm
from experiments import evaluate as ev


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.mark.parametrize("kind,seed", [("2d", 0), ("3d_1", 3), ("3d_3", 1)])
def test_the_baseline_loop_reproduces_the_evaluation_suite(kind, seed):
    """Q-learning in this loop is the suite's Q-learning: same maze, same exploration
    seed, same random draws, so the same run, check for check."""
    episodes, window = 60, 10
    record, _, raw = quiet(cm.train, kind, "q_learning", {}, seed, "dev", episodes=episodes, window=window)
    suite = ev.make_suite(mazes=1, episodes=episodes, heldout=1, window=window)
    expected, _ = quiet(ev.train_run, kind, "current", seed, suite)
    for field in ("env_steps", "shortest_route", "first_success_episode", "first_success_env_steps",
                  "first_greedy_shortest_episode", "converged_episode", "converged_env_steps",
                  "final_greedy_route", "training_success_last100", "greedy_shortest_final",
                  "greedy_optimal_share", "visited_share"):
        assert record[field] == expected[field], field
    assert record["updates"] == record["env_steps"]    # one update per real step


def test_learned_depth_reaches_the_route_length_exactly_when_the_greedy_route_is_shortest():
    record, curves, raw = quiet(cm.train, "3d_1", "dyna_q", {"planning_steps": 10}, 2, "dev", episodes=80)
    route = record["shortest_route"]
    assert any(g == route for g in raw["greedy"])     # the run gets there, so both sides are exercised
    for greedy, depth in zip(raw["greedy"], raw["depth"]):
        assert (depth == route) == (greedy == route)
        assert 0 <= depth <= route
    assert curves["depth"][-1] == pytest.approx(raw["depth"][-1] / route, abs=1e-3)
    assert record["updates"] == record["env_steps"] * 11 and record["planning_updates"] == record["env_steps"] * 10


def runs_for(method, key, converged, median_steps, n=60):
    out = []
    for i in range(n):
        done = i < converged
        out.append({"method": method, "params": key, "converged_episode": 500 if done else None,
                    "converged_env_steps": median_steps if done else None, "updates": 1, "agent_seconds": 1.0})
    return out


def test_selection_rules():
    runs = []
    q = {"replacing=True,trace_decay=0.5": (40, 300_000), "replacing=True,trace_decay=0.7": (45, 200_000),
         "replacing=True,trace_decay=0.9": (45, 150_000), "replacing=False,trace_decay=0.9": (30, 400_000)}
    for key, (conv, steps) in q.items():
        runs += runs_for("q_lambda", key, conv, steps)
    d = {5: (50, 200_000), 10: (55, 150_000), 20: (56, 140_000), 50: (56, 130_000)}
    for n, (conv, steps) in d.items():
        runs += runs_for("dyna_q", f"planning_steps={n}", conv, steps)
    chosen, tables = cm.select_settings(runs)
    assert chosen["q_lambda"] == {"trace_decay": 0.9, "replacing": True}   # 45 converged, fewer steps
    # Best is n=50 (56, 130k); the smallest n within one run and 10% of it is n=20 (56, 140k);
    # n=10 converged on 55 but needs 150k > 143k.
    assert chosen["dyna_q"] == {"planning_steps": 20}
    assert chosen["q_learning"] == {}


def test_the_final_runs_refuse_to_start_without_frozen_settings(tmp_path):
    args = argparse.Namespace(out=str(tmp_path), workers=1, episodes=5, mazes=1)
    with pytest.raises(SystemExit, match="run `tune` first"):
        cm.cmd_final(args)


def test_tune_then_final_end_to_end(tmp_path):
    args = argparse.Namespace(out=str(tmp_path), workers=1, episodes=12, mazes=1, long_count=1, quick=True)
    quiet(cm.cmd_tune, args)
    with open(tmp_path / "frozen_settings.json", encoding="utf-8") as f:
        frozen = json.load(f)
    assert set(frozen["methods"]) == {"q_learning", "q_lambda", "dyna_q"}
    assert frozen["development"]["maze_seeds"] == [1000]
    long_seed = frozen["final_sets"]["long"]["maze_seeds"]
    assert len(long_seed) == 1 and long_seed[0] >= cm.LONG_ROUTES["first_seed"]

    quiet(cm.cmd_final, args)
    out = tmp_path / "results"
    for name in ("runs.csv", "curves.json", "episodes.jsonl.gz", "provenance.json", "summary.json", "tables.md"):
        assert os.path.exists(out / name), name
    runs = cm.read_runs(out / "runs.csv")
    assert len(runs) == 3 * 3 + 3                      # 3 types x 3 methods, plus the long-route maze x 3
    assert {r["split"] for r in runs} == {"suite", "long"}
    with open(out / "provenance.json", encoding="utf-8") as f:
        provenance = json.load(f)
    assert provenance["frozen_settings_sha256"] == cm.file_sha256(tmp_path / "frozen_settings.json")
    summary = json.load(open(out / "summary.json", encoding="utf-8"))
    assert "suite:3d_3" in summary["groups"] and "long:3d_3" in summary["groups"]
    assert len(cm.read_raw(out / "episodes.jsonl.gz")) == len(runs)

    # Committed records are not replaced by accident.
    with pytest.raises(SystemExit, match="already exists"):
        cm.cmd_tune(args)
    with pytest.raises(SystemExit, match="already exists"):
        cm.cmd_final(args)

    # A reproduction reads the committed settings and writes to its own folder; `verify`
    # finds it identical to the published runs apart from timing.
    rerun = tmp_path / "rerun"
    rerun_args = argparse.Namespace(**{**vars(args), "out": str(rerun),
                                       "settings": str(tmp_path / "frozen_settings.json")})
    quiet(cm.cmd_final, rerun_args)
    assert not os.path.exists(rerun / "frozen_settings.json")
    with open(rerun / "results" / "provenance.json", encoding="utf-8") as f:
        assert json.load(f)["frozen_settings_sha256"] == provenance["frozen_settings_sha256"]
    verify_args = argparse.Namespace(out=str(rerun), reference=str(out / "runs.csv"))
    report = quiet(cm.cmd_verify, verify_args)
    assert report["compared"] == len(runs) and not report["differences"] and not report["missing"]


def test_a_recorded_run_replays_exactly_for_the_failure_analysis():
    """The Dyna-Q failure trace in analysis.json replays a final run; the replay must be the
    recorded run (it stops otherwise) and must see the cells the run visited."""
    pytest.importorskip("matplotlib")
    from experiments import plot_value_propagation as pv

    record, _, _ = quiet(cm.train, "3d_1", "dyna_q", {"planning_steps": 10}, 2, "suite", episodes=30)
    agent, facts, seen = pv.replay_final_run(record, {"planning_steps": 10})
    assert facts["start"] in seen and len(seen) > 1
    assert agent.planning_updates == record["planning_updates"]


def test_verify_reports_a_changed_field_and_ignores_timing(tmp_path):
    fields = {f: "1" for f in cm.RUN_FIELDS}
    fields.update(split="suite", kind="2d", method="dyna_q", params="planning_steps=50", maze_seed="7")
    reference = tmp_path / "reference.csv"
    ev.write_csv(str(reference), cm.RUN_FIELDS, [fields])
    rerun = tmp_path / "rerun" / "results"
    rerun.mkdir(parents=True)
    ev.write_csv(str(rerun / "runs.csv"), cm.RUN_FIELDS, [dict(fields, agent_seconds="9", run_seconds="9")])
    args = argparse.Namespace(out=str(tmp_path / "rerun"), reference=str(reference))
    assert not quiet(cm.cmd_verify, args)["differences"]
    ev.write_csv(str(rerun / "runs.csv"), cm.RUN_FIELDS, [dict(fields, converged_env_steps="2")])
    with pytest.raises(SystemExit, match="runs that differ: 1;"):
        quiet(cm.cmd_verify, args)
