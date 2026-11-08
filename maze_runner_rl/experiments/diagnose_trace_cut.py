"""Diagnostic intervention for the Q(λ) long-trace failure, on development mazes only.

This specification was written before the intervention was first run. The repository history
does not timestamp that order: the script was committed after its first run.

Question. With λ = 0.9 and accumulating traces, Watkins's Q(λ) discovers a median 16% of
reachable cells in 50,000 development steps, and converges on 0 of 60 development runs. The
report's proposed mechanism concerns the cut rule. Watkins's Q(λ) keeps its traces after any
greedy next action, and a choice among untried actions (all valued 0) counts as greedy. The
penalties of an exploratory walk are therefore credited back along the whole path walked.

Intervention. The agent, settings and random seeds are unchanged, except for one rule: the
traces are also cut when the next behaviour action a' has never been tried in s' (Q(s', a')
still at its initial 0). Every other update is unchanged. This is a diagnostic ablation of one
rule, not a candidate method, and it is used in no comparison of methods.

Scope:
- the 60 development mazes, seeds 1000–1019 of each type;
- λ = 0.9 with accumulating traces;
- each run trains to the first episode end at or past 50,000 real steps;
- exploration seed 500000 + maze seed.
No final maze and no other setting is run.

Variants on the same mazes and seeds:
- unmodified Q(λ), checked against the published tuning records episode by episode;
- the intervention;
- Q-learning.

Measure. The share of reachable cells discovered after 50,000 real steps, as in the published
λ sweep, and whether any episode reached the goal within those steps.

Reading. If coverage recovers toward Q-learning's, the cut rule is implicated in the collapse.
If coverage stays near the unmodified agent's, the stated mechanism does not explain it.

Run from maze_runner_rl/:
    python -m experiments.diagnose_trace_cut          # under a minute on 20 workers
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments import compare_methods as cm
from experiments import evaluate as ev
from src.agents.q_lambda_agent import WatkinsQLambdaAgent

BUDGET = 50_000
PARAMS = {"trace_decay": 0.9, "replacing": False}
VARIANTS = ("q_lambda", "q_lambda_cut_untried", "q_learning")
OUT = os.path.join(cm.OUT, "figures", "diagnostic_trace_cut.json")


class CutAtUntried(WatkinsQLambdaAgent):
    """Watkins's Q(λ) whose traces are also cut when the next action has never been tried."""

    def update_q_table(self, state, action, reward, next_state, done=False):
        current = self.q_table[state][action]
        if done:
            target, keep_traces = reward, False
        else:
            values = self.action_values(next_state)
            best = max(values)
            next_action = self.behaviour_action(next_state)
            self._next = (next_state, next_action)
            tried = next_action in self.q_table.get(next_state, {})
            keep_traces = values[next_action] == best and tried            # the one change
            target = reward + self.discount_factor * best
        delta = target - current
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


def run(task):
    kind, seed, variant = task
    with contextlib.redirect_stdout(io.StringIO()):
        env = ev.make_env(kind, seed)
    facts = cm.maze_facts(env)
    random.seed(ev.exploration_seed(seed))
    if variant == "q_lambda_cut_untried":
        a = ev.SETTINGS["agent"]
        agent = CutAtUntried(env.num_states, env.action_space.n, learning_rate=a["learning_rate"],
                             epsilon=a["epsilon"], epsilon_decay=a["epsilon_decay"], discount_factor=a["gamma"],
                             epsilon_min=a["epsilon_min"], **PARAMS)
    else:
        agent = cm.make_agent(variant, PARAMS if variant == "q_lambda" else {}, env, seed)
    seen, total, steps, new_cells, reached = {facts["start"]}, 0, [], [], False
    while total < BUDGET:
        _, info = env.reset()
        agent.begin_episode() if hasattr(agent, "begin_episode") else None
        state, n, found = info["state"], 0, 0
        for _ in range(cm.MAX_STEPS):
            action = agent.choose_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            agent.update_q_table(state, action, reward, info["state"], terminated)
            state = info["state"]
            n += 1
            total += 1
            if state not in seen:
                seen.add(state)
                found += 1
            if terminated or truncated:
                reached = reached or terminated
                break
        (agent.end_episode if hasattr(agent, "end_episode") else agent.decay_epsilon)()
        steps.append(n)
        new_cells.append(found)
    return {"kind": kind, "maze_seed": seed, "variant": variant, "env_steps": total, "episodes": len(steps),
            "discovered": len(seen), "reachable": len(facts["cells"]), "share": len(seen) / len(facts["cells"]),
            "reached_goal": reached, "steps": steps, "new_cells": new_cells}


def check_against_tuning(results):
    """The unmodified runs must be the published development runs, episode by episode."""
    raw = cm.read_raw(os.path.join(cm.OUT, "tuning", "episodes.jsonl.gz"))
    key = cm.params_key(PARAMS)
    published = {(x["kind"], x["maze_seed"]): x for x in raw if x["method"] == "q_lambda" and x["params"] == key}
    for r in results:
        if r["variant"] != "q_lambda":
            continue
        ref = published[(r["kind"], r["maze_seed"])]
        n = r["episodes"]
        if r["steps"] != ref["steps"][:n] or r["new_cells"] != ref["new_cells"][:n]:
            raise SystemExit(f"unmodified Q(λ) on {r['kind']} maze {r['maze_seed']} differs from the tuning records")
    return len([r for r in results if r["variant"] == "q_lambda"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, min(20, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    tasks = [(kind, seed, v) for kind in cm.KINDS for seed in cm.DEV_SEEDS for v in VARIANTS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, tasks))
    checked = check_against_tuning(results)
    summary = {}
    for v in VARIANTS:
        mine = [r for r in results if r["variant"] == v]
        shares = np.array([r["share"] for r in mine])
        summary[v] = {"runs": len(mine), "median_share": float(np.median(shares)),
                      "quartiles": [float(np.percentile(shares, 25)), float(np.percentile(shares, 75))],
                      "reached_goal": int(sum(r["reached_goal"] for r in mine))}
    with open(__file__, "rb") as f:
        script_sha256 = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    out = {"specification": __doc__.split("Run from")[0].strip(), "budget_env_steps": BUDGET, "params": PARAMS,
           "checked_unmodified_runs": checked, "summary": summary,
           "runs": [{k: v for k, v in r.items() if k not in ("steps", "new_cells")} for r in results],
           "provenance": {**cm.snapshot(), "script_sha256": script_sha256,
                          "environment": {k: v for k, v in cm.environment(args.workers, 0).items()
                                          if k != "wall_seconds"}}}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    for v, s in summary.items():
        print(f"{v}: median share {s['median_share']:.0%} (IQR {s['quartiles'][0]:.0%}–{s['quartiles'][1]:.0%}), "
              f"reached the goal in {s['reached_goal']}/{s['runs']} runs")
    print(f"{checked} unmodified runs match the tuning records -> {OUT}")


if __name__ == "__main__":
    main()
