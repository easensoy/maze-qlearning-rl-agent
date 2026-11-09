"""The interaction-computation trade-off of Dyna-Q against Q-learning and Watkins's Q(lambda).

Drawn from the committed run records only; nothing is trained again.
Inputs (docs/value-propagation):
- results/runs.csv, results/episodes.jsonl.gz: the 630 final runs (3 methods x 210 mazes);
- tuning/runs.csv, tuning/episodes.jsonl.gz: the development sweep (seeds 1000-1019 of each
  maze type, 60 runs per setting), on which lambda and n were chosen.
Outputs (docs/value-propagation/figures):
- tradeoff.png: (A) the share of each method's 210 final runs whose frozen greedy route is a
  shortest route, against real environment steps, and (B) against Q-value updates; (D) the
  development sweep, median real steps against median updates to convergence per setting;
  (E) agent wall-clock time per 1,000-episode run;
- tradeoff_paired.png: (C) maze by maze on the suite, real steps and updates to convergence of
  Dyna-Q and of Q(lambda) against Q-learning's;
- tradeoff.json: every number the two figures show or annotate, and the checks of the numbers
  they share with analysis.json, summary.json, curves.json and frozen_settings.json.

Conventions:
- A run's state at x real steps (or updates) is its greedy check after the last episode that
  ended within x: not shortest before its first episode ends, and its final state carried
  forward once it has finished its 1,000 episodes (as in curves.json).
- Updates to convergence: Q-value updates (real, trace and planning) through the converged
  episode, as analysis.json counts them.
- A run that did not converge counts as +inf in a median, so a median can be 'not reached'.

Run from maze_runner_rl/:
    python -m experiments.plot_tradeoff
"""
import argparse
import json
import os
import sys

import numpy as np

from experiments import asset_style as st
from experiments import compare_methods as cm
from experiments import evaluate as ev
from experiments import plot_value_propagation as pv
from experiments.plot_value_propagation import (COLOURS, SETS, SHORT, axes, figure, label_end, save,
                                                share_with_band, text)
from experiments.stats import wilson_interval

INF = cm.INF
SUITE = [s for s in SETS if s[0] == "suite"]
STEPS_RANGE, UPDATES_RANGE = (1e3, 1e6), (1e3, 6e7)   # B ends past the last run (50.9M)
DECADE = 0.79            # inches per decade in A and B: Q-learning's curve (one update per step) is the same in both
LEVEL = 0.5              # the share at which A and B compare the methods
AT = {"steps": (1e4, 1e5, 1e6), "updates": (1e5, 1e6, 1e7)}
SMALLEST_FONT = 8.5      # points, on a 1600-pixel-wide figure shown at 800 pixels
SET_MARKERS = {"2d": "o", "3d_1": "s", "3d_3": "^"}
FROZEN = {"q_lambda": "replacing=False,trace_decay=0.3", "dyna_q": "planning_steps=50"}


# --- the numbers ---------------------------------------------------------------------------------

def run_key(r):
    return r["split"], r["kind"], r["method"], r["params"], r["maze_seed"]


def updates_to_converge(run, raw):
    """Q-value updates (real, trace and planning) through the run's converged episode; None if
    the run did not converge."""
    if run["converged_episode"] is None:
        return None
    return int(sum(raw["updates"][:run["converged_episode"]]))


def method_series(runs, raw, method, axis):
    """Per final run of `method`: cumulative real steps ("steps") or Q-value updates ("updates")
    at each episode end, and whether the frozen greedy route was a shortest route after it."""
    route = {run_key(r): r["shortest_route"] for r in runs}
    mine = [x for x in raw if x["method"] == method]
    return ([np.cumsum(x[axis]) for x in mine],
            [np.asarray(x["greedy"]) == route[run_key(x)] for x in mine])


def pooled_count(cums, flags, grid):
    """Runs whose flag is set at each grid point. A run's state at x is its flag after the last
    episode that ended within x (ev.sample_at): unset before its first episode ends, and its
    final flag carried forward once it has finished."""
    count = np.zeros(len(grid), dtype=int)
    for cum, f in zip(cums, flags):
        count += ev.sample_at(cum, f, grid)
    return count


def first_reaching(cums, flags, share):
    """The smallest x at which at least `share` of the runs have their flag set, or None if the
    pooled share never gets there. Exact: the share changes only where some run's episode ends."""
    grid = np.unique(np.concatenate(cums))
    count = pooled_count(cums, flags, grid)
    hit = np.flatnonzero(count >= np.ceil(share * len(cums) - 1e-9))
    return None if hit.size == 0 else float(grid[hit[0]])


def curve(cums, flags, grid):
    """Pooled share on a grid, with 95% Wilson bands."""
    rows = [{"shortest": ev.bits(ev.sample_at(c, f, grid))} for c, f in zip(cums, flags)]
    return share_with_band(rows, "shortest")


def censored_median(values):
    """Median over all runs, a run that did not converge (None) counted as +inf: the median is
    'not reached' (inf) unless more than half of the runs converged."""
    return float(np.median([INF if v is None else float(v) for v in values]))


def dominates(a, b):
    """a dominates b: converged on at least as many runs, with no more median real steps and no
    more median updates to convergence, and strictly better in at least one of the three. A
    median not reached is +inf, so it is no better than another one not reached."""
    keys = ("median_real_steps", "median_updates")
    no_worse = a["converged"] >= b["converged"] and all(a[k] <= b[k] for k in keys)
    better = a["converged"] > b["converged"] or any(a[k] < b[k] for k in keys)
    return no_worse and better


def non_dominated(configs):
    """For each configuration: True if no other measured configuration dominates it."""
    return [not any(dominates(other, c) for other in configs if other is not c) for c in configs]


def config_label(c):
    if c["method"] == "q_learning":
        return "Q-learning"
    params = dict(kv.split("=") for kv in c["params"].split(","))
    if c["method"] == "dyna_q":
        return f"Dyna-Q, n {params['planning_steps']}"
    trace = "replacing" if params["replacing"] == "True" else "accumulating"
    return f"Q(λ), λ {params['trace_decay']}, {trace}"


def development(dev_runs, dev_raw):
    """Each development setting: runs converged out of 60, and the median real steps and median
    updates to convergence over all 60 runs (each a separate median, a run that did not converge
    counted as +inf), with its dominance among the measured settings."""
    raw = {run_key(x): x for x in dev_raw}
    configs = []
    for method in cm.METHODS:
        for params in cm.METHODS[method]["grid"]:
            key = cm.params_key(params)
            mine = [r for r in dev_runs if r["method"] == method and r["params"] == key]
            configs.append({
                "method": method, "params": key, "runs": len(mine),
                "converged": sum(r["converged_episode"] is not None for r in mine),
                "median_real_steps": censored_median([r["converged_env_steps"] for r in mine]),
                "median_updates": censored_median([updates_to_converge(r, raw[run_key(r)]) for r in mine]),
                "frozen": FROZEN.get(method) == key,
            })
    for c, free in zip(configs, non_dominated(configs)):
        c["label"] = config_label(c)
        c["non_dominated"] = free
        c["dominated_by"] = [config_label(o) for o in configs if dominates(o, c)]
    return configs


def paired(runs, raw, split, kind, method):
    """One maze set, maze by maze: `method` against Q-learning. Each point is (real steps, updates)
    to convergence, None if the run did not converge. The mazes are taken in the order
    analysis.json takes them, so its bootstrap intervals are reproduced exactly."""
    rawd = {run_key(x): x for x in raw}
    group = {m: {r["maze_seed"]: r for r in runs if (r["split"], r["kind"], r["method"]) == (split, kind, m)}
             for m in ("q_learning", method)}

    def point(r):
        if r["converged_episode"] is None:
            return None
        return r["converged_env_steps"], updates_to_converge(r, rawd[run_key(r)])

    out = {"both": [], "only_method": [], "only_q_learning": [], "neither": []}
    for seed, r in group[method].items():
        base, other = point(group["q_learning"][seed]), point(r)
        if base and other:
            out["both"].append({"maze_seed": seed, "q_learning": base, "method": other})
        elif other:
            out["only_method"].append({"maze_seed": seed, "method": other})
        elif base:
            out["only_q_learning"].append({"maze_seed": seed, "q_learning": base})
        else:
            out["neither"].append({"maze_seed": seed})
    both = out["both"]
    out["real_steps_ratio"] = pv.geometric_ratio([(p["q_learning"][0], p["method"][0]) for p in both])
    out["updates_ratio"] = pv.geometric_ratio([(p["method"][1], p["q_learning"][1]) for p in both])
    out["fewer_real_steps"] = sum(p["method"][0] < p["q_learning"][0] for p in both)
    out["more_updates"] = sum(p["method"][1] > p["q_learning"][1] for p in both)
    return out


def agent_time(runs):
    """Agent wall-clock seconds per 1,000-episode run: median and interquartile range."""
    out = {}
    for split, kind, name in SETS:
        entry = out[f"{split}:{kind}"] = {"label": name}
        for m in cm.METHODS:
            secs = [r["agent_seconds"] for r in runs if (r["split"], r["kind"], r["method"]) == (split, kind, m)]
            q25, med, q75 = np.percentile(secs, [25, 50, 75])
            entry[m] = {"runs": len(secs), "median": float(med), "q25": float(q25), "q75": float(q75)}
        entry["dyna_q_median_over_q_learning_median"] = entry["dyna_q"]["median"] / entry["q_learning"]["median"]
    return out


def shares(runs, raw):
    """A and B: per method and axis, the pooled share at a few points, where it first reaches
    LEVEL, and its final value (every run finished)."""
    out = {}
    for axis in ("steps", "updates"):
        out[axis] = {}
        for m in cm.METHODS:
            cums, flags = method_series(runs, raw, m, axis)
            n = len(cums)
            at = pooled_count(cums, flags, np.array(AT[axis]))
            final = int(sum(f[-1] for f in flags))
            out[axis][m] = {
                "runs": n,
                "at": [{"x": x, "k": int(k), "share": k / n, "ci": list(wilson_interval(int(k), n))}
                       for x, k in zip(AT[axis], at)],
                "first_reaches_half": first_reaching(cums, flags, LEVEL),
                "last_run_ends": float(max(c[-1] for c in cums)),
                "final": {"k": final, "share": final / n, "ci": list(wilson_interval(final, n))},
            }
    return out


# --- checks against the published records ---------------------------------------------------------

def check(runs, raw, data, published):
    """Every number this figure shares with a published record, compared exactly. Returns
    {name: True/False}; main() stops if any is False."""
    analysis, summary, curves, frozen = (published[k] for k in ("analysis", "summary", "curves", "frozen"))
    out = {}
    # A: the per-run flags, sampled on the committed step grid, are curves.json's.
    flags_ok = True
    by_key = {(c["split"], c["kind"], c["method"], c["maze_seed"]): c["shortest"] for c in curves["runs"]}
    route = {run_key(r): r["shortest_route"] for r in runs}
    for x in raw:
        bits = ev.bits(ev.sample_at(np.cumsum(x["steps"]), np.asarray(x["greedy"]) == route[run_key(x)],
                                    cm.STEP_GRID))
        flags_ok &= bits == by_key[(x["split"], x["kind"], x["method"], x["maze_seed"])]
    out["A: per-run shortest flags on the step grid equal curves.json"] = bool(flags_ok)
    # A, B: the final shares are summary.json's final greedy checks, summed over the four sets.
    for m in cm.METHODS:
        k = sum(summary["groups"][f"{s}:{kd}"][m]["greedy_shortest_final"]["k"] for s, kd, _ in SETS)
        out[f"A, B: {m} final share k equals summary.json"] = data["shares"]["steps"][m]["final"]["k"] == k == \
            data["shares"]["updates"][m]["final"]["k"]
    # C: ratios, intervals, pairs and discordant counts.
    for split, kind, _ in SUITE:
        for m in ("q_lambda", "dyna_q"):
            p = data["paired"][f"{split}:{kind}"][m]
            ref = analysis["sets"][f"{split}:{kind}"][m]["vs_q_learning_where_both_converged"]
            out[f"C: {split}:{kind} {m} ratios and intervals equal analysis.json"] = (
                p["real_steps_ratio"] == ref["real_steps_ratio"] and p["updates_ratio"] == ref["updates_ratio"])
            vs = summary["groups"][f"{split}:{kind}"][m]["vs_q_learning"]
            out[f"C: {split}:{kind} {m} discordant counts equal summary.json"] = (
                len(p["only_method"]) == vs["converged_only_method"]
                and len(p["only_q_learning"]) == vs["converged_only_baseline"])
    # C, D: updates to convergence counted as analysis.json counts them (its medians over converged runs).
    rawd = {run_key(x): x for x in raw}
    for split, kind, _ in SETS:
        for m in cm.METHODS:
            upd = [updates_to_converge(r, rawd[run_key(r)]) for r in runs
                   if (r["split"], r["kind"], r["method"]) == (split, kind, m) and r["converged_episode"] is not None]
            mine = float(np.median(upd)) if upd else None
            out[f"updates to convergence: {split}:{kind} {m} median over converged runs equals analysis.json"] = (
                mine == analysis["sets"][f"{split}:{kind}"][m]["updates_to_converge_median_converged_runs"])
    # D: converged counts and median real steps against analysis.json and frozen_settings.json.
    scores = {s["key"]: s for m in ("q_lambda", "dyna_q") for s in frozen["development"]["scores"][m]}
    for c in data["development"]:
        if c["method"] == "q_learning":
            ref = analysis["development"]["q_learning"]
            out["D: Q-learning converged count equals analysis.json"] = c["converged"] == ref["converged"]
            continue
        s = scores[c["params"]]
        steps = None if c["median_real_steps"] == INF else c["median_real_steps"]
        out[f"D: {c['label']} count and median steps equal frozen_settings.json"] = (
            c["converged"] == s["converged"] and c["runs"] == s["runs"] and steps == s["median_converged_env_steps"])
        if c["method"] == "q_lambda":
            out[f"D: {c['label']} count equals analysis.json"] = (
                c["converged"] == analysis["development"][c["params"]]["converged"])
    # E: medians as analysis.json and summary.json report them.
    for split, kind, _ in SETS:
        for m in cm.METHODS:
            med = data["agent_time"][f"{split}:{kind}"][m]["median"]
            out[f"E: {split}:{kind} {m} median agent time equals analysis.json and summary.json"] = (
                med == analysis["sets"][f"{split}:{kind}"][m]["agent_seconds_per_run_median"]
                == summary["groups"][f"{split}:{kind}"][m]["agent_seconds"])
    return out


# --- drawing ----------------------------------------------------------------------------------------

def paragraph(fig, x, y, s, width, size=SMALLEST_FONT, colour=st.INK2, leading=1.32, weight="normal"):
    """Text wrapped to `width` inches at a fixed size (never shrunk), first baseline at y.
    Returns the baseline of the next line."""
    from matplotlib.font_manager import FontProperties

    renderer = fig.canvas.get_renderer()
    prop = FontProperties(family=st.FONT_FAMILY, size=size, weight=weight)
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        wide = renderer.get_text_width_height_descent(trial, prop, ismath=False)[0] / fig.dpi
        if line and wide > width:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    step = size * leading / 72
    for i, ln in enumerate(lines):
        text(fig, x, y - i * step, ln, size=size, colour=colour, weight=weight)
    return y - len(lines) * step


def panel_title(ax, letter, s, pad=7):
    ax.set_title(f"{letter}   {s}", fontsize=8.8, color=st.INK, loc="left", family=st.FONT_FAMILY, pad=pad)


def label(ax, x, y, s, colour, dx=0, dy=0, ha="left", va="center", weight="bold", coords="data", mask=False,
          rotation=0):
    """Text at (x, y) in data or "axes fraction" coordinates, offset by (dx, dy) points; `mask`
    hides the lines it crosses behind a patch of the background colour."""
    box = dict(boxstyle="square,pad=0.12", facecolor=st.BG, edgecolor="none") if mask else None
    return ax.annotate(s, (x, y), xycoords=coords, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       color=colour, fontsize=SMALLEST_FONT, fontweight=weight, family=st.FONT_FAMILY,
                       annotation_clip=False, bbox=box, rotation=rotation, rotation_mode="anchor", zorder=6)


def log_ticks(ax, axis, low, high, unit=""):
    names = {1: "1", 10: "10", 100: "100", 1e3: "1k", 1e4: "10k", 1e5: "100k", 1e6: "1M", 1e7: "10M",
             1e8: "100M"}
    ticks = [t for t in names if low <= t <= high]
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")([names[t] + unit for t in ticks])


def strip_axes(fig, left, bottom, width, height, xscale="linear", yscale="linear", xlim=(0, 1), ylim=(0, 1)):
    """A margin strip beside a scatter: shaded, without ticks or spines, on the scatter's scale
    along the side it shares with it."""
    from matplotlib.ticker import NullLocator

    a = axes(fig, left, bottom, width, height)
    a.set_facecolor(st.PANEL)
    a.grid(False)
    for side in ("left", "bottom"):
        a.spines[side].set_visible(False)
    a.set_xscale(xscale)    # a scale installs its own tick locators, so they are removed after it
    a.set_yscale(yscale)
    a.set_xlim(*xlim)
    a.set_ylim(*ylim)
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_locator(NullLocator())
        axis.set_minor_locator(NullLocator())
    return a


def angle(ax, x0, y0, x1, y1):
    """The on-page angle, in degrees, of the segment from (x0, y0) to (x1, y1) in data coordinates."""
    (a, b), (c, d) = ax.transData.transform([(x0, y0), (x1, y1)])
    return float(np.degrees(np.arctan2(d - b, c - a)))


HALF_LABELS = {   # where the 50% values go beside their dots, clear of the curves: (dx, dy) points, ha
    "steps": {"dyna_q": (-7, 0, "right"), "q_lambda": (-7, 0, "right"), "q_learning": (6, -8, "left")},
    "updates": {"q_learning": (-6, 8, "right"), "dyna_q": (-3, 0, "right"), "q_lambda": (6, -8, "left")},
}


def share_panels(fig, runs, raw, data, bottom, height):
    """A and B: the pooled share against real steps and against Q-value updates, a decade the same
    width in both, so Q-learning's curve (one update per real step) is drawn identically."""
    left = 0.72
    specs = (("A", "steps", STEPS_RANGE, "Against real environment steps", "real environment steps (log scale)"),
             ("B", "updates", UPDATES_RANGE, "Against Q-value updates: real, trace and planning",
              "Q-value updates (log scale)"))
    for letter, axis, (low, high), name, xlabel in specs:
        width = DECADE * np.log10(high / low)
        ax = axes(fig, left, bottom, width, height)
        grid = np.logspace(np.log10(low), np.log10(high), int(40 * np.log10(high / low)) + 1)
        info = data["shares"][axis]
        for m in cm.METHODS:
            cums, flags = method_series(runs, raw, m, axis)
            share, lo, hi = curve(cums, flags, grid)
            end, final = info[m]["last_run_ends"], info[m]["final"]["share"]
            live = grid <= end
            band = live & (np.arange(len(grid)) >= np.argmax(share > 0))
            ax.fill_between(grid[band], lo[band], hi[band], color=COLOURS[m], alpha=0.16, linewidth=0)
            ax.plot(grid[live], share[live], color=COLOURS[m], linewidth=1.5)
            if end < high:    # every run of the method has finished: its final states, carried forward
                ax.plot([end, high], [final, final], color=COLOURS[m], linewidth=1.2, linestyle=(0, (1.2, 2.2)))
        # Where each share first reaches 50%: a dot, and the value beside it where the curves leave room.
        for m in cm.METHODS:
            x50 = info[m]["first_reaches_half"]
            ax.plot([x50], [LEVEL], "o", color=COLOURS[m], markersize=5.5, markeredgecolor=st.BG,
                    markeredgewidth=1.0, zorder=5)
            dx, dy, ha = HALF_LABELS[axis][m]
            label(ax, x50, LEVEL, ev.fmt_steps(x50), COLOURS[m], dx=dx, dy=dy, ha=ha)
        ax.set_xscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0%", "50%", "100%"] if letter == "A" else [])
        log_ticks(ax, "x", low, high)
        ax.set_xlabel(xlabel, fontsize=SMALLEST_FONT, color=st.INK2, family=st.FONT_FAMILY)
        panel_title(ax, letter, name)
        if letter == "A":
            ax.set_ylabel("runs with a shortest greedy route", fontsize=SMALLEST_FONT, color=st.INK2,
                          family=st.FONT_FAMILY)
        else:
            last = -1.0
            for value, m in sorted((info[m]["final"]["share"], m) for m in cm.METHODS):
                y = max(value, last + 0.09)
                label_end(ax, high, y, SHORT[m], COLOURS[m])
                last = y
        left += width + 0.36


D_LABELS = {   # where each development setting's label goes: (dx, dy) points, ha, va
    "Q-learning": (-8, 0, "right", "center"),
    "Q(λ), λ 0.1, accumulating": (8, 0, "left", "center"), "Q(λ), λ 0.1, replacing": (8, 0, "left", "center"),
    "Q(λ), λ 0.3, accumulating": (-8, 0, "right", "center"), "Q(λ), λ 0.3, replacing": (8, 0, "left", "center"),
    "Q(λ), λ 0.5, accumulating": (-8, 0, "right", "center"), "Q(λ), λ 0.5, replacing": (8, 0, "left", "center"),
}


def short_label(c):
    if c["method"] == "dyna_q":
        return f"n {c['params'].split('=')[1]}\n{c['converged']}/{c['runs']}"
    name = "Q-learning" if c["method"] == "q_learning" else c["label"].split(", ")[1]
    return f"{name} · {c['converged']}/{c['runs']}"


def development_panel(fig, data, left, bottom, width, height, strip):
    """D: the development sweep, each setting's median real steps against median updates to
    convergence; the settings whose medians were not reached in a strip at the right."""
    ax = axes(fig, left, bottom, width, height)
    xlim, ylim = (7e3, 4e5), (1e5, 5e6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # Constant updates per real step, labelled where each line has room.
    for ratio, name, at in ((1, "1 per step", 1.75e5), (10, "10 per step", 1.12e4), (100, "100 per step", 2.4e4)):
        ax.plot(xlim, [ratio * xlim[0], ratio * xlim[1]], color=st.INK3, linewidth=0.8, linestyle=(0, (1, 2.4)),
                zorder=1)
        label(ax, at, ratio * at, name, st.INK3, dy=3, va="bottom", weight="normal",
              rotation=angle(ax, xlim[0], ratio * xlim[0], xlim[1], ratio * xlim[1]))
    for c in data["development"]:
        if c["median_real_steps"] == INF:
            continue
        colour = COLOURS[c["method"]]
        x, y = c["median_real_steps"], c["median_updates"]
        ax.plot([x], [y], "s" if c["label"].endswith("replacing") else "o", markersize=6.5, zorder=4,
                markerfacecolor=colour if c["non_dominated"] else st.BG, markeredgecolor=colour, markeredgewidth=1.4)
        dx, dy, ha, va = D_LABELS.get(c["label"], (0, -7, "center", "top"))
        label(ax, x, y, short_label(c), colour, dx=dx, dy=dy, ha=ha, va=va, mask=True,
              weight="bold" if c["frozen"] else "normal")
    log_ticks(ax, "x", *xlim)
    log_ticks(ax, "y", *ylim)
    ax.set_xlabel("median real steps to convergence", fontsize=SMALLEST_FONT, color=st.INK2, family=st.FONT_FAMILY)
    ax.set_ylabel("median Q-value updates to convergence", fontsize=SMALLEST_FONT, color=st.INK2,
                  family=st.FONT_FAMILY)
    panel_title(ax, "D", f"DEVELOPMENT sweep, {data['development_runs']} runs per configuration: not the final "
                          "runs")
    # Settings whose medians were not reached: more than half of their runs did not converge.
    side = strip_axes(fig, left + width + 0.12, bottom, strip, height)
    label(side, 0.08, 0.95, "median\nnot reached", st.INK2, ha="left", va="top", weight="normal")
    missing = [c for c in data["development"] if c["median_real_steps"] == INF]
    for i, c in enumerate(missing):
        y = 0.68 - i * 0.15
        side.plot([0.14], [y], "s" if c["label"].endswith("replacing") else "o", markersize=6.5,
                  markerfacecolor=st.BG, markeredgecolor=COLOURS[c["method"]], markeredgewidth=1.4)
        label(side, 0.14, y, short_label(c), COLOURS[c["method"]], dx=8, weight="normal")
    return ax


def time_panel(fig, data, left, bottom, width, height):
    """E: agent wall-clock seconds per 1,000-episode run, median and interquartile range."""
    ax = axes(fig, left, bottom, width, height)
    offsets = {"q_learning": -0.24, "q_lambda": 0.0, "dyna_q": 0.24}
    for i, (split, kind, _) in enumerate(SETS):
        for m in cm.METHODS:
            t = data["agent_time"][f"{split}:{kind}"][m]
            x = i + offsets[m]
            ax.plot([x, x], [t["q25"], t["q75"]], color=COLOURS[m], linewidth=1.6, solid_capstyle="round")
            ax.plot([x], [t["median"]], "o", color=COLOURS[m], markersize=5.5, markeredgecolor=st.BG,
                    markeredgewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 400)
    log_ticks(ax, "y", 0.6, 400, unit=" s")
    ax.set_xlim(-0.55, len(SETS) - 0.45)
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels(["2D", "One\nlevel", "Three\nlevels", "Long\nroutes"], fontsize=SMALLEST_FONT, color=st.INK2)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.grid(False, axis="x")
    panel_title(ax, "E", "Agent time per run (indicative)")
    for j, m in enumerate(("dyna_q", "q_lambda", "q_learning")):     # the legend, in the empty upper left
        label(ax, 0.04, 0.96 - 0.1 * j, SHORT[m], COLOURS[m], ha="left", va="top", coords="axes fraction")
    return ax


def times(x):
    """A ratio as the titles give it: whole numbers from 10 up, one decimal below."""
    return f"{x:.0f}" if round(x, 1) >= 10 else f"{x:.1f}"


def titles(data):
    """The two titles, their numbers taken from the data they head."""
    half = data["at_half"]["dyna_q"]
    steps = [data["paired"][f"{s}:{k}"]["dyna_q"]["real_steps_ratio"]["ratio"] for s, k, _ in SUITE]
    upd = [data["paired"][f"{s}:{k}"]["dyna_q"]["updates_ratio"]["ratio"] for s, k, _ in SUITE]
    return {"tradeoff": f"Dyna-Q reaches 50% with {times(half['real_steps_fewer'])}× fewer real steps than "
                        f"Q-learning, but {times(half['updates_more'])}× more updates",
            "paired": f"Per maze, Dyna-Q used {times(min(steps))}–{times(max(steps))}× fewer real steps than "
                      f"Q-learning and {times(min(upd))}–{times(max(upd))}× its updates"}


def tradeoff_figure(runs, raw, data, title):
    fig = figure(8, 8.3)
    _, h = fig.get_size_inches()
    text(fig, 0.32, h - 0.36, title, size=12.5, weight="bold")
    sizes, n = data["set_sizes"], data["planning_steps"]
    y = paragraph(fig, 0.32, h - 0.64,
                  f"A, B: share of each method's {sum(sizes)} final runs (the four maze sets pooled: "
                  f"{' + '.join(map(str, sizes))} mazes) whose frozen greedy route is a shortest route at that point. "
                  f"A run that has finished its {cm.EPISODES:,} episodes keeps its final state; dotted, every run of "
                  "the method has finished. Bands: 95% Wilson intervals, from where a share first rises above zero. "
                  "Dots: where each share first reaches 50%. Q-learning makes one update per real step, so its curve "
                  f"is the same in A and B; Dyna-Q (n = {n}) makes {n + 1}.", 7.36)
    top = y - 0.3
    share_panels(fig, runs, raw, data, bottom=top - 2.0, height=2.0)
    top = top - 2.0 - 0.9
    development_panel(fig, data, left=0.72, bottom=top - 2.3, width=3.45, height=2.3, strip=1.02)
    time_panel(fig, data, left=5.9, bottom=top - 2.3, width=1.8, height=2.3)
    foot = top - 2.3 - 0.66
    n = data["development_runs"]
    paragraph(fig, 0.32, foot,
              f"D: each axis is a separate median over the configuration's {n} runs, a run that did not converge "
              f"counted as +∞; labels give runs converged of {n}, and n, Dyna-Q's planning updates per real step. "
              "Filled: non-dominated among measured configurations (no other converged on as many runs with no more "
              "median steps and no more median updates); hollow: dominated. Q(λ): circles accumulating, squares "
              "replacing traces. Bold: the configurations frozen for the final runs. Dotted: constant updates per "
              "real step.", 4.95)
    paragraph(fig, 5.45, foot,
              "E: dots, medians; bars, interquartile ranges. Wall-clock time inside agent calls, with "
              f"{data['measurement']['workers']} concurrent worker processes on one hybrid-core laptop CPU: "
              "indicative, not a controlled benchmark.", 2.3)
    return fig


def paired_figure(runs, raw, data, title):
    """C: maze by maze on the suite, Dyna-Q's and Q(λ)'s real steps and updates to convergence
    against Q-learning's; the mazes only one of the pair converged in margin strips."""
    fig = figure(8, 7.7)
    _, h = fig.get_size_inches()
    text(fig, 0.32, h - 0.36, title, size=12.5, weight="bold")
    sizes, long = data["set_sizes"][:3], data["long_not_paired"]
    each = f"{sizes[0]} each" if len(set(sizes)) == 1 else " + ".join(map(str, sizes))
    said = {m: "none" if k == 0 else str(k) for m, k in long["converged"].items()}
    y = paragraph(fig, 0.32, h - 0.64,
                  f"The {sum(sizes)} suite mazes ({each} of 2D, one-level and three-level), one point per maze. "
                  "Inside the "
                  "axes: mazes on which both methods converged, with the geometric-mean ratio over them per maze set "
                  "(as in analysis.json; pairs in brackets). Strips: mazes on which only one of the pair converged, "
                  f"drawn at that method's value; corner: the count that neither converged. Not paired: the "
                  f"{long['mazes']} long-route mazes, on which Q-learning converged on {said['q_learning']}, Dyna-Q on "
                  f"{said['dyna_q']} and Q(λ) on {said['q_lambda']}.", 7.36)
    size, strip, gap = 2.05, 0.26, 0.05
    columns = ((0.98, "steps", (5e2, 1e6), "real steps to convergence"),
               (4.62, "updates", (1e4, 3e7), "Q-value updates to convergence"))
    names = {"2d": "2D", "3d_1": "1 level", "3d_3": "3 levels"}
    top = y - 0.1                                     # the row header's baseline
    for m in ("dyna_q", "q_lambda"):
        n = data["suite_counts"][m]
        if m == "dyna_q":
            text(fig, 0.32, top, "C", size=8.8)
        text(fig, 0.56, top, f"{SHORT[m]} against Q-learning: both converged on {n['both']} of {n['mazes']} mazes; "
             f"only {SHORT[m]} on {n['only_method']}, only Q-learning on {n['only_q_learning']}, neither on "
             f"{n['neither']}", size=8.8)
        bottom = top - 0.1 - strip - gap - size
        for left, axis, (low, high), name in columns:
            i = 0 if axis == "steps" else 1
            ax = axes(fig, left, bottom, size, size)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            right = strip_axes(fig, left + size + gap, bottom, strip, size, yscale="log", ylim=(low, high))
            upper = strip_axes(fig, left, bottom + size + gap, size, strip, xscale="log", xlim=(low, high))
            corner = strip_axes(fig, left + size + gap, bottom + size + gap, strip, strip)
            ax.plot([low, high], [low, high], color=st.INK3, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
            label(ax, high / 2.2, high / 2.2, "equal", st.INK3, dx=-2, dy=3, ha="right", va="bottom",
                  weight="normal", rotation=angle(ax, low, low, high, high))
            rng = np.random.default_rng(0)    # spreads the strip points sideways, the same way every time
            for split, kind, _ in SUITE:
                p = data["paired_points"][f"{split}:{kind}"][m]
                style = dict(marker=SET_MARKERS[kind], alpha=0.8, linewidths=0.4, edgecolors=st.BG, zorder=3)
                ax.scatter([q["q_learning"][i] for q in p["both"]], [q["method"][i] for q in p["both"]],
                           s=14, color=COLOURS[m], **style)
                if p["only_method"]:
                    right.scatter(rng.uniform(0.25, 0.75, len(p["only_method"])),
                                  [q["method"][i] for q in p["only_method"]], s=18, color=COLOURS[m], **style)
                if p["only_q_learning"]:
                    upper.scatter([q["q_learning"][i] for q in p["only_q_learning"]],
                                  rng.uniform(0.3, 0.7, len(p["only_q_learning"])), s=30,
                                  color=COLOURS["q_learning"], **style)
            label(upper, 0.0, 0.5, "only Q-learning", st.INK2, dx=-6, ha="right", weight="normal",
                  coords="axes fraction")
            label(right, 1.0, 0.5, f"only {SHORT[m]}", st.INK2, dx=4, ha="center", va="top", weight="normal",
                  coords="axes fraction", rotation=90)
            label(corner, 0.5, 0.5, str(n["neither"]), st.INK2, ha="center", weight="normal", coords="axes fraction")
            log_ticks(ax, "x", low, high)
            log_ticks(ax, "y", low, high)
            ax.set_xlabel(f"Q-learning: {name}", fontsize=SMALLEST_FONT, color=st.INK2, family=st.FONT_FAMILY)
            ax.set_ylabel(f"{SHORT[m]}: {name}", fontsize=SMALLEST_FONT, color=st.INK2, family=st.FONT_FAMILY)
            # Geometric-mean ratios per maze set, in the empty half, each line led by its set's marker:
            # with the set names on the left (the marker legend), without them on the right.
            key = "real_steps_ratio" if axis == "steps" else "updates_ratio"
            x0, y0, head = (0.03, 0.97, "fewer real steps") if axis == "steps" else (0.58, 0.44, "more updates")
            label(ax, x0, y0, head, st.INK2, va="top", weight="normal", coords="axes fraction")
            for j, (split, kind, _) in enumerate(SUITE):
                r = data["paired"][f"{split}:{kind}"][m][key]
                yy = y0 - 0.105 * (j + 1)
                ax.plot([x0 + 0.03], [yy - 0.034], SET_MARKERS[kind], color=COLOURS[m], markersize=4.5,
                        markeredgewidth=0, transform=ax.transAxes, clip_on=False)
                where = f"{names[kind]} " if axis == "steps" else ""
                label(ax, x0 + 0.08, yy, f"{where}{cm.fmt_ratio(r['ratio'])}× ({r['mazes']})", st.INK, va="top",
                      weight="normal", coords="axes fraction")
        top = bottom - 0.82
    return fig


def font_problems(fig):
    """Visible text smaller than SMALLEST_FONT, or reaching outside the figure."""
    import matplotlib.text

    fig.canvas.draw()
    box = fig.bbox
    problems = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < SMALLEST_FONT - 1e-6:
            problems.append(f"{t.get_text()!r} at {t.get_fontsize():.1f} pt")
        e = t.get_window_extent()
        if e.x0 < box.x0 - 1 or e.x1 > box.x1 + 1 or e.y0 < box.y0 - 1 or e.y1 > box.y1 + 1:
            problems.append(f"{t.get_text()!r} outside the figure")
    return problems


def build(runs, raw, dev_runs, dev_raw):
    """Every number the figures show, from the run records."""
    data = {"shares": shares(runs, raw), "development": development(dev_runs, dev_raw),
            "agent_time": agent_time(runs), "paired": {}, "paired_points": {}}
    for split, kind, _ in SUITE:
        for m in ("q_lambda", "dyna_q"):
            p = paired(runs, raw, split, kind, m)
            data["paired_points"].setdefault(f"{split}:{kind}", {})[m] = p
            data["paired"].setdefault(f"{split}:{kind}", {})[m] = {
                "pairs": len(p["both"]), "only_method": [q["maze_seed"] for q in p["only_method"]],
                "only_q_learning": [q["maze_seed"] for q in p["only_q_learning"]],
                "neither": [q["maze_seed"] for q in p["neither"]],
                "real_steps_ratio": p["real_steps_ratio"], "updates_ratio": p["updates_ratio"],
                "fewer_real_steps": p["fewer_real_steps"], "more_updates": p["more_updates"]}
    data["suite_counts"] = {
        m: {"mazes": sum(len(data["paired_points"][f"{s}:{k}"][m][c]) for s, k, _ in SUITE
                         for c in ("both", "only_method", "only_q_learning", "neither")),
            **{c: sum(len(data["paired_points"][f"{s}:{k}"][m][c]) for s, k, _ in SUITE)
               for c in ("both", "only_method", "only_q_learning", "neither")}}
        for m in ("q_lambda", "dyna_q")}
    data["long_not_paired"] = {
        "mazes": sum(r["split"] == "long" and r["method"] == "q_learning" for r in runs),
        "converged": {m: sum(r["converged_episode"] is not None for r in runs
                             if r["split"] == "long" and r["method"] == m) for m in cm.METHODS}}
    s = data["shares"]
    data["at_half"] = {m: {
        "real_steps_fewer": s["steps"]["q_learning"]["first_reaches_half"] / s["steps"][m]["first_reaches_half"],
        "updates_more": s["updates"][m]["first_reaches_half"] / s["updates"]["q_learning"]["first_reaches_half"]}
        for m in ("q_lambda", "dyna_q")}
    sizes = {c["runs"] for c in data["development"]}
    if len(sizes) != 1:
        raise SystemExit(f"development configurations have different numbers of runs: {sorted(sizes)}")
    data["development_runs"] = sizes.pop()
    data["set_sizes"] = [sum((r["split"], r["kind"], r["method"]) == (s, k, "q_learning") for r in runs)
                         for s, k, _ in SETS]
    data["planning_steps"] = int(pv.runs_param(runs, "dyna_q", "planning_steps"))
    return data


CONVENTIONS = {
    "share": "A, B: share of a method's 210 final runs (the four maze sets pooled) whose frozen greedy route is a "
             "shortest route at x real steps or x Q-value updates. A run's state at x is its check after the last "
             "episode that ended within x: not shortest before its first episode ends, and its final state carried "
             "forward once it has finished its 1,000 episodes. Intervals: 95% Wilson.",
    "first_reaches_half": "A, B: the smallest x at which the pooled share is at least 50%, exact over every episode "
                          "end; at_half gives Q-learning's x over the method's (real steps) and the method's over "
                          "Q-learning's (updates).",
    "updates_to_converge": "Q-value updates (real, trace and planning) summed through the converged episode, as "
                           "analysis.json counts them.",
    "paired": "C: suite mazes; geometric means over the mazes on which both methods converged, real steps as "
              "Q-learning's over the method's and updates as the method's over Q-learning's, with 95% "
              "percentile-bootstrap intervals (the same numbers as analysis.json).",
    "development": "D: development mazes (seeds 1000-1019 of each type), not the final runs. Each axis is a separate "
                   "median over the configuration's runs, a run that did not converge counted as +inf (null: not "
                   "reached).",
    "dominance": "D: a configuration dominates another if it converged on at least as many runs, with no more "
                 "median real steps and no more median updates to convergence, and strictly better in at least one "
                 "of the three. non_dominated: non-dominated among the 15 measured configurations, not a Pareto "
                 "frontier.",
    "agent_time": "E: wall-clock seconds inside agent calls per 1,000-episode run, median and interquartile range "
                  "(numpy percentiles, linear interpolation). Indicative: measured with concurrent worker processes "
                  "on one hybrid-core laptop CPU, not a controlled benchmark.",
}


def write_json(data, checks):
    """tradeoff.json: the numbers in the figures, by panel, and the checks against the published records."""
    keep = ("set_sizes", "planning_steps", "shares", "at_half", "paired", "suite_counts", "long_not_paired",
            "agent_time", "measurement")
    out = {"source": "experiments/plot_tradeoff.py, from docs/value-propagation/results and tuning",
           "conventions": CONVENTIONS, **{k: data[k] for k in keep},
           "development": {"runs_per_setting": data["development_runs"], "settings": data["development"]},
           "checks": checks}
    path = os.path.join(pv.FIGURES, "tradeoff.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cm.ev._finite(out), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    st.style()
    runs = cm.read_runs(os.path.join(pv.RESULTS, "runs.csv"))
    raw = cm.read_raw(os.path.join(pv.RESULTS, "episodes.jsonl.gz"))
    dev_runs = cm.read_runs(os.path.join(pv.TUNING, "runs.csv"))
    dev_raw = cm.read_raw(os.path.join(pv.TUNING, "episodes.jsonl.gz"))
    published = {}
    for name, path in (("analysis", os.path.join(pv.FIGURES, "analysis.json")),
                       ("summary", os.path.join(pv.RESULTS, "summary.json")),
                       ("curves", os.path.join(pv.RESULTS, "curves.json")),
                       ("provenance", os.path.join(pv.RESULTS, "provenance.json")),
                       ("frozen", os.path.join(cm.OUT, "frozen_settings.json"))):
        with open(path, encoding="utf-8") as f:
            published[name] = json.load(f)
    data = build(runs, raw, dev_runs, dev_raw)
    environment = published["provenance"]["environment"]
    data["measurement"] = {"workers": environment["workers"], "processor": environment["processor"],
                           "platform": environment["platform"]}
    checks = check(runs, raw, data, published)
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("numbers that differ from the published records:\n  " + "\n  ".join(failed))
    heads = titles(data)
    for name, fig in (("tradeoff.png", tradeoff_figure(runs, raw, data, heads["tradeoff"])),
                      ("tradeoff_paired.png", paired_figure(runs, raw, data, heads["paired"]))):
        problems = font_problems(fig)
        if problems:
            raise SystemExit(f"{name}: " + "; ".join(problems))
        print(save(fig, name))
    print(write_json(data, checks))


if __name__ == "__main__":
    main()
