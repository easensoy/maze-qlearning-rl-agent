"""Quantities the visual analyses plot, computed from checkpointed replays.

The input is a case file written by capture_analysis.py: the maze with its exact solution, and
snapshots of real runs. Nothing here trains or plans. For a snapshot with state values
V(s) = max_a Q(s, a), where untried actions count as 0:

- greedy walk: follow the recorded greedy action from the start, until the goal or the first
  repeated state;
- greedy-reach region: the states from which that walk reaches the goal;
- value error: e(s) = V(s) - V*(s), with V* from value iteration on the known model;
- Bellman residual: rho(s) = max_a [r(s, a) + gamma (1 - done(s, a)) V(s'(s, a))] - V(s), the
  one-step optimality defect of V under the exact model (zero for every state at V = V*);
- frontier distance: the graph distance from a state to the nearest never-visited state.

The values were rounded to 4 decimals when they were captured, so e and rho carry an error
below 1e-3. Greedy actions were recorded exactly.
"""
import json
import os
from collections import deque

import numpy as np

from experiments import asset_style as st

DATA = os.path.join(st.DATA, "analysis")


class Case:
    def __init__(self, name, data=None):
        if data is None:
            with open(os.path.join(DATA, f"{name}.json"), encoding="utf-8") as f:
                data = json.load(f)
        self.name, self.data = name, data
        m = data["maze"]
        self.maze = m
        self.states = m["states"]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.pos = {s: tuple(p) for s, p in zip(self.states, m["positions"])}
        self.nxt = {s: row for s, row in zip(self.states, m["nxt"])}
        self.rew = {s: row for s, row in zip(self.states, m["rew"])}
        self.done = {s: row for s, row in zip(self.states, m["done"])}
        self.v_star = dict(zip(self.states, m["v_star"]))
        self.distance = dict(zip(self.states, m["distance"]))
        self.on_route = {s for s, flag in zip(self.states, m["on_shortest_route"]) if flag}
        self.start, self.goal, self.gamma = m["start"], m["goal"], m["gamma"]
        self.route_length = m["shortest_route"]
        self.runs = data["runs"]
        self.bfs = shortest_route(self.nxt, self.start, self.goal)

    def snapshot(self, method, label):
        return next(s for s in self.runs[method]["snapshots"] if s["label"] == label)

    def values(self, snap):
        return dict(zip(self.states, snap["v"]))

    def greedy(self, snap):
        return dict(zip(self.states, snap["greedy"]))

    def tried(self, snap):
        return dict(zip(self.states, snap["tried"]))

    def q_rows(self, snap):
        return dict(zip(self.states, snap["q"]))


def shortest_route(nxt, start, goal):
    """One shortest route by breadth-first search, trying actions in index order (the same
    route capture_traces.shortest_route_cells returns)."""
    parent, queue = {start: None}, deque([start])
    while queue:
        state = queue.popleft()
        if state == goal:
            break
        for t in nxt[state]:
            if t not in parent:
                parent[t] = state
                queue.append(t)
    route, state = [], goal
    while state is not None:
        route.append(state)
        state = parent[state]
    return route[::-1]


def greedy_walk(case, snap):
    """States the frozen greedy policy visits from the start, and whether it reaches the goal."""
    greedy = case.greedy(snap)
    state, walk = case.start, [case.start]
    while True:
        state = case.nxt[state][greedy[state]]
        walk.append(state)
        if state == case.goal:
            return walk, True
        if state in walk[:-1]:
            return walk, False


def reach_region(case, snap):
    """Non-goal states from which the frozen greedy policy reaches the goal."""
    greedy = case.greedy(snap)
    known = {case.goal: True}
    for cell in case.states:
        path, seen, state = [], set(), cell
        while state not in known and state not in seen:
            seen.add(state)
            path.append(state)
            state = case.nxt[state][greedy[state]]
        reached = known.get(state, False)
        for s in path:
            known[s] = reached
    return {s for s, ok in known.items() if ok and s != case.goal}


def value_error(case, snap):
    v = case.values(snap)
    return {s: v[s] - case.v_star[s] for s in case.states if s != case.goal}


def bellman_residual(case, snap):
    v = case.values(snap)
    v[case.goal] = 0.0                    # terminal: never acted from, and never bootstrapped
    out = {}
    for s in case.states:
        if s == case.goal:
            continue
        best = max(r + case.gamma * (0.0 if d else v[t])
                   for r, d, t in zip(case.rew[s], case.done[s], case.nxt[s]))
        out[s] = best - v[s]
    return out


def frontier_distance(case, visited):
    """Graph distance from each reachable state to the nearest never-visited state (multi-source
    BFS backwards over the true transitions); None when every state has been visited."""
    unvisited = [s for s in case.states if s not in visited and s != case.goal]
    if not unvisited:
        return None
    back = {}
    for s in case.states:
        for t in case.nxt[s]:
            if t != s:
                back.setdefault(t, set()).add(s)
    dist, queue = {u: 0 for u in unvisited}, deque(unvisited)
    while queue:
        t = queue.popleft()
        for s in back.get(t, ()):
            if s not in dist:
                dist[s] = dist[t] + 1
                queue.append(s)
    return dist


def frontier_directed(case, snap, min_distance=2):
    """Among visited non-goal states at least `min_distance` moves from the nearest
    never-visited state, the share whose greedy action moves one step closer to it, and the
    share expected if the greedy action were uniform over the state's moves (wall bumps
    included). Cells next to unexplored ones are left out: there an untried action (Q = 0)
    leads straight in, whatever the learned values say."""
    visited = set(snap["visited"])
    dist = frontier_distance(case, visited)
    if dist is None:
        return None
    greedy = case.greedy(snap)
    hits, chance, n = 0, 0.0, 0
    for s in visited:
        if s == case.goal or s not in dist or dist[s] < min_distance:
            continue
        closer = [a for a, t in enumerate(case.nxt[s]) if dist.get(t, 10 ** 9) == dist[s] - 1]
        n += 1
        hits += greedy[s] in closer
        chance += len(closer) / len(case.nxt[s])
    return None if n == 0 else {"share": hits / n, "chance": chance / n, "states": n}


def route_verdict(case, walk, reaches):
    if not reaches:
        return "loops"
    moves = len(walk) - 1
    return "shortest" if moves == case.route_length else "reaches"
