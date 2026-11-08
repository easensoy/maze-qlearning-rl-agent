"""Seeded maze sessions: the rules every tool call follows.

A session is one episode in one maze, isolated from every other session. The maze comes
from the repository's 3D generator (Maze3DEnvironment) with the viewer's density settings,
built from the session's seed, so the same seed always gives the same maze.

What an agent may know, by design:
- its own position and the goal's position, as (x, y, level) coordinates;
- which of the six moves are open from its current cell (a move into a wall is not);
- how many moves it has used and how many are left.
It never sees the map. Anything else it knows, it must remember from earlier observations.

Moves: north is -y, south is +y, west is -x, east is +x, down is -level, up is +level.
A move into a wall uses a move and leaves the agent where it was (a "bump"). The episode
ends when the agent reaches the goal or runs out of moves; after that, moves are refused.
"""
import contextlib
import io
import itertools
import os

import yaml

from src.environments.maze_3d_env import Maze3DEnvironment

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PACKAGE_ROOT, "websocket", "config.yaml"), encoding="utf-8") as f:
    MAZE_SETTINGS = yaml.safe_load(f)["maze"]

# Action indices of Maze3DEnvironment: 0 -y, 1 +y, 2 -x, 3 +x, 4 -z, 5 +z.
DIRECTIONS = {"north": 0, "south": 1, "west": 2, "east": 3, "down": 4, "up": 5}
LIMITS = {"width": (5, 41), "height": (5, 41), "levels": (1, 5), "max_moves": (1, 10_000)}


class SessionError(Exception):
    """A call the rules refuse: an unknown session, bad settings, or a move after the end."""


def build_maze(seed, width, height, levels):
    """The environment for a seed. The generator prints progress to stdout, which is the
    MCP server's protocol channel, so it is silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = Maze3DEnvironment(width, height, levels, wall_density=MAZE_SETTINGS["wall_density"],
                                branching_factor=MAZE_SETTINGS["branching_factor"],
                                dead_end_percentage=MAZE_SETTINGS["dead_end_percentage"])
        env.reset(seed=seed)
    return env


def point(position):
    x, y, level = (int(v) for v in position)
    return {"x": x, "y": y, "level": level}


class MazeSession:
    def __init__(self, session_id, seed, width, height, levels, max_moves):
        self.session_id = session_id
        self.seed = seed
        self.size = (width, height, levels)
        self.max_moves = max_moves
        self.env = build_maze(seed, width, height, levels)
        generator = self.env.maze_generator
        self.start = tuple(int(v) for v in generator.get_start())
        self.goal = tuple(int(v) for v in generator.get_end())
        self.position = self.start
        self.moves = 0
        self.bumps = 0
        self.refused = 0
        self.path = [self.start]

    @property
    def reached_goal(self):
        return self.position == self.goal

    @property
    def done(self):
        return self.reached_goal or self.moves >= self.max_moves

    def open_directions(self):
        x, y, z = self.position
        is_open = self.env.maze_generator.is_valid_position
        return [name for name, action in DIRECTIONS.items()
                if is_open(*(a + b for a, b in zip((x, y, z), self.env.actions[action])))]

    def observe(self):
        return {
            "session_id": self.session_id,
            "position": point(self.position),
            "goal": point(self.goal),
            "open": self.open_directions(),
            "moves_used": self.moves,
            "moves_left": self.max_moves - self.moves,
            "reached_goal": self.reached_goal,
            "done": self.done,
        }

    def step(self, direction):
        if direction not in DIRECTIONS:
            self.refused += 1
            raise SessionError(f"unknown direction {direction!r}; use one of {', '.join(DIRECTIONS)}")
        if self.done:
            self.refused += 1
            raise SessionError("the episode is over: " + ("the goal was reached" if self.reached_goal
                                                          else "no moves are left"))
        new_position, _, _ = self.env._transition(self.position, DIRECTIONS[direction])
        new_position = tuple(int(v) for v in new_position)
        moved = new_position != self.position
        self.moves += 1
        if moved:
            self.position = new_position
            self.path.append(new_position)
        else:
            self.bumps += 1
        return {"moved": moved, "direction": direction, **({} if moved else {"blocked_by": "wall"}), **self.observe()}

    def status(self):
        return {
            "session_id": self.session_id,
            "reached_goal": self.reached_goal,
            "done": self.done,
            "moves_used": self.moves,
            "moves_left": self.max_moves - self.moves,
            "bumps": self.bumps,
            "refused_calls": self.refused,
        }


class SessionManager:
    """Holds isolated sessions by id. Ids are sequential per manager: maze-<seed>-<n>."""

    def __init__(self):
        self.sessions = {}
        self._counter = itertools.count(1)

    def reset(self, seed, width=11, height=11, levels=1, max_moves=120):
        settings = {"width": width, "height": height, "levels": levels, "max_moves": max_moves}
        for key, value in settings.items():
            low, high = LIMITS[key]
            if not isinstance(value, int) or not low <= value <= high:
                raise SessionError(f"{key} must be an integer from {low} to {high}")
        if width % 2 == 0 or height % 2 == 0:
            raise SessionError("width and height must be odd")
        session_id = f"maze-{seed}-{next(self._counter)}"
        session = MazeSession(session_id, seed, width, height, levels, max_moves)
        self.sessions[session_id] = session
        return session

    def get(self, session_id):
        try:
            return self.sessions[session_id]
        except KeyError:
            raise SessionError(f"unknown session {session_id!r}") from None
