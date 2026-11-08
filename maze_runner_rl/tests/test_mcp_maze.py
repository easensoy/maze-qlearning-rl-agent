"""The maze sessions, their MCP server, and the Claude evaluation harness (with a fake client).

No test calls the Claude API: the harness is driven by a scripted client that returns
Messages-API-shaped responses, so the accounting, the transcript and the stopping rules
are checked deterministically and for free.
"""
import asyncio
import json
import re
from collections import deque
from types import SimpleNamespace

import pytest
from mcp import Client

from experiments import claude_navigation as cn
from mcp_maze.server import build_server
from mcp_maze.sessions import DIRECTIONS, SessionError, SessionManager

SEED = 30000


def bfs_directions(session):
    """Directions of a shortest route from the session's position to its goal."""
    start, goal = session.position, session.goal
    previous = {start: None}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        if here == goal:
            break
        for name in DIRECTIONS:
            there = cn.neighbour(here, name)
            if there not in previous and session.env.maze_generator.is_valid_position(*there):
                previous[there] = (here, name)
                queue.append(there)
    route, here = [], goal
    while previous[here] is not None:
        here, name = previous[here]
        route.append(name)
    return route[::-1]


def wall_direction(session):
    return next(d for d in DIRECTIONS if d not in session.open_directions())


# --- the session rules ------------------------------------------------------------------------------

def test_sessions_are_seeded_and_isolated():
    manager = SessionManager()
    a, b = manager.reset(SEED), manager.reset(SEED)
    assert a.session_id != b.session_id
    assert (a.env.maze == b.env.maze).all() and a.start == b.start and a.goal == b.goal
    a.step(bfs_directions(a)[0])
    assert b.position == b.start and b.moves == 0           # a's move did not touch b
    other = manager.reset(SEED + 1)
    assert not (other.env.maze == a.env.maze).all()


def test_observation_shows_only_the_local_cell_and_the_goal():
    session = SessionManager().reset(SEED)
    obs = session.observe()
    assert set(obs) == {"session_id", "position", "goal", "open", "moves_used", "moves_left", "reached_goal", "done"}
    for direction in DIRECTIONS:
        target = cn.neighbour(session.position, direction)
        assert (direction in obs["open"]) == session.env.maze_generator.is_valid_position(*target)
    assert "up" not in obs["open"] and "down" not in obs["open"]   # one level


def test_a_bump_costs_a_move_and_goes_nowhere():
    session = SessionManager().reset(SEED)
    result = session.step(wall_direction(session))
    assert result["moved"] is False and result["blocked_by"] == "wall"
    assert session.position == session.start and session.moves == 1 and session.bumps == 1


def test_the_shortest_route_reaches_the_goal_and_ends_the_episode():
    session = SessionManager().reset(SEED)
    route = bfs_directions(session)
    for direction in route:
        result = session.step(direction)
    assert result["reached_goal"] and result["done"] and session.moves == len(route)
    assert len(route) == cn.maze_facts(SEED)["shortest_route"]
    with pytest.raises(SessionError, match="episode is over"):
        session.step(route[-1])
    assert session.refused == 1


def test_running_out_of_moves_ends_the_episode():
    session = SessionManager().reset(SEED, max_moves=3)
    for _ in range(3):
        session.step(wall_direction(session))
    assert session.done and not session.reached_goal
    with pytest.raises(SessionError, match="no moves are left"):
        session.step("north")


@pytest.mark.parametrize("settings", [{"width": 10}, {"levels": 0}, {"max_moves": 0}, {"height": 99}])
def test_bad_settings_are_refused(settings):
    with pytest.raises(SessionError):
        SessionManager().reset(SEED, **settings)


# --- the MCP server -----------------------------------------------------------------------------------

def test_the_server_exposes_four_tools_with_a_direction_enum():
    async def go():
        server, _ = build_server()
        async with Client(server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == {"reset", "observe", "step", "episode_status"}
    assert tools["step"].input_schema["properties"]["direction"]["enum"] == list(
        ["north", "south", "east", "west", "up", "down"])
    assert tools["step"].input_schema["required"] == ["session_id", "direction"]
    claude_tools = cn.tool_definitions(tools.values())
    assert [t["name"] for t in claude_tools] == ["episode_status", "observe", "step"]  # no reset for Claude


def test_the_server_follows_the_session_rules():
    async def go():
        server, manager = build_server()
        reference = SessionManager().reset(SEED)
        route = bfs_directions(reference)
        async with Client(server) as client:
            first = json.loads((await client.call_tool("reset", {"seed": SEED})).content[0].text)
            sid = first["session_id"]
            assert first == {**reference.observe(), "session_id": sid}
            bump = await client.call_tool("step", {"session_id": sid, "direction": wall_direction(reference)})
            assert json.loads(bump.content[0].text)["moved"] is False and not bump.is_error
            reference.step(wall_direction(reference))
            for direction in route:
                got = json.loads((await client.call_tool("step", {"session_id": sid, "direction": direction})).content[0].text)
                assert got == {**reference.step(direction), "session_id": sid}
            assert got["reached_goal"]
            late = await client.call_tool("step", {"session_id": sid, "direction": route[-1]})
            assert late.is_error and "episode is over" in late.content[0].text
            unknown = await client.call_tool("observe", {"session_id": "maze-0-99"})
            assert unknown.is_error and "unknown session" in unknown.content[0].text
            bad = await client.call_tool("step", {"session_id": sid, "direction": "sideways"})
            assert bad.is_error
            status = json.loads((await client.call_tool("episode_status", {"session_id": sid})).content[0].text)
            assert status == {"session_id": sid, "reached_goal": True, "done": True, "moves_used": len(route) + 1,
                              "moves_left": 120 - len(route) - 1, "bumps": 1, "refused_calls": 1}

    asyncio.run(go())


# --- the harness, with a scripted client ----------------------------------------------------------------

class Block(SimpleNamespace):
    def model_dump(self, **_):
        return dict(self.__dict__)


class ScriptedClaude:
    """Returns Messages-API-shaped responses: observe once, then the shortest route one step
    per response, then a text answer. Usage numbers are fixed so the cost can be checked."""

    def __init__(self, route):
        self.route = list(route)
        self.calls = 0
        self.messages = SimpleNamespace(create=self.create)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        sid = re.search(r"session_id is (\S+)\.", kwargs["messages"][0]["content"]).group(1)
        self.calls += 1
        usage = SimpleNamespace(input_tokens=100, output_tokens=50, cache_creation_input_tokens=200,
                                cache_read_input_tokens=1000)
        if self.calls == 1:
            content = [Block(type="thinking", thinking="Look first.", signature="s"),
                       Block(type="tool_use", id=f"t{self.calls}", name="observe", input={"session_id": sid})]
            stop = "tool_use"
        elif self.route:
            content = [Block(type="tool_use", id=f"t{self.calls}", name="step",
                             input={"session_id": sid, "direction": self.route.pop(0)})]
            stop = "tool_use"
        else:
            content = [Block(type="text", text="Done.")]
            stop = "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop, usage=usage, model=kwargs["model"],
                               _request_id=f"req_{self.calls}")


def run_scripted(tmp_path, route, cap=100.0, mazes=1, trials=1):
    args = SimpleNamespace(out=str(tmp_path), model="claude-opus-5-5", effort="medium", trials=trials,
                           max_usd=cap, mazes=mazes, base_url=None)
    server, _ = build_server()
    fake = ScriptedClaude(route)
    records = asyncio.run(cn.run_claude(args, mcp_server=server, client=fake))
    return records, fake


def test_the_harness_runs_an_episode_and_accounts_for_every_call(tmp_path):
    route = bfs_directions(SessionManager().reset(SEED))
    records, fake = run_scripted(tmp_path, route)
    (record,) = records
    assert record["reached_goal"] and record["moves"] == len(route) == record["shortest_route"]
    assert record["end"] == "goal reached"                 # the harness stops once the session is done
    assert record["api_calls"] == 1 + len(route) and record["tool_calls"] == 1 + len(route)
    assert record["bumps"] == 0 and record["tool_errors"] == 0
    assert record["input_tokens"] == 100 * record["api_calls"]
    per_call = (100 * 4.00 + 50 * 20.00 + 200 * 5.00 + 1000 * 0.20) / 1e6
    assert record["cost_usd"] == pytest.approx(per_call * record["api_calls"], abs=1e-6)
    request = fake.requests[0]
    assert [t["name"] for t in request["tools"]] == ["episode_status", "observe", "step"]
    assert request["cache_control"] == {"type": "ephemeral"} and request["output_config"] == {"effort": "medium"}
    assert "120 moves" in request["system"]
    lines = [json.loads(line) for line in open(tmp_path / "results" / "claude" / "claude-opus-5-5" / "transcripts"
                                               / f"{SEED}-1.jsonl", encoding="utf-8")]
    assert lines[0]["type"] == "episode" and lines[-1]["type"] == "end"
    assert sum(line["type"] == "response" for line in lines) == record["api_calls"]
    assert lines[1]["content"][0] == {"type": "thinking", "thinking": "Look first.", "signature": "s"}


def test_the_harness_stops_at_the_spending_cap_and_resumes_without_repeating(tmp_path):
    per_call = (100 * 4.00 + 50 * 20.00 + 200 * 5.00 + 1000 * 0.20) / 1e6
    records, _ = run_scripted(tmp_path, ["north"] * 50, cap=per_call * 3.5)
    assert records[0]["end"] == "spending cap reached" and records[0]["api_calls"] == 4
    # A rerun skips the finished episode and, with the cap spent, starts nothing.
    again, fake = run_scripted(tmp_path, ["north"] * 50, cap=per_call * 3.5)
    assert again == [] and fake.calls == 0


def test_a_model_that_stops_calling_tools_ends_its_episode(tmp_path):
    records, _ = run_scripted(tmp_path, [])            # observe, then a text answer
    assert records[0]["end"] == "stopped calling tools (end_turn)" and not records[0]["reached_goal"]


# --- the references -------------------------------------------------------------------------------------

def test_the_dfs_explorer_reaches_the_goal_using_only_local_observations():
    result = cn.run_dfs_explorer(SEED)
    assert result["reached_goal"] and result["bumps"] == 0
    assert result["moves"] >= cn.maze_facts(SEED)["shortest_route"]


def test_random_walks_are_reproducible():
    assert cn.run_random_walks(SEED, walks=50) == cn.run_random_walks(SEED, walks=50)
