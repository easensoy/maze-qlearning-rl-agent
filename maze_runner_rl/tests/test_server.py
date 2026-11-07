"""MazeRLServer: training lifecycle (in-process) and the WebSocket protocol (real socket)."""
import asyncio
import json
import socket
import threading
import time

import numpy as np
import pytest
from websockets.sync.client import connect

from websocket.websocket_server import MazeRLServer


def make_server(**training_params):
    server = MazeRLServer()
    server.initialize_environment()
    server.training_params.update(training_params)
    return server


def send(server, message):
    """Feed one client message to the server, as the WebSocket handler does."""
    return server.handle_message(None, json.dumps(message))


# ---------------------------------------------------------------- lifecycle (in-process)

def test_stop_interrupts_a_slow_run_and_a_restart_keeps_one_worker():
    """Regression: at 0.1 steps/s the worker waits 10 s between steps. Stop used to wait
    5 s and give up, and a restart then left two workers training at once."""
    async def scenario():
        server = make_server(speed=0.1)
        await server.start_training()
        await asyncio.sleep(0.3)  # the worker is now waiting between steps
        first = server.training_thread
        started = time.monotonic()
        await server.stop_training()
        stop_seconds = time.monotonic() - started
        first_alive_after_stop = first.is_alive()
        await server.start_training()
        second = server.training_thread
        second_alive = second.is_alive()
        await server.stop_training()
        return stop_seconds, first_alive_after_stop, second is first, second_alive, second.is_alive()

    stop_seconds, first_alive, same_thread, second_alive, second_alive_after_stop = asyncio.run(scenario())
    assert stop_seconds < 1.0
    assert not first_alive
    assert not same_thread and second_alive
    assert not second_alive_after_stop


def test_speed_change_applies_immediately():
    async def scenario():
        server = make_server(speed=0.1)
        steps = []
        real_step = server.env.step

        def counting_step(action):
            steps.append(action)
            return real_step(action)

        server.env.step = counting_step
        await server.start_training()
        await asyncio.sleep(0.3)
        before = len(steps)
        await send(server, {"type": "update_params", "params": {"speed": 1000}})
        await asyncio.sleep(0.3)
        after = len(steps)
        await server.stop_training()
        return before, after

    before, after = asyncio.run(scenario())
    assert before <= 2  # one step, then a 10 s wait
    assert after - before > 100  # the wait ended as soon as the speed changed


def test_a_run_ends_by_itself_after_max_episodes():
    async def scenario():
        server = make_server(speed=1000, max_episodes=3)
        await server.start_training()
        await asyncio.to_thread(server.training_thread.join, 30)
        return server

    server = asyncio.run(scenario())
    assert not server.training_thread.is_alive()
    assert server.is_training is False
    assert server.training_stats["episode"] == 3
    assert server.training_stats["episodes_history"] == [1, 2, 3]


def test_update_params_accepts_known_keys_only():
    server = make_server()
    asyncio.run(send(server, {"type": "update_params", "params": {
        "speed": "5", "max_episodes": "7", "learning_rate": "0.5", "epsilon": "0.2", "unknown": 1}}))
    assert server.training_params == {"speed": 5.0, "max_episodes": 7,
                                      "max_steps_per_episode": server.config["training"]["max_steps_per_episode"]}
    assert server.agent.learning_rate == 0.5
    assert server.agent.epsilon == 0.2


def test_reset_maze_keeps_the_size_and_starts_a_fresh_agent():
    async def scenario():
        server = make_server()
        await send(server, {"type": "init_maze", "width": 21, "height": 21, "depth": 3})
        old_maze, old_agent = server.env.maze.copy(), server.agent
        server.training_stats["episode"] = 5
        await send(server, {"type": "reset_maze"})
        return server, old_maze, old_agent

    server, old_maze, old_agent = asyncio.run(scenario())
    generator = server.env.maze_generator
    assert (generator.width, generator.height, generator.depth) == (21, 21, 3)
    assert server.agent is not old_agent
    assert server.training_stats["episode"] == 0
    assert not np.array_equal(server.env.maze, old_maze)


def test_q_values_are_keyed_by_cell_coordinates():
    server = make_server()
    env, agent = server.env, server.agent
    _, info = env.reset()
    state = info["state"]
    for action in (1, 3, 1, 3, 0, 2):
        _, reward, terminated, _, info = env.step(action)
        agent.update_q_table(state, action, reward, info["state"], terminated)
        state = info["state"]
    best_actions = server.q_values_message()["best_actions"]
    assert best_actions
    for key, entry in best_actions.items():
        x, y, z = map(int, key.split(","))  # the heatmap parses "x,y,z"
        assert env.maze[z, y, x] == 0
        assert set(entry) == {"action", "value"}


# ---------------------------------------------------------------- protocol (real WebSocket)

@pytest.fixture
def live_server():
    server = MazeRLServer()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        server.host, server.port = "127.0.0.1", probe.getsockname()[1]
    loop = asyncio.new_event_loop()
    task = loop.create_task(server.start_server())

    def serve():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            socket.create_connection((server.host, server.port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    yield server
    server.is_training = False
    server.wake_event.set()
    loop.call_soon_threadsafe(task.cancel)
    thread.join(10)


def message_types(ws, count):
    return [json.loads(ws.recv(timeout=10))["type"] for _ in range(count)]


def test_a_new_client_gets_a_snapshot(live_server):
    with connect(f"ws://{live_server.host}:{live_server.port}") as ws:
        assert message_types(ws, 3) == ["maze_state", "training_update", "q_values"]


def test_a_training_run_streams_updates_and_ends_with_training_stopped(live_server):
    with connect(f"ws://{live_server.host}:{live_server.port}") as ws:
        message_types(ws, 3)  # snapshot
        ws.send(json.dumps({"type": "update_params", "params": {"speed": 1000, "max_episodes": 5}}))
        ws.send(json.dumps({"type": "start_training"}))
        seen = []
        deadline = time.monotonic() + 60
        while "training_stopped" not in seen and time.monotonic() < deadline:
            seen.append(json.loads(ws.recv(timeout=30))["type"])
    assert "training_started" in seen and "training_update" in seen
    assert seen[-1] == "training_stopped"
    assert seen.count("maze_state") <= 1  # sent when the run starts, not on every step
