import asyncio
import json
import os
import threading
import time

import websockets
import yaml

from src.environments.maze_3d_env import Maze3DEnvironment
from src.agents.q_learning_agent import QLearningAgent


class MazeRLServer:
    """WebSocket server that trains a Q-learning agent on a maze and streams progress.

    Client -> server messages: init_maze, start_training, stop_training, reset_maze, update_params
    Server -> client messages: maze_state, training_update, q_values, training_started, training_stopped

    Training runs in a worker thread. Messages from that thread are handed to the
    server's asyncio loop with run_coroutine_threadsafe(); per-step updates are capped
    at training.max_updates_per_second so fast training does not flood the clients.
    """

    def __init__(self, config_path='config.yaml'):
        with open(os.path.join(os.path.dirname(__file__), config_path), 'r') as f:
            self.config = yaml.safe_load(f)

        self.host = os.environ.get('MAZE_SERVER_HOST', self.config['server']['host'])
        self.port = int(os.environ.get('MAZE_SERVER_PORT', self.config['server']['port']))
        self.clients = set()
        self.env = None
        self.agent = None
        self.loop = None
        self.training_thread = None
        self.lock = threading.Lock()
        self.is_training = False
        self.history_size = self.config['stats']['history_size']
        self.training_params = {
            'speed': self.config['training']['speed'],
            'max_episodes': self.config['training']['max_episodes'],
            'max_steps_per_episode': self.config['training']['max_steps_per_episode']
        }
        self.reset_stats()

    def reset_stats(self):
        self.training_stats = {
            'episode': 0,
            'step': 0,
            'total_reward': 0,
            'epsilon': self.agent.epsilon if self.agent else self.config['agent']['epsilon'],
            'goals_reached': 0,
            'rewards_history': [],
            'steps_history': [],
            'episodes_history': []
        }

    # ------------------------------------------------------------------ messages
    def maze_state_message(self):
        generator = self.env.maze_generator
        return {
            'type': 'maze_state',
            'maze': self.env.maze.tolist(),
            'agent_position': list(self.env.agent_position),
            'start_position': list(generator.get_start()),
            'goal_position': list(generator.get_end()),
            'dimensions': {
                'width': generator.width,
                'height': generator.height,
                'depth': generator.depth
            }
        }

    def training_update_message(self, step_data=None):
        stats = {key: list(value) if isinstance(value, list) else value
                 for key, value in self.training_stats.items()}
        update = {'type': 'training_update', 'stats': stats, 'is_training': self.is_training}
        if step_data:
            update['step_data'] = step_data
        return update

    def q_values_message(self):
        """Best action and its value per visited cell, keyed "x,y,z"."""
        best_actions = {}
        if self.agent is not None and self.env is not None:
            with self.lock:
                states = [state for state, actions in self.agent.q_table.items() if actions]
                values = {state: self.agent.action_values(state) for state in states}
            for state, action_values in values.items():
                action = max(range(len(action_values)), key=action_values.__getitem__)
                x, y, z = self.env.position_from_state(state)
                best_actions[f'{x},{y},{z}'] = {'action': action, 'value': action_values[action]}
        return {'type': 'q_values', 'best_actions': best_actions}

    # ------------------------------------------------------------------ transport
    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")

        if self.env:
            for message in (self.maze_state_message(), self.training_update_message(), self.q_values_message()):
                await websocket.send(json.dumps(message))

    async def unregister_client(self, websocket):
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast_message(self, message):
        if self.clients:
            payload = json.dumps(message)
            await asyncio.gather(
                *[client.send(payload) for client in list(self.clients)],
                return_exceptions=True
            )

    def post(self, message):
        """Thread-safe broadcast, usable from the training thread."""
        if self.loop is not None and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.broadcast_message(message), self.loop)

    # ------------------------------------------------------------------ environment
    def initialize_environment(self, width=None, height=None, depth=None, **maze_params):
        width = width or self.config['maze']['default_width']
        height = height or self.config['maze']['default_height']
        depth = depth or self.config['maze']['default_depth']

        self.maze_params = {
            'wall_density': maze_params.get('wall_density', self.config['maze']['wall_density']),
            'branching_factor': maze_params.get('branching_factor', self.config['maze']['branching_factor']),
            'dead_end_percentage': maze_params.get('dead_end_percentage', self.config['maze']['dead_end_percentage'])
        }

        self.env = Maze3DEnvironment(width, height, depth, **self.maze_params)
        self.env.reset()

        agent_config = self.config['agent']
        self.agent = QLearningAgent(
            self.env.num_states,
            self.env.action_space.n,
            learning_rate=agent_config['learning_rate'],
            epsilon=agent_config['epsilon'],
            epsilon_decay=agent_config['epsilon_decay'],
            discount_factor=agent_config['gamma'],
            epsilon_min=agent_config['epsilon_min']
        )
        self.reset_stats()

        g = self.env.maze_generator
        complexity_info = ", ".join(f"{key}={value:.2f}" for key, value in self.maze_params.items())
        print(f"Environment initialized: {g.width}x{g.height}x{g.depth} ({complexity_info})")

    async def reinitialize(self, width=None, height=None, depth=None, **maze_params):
        """New maze and a fresh agent: a Q-table learned on one maze is meaningless on another."""
        await self.stop_training()
        self.initialize_environment(width, height, depth, **maze_params)
        for message in (self.maze_state_message(), self.training_update_message(), self.q_values_message()):
            await self.broadcast_message(message)

    # ------------------------------------------------------------------ training
    async def start_training(self):
        if self.is_training or self.env is None:
            return

        self.loop = asyncio.get_running_loop()
        self.is_training = True
        self.training_thread = threading.Thread(target=self.run_training_loop, daemon=True)
        self.training_thread.start()

        await self.broadcast_message({'type': 'training_started'})

    async def stop_training(self):
        was_running = self.training_thread is not None and self.training_thread.is_alive()
        self.is_training = False
        if was_running:
            await asyncio.to_thread(self.training_thread.join, 5.0)
        else:
            await self.broadcast_message({'type': 'training_stopped'})

    def run_training_loop(self):
        env, agent = self.env, self.agent
        training_config = self.config['training']
        min_update_gap = 1.0 / training_config.get('max_updates_per_second', 30)
        q_value_every = max(1, int(training_config.get('update_interval', 10)))
        last_update = last_q_values = 0.0
        episodes_this_run = 0

        try:
            self.post(self.maze_state_message())
            while self.is_training and episodes_this_run < int(self.training_params['max_episodes']):
                episodes_this_run += 1
                self.training_stats['episode'] += 1
                episode = self.training_stats['episode']

                _, info = env.reset()
                state = info['state']
                total_reward = 0.0
                steps = 0
                terminated = truncated = False
                step_data = None
                max_steps = int(self.training_params['max_steps_per_episode'])

                while self.is_training and steps < max_steps:
                    with self.lock:
                        action = agent.choose_action(state)
                        _, reward, terminated, truncated, info = env.step(action)
                        next_state = info['state']
                        agent.update_q_table(state, action, reward, next_state, terminated)

                    state = next_state
                    total_reward += reward
                    steps += 1
                    self.training_stats['step'] = steps
                    self.training_stats['total_reward'] = total_reward
                    self.training_stats['epsilon'] = agent.epsilon

                    step_data = {
                        'action': action,
                        'reward': reward,
                        'position': list(env.agent_position),
                        'done': terminated
                    }

                    if terminated or truncated:
                        break

                    now = time.monotonic()
                    if now - last_update >= min_update_gap:
                        last_update = now
                        self.post(self.training_update_message(step_data))
                    if steps % q_value_every == 0 and now - last_q_values >= 0.5:
                        last_q_values = now
                        self.post(self.q_values_message())

                    speed = float(self.training_params['speed'])
                    if 0 < speed < 1000:
                        time.sleep(1.0 / speed)

                if not self.is_training and not terminated:
                    break  # stopped mid-episode: do not record a partial episode

                if terminated:
                    self.training_stats['goals_reached'] += 1
                    print(f"Episode {episode} completed in {steps} steps, reward: {total_reward:.1f}")

                for key, value in (('rewards_history', total_reward), ('steps_history', steps),
                                   ('episodes_history', episode)):
                    history = self.training_stats[key]
                    history.append(value)
                    del history[:-self.history_size]

                agent.decay_epsilon()
                self.training_stats['epsilon'] = agent.epsilon
                # Every update carries the whole recent history, so skipping some at high
                # speed loses nothing; the final state is always sent when the run ends.
                now = time.monotonic()
                if now - last_update >= min_update_gap:
                    last_update = now
                    self.post(self.training_update_message(step_data))
                if now - last_q_values >= 0.5:
                    last_q_values = now
                    self.post(self.q_values_message())
        finally:
            self.is_training = False
            self.post(self.training_update_message())
            self.post(self.q_values_message())
            self.post({'type': 'training_stopped'})

    # ------------------------------------------------------------------ protocol
    async def handle_message(self, websocket, message):
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'init_maze':
                maze_params = {k: v for k, v in data.items()
                               if k in ['wall_density', 'branching_factor', 'dead_end_percentage']}
                await self.reinitialize(data.get('width'), data.get('height'), data.get('depth'), **maze_params)

            elif msg_type == 'start_training':
                await self.start_training()

            elif msg_type == 'stop_training':
                await self.stop_training()

            elif msg_type == 'reset_maze':
                if self.env:
                    g = self.env.maze_generator
                    await self.reinitialize(g.width, g.height, g.depth, **self.maze_params)

            elif msg_type == 'update_params':
                params = data.get('params', {})
                if 'speed' in params:
                    self.training_params['speed'] = float(params['speed'])
                for key in ('max_episodes', 'max_steps_per_episode'):
                    if key in params:
                        self.training_params[key] = int(params[key])
                if self.agent and 'learning_rate' in params:
                    self.agent.learning_rate = float(params['learning_rate'])
                if self.agent and 'epsilon' in params:
                    self.agent.epsilon = float(params['epsilon'])
                    self.training_stats['epsilon'] = self.agent.epsilon

        except json.JSONDecodeError:
            print(f"Invalid JSON received: {message}")
        except Exception as e:
            print(f"Error handling message: {e}")

    async def websocket_handler(self, websocket):
        await self.register_client(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    async def start_server(self):
        print(f"Starting WebSocket server on {self.host}:{self.port}")

        self.loop = asyncio.get_running_loop()
        self.initialize_environment()

        async with websockets.serve(self.websocket_handler, self.host, self.port):
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            await asyncio.Future()


if __name__ == "__main__":
    server = MazeRLServer()
    asyncio.run(server.start_server())
