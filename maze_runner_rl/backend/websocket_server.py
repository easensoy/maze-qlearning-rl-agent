import asyncio
import websockets
import json
import sys
import os
import threading
import time
from typing import Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.maze_3d_env import Maze3DEnvironment
from agents.q_learning_agent import QLearningAgent

class MazeRLServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.env = None
        self.agent = None
        self.is_training = False
        self.training_stats = {
            'episode': 0,
            'step': 0,
            'total_reward': 0,
            'epsilon': 0.1,
            'rewards_history': [],
            'steps_history': []
        }
        self.training_params = {
            'speed': 1.0,
            'max_episodes': 1000,
            'max_steps_per_episode': 200
        }
        
    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        if self.env:
            await self.send_maze_state(websocket)

    async def unregister_client(self, websocket):
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast_message(self, message):
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients],
                return_exceptions=True
            )

    async def send_maze_state(self, websocket=None):
        if not self.env:
            return
            
        maze_data = {
            'type': 'maze_state',
            'maze': self.env.maze.tolist(),
            'agent_position': self.env.agent_position,
            'goal_position': list(self.env.maze_generator.get_end()),
            'dimensions': {
                'width': self.env.maze_generator.width,
                'height': self.env.maze_generator.height, 
                'depth': self.env.maze_generator.depth
            }
        }
        
        if websocket:
            await websocket.send(json.dumps(maze_data))
        else:
            await self.broadcast_message(maze_data)

    async def send_training_update(self, step_data=None):
        update = {
            'type': 'training_update',
            'stats': self.training_stats.copy(),
            'is_training': self.is_training
        }
        
        if step_data:
            update['step_data'] = step_data
            
        await self.broadcast_message(update)

    async def send_q_values(self):
        if not self.agent or not hasattr(self.agent, 'q_table'):
            return
            
        q_data = {
            'type': 'q_values',
            'q_table': {},
            'best_actions': {}
        }
        
        for pos, actions in self.agent.q_table.items():
            if actions:
                best_action = max(actions, key=actions.get)
                q_data['best_actions'][str(pos)] = {
                    'action': best_action,
                    'value': actions[best_action]
                }
        
        await self.broadcast_message(q_data)

    def initialize_environment(self, width=21, height=21, depth=1, **maze_params):
        wall_density = maze_params.get('wall_density', 0.72)
        branching_factor = maze_params.get('branching_factor', 0.35)
        dead_end_percentage = maze_params.get('dead_end_percentage', 0.4)
        
        self.env = Maze3DEnvironment(width, height, depth, 
                                   wall_density=wall_density,
                                   branching_factor=branching_factor,
                                   dead_end_percentage=dead_end_percentage)
        self.env.reset()
        
        self.agent = QLearningAgent(
            width * height * depth,
            6,
            learning_rate=0.1,
            epsilon=0.1,
            epsilon_decay=0.995
        )
        
        complexity_info = f"wall_density={wall_density:.2f}, branching={branching_factor:.2f}, dead_ends={dead_end_percentage:.2f}"
        print(f"Environment initialized: {width}x{height}x{depth} ({complexity_info})")

    async def start_training(self):
        if self.is_training:
            return
            
        self.is_training = True
        self.training_stats['episode'] = 0
        
        training_thread = threading.Thread(target=self.run_training_loop)
        training_thread.daemon = True
        training_thread.start()
        
        await self.broadcast_message({'type': 'training_started'})

    async def stop_training(self):
        self.is_training = False
        await self.broadcast_message({'type': 'training_stopped'})

    def run_training_loop(self):
        episode = 0
        
        while self.is_training and episode < self.training_params['max_episodes']:
            episode += 1
            self.training_stats['episode'] = episode
            
            observation = self.env.reset()
            total_reward = 0
            step = 0
            
            for step in range(self.training_params['max_steps_per_episode']):
                if not self.is_training:
                    break
                    
                state = self.position_to_state(self.env.agent_position)
                action = self.agent.choose_action(state)
                
                observation, reward, done, _ = self.env.step(action)
                new_state = self.position_to_state(self.env.agent_position)
                
                self.agent.update_q_table(state, action, reward, new_state)
                
                total_reward += reward
                self.training_stats['step'] = step
                self.training_stats['total_reward'] = total_reward
                self.training_stats['epsilon'] = self.agent.epsilon
                
                step_data = {
                    'action': action,
                    'reward': reward,
                    'position': self.env.agent_position.copy(),
                    'done': done
                }
                
                asyncio.run_coroutine_threadsafe(
                    self.send_maze_state(), 
                    asyncio.get_event_loop()
                )
                asyncio.run_coroutine_threadsafe(
                    self.send_training_update(step_data),
                    asyncio.get_event_loop()
                )
                
                if step % 10 == 0:
                    asyncio.run_coroutine_threadsafe(
                        self.send_q_values(),
                        asyncio.get_event_loop()
                    )
                
                if done:
                    print(f"Episode {episode} completed in {step} steps, reward: {total_reward}")
                    break
                    
                time.sleep(1.0 / self.training_params['speed'])
            
            self.training_stats['rewards_history'].append(total_reward)
            self.training_stats['steps_history'].append(step)
            
            if len(self.training_stats['rewards_history']) > 100:
                self.training_stats['rewards_history'] = self.training_stats['rewards_history'][-100:]
                self.training_stats['steps_history'] = self.training_stats['steps_history'][-100:]
            
            self.agent.decay_epsilon()

    def position_to_state(self, position):
        x, y, z = position
        return z * (self.env.maze_generator.width * self.env.maze_generator.height) + y * self.env.maze_generator.width + x

    async def handle_message(self, websocket, message):
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'init_maze':
                width = data.get('width', 11)
                height = data.get('height', 11)
                depth = data.get('depth', 5)
                maze_params = {k: v for k, v in data.items() 
                             if k in ['wall_density', 'branching_factor', 'dead_end_percentage']}
                self.initialize_environment(width, height, depth, **maze_params)
                await self.send_maze_state()
                
            elif msg_type == 'start_training':
                await self.start_training()
                
            elif msg_type == 'stop_training':
                await self.stop_training()
                
            elif msg_type == 'reset_maze':
                if self.env:
                    self.env.reset()
                    await self.send_maze_state()
                    
            elif msg_type == 'update_params':
                params = data.get('params', {})
                self.training_params.update(params)
                if self.agent and 'learning_rate' in params:
                    self.agent.learning_rate = params['learning_rate']
                if self.agent and 'epsilon' in params:
                    self.agent.epsilon = params['epsilon']
                    
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
        
        self.initialize_environment()
        
        start_server = websockets.serve(self.websocket_handler, self.host, self.port)
        print(f"WebSocket server running on ws://{self.host}:{self.port}")
        
        await start_server
        await asyncio.Future()

if __name__ == "__main__":
    server = MazeRLServer()
    asyncio.run(server.start_server())