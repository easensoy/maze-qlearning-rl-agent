import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environments.maze_env import MazeEnvironment
from agents.q_learning_agent import QLearningAgent

def main():
    env = MazeEnvironment()
    observation = env.reset()
    
    print("Generated Maze:")
    print(env.render())
    print(f"Start: {env.maze_generator.get_start()}")
    print(f"Goal: {env.maze_generator.get_end()}")
    
    agent = QLearningAgent(env.maze_generator.width, env.maze_generator.height)
    
    for step in range(10):
        action = agent.choose_action(observation)
        observation, reward, done, _ = env.step(action)
        
        print(f"\nStep {step + 1}: Action {action}, Reward: {reward}")
        print(env.render())
        
        if done:
            print("Goal reached!")
            break

if __name__ == "__main__":
    main()