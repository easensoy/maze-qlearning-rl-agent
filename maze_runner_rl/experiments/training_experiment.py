import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np


class TrainingExperiment:
    """Runs a tabular agent on a Gymnasium-style maze environment.

    The environment must return info["state"] (a hashable state index) from reset()
    and step(); both maze environments do.
    """

    def __init__(self, env, agent, name="experiment"):
        self.env = env
        self.agent = agent
        self.name = name
        self.results = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'successes': [],
            'epsilon_values': [],
            'training_time': 0,
            'total_episodes': 0
        }

    def run(self, num_episodes=1000, max_steps=200, log_every=100):
        start_time = time.time()

        for episode in range(num_episodes):
            _, info = self.env.reset()
            state = info["state"]
            episode_reward = 0
            steps = 0
            terminated = False

            for step in range(max_steps):
                action = self.agent.choose_action(state)
                _, reward, terminated, truncated, info = self.env.step(action)
                next_state = info["state"]
                self.agent.update_q_table(state, action, reward, next_state, terminated)

                state = next_state
                episode_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            self.agent.decay_epsilon()

            self.results['episodes'].append(episode)
            self.results['rewards'].append(episode_reward)
            self.results['steps'].append(steps)
            self.results['successes'].append(bool(terminated))
            self.results['epsilon_values'].append(self.agent.epsilon)

            if log_every and (episode % log_every == 0 or episode == num_episodes - 1):
                avg_reward = np.mean(self.results['rewards'][-log_every:])
                success = np.mean(self.results['successes'][-log_every:]) * 100
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Success: {success:.0f}%, Epsilon: {self.agent.epsilon:.3f}")

        self.results['training_time'] = time.time() - start_time
        self.results['total_episodes'] = len(self.results['episodes'])

        return self.results

    def save_results(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        return filename


def main():
    """Train on one maze with the settings in websocket/config.yaml.

    Run from maze_runner_rl/:  python -m experiments.training_experiment --env 2d
    """
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, package_root)
    import yaml
    from src.agents.q_learning_agent import QLearningAgent
    from src.environments.maze_env import MazeEnvironment
    from src.environments.maze_3d_env import Maze3DEnvironment

    with open(os.path.join(package_root, "websocket", "config.yaml")) as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--env", choices=["2d", "3d"], default="2d")
    parser.add_argument("--depth", type=int, default=3, help="levels for --env 3d")
    parser.add_argument("--episodes", type=int, default=config["training"]["max_episodes"])
    parser.add_argument("--max-steps", type=int, default=config["training"]["max_steps_per_episode"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", default=None, help="write the results to this JSON file")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    maze = config["maze"]
    if args.env == "2d":
        env = MazeEnvironment()
    else:
        env = Maze3DEnvironment(maze["default_width"], maze["default_height"], args.depth,
                                wall_density=maze["wall_density"], branching_factor=maze["branching_factor"],
                                dead_end_percentage=maze["dead_end_percentage"])
    env.reset(seed=args.seed)

    agent_cfg = config["agent"]
    agent = QLearningAgent(env.num_states, env.action_space.n,
                           learning_rate=agent_cfg["learning_rate"], epsilon=agent_cfg["epsilon"],
                           epsilon_decay=agent_cfg["epsilon_decay"], discount_factor=agent_cfg["gamma"],
                           epsilon_min=agent_cfg["epsilon_min"])
    experiment = TrainingExperiment(env, agent, name=f"maze_{args.env}")
    results = experiment.run(num_episodes=args.episodes, max_steps=args.max_steps)
    print(f"Goal reached in {np.mean(results['successes']) * 100:.1f}% of {results['total_episodes']} episodes "
          f"({np.mean(results['successes'][-100:]) * 100:.0f}% of the last 100) in {results['training_time']:.1f}s")
    if args.save:
        print("Saved", experiment.save_results(args.save))


if __name__ == "__main__":
    main()
