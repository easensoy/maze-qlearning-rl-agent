import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np


def shortest_route_length(env):
    """Moves on a shortest start-to-goal route, found by breadth-first search over the
    environment's own transition model (env.step_from_state). None if unreachable."""
    start = env.state_index(env.maze_generator.get_start())
    goal = env.state_index(env.maze_generator.get_end())
    depth = {start: 0}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state == goal:
            return depth[state]
        for action in range(env.action_space.n):
            next_state, _, _ = env.step_from_state(state, action)
            if next_state not in depth:
                depth[next_state] = depth[state] + 1
                queue.append(next_state)
    return None


def greedy_route_length(env, agent, limit):
    """Moves the agent's greedy policy (no exploration) needs from start to goal under
    the environment's transition model, or None if it does not arrive within `limit`.

    The policy and the maze are both deterministic, so a route that comes back to a
    cell it has already visited (including bumping into a wall) loops forever; the
    check stops there instead of using up the limit."""
    state = env.state_index(env.maze_generator.get_start())
    goal = env.state_index(env.maze_generator.get_end())
    visited = {state}
    for moves in range(1, limit + 1):
        state, _, _ = env.step_from_state(state, agent.greedy_action(state))
        if state == goal:
            return moves
        if state in visited:
            return None
        visited.add(state)
    return None


def convergence_episode(shortest_flags, window=100):
    """First episode (1-based) from which the greedy route is a shortest route at every
    later check, provided that run of checks is at least `window` long. None otherwise:
    the policy did not settle within the run, or settled too late to tell."""
    settled = 0
    for flag in reversed(shortest_flags):
        if not flag:
            break
        settled += 1
    if settled < window:
        return None
    return len(shortest_flags) - settled + 1


def summarize(results, window=100):
    """Training success and learned-policy quality, reported separately.

    Training success counts episodes that reached the goal while still exploring
    (epsilon-greedy). The greedy checks follow the learned policy with no exploration,
    which is what "the agent has learned the maze" means.
    """
    successes = results['successes']
    summary = {
        'episodes': len(successes),
        'window': window,
        'training_success_last_window': float(np.mean(successes[-window:])) if successes else None,
        'first_training_success_episode': next((i + 1 for i, s in enumerate(successes) if s), None),
    }
    greedy = results.get('greedy_steps') or []
    if greedy:
        shortest = results['shortest_route']
        reaches = [g is not None for g in greedy]
        is_shortest = [g is not None and g == shortest for g in greedy]
        summary.update({
            'shortest_route': shortest,
            'greedy_reaches_goal_last_window': float(np.mean(reaches[-window:])),
            'greedy_shortest_last_window': float(np.mean(is_shortest[-window:])),
            'first_greedy_goal_episode': next((i + 1 for i, r in enumerate(reaches) if r), None),
            'converged_episode': convergence_episode(is_shortest, window),
            'final_greedy_route': greedy[-1],
        })
    return summary


def run_info(package_root, args, config, env):
    """Where a result came from: code version, settings and library versions."""
    def git(*command):
        try:
            return subprocess.run(['git', *command], cwd=package_root, capture_output=True,
                                  text=True, timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    import gymnasium
    status = git('status', '--porcelain', '--untracked-files=no')
    generator = env.maze_generator
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_commit': git('rev-parse', 'HEAD'),
        'git_uncommitted_changes': None if status is None else bool(status),
        'command': sys.argv,
        'args': vars(args),
        'maze_size': [generator.width, generator.height, getattr(generator, 'depth', 1)],
        'config': {'agent': config['agent'], 'maze': config['maze']},
        'python': platform.python_version(),
        'numpy': np.__version__,
        'gymnasium': gymnasium.__version__,
    }


class TrainingExperiment:
    """Runs a tabular agent on a Gymnasium-style maze environment.

    The environment must return info["state"] (a hashable state index) from reset()
    and step(); both maze environments do. If it also provides the transition model
    (step_from_state, state_index), the greedy policy is checked after every episode
    against a shortest route.
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
            'greedy_steps': [],
            'shortest_route': None,
            'training_time': 0,
            'total_episodes': 0
        }

    def run(self, num_episodes=1000, max_steps=200, log_every=100, evaluate_greedy=True, greedy_limit=None):
        """Train for num_episodes of at most max_steps steps each. The greedy check after
        each episode allows greedy_limit moves (default max_steps)."""
        evaluate_greedy = evaluate_greedy and hasattr(self.env, 'step_from_state')
        greedy_limit = greedy_limit or max_steps
        start_time = time.time()

        for episode in range(num_episodes):
            _, info = self.env.reset()
            if evaluate_greedy and episode == 0:
                self.results['shortest_route'] = shortest_route_length(self.env)
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
            if evaluate_greedy:
                self.results['greedy_steps'].append(greedy_route_length(self.env, self.agent, greedy_limit))

            if log_every and (episode % log_every == 0 or episode == num_episodes - 1):
                avg_reward = np.mean(self.results['rewards'][-log_every:])
                success = np.mean(self.results['successes'][-log_every:]) * 100
                line = (f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                        f"Training success: {success:.0f}%, Epsilon: {self.agent.epsilon:.3f}")
                if evaluate_greedy:
                    route = self.results['greedy_steps'][-1]
                    line += (f", Greedy route: {'does not reach the goal' if route is None else f'{route} moves'}"
                             f" (shortest {self.results['shortest_route']})")
                print(line)

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
    parser.add_argument("--window", type=int, default=100,
                        help="episodes the greedy route must stay shortest to count as converged")
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
    summary = summarize(results, args.window)
    w = args.window

    print(f"Training episodes reaching the goal (with exploration): "
          f"{summary['training_success_last_window'] * 100:.0f}% of the last {w}, "
          f"first at episode {summary['first_training_success_episode']} ({results['training_time']:.1f}s)")
    if 'shortest_route' in summary:
        print(f"Greedy policy (no exploration): reaches the goal in {summary['greedy_reaches_goal_last_window'] * 100:.0f}% "
              f"of the last {w} checks, takes a shortest route ({summary['shortest_route']} moves) in "
              f"{summary['greedy_shortest_last_window'] * 100:.0f}%")
        if summary['converged_episode'] is not None:
            print(f"Converged: shortest route at every check from episode {summary['converged_episode']} on")
        else:
            print(f"Not converged: the greedy route was not a shortest route at every one of the last {w} checks")

    if args.save:
        results['summary'] = summary
        results['run_info'] = run_info(package_root, args, config, env)
        print("Saved", experiment.save_results(args.save))


if __name__ == "__main__":
    main()
