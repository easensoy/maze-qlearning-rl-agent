import numpy as np
import json
import time
from datetime import datetime

class TrainingExperiment:
    def __init__(self, env, agent, name="experiment"):
        self.env = env
        self.agent = agent
        self.name = name
        self.results = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'epsilon_values': [],
            'training_time': 0,
            'total_episodes': 0
        }
    
    def run(self, num_episodes=1000, max_steps=200):
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                action = self.agent.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.agent.update_q_table(state, action, reward, next_state)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            self.agent.decay_epsilon()
            
            self.results['episodes'].append(episode)
            self.results['rewards'].append(episode_reward)
            self.results['steps'].append(steps)
            self.results['epsilon_values'].append(self.agent.epsilon)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.results['rewards'][-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
        
        self.results['training_time'] = time.time() - start_time
        self.results['total_episodes'] = num_episodes
        
        return self.results
    
    def save_results(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return filename