# 3D Maze as test field for the RL agent

<img width="1903" height="1033" alt="Screenshot 2025-08-18 152040" src="https://github.com/user-attachments/assets/4261a20b-9d9b-4f31-b37a-b167521cce3a" />

The 3D maze serves as both the testing environment and visual output for the reinforcement learning algorithm. The 3D view displays a blue maze structure with light pathways (a multi-level maze is shown level by level, side by side), where the red sphere represents the agent's current position and the green sphere marks the target goal. This maze acts as the testing ground where the Q-learning agent must navigate from start to finish, learning optimal paths through trial and error across multiple episodes. With the 3-level maze, vertical movement between levels adds complexity, creating a more challenging navigation problem than traditional 2D mazes.

The maze functions as real-time feedback for algorithm performance, visually demonstrating how the agent explores the environment and gradually improves its path-finding strategy. Each training episode resets the agent to the starting position in the same maze, so the Q-table can learn that maze; Reset Maze or a new size generates a new maze and starts a fresh agent. (The Q-table only knows the agent's position, so it cannot carry what it learned over to a different maze.) The visual representation allows observers to track the agent's decision-making process, showing whether it's exploring new areas, exploiting known good paths, or getting stuck in suboptimal routes. This immediate visual feedback makes the abstract learning process tangible and helps identify when the algorithm is converging towards optimal solutions.

## Left Tab Control Interface

Training Controls

The training controls provide the main interface for managing the reinforcement learning process. The Start Training button runs the Q-learning algorithm, while Reset Maze generates a new maze and resets the agent. The statistics panel shows real-time metrics such as the current episode, step count, cumulative reward, and epsilon value (exploration rate); the footer also counts how often the goal was reached. These give immediate feedback on the agent’s learning progress.

Parameters

The parameters section includes sliders for adjusting key Q-learning settings during training. The learning rate slider sets how quickly the agent updates its Q-table. Higher values mean faster updates but can reduce stability. The epsilon slider controls the balance between exploration and exploitation. A higher epsilon encourages more exploration, while a lower value focuses on using known paths. The speed slider sets training steps per second (0.1 to 1000). Adjusting these during training makes it possible to refine learning without restarting.

Maze Configuration

The maze configuration controls allow testing with different levels of complexity. The size options (Small 15x15, Medium 21x21, Large 31x31, and a 21x21 maze with 3 levels) define the maze dimensions. Wall density sets the minimum share of the maze that is wall (dead ends are trimmed to reach it), while the branching factor controls how many extra loop corridors, and so alternative routes, are added. These settings make it possible to study how environment complexity influences learning speed and performance.
