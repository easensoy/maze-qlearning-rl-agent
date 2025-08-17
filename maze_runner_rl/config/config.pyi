from typing import List

class Actions:
    up: List[int]
    down: List[int] 
    left: List[int]
    right: List[int]

class Penalties:
    step_penalty: float
    wall_penalty: float

class Rewards:
    goal_reward: int

class Environment:
    observation_low: int
    observation_high: int
    observation_width: int
    observation_height: int

class Maze:
    boundary_min: int
    empty_cell: int
    x_index: int
    y_index: int
    agent_value: int

class Rendering:
    agent_symbol: str
    start_symbol: str
    end_symbol: str
    wall_symbol: str
    empty_symbol: str
    line_separator: str
    empty_string: str

class Agent:
    learning_rate: float
    epsilon: float
    default_action: int
    min_positions: int

class Config:
    actions: Actions
    penalties: Penalties
    rewards: Rewards
    environment: Environment
    maze: Maze
    rendering: Rendering
    agent: Agent

CONFIG: Config