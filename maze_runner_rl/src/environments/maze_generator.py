import random

from config.config import CONFIG


class MazeGenerator:
    def __init__(self, width=None, height=None, rng=None):
        self.width = width or CONFIG.maze_generator.default_width
        self.height = height or CONFIG.maze_generator.default_height
        self.rng = rng or random.Random()
        self.maze = None
        self.start = (CONFIG.maze_generator.start_x, CONFIG.maze_generator.start_y)
        self.end = (self.width - CONFIG.maze_generator.end_offset, self.height - CONFIG.maze_generator.end_offset)

    def generate_maze(self):
        """Recursive-backtracker (depth-first) maze: every open cell is connected and
        there is exactly one path from start to end. Uses an explicit stack, so large
        mazes do not hit Python's recursion limit."""
        self.maze = [[CONFIG.maze_generator.wall_value] * self.width for _ in range(self.height)]

        start_x, start_y = self.start
        self.maze[start_y][start_x] = CONFIG.maze_generator.path_value
        visited = {(start_x, start_y)}
        stack = [(start_x, start_y)]

        while stack:
            x, y = stack[-1]
            candidates = [(x, y - 2), (x, y + 2), (x - 2, y), (x + 2, y)]
            self.rng.shuffle(candidates)
            for nx, ny in candidates:
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    wall_x, wall_y = (x + nx) // 2, (y + ny) // 2
                    self.maze[wall_y][wall_x] = CONFIG.maze_generator.path_value
                    self.maze[ny][nx] = CONFIG.maze_generator.path_value
                    visited.add((nx, ny))
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()

        self.maze[self.end[CONFIG.maze.y_index]][self.end[CONFIG.maze.x_index]] = CONFIG.maze_generator.path_value

    def get_maze(self):
        return self.maze

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end
