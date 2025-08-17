from config.config import CONFIG

class MazeGenerator:
    def __init__(self, width=None, height=None):
        self.width = width or CONFIG.maze_generator.default_width
        self.height = height or CONFIG.maze_generator.default_height
        self.maze = None
        self.start = (CONFIG.maze_generator.start_x, CONFIG.maze_generator.start_y)
        self.end = (self.width - CONFIG.maze_generator.end_offset, self.height - CONFIG.maze_generator.end_offset)

    def generate_maze(self):
        self.maze = [[CONFIG.maze_generator.wall_value] * self.width for _ in range(self.height)]
        self._carve_path(self.start[0], self.start[1])
        self.maze[self.end[CONFIG.maze.y_index]][self.end[CONFIG.maze.x_index]] = CONFIG.maze_generator.path_value

    def _carve_path(self, x, y):
        self.maze[y][x] = CONFIG.maze_generator.path_value
        
        neighbours = [
            (x, y + CONFIG.maze_generator.step_up),
            (x, y + CONFIG.maze_generator.step_down),
            (x + CONFIG.maze_generator.step_left, y),
            (x + CONFIG.maze_generator.step_right, y)
        ]
                
        for nx, ny in neighbours:
            if CONFIG.maze_generator.boundary_min <= nx < self.width and CONFIG.maze_generator.boundary_min <= ny < self.height:
                if self.maze[ny][nx] == CONFIG.maze_generator.wall_value:
                    self.maze[ny][nx] = CONFIG.maze_generator.path_value
                    self._carve_path(nx, ny)

    def get_maze(self):
        return self.maze
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end