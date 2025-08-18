from config.config import CONFIG

class MazeGenerator:
    def __init__(self, width=None, height=None):
        self.width = width or CONFIG.maze_generator.default_width
        self.height = height or CONFIG.maze_generator.default_height
        self.maze = None
        self.start = (CONFIG.maze_generator.start_x, CONFIG.maze_generator.start_y)
        self.end = (self.width - CONFIG.maze_generator.end_offset, self.height - CONFIG.maze_generator.end_offset)

    def generate_maze(self):
        import random
        self.maze = [[CONFIG.maze_generator.wall_value] * self.width for _ in range(self.height)]
        
        visited = set()
        self._carve_path_recursive(self.start[0], self.start[1], visited)
        
        self.maze[self.end[CONFIG.maze.y_index]][self.end[CONFIG.maze.x_index]] = CONFIG.maze_generator.path_value

    def _carve_path_recursive(self, x, y, visited):
        import random
        
        visited.add((x, y))
        self.maze[y][x] = CONFIG.maze_generator.path_value
        
        directions = [
            (x, y - 2),
            (x, y + 2),
            (x - 2, y),
            (x + 2, y)
        ]
        
        random.shuffle(directions)
        
        for nx, ny in directions:
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                (nx, ny) not in visited):
                
                wall_x, wall_y = (x + nx) // 2, (y + ny) // 2
                self.maze[wall_y][wall_x] = CONFIG.maze_generator.path_value
                
                self._carve_path_recursive(nx, ny, visited)

    def get_maze(self):
        return self.maze
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end