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
        # Start with all walls
        self.maze = [[CONFIG.maze_generator.wall_value] * self.width for _ in range(self.height)]
        
        # Create paths using depth-first search maze generation
        visited = set()
        self._carve_path_recursive(self.start[0], self.start[1], visited)
        
        # Ensure goal is accessible
        self.maze[self.end[CONFIG.maze.y_index]][self.end[CONFIG.maze.x_index]] = CONFIG.maze_generator.path_value

    def _carve_path_recursive(self, x, y, visited):
        import random
        
        # Mark current cell as visited and carve it
        visited.add((x, y))
        self.maze[y][x] = CONFIG.maze_generator.path_value
        
        # Get all possible directions (2 cells away to leave walls between paths)
        directions = [
            (x, y - 2),  # North
            (x, y + 2),  # South  
            (x - 2, y),  # West
            (x + 2, y)   # East
        ]
        
        # Randomize directions
        random.shuffle(directions)
        
        for nx, ny in directions:
            # Check if the cell is within bounds and unvisited
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                (nx, ny) not in visited):
                
                # Carve the wall between current cell and next cell
                wall_x, wall_y = (x + nx) // 2, (y + ny) // 2
                self.maze[wall_y][wall_x] = CONFIG.maze_generator.path_value
                
                # Recursively carve from the new cell
                self._carve_path_recursive(nx, ny, visited)

    def get_maze(self):
        return self.maze
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end