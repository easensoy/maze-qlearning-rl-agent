class MazeGenerator:
    def __init__(self, width=21, height=21):
        self.width = width
        self.height = height
        self.maze = None
        self.start = (1, 1)
        self.end = (width-2, height-2)

    def generate_maze(self):
        self.maze = [[1] * self.width for _ in range(self.height)]
        self._carve_path(self.start[0], self.start[1])
        self.maze[self.end[1]][self.end[0]] = 0

    def _carve_path(self, x, y):
        self.maze[y][x] = 0
        
        neighbours = [
            (x, y-1),
            (x, y+1),
            (x-1, y),
            (x+1, y)
        ]
                
        for nx, ny in neighbours:
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.maze[ny][nx] == 1:
                    self.maze[ny][nx] = 0
                    self._carve_path(nx, ny)

    def get_maze(self):
        return self.maze
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end