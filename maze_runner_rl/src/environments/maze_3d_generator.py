import numpy as np
import random

class Maze3DGenerator:
    def __init__(self, width=None, height=None, depth=None, wall_density=0.65, branching_factor=0.4, dead_end_percentage=0.3):
        # Ensure odd dimensions for proper maze generation (minimum 15 for complexity)
        self.width = max(width or 21, 15)
        self.height = max(height or 21, 15)  
        self.depth = max(depth or 3, 1)
        
        # Make dimensions odd to ensure proper wall/path structure
        if self.width % 2 == 0:
            self.width += 1
        if self.height % 2 == 0:
            self.height += 1
        if self.depth % 2 == 0:
            self.depth += 1
            
        # Maze complexity parameters
        self.wall_density = wall_density  # Target 70-80% wall coverage
        self.branching_factor = branching_factor  # Controls decision points
        self.dead_end_percentage = dead_end_percentage  # Configurable dead end percentage
        
        self.maze = None
        self.start = (1, 1, 0)  # Start at ground level
        self.end = (self.width - 2, self.height - 2, min(self.depth - 1, 2))  # End position

    def generate_maze(self):
        """Generate a proper maze with carved corridors using recursive backtracking"""
        # Initialize completely filled with walls (1 = wall, 0 = path/corridor)
        self.maze = np.ones((self.depth, self.height, self.width), dtype=int)
        
        # Start with a clean slate - carve out the starting area
        start_x, start_y, start_z = self.start
        self.maze[start_z, start_y, start_x] = 0
        
        # Use proper recursive backtracking to carve corridors
        self._carve_maze_recursive(start_x, start_y, start_z)
        
        # Ensure end position is accessible
        end_x, end_y, end_z = self.end
        self.maze[end_z, end_y, end_x] = 0
        
        # Connect start to end if not already connected
        if not self._is_reachable(self.start, self.end):
            self._carve_direct_path(self.start, self.end)
        
        # Add some additional branching for complexity
        self._add_branching_corridors()
        
        # Ensure dead ends are clearly visible and well-distributed
        self._enhance_dead_end_visibility()
        
        print(f"Generated maze: {self._calculate_wall_percentage():.1f}% walls, Dead ends enhanced")

    def _connect_isolated_regions(self):
        """Connect any isolated maze regions to ensure full connectivity"""
        # Find all path cells
        path_cells = []
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    if self.maze[z, y, x] == 0:
                        path_cells.append((x, y, z))
        
        # If we have multiple disconnected regions, connect them
        if len(path_cells) > 0:
            # Use a simple approach - occasionally carve connecting corridors
            for _ in range(max(3, len(path_cells) // 20)):
                # Pick two random path cells and try to connect them
                if len(path_cells) >= 2:
                    p1 = random.choice(path_cells)
                    p2 = random.choice(path_cells)
                    self._carve_connecting_path(p1, p2)

    def _carve_connecting_path(self, p1, p2):
        """Carve a connecting path between two points"""
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        
        # Simple L-shaped connection
        current_x, current_y, current_z = x1, y1, z1
        
        # Move towards target x
        while current_x != x2:
            step = 1 if x2 > current_x else -1
            current_x += step
            if (0 < current_x < self.width - 1 and 
                0 < current_y < self.height - 1 and
                0 < current_z < self.depth - 1):
                self.maze[current_z, current_y, current_x] = 0
        
        # Move towards target y
        while current_y != y2:
            step = 1 if y2 > current_y else -1
            current_y += step
            if (0 < current_x < self.width - 1 and 
                0 < current_y < self.height - 1 and
                0 < current_z < self.depth - 1):
                self.maze[current_z, current_y, current_x] = 0

    def _add_maze_complexity(self):
        """Add final touches to increase maze complexity and organic feel"""
        # Add some random corridor widening in places
        path_cells = []
        for z in range(1, self.depth - 1):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.maze[z, y, x] == 0:
                        path_cells.append((x, y, z))
        
        # Randomly widen some corridors for variety
        for _ in range(len(path_cells) // 10):
            if path_cells:
                x, y, z = random.choice(path_cells)
                # Try to widen corridor in random direction
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (1 <= nx < self.width - 1 and 
                        1 <= ny < self.height - 1 and
                        random.random() < 0.3):  # 30% chance to widen
                        self.maze[z, ny, nx] = 0

    def _carve_maze_recursive(self, x, y, z):
        """Proper recursive backtracking to carve continuous corridors"""
        # Mark current position as a corridor (already carved)
        self.maze[z, y, x] = 0
        
        # Define movement directions (only cardinal directions, no diagonals)
        directions = [
            (0, -2, 0),   # North - move 2 cells to skip intermediate wall
            (0, 2, 0),    # South
            (-2, 0, 0),   # West  
            (2, 0, 0),    # East
        ]
        
        # Add vertical movement occasionally for 3D complexity
        if self.depth > 1 and random.random() < 0.3:  # 30% chance for vertical
            if z > 1: directions.append((0, 0, -2))  # Down
            if z < self.depth - 2: directions.append((0, 0, 2))  # Up
        
        # Shuffle directions for random maze generation
        random.shuffle(directions)
        
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Check if target cell is valid and still a wall (uncarved)
            if (self._is_valid_maze_cell(nx, ny, nz) and self.maze[nz, ny, nx] == 1):
                
                # Carve the corridor from current to target cell
                if dx != 0:  # Moving horizontally
                    wall_x = x + dx // 2
                    self.maze[z, y, wall_x] = 0  # Carve intermediate wall
                elif dy != 0:  # Moving vertically in 2D plane
                    wall_y = y + dy // 2
                    self.maze[z, wall_y, x] = 0  # Carve intermediate wall
                elif dz != 0:  # Moving between levels
                    wall_z = z + dz // 2
                    self.maze[wall_z, y, x] = 0  # Carve vertical connection
                
                # Recursively carve from the new position
                self._carve_maze_recursive(nx, ny, nz)

    def _is_valid_maze_cell(self, x, y, z):
        """Check if coordinates are within maze bounds for carving"""
        return (1 <= x < self.width - 1 and 
                1 <= y < self.height - 1 and 
                0 <= z < self.depth)
    
    def _carve_direct_path(self, start, end):
        """Carve a direct path from start to end"""
        x1, y1, z1 = start
        x2, y2, z2 = end
        
        # Carve horizontal path first
        current_x, current_y, current_z = x1, y1, z1
        
        while current_x != x2:
            step = 1 if x2 > current_x else -1
            current_x += step
            if 0 <= current_x < self.width:
                self.maze[current_z, current_y, current_x] = 0
                
        while current_y != y2:
            step = 1 if y2 > current_y else -1
            current_y += step
            if 0 <= current_y < self.height:
                self.maze[current_z, current_y, current_x] = 0
                
        while current_z != z2:
            step = 1 if z2 > current_z else -1
            current_z += step
            if 0 <= current_z < self.depth:
                self.maze[current_z, current_y, current_x] = 0
    
    def _add_branching_corridors(self):
        """Add some additional corridors for complexity"""
        # Find existing corridor cells
        corridors = []
        for z in range(self.depth):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.maze[z, y, x] == 0:
                        corridors.append((x, y, z))
        
        # Add some random branches from existing corridors
        num_branches = min(10, len(corridors) // 4)
        for _ in range(num_branches):
            if corridors:
                x, y, z = random.choice(corridors)
                # Try to carve a short branch
                directions = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
                dx, dy, dz = random.choice(directions)
                
                for step in range(1, 4):  # Short branches
                    nx, ny, nz = x + dx * step, y + dy * step, z + dz * step
                    if (1 <= nx < self.width - 1 and 1 <= ny < self.height - 1 and 
                        0 <= nz < self.depth):
                        self.maze[nz, ny, nx] = 0
                    else:
                        break

    def _add_strategic_dead_ends(self):
        """Add strategic dead ends to increase maze complexity"""
        path_cells = []
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    if self.maze[z, y, x] == 0:
                        path_cells.append((x, y, z))
        
        # Add dead ends to reach target percentage
        target_dead_ends = int(len(path_cells) * self.dead_end_percentage)
        current_dead_ends = 0
        
        for x, y, z in random.sample(path_cells, min(target_dead_ends, len(path_cells))):
            if self._can_create_dead_end(x, y, z):
                self._create_dead_end(x, y, z)
                current_dead_ends += 1
                if current_dead_ends >= target_dead_ends:
                    break

    def _can_create_dead_end(self, x, y, z):
        """Check if we can create a dead end from this position"""
        directions = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
        wall_neighbors = 0
        
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth):
                if self.maze[nz, ny, nx] == 1:
                    wall_neighbors += 1
        
        return wall_neighbors >= 2  # Can create dead end if mostly surrounded by walls

    def _create_dead_end(self, x, y, z):
        """Create a dead end branch from the given position"""
        directions = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
        random.shuffle(directions)
        
        for dx, dy, dz in directions:
            branch_length = random.randint(1, 4)
            current_x, current_y, current_z = x, y, z
            
            for step in range(branch_length):
                current_x += dx
                current_y += dy
                current_z += dz
                
                if (1 <= current_x < self.width - 1 and 
                    1 <= current_y < self.height - 1 and
                    1 <= current_z < self.depth - 1 and
                    self.maze[current_z, current_y, current_x] == 1):
                    self.maze[current_z, current_y, current_x] = 0
                else:
                    break
            break  # Only create one dead end branch per call

    def _create_dead_end_branch(self, cx, cy, cz):
        """Create random dead-end branches for maze complexity"""
        directions = [(0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
        random.shuffle(directions)
        
        for dx, dy, dz in directions:
            branch_length = random.randint(1, 3)
            x, y, z = cx, cy, cz
            
            for _ in range(branch_length):
                x, y, z = x + dx, y + dy, z + dz
                if (1 <= x < self.width - 1 and 
                    1 <= y < self.height - 1 and 
                    1 <= z < self.depth - 1 and
                    self.maze[z, y, x] == 1):
                    self.maze[z, y, x] = 0
                else:
                    break
            break  # Only create one dead-end branch per call

    def _ensure_connectivity(self):
        """Ensure maze connectivity and create solution path"""
        if not self._is_reachable(self.start, self.end):
            # Create minimal connecting path
            self._carve_solution_path()

    def _carve_solution_path(self):
        """Carve a minimal path from start to end"""
        x1, y1, z1 = self.start
        x2, y2, z2 = self.end
        
        # Create L-shaped path
        current_x, current_y, current_z = x1, y1, z1
        
        # Move horizontally first
        while current_x != x2:
            self.maze[current_z, current_y, current_x] = 0
            current_x += 1 if x2 > current_x else -1
            
        while current_y != y2:
            self.maze[current_z, current_y, current_x] = 0
            current_y += 1 if y2 > current_y else -1
            
        # Move vertically if needed
        while current_z != z2:
            self.maze[current_z, current_y, current_x] = 0
            current_z += 1 if z2 > current_z else -1
        
        # Ensure end is carved
        self.maze[z2, y2, x2] = 0

    def _adjust_wall_density(self):
        """Adjust maze to achieve target wall density"""
        current_density = self._calculate_wall_percentage() / 100.0
        target_density = self.wall_density
        
        if current_density < target_density:
            # Need more walls - close some passages strategically
            self._close_passages(target_density - current_density)
        elif current_density > target_density + 0.05:  # 5% tolerance
            # Need fewer walls - open more passages
            self._open_passages(current_density - target_density)

    def _enhance_dead_end_visibility(self):
        """Enhance dead end visibility by ensuring they are well-distributed and clearly marked"""
        dead_ends = self._find_dead_ends()
        target_dead_ends = int(self._count_path_cells() * self.dead_end_percentage)
        
        print(f"Found {len(dead_ends)} dead ends, target: {target_dead_ends}")
        
        if len(dead_ends) < target_dead_ends:
            # Need more dead ends - create some by selectively closing passages
            self._create_additional_dead_ends(target_dead_ends - len(dead_ends))
        elif len(dead_ends) > target_dead_ends * 1.5:
            # Too many dead ends - open some to create more connectivity
            self._reduce_excessive_dead_ends(len(dead_ends) - target_dead_ends)
    
    def _find_dead_ends(self):
        """Find all current dead ends in the maze"""
        dead_ends = []
        for z in range(self.depth):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.maze[z, y, x] == 0:  # This is a path
                        # Count neighboring paths
                        neighbors = 0
                        directions = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)]
                        if self.depth > 1:
                            directions.extend([(0, 0, 1), (0, 0, -1)])
                        
                        for dx, dy, dz in directions:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (0 <= nx < self.width and 
                                0 <= ny < self.height and 
                                0 <= nz < self.depth and
                                self.maze[nz, ny, nx] == 0):
                                neighbors += 1
                        
                        if neighbors == 1:  # Dead end has exactly 1 neighbor
                            dead_ends.append((x, y, z))
        return dead_ends
    
    def _count_path_cells(self):
        """Count total number of path cells"""
        return np.sum(self.maze == 0)
    
    def _create_additional_dead_ends(self, needed):
        """Create additional dead ends by selectively closing passages"""
        for _ in range(needed):
            # Find a random corridor segment that can be closed to create a dead end
            for attempts in range(100):  # Limit attempts to prevent infinite loops
                x = random.randint(2, self.width - 3)
                y = random.randint(2, self.height - 3)
                z = random.randint(0, min(self.depth - 1, 2))
                
                if self.maze[z, y, x] == 0:  # This is a path
                    # Check if closing this would create a dead end without breaking connectivity
                    neighbors = []
                    directions = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)]
                    
                    for dx, dy, dz in directions:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < self.width and 
                            0 <= ny < self.height and 
                            0 <= nz < self.depth and
                            self.maze[nz, ny, nx] == 0):
                            neighbors.append((nx, ny, nz))
                    
                    if len(neighbors) >= 3:  # Can close this to create a dead end
                        # Close one of the connections to create a dead end
                        nx, ny, nz = random.choice(neighbors)
                        if (nx, ny, nz) not in [self.start, self.end]:
                            # Create a small dead-end corridor
                            self.maze[nz, ny, nx] = 1
                            break
    
    def _reduce_excessive_dead_ends(self, excess):
        """Reduce excessive dead ends by connecting some of them"""
        dead_ends = self._find_dead_ends()
        
        for _ in range(min(excess, len(dead_ends) // 2)):
            if len(dead_ends) < 2:
                break
                
            # Pick a random dead end to extend
            dead_end = random.choice(dead_ends)
            dead_ends.remove(dead_end)
            
            # Try to extend this dead end to connect to another path
            x, y, z = dead_end
            directions = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)]
            random.shuffle(directions)
            
            for dx, dy, dz in directions:
                nx, ny = x + dx * 2, y + dy * 2  # Skip one cell to create proper corridor
                if (1 <= nx < self.width - 1 and 
                    1 <= ny < self.height - 1 and
                    self.maze[z, ny, nx] == 1):  # This is a wall we can break
                    # Create a short corridor
                    self.maze[z, y + dy, x + dx] = 0  # Intermediate cell
                    self.maze[z, ny, nx] = 0  # End cell
                    break

    def _calculate_wall_percentage(self):
        """Calculate current wall density percentage"""
        total_cells = self.maze.size
        wall_cells = np.sum(self.maze == 1)
        return (wall_cells / total_cells) * 100

    def _close_passages(self, density_to_add):
        """Strategically close passages to increase wall density"""
        path_cells = []
        for z in range(1, self.depth - 1):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.maze[z, y, x] == 0 and (x, y, z) != self.start and (x, y, z) != self.end:
                        path_cells.append((x, y, z))
        
        # Calculate how many cells to close
        cells_to_close = int(self.maze.size * density_to_add)
        cells_to_close = min(cells_to_close, len(path_cells) // 3)  # Don't close too many
        
        # Close random path cells that don't break connectivity
        random.shuffle(path_cells)
        closed = 0
        
        for x, y, z in path_cells:
            if closed >= cells_to_close:
                break
                
            # Temporarily close the cell
            self.maze[z, y, x] = 1
            
            # Check if start and end are still connected
            if self._is_reachable(self.start, self.end):
                closed += 1
            else:
                # Reopen if it breaks connectivity
                self.maze[z, y, x] = 0

    def _open_passages(self, density_to_remove):
        """Open strategic passages to decrease wall density"""
        wall_cells = []
        for z in range(1, self.depth - 1):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.maze[z, y, x] == 1:
                        wall_cells.append((x, y, z))
        
        cells_to_open = int(self.maze.size * density_to_remove)
        cells_to_open = min(cells_to_open, len(wall_cells) // 4)
        
        # Open random wall cells
        for x, y, z in random.sample(wall_cells, min(cells_to_open, len(wall_cells))):
            self.maze[z, y, x] = 0

    def _is_reachable(self, start, end):
        """Simple BFS to check if end is reachable from start"""
        from collections import deque
        
        queue = deque([start])
        visited = set([start])
        
        directions = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
        
        while queue:
            x, y, z = queue.popleft()
            
            if (x, y, z) == end:
                return True
                
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                
                if (0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth and
                    (nx, ny, nz) not in visited and self.maze[nz, ny, nx] == 0):
                    visited.add((nx, ny, nz))
                    queue.append((nx, ny, nz))
        
        return False

    def get_maze(self):
        return self.maze
    
    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end

    def is_valid_position(self, x, y, z):
        return (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth and
                self.maze[z, y, x] == 0)