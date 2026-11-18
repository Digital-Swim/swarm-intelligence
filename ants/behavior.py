from scipy.ndimage import gaussian_filter
import numpy as np
from numpy.typing import NDArray

# Target Home, Goal 
# Initial postion random , velocity random 
# Make a grid of markers , set them to 0 and when ever a particle on it raise the value 
# PArticles attract towards stronger value on the grid and also have randomness 
# Make the grid loose value with time 

MAX_FORCE = 50
MAX_SPEED = 100
ALIGNMENT_WEIGHT  = 0.0
RANDOM_WEIGHT = 0.5

class AntBehavior:
  
    def __init__(self) -> None:
        self.evaporation_rate = 0.1
        self.pheromone_deposition = 1
        self.max_pheromone = 50
        self.diffusion_sigma = 0.1
        self.grid = np.zeros((1101, 751))

    
    def get_grid(self):
        return self.grid

    def limit(self, value, max):
        mag = np.linalg.norm(value)
        if mag > max:
            return (value / mag) * max
        return value
    
    def steering_force(self, target_velocity, current_velocity):
        if np.linalg.norm(target_velocity) < 1e-6:
            return np.zeros(2)
        desired = target_velocity / np.linalg.norm(target_velocity) * MAX_SPEED
        return self.limit(desired - current_velocity, MAX_FORCE)
    
    def get_nearest_high_value_pos(self, position):

        x, y = map(int, position)

        best_value = float('-inf')
        best_pos = None

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:

                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                if not (0 <= nx < self.grid.shape[0]):
                    continue

                if not (0 <= ny < self.grid.shape[1]):
                    continue

                current_value = self.grid[nx, ny]

                if current_value > best_value:
                    best_value = current_value
                    best_pos = (nx, ny)

        return best_pos, best_value
    

    def find_best_8way_step(self,grid, x, y, filter_radius=1):
        """
        Finds the best 8-way neighboring cell to move to based on local neighborhood values.
        
        Parameters:
        - grid: 2D NumPy array
        - x, y: Current column and row coordinates of the particle
        - filter_radius: Size of the local window around each potential step (1 = 3x3 window)
        
        Returns:
        - best_dx, best_dy: The optimal step direction (-1, 0, or 1)
        - best_score: The aggregated window value at the destination
        """
        H, W = grid.shape
        patch_r = filter_radius + 1
        
        # Pad the grid to handle edge boundaries gracefully
        padded = np.pad(grid, pad_width=patch_r, mode='edge')
        px, py = x + patch_r, y + patch_r
        
        # Define all 8 possible movement directions plus staying still (0, 0)
        directions = [
            (-1,  0), # Up
            ( 1,  0), # Down
            ( 0, -1), # Left
            ( 0,  1), # Right
            (-1, -1), # Up-Left
            (-1,  1), # Up-Right
            ( 1, -1), # Down-Left
            ( 1,  1), # Down-Right
            ( 0,  0)  # Stay put if surrounded by lower values
        ]
        
        best_score = -np.inf
        best_dx, best_dy = 0, 0
        
        # Evaluate each neighboring target position
        for dx, dy in directions:
            # Center of the local neighborhood patch if the particle moves to (x + dx, y + dy)
            target_cx = px + dx
            target_cy = py + dy
            
            # Extract the local window around that target position
            window = padded[target_cy - filter_radius : target_cy + filter_radius + 1,
                            target_cx - filter_radius : target_cx + filter_radius + 1]
            
            # Aggregate the window (you can use np.sum or np.mean depending on your preference)
            score = np.sum(window)
            
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy
                
        return best_dx, best_dy

    def update_velocities1(
        self,
        positions: NDArray[np.float32],
        velocities: NDArray[np.float32],
        obstacles: NDArray[np.float32],
        dt: float,
        mouse_position: NDArray[np.float32],
        mouse_mode: int,
        bounds: tuple[int, int],
        targets: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
            
            N = len(positions)
            new_vel = velocities.copy()
            
            # Removing points from grid 
            #self.grid = np.maximum(self.grid - self.evaporation_rate, 0)
            self.grid = gaussian_filter(
                self.grid,
                sigma=self.diffusion_sigma
            )
            self.grid *= np.exp(-self.evaporation_rate * dt)
            
            for n in range(N):
                position = positions[n]
                x, y =  position.astype(int)
                
                # Towards best 
                best_pos = self.get_nearest_high_value_pos((x,y))
                alignment = self.steering_force(best_pos - position, velocities[n])
                
                # Random exploration
                random_position = np.array([
                    np.random.uniform(x - 1.0, x + 1.0),
                    np.random.uniform(y - 1.0, y + 1.0)
                ])
                                                                        
                random_force = self.steering_force(random_position - position, velocities[n])
                
                acceleration = alignment * ALIGNMENT_WEIGHT + random_force * RANDOM_WEIGHT
                new_vel[n] += acceleration
                self.grid[x, y] = 1
                                                        
            return new_vel
     
        
    def update_velocities(
           self,
           positions: NDArray[np.float32],
           velocities: NDArray[np.float32],
           obstacles: NDArray[np.float32],
           dt: float,
           mouse_position: NDArray[np.float32],
           mouse_mode: int,
           bounds: tuple[int, int],
           targets: NDArray[np.float32] | None = None,
       ) -> NDArray[np.float32]:
        
        new_velocity = velocities.copy()

        N = len(positions)

        # Removing points from grid 
        #self.grid = np.maximum(self.grid - self.evaporation_rate, 0)
        
        self.grid = gaussian_filter(
                self.grid,
                sigma=self.diffusion_sigma
            )
        self.grid *= np.exp(-self.evaporation_rate * dt)
        
        for n in range(N):

            # Force is propertional to accelaration which is change in velocity 
            position = positions[n]
            x, y =  position.astype(int)
                                    
            # Look slightly ahead in the direction of current velocity
            look_ahead_distance = 3.0
            norm_vel = velocities[n] / (np.linalg.norm(velocities[n]) + 1e-5)
            ahead_x = int(round(x + norm_vel[0] * look_ahead_distance))
            ahead_y = int(round(y + norm_vel[1] * look_ahead_distance))

            # Find the best step from that future look-ahead position
            dx, dy = self.find_best_8way_step(self.grid, ahead_x, ahead_y, 3)


            current_vel = velocities[n]
            current_speed = np.linalg.norm(current_vel)

            if dx != 0 or dy != 0:
                # Desired heading from the grid
                target_dir = np.array([dx, dy], dtype=float)
                target_dir /= np.linalg.norm(target_dir)
                
                # Blend old velocity with new target direction (Inertia factor = 0.7)
                # This prevents abrupt snapping and keeps them moving smoothly through local bumps
                new_dir = 0.7 * (current_vel / (current_speed + 1e-5)) + 0.3 * target_dir
                new_dir /= np.linalg.norm(new_dir)
                
                new_velocity[n] = current_speed * new_dir
            else:
                # If no higher gradient is found, maintain forward momentum instead of stopping!
                # They coast straight ahead until they catch a new scent gradient.
                if current_speed > 0:
                    new_velocity[n] = current_vel
                else:
                    # Give them a tiny random kick if completely dead-stopped
                    new_velocity[n] = np.random.randn(2) * 0.5
                    
                    
            #best_pos, best_val = self.get_nearest_high_value_pos((x,y))
            #print(best_val)
            #if best_val == 0:
            # Get a random velocity 
            #r = np.random.randint(1, 300)
            #if r == 31111:
                # change velocity 
            #    new_velocity[n] = np.random.uniform(-1, 1,size=2) * np.random.randint(20, 30)
            #else:
            #    # Towards best 
            #    direction = best_pos - position
            #    mag_dir = np.linalg.norm(direction)
            #    new_velocity[n] = np.linalg.norm(velocities[n]) * direction / mag_dir
                
            self.grid[x, y] = min(
                            self.grid[x, y] + self.pheromone_deposition,
                            self.max_pheromone
                        )            
            
        return new_velocity
           
    