# Showing warehouse, house and store as 2D scatter plot using matplotlib
import matplotlib.pyplot as plt
import numpy as np

from warehouse.env import WarehouseEnv

class WarehouseVisualizer:
    def __init__(self, warehouse_env:WarehouseEnv):
        self.warehouse_env = warehouse_env

    def plot_environment(self):
        # Get positions
        swarm_agent_positions = self.warehouse_env.get_swarm_agent_positions()
        store_positions = self.warehouse_env.store_positions
        house_positions = self.warehouse_env.house_positions
        best_agent_position = self.warehouse_env.find_best_position()
        
        # Create a scatter plot
        plt.figure(figsize=(10, 10))
        heatmap = plt.imshow(self.get_heatmap_data(), 
                    extent=[0, self.warehouse_env.world_size, 0, self.warehouse_env.world_size], #type:ignore 
                    origin='lower', 
                    cmap='coolwarm', alpha=0.6, interpolation='bilinear') 
        plt.colorbar(heatmap, label='Environment Reward/Score Value', fraction=0.046, pad=0.04)

        plt.scatter(swarm_agent_positions[:, 0], swarm_agent_positions[:, 1], c='blue', label='Swarm Agents', s=50)
        
        for (x, y), label_val in zip(swarm_agent_positions[:, :2], self.warehouse_env.agent_scores):
            plt.text(
                x + 1,            # Offset step to prevent overlapping the dot marker
                y + 1,            
                str(round(label_val)),     
                fontsize=9,         
                color='black',      
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.15')
            )
        
        plt.scatter(
            best_agent_position[0], 
            best_agent_position[1], 
            c='gold',               # Clean golden-yellow fill color
            marker='*',             # Matplotlib star symbol flag
            s=250,                  # Large size so it stands out over standard dots
            edgecolors='black',     # High-contrast black outline border
            linewidths=1.2,
            zorder=5                # Ensures it renders on top of the heatmap and other agents
        )
                    
        plt.scatter(store_positions[:, 0], store_positions[:, 1], c='green', label='Store', s=50)
        plt.scatter(house_positions[:, 0], house_positions[:, 1], c='red', label='Houses', s=50)

        # Set plot limits and labels
        plt.xlim(0, self.warehouse_env.world_size)
        plt.ylim(0, self.warehouse_env.world_size)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Warehouse Environment Visualization')
        plt.legend(bbox_to_anchor=(0, 1.04), loc='lower left', borderaxespad=0.)
        plt.grid(True)
        plt.show()
    
    def get_heatmap_data(self):
        # 2. Generate and display the background score heatmap
        # Grid resolution (e.g., 100x100 calculations across the environment grid)
        GRID_RESOLUTION = 100 
        x_ticks = np.linspace(0, self.warehouse_env.world_size, GRID_RESOLUTION)
        y_ticks = np.linspace(0, self.warehouse_env.world_size, GRID_RESOLUTION)
        X, Y = np.meshgrid(x_ticks, y_ticks)

        # Initialize an empty matrix for score values
        Z = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))
        
        # Calculate the score for every discrete point in the warehouse space
        # Assumes your score function accepts an array/tuple like (x, y) or separate arguments
        for i in range(GRID_RESOLUTION):
            for j in range(GRID_RESOLUTION):
                # Query the environment's score logic at this spatial coordinate
                # Adjust signature to self.warehouse_env.score_function(X[i,j], Y[i,j]) if needed
                Z[i, j] = self.warehouse_env.get_score_at_position((X[i, j], Y[i, j]))

        return Z