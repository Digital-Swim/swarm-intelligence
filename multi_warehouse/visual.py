import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from multi_warehouse.warehouse import WareHouseEnvironment


class MultiWarehouseVisualizer:
    def __init__(self, warehouse_env: WareHouseEnvironment):
        """
        Initializes the visualization engine wrapper.
        
        Args:
            warehouse_env (MultiWarehouseEnv): An active instance of your swarm environment.
        """
        self.warehouse_env = warehouse_env
        
        # 1. Pre-initialize persistent fig and ax variables to keep memory usage low
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # 2. Cache the background score heatmap matrix so your computer doesn't 
        # re-calculate 10,000 spatial grids every single frame update.
        self.heatmap_data = self.get_heatmap_data()
        
        # Fallback interval storage initialization
        self.current_interval = 30 

    def _render_frame(self):
        """Clears the axis and repaints all static/dynamic environmental assets."""
        self.ax.cla() # Wipe the canvas to prevent previous positions from bleeding over
        
        # Fetch up-to-date position matrices from the environment wrapper
        swarm_agent_positions = self.warehouse_env.get_swarm_agent_positions()
        store_positions = self.warehouse_env.store_positions
        house_positions = self.warehouse_env.house_positions
        best_agent_positions = self.warehouse_env.best_group_positions
        world_size = self.warehouse_env.world_size

        # Layer 1: Plot the static smooth score gradient heatmap background
        self.ax.imshow(
            self.heatmap_data, 
            extent=[0, world_size, 0, world_size], #type:ignore
            origin='lower', 
            cmap='coolwarm', 
            alpha=0.6, 
            interpolation='bilinear', 
            zorder=1
        ) 

        # Layer 2: Plot the moving swarm agent positions
        for (x, y), color in zip(swarm_agent_positions[:, :2], self.warehouse_env.agent_colors):
            self.ax.scatter(
                x, 
                y, 
                c=color, 
                label='Swarm Agents', 
                s=50, 
                zorder=3
            )
        
        #print("Swarm Agent Scores:\n", self.warehouse_env.agent_scores)
        # Layer 3: Annotate rounded integer score values directly on top of each dot
        for (x, y), label_val in zip(swarm_agent_positions[:, :2], self.warehouse_env.agent_scores):
            self.ax.text(
                x + 1, 
                y + 1,            
                str(round(label_val)),     
                fontsize=9, 
                color='black', 
                weight='semibold', 
                zorder=4, # High zorder forces text to stay cleanly visible on top
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.15')
            )
        
        
        for position in best_agent_positions:
            self.ax.scatter(
                position[0], 
                position[1], 
                c='gold', 
                marker='*', 
                s=250, 
                edgecolors='black', 
                linewidths=1.2, 
                label='Best Position', 
                zorder=5
            )
            
                   
        # Layer 5: Plot static infrastructure markers (Store / Houses)
        self.ax.scatter(store_positions[:, 0], store_positions[:, 1], c='green', label='Store', s=50, zorder=3)
        self.ax.scatter(house_positions[:, 0], house_positions[:, 1], c='red', label='Houses', s=50, zorder=3)

        # Layer 6: Formatting, boundary limits, and labels
        self.ax.set_xlim(0, world_size)
        self.ax.set_ylim(0, world_size)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Warehouse Environment Continuous Simulation')
        
        # Position the legend completely outside on the upper edge of the plot
        #self.ax.legend(bbox_to_anchor=(0, 1.04), loc='lower left', borderaxespad=0., ncol=4)
        self.ax.grid(True, linestyle='--', alpha=0.5)

    def _animation_update(self, frame):
        """The internal driver clock loop that links physics math to frames."""
        # Calculate physics delta-time (dt) dynamically from the visual frame interval
        # E.g., if interval is 30 milliseconds, dynamic_dt = 0.03 seconds of elapsed time
        dynamic_dt = self.current_interval / 1000.0
        
        # Step the environment physics forward using your step execution method
        self.warehouse_env.step(dt=dynamic_dt)
        
        # Re-render the newly shifted coordinate system
        self._render_frame()
        return self.ax,

    def animate_environment(self, interval=30):
        """
        Launches an interactive, infinite continuous simulation runtime loop window.
        
        Args:
            interval (int): Speed of each frame step in milliseconds (default: 30ms ~33 FPS).
        """
        self.current_interval = interval
        
        # Paint the baseline visual frame context layout
        self._render_frame()
        
        # Generate and attach the side scale bar once (prevents multiple bar duplications)
        sm = plt.cm.ScalarMappable(cmap='coolwarm')
        sm.set_array(self.heatmap_data)
        self.fig.colorbar(sm, ax=self.ax, label='Environment Reward/Score Value', fraction=0.046, pad=0.04)

        # Configure and execute the persistent infinite loop engine
        self.ani = FuncAnimation(
            self.fig, 
            self._animation_update, 
            frames=None,        # Setting frames to None causes it to loop infinitely
            interval=interval, 
            blit=False
        )
        
        plt.tight_layout()
        plt.show()

    def get_heatmap_data(self):
        """Constructs a 2D meshgrid matrix calculating coordinates across score metrics."""
        GRID_RESOLUTION = 100 
        x_ticks = np.linspace(0, self.warehouse_env.world_size, GRID_RESOLUTION)
        y_ticks = np.linspace(0, self.warehouse_env.world_size, GRID_RESOLUTION)
        X, Y = np.meshgrid(x_ticks, y_ticks)
        
        # Initialize canvas matrix layout
        Z = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))
        
        # Scan and evaluate every coordinate point sector inside the warehouse
        for i in range(GRID_RESOLUTION):
            for j in range(GRID_RESOLUTION):
                Z[i, j] = self.warehouse_env.get_score_at_position((X[i, j], Y[i, j]))
        return Z
