import numpy as np

from warehouse.env import WarehouseEnv
from warehouse.visual import WarehouseVisualizer



# stroe_positions = np.array([[50, 50], [10, 10], [30, 40]])
store_positions = np.random.rand(5, 2) * 100

#house_positions = np.array([[10, 10], [90, 90]])
house_positions = np.random.rand(10, 2) * 100


env = WarehouseEnv(
    num_of_swarm_agents=99,
    store_positions=store_positions,
    house_positions=house_positions,
    world_size=150
)

visualizer = WarehouseVisualizer(env)
visualizer.animate_environment()
