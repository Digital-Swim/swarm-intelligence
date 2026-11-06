import numpy as np

from warehouse.config import (
    NUM_HOUSES,
    NUM_STORES,
    NUM_SWARM_AGENTS,
    POSITION_RANGE,
    WORLD_SIZE,
)
from warehouse.env import WarehouseEnv
from warehouse.visual import WarehouseVisualizer


store_positions = np.random.rand(NUM_STORES, 2) * POSITION_RANGE
house_positions = np.random.rand(NUM_HOUSES, 2) * POSITION_RANGE


env = WarehouseEnv(
    num_of_swarm_agents=NUM_SWARM_AGENTS,
    store_positions=store_positions,
    house_positions=house_positions,
    world_size=WORLD_SIZE,
)

visualizer = WarehouseVisualizer(env)
visualizer.animate_environment(interval=30)
