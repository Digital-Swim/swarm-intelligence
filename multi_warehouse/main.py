import numpy as np

from multi_warehouse.config import ( 
                                    DIMENSION,
                                    MAX_VELOCITY,
                                    MIN_HOUSE_DISTANCE,
                                    NUM_HOUSES,
                                    NUM_STORES,
                                    NUM_SWARM_AGENTS,
                                    NUM_WAREHOUSES,
                                    VELOCITY_INERTIA,
                                    VELOCITY_TOWARDS_BEST_POSITION,
                                    VELOCITY_TOWARDS_GLOBAL_BEST, 
                                    WORLD_SIZE 
                                    
                                    )
from multi_warehouse.visual import MultiWarehouseVisualizer
from multi_warehouse.warehouse import WareHouseEnvironment

env = WareHouseEnvironment(
    dimension=DIMENSION,
    world_size=WORLD_SIZE,
    no_of_agents=NUM_SWARM_AGENTS,
    no_of_houses=NUM_HOUSES,
    no_of_stores=NUM_STORES,
    no_of_warehouses=NUM_WAREHOUSES,
    inertia_weight=VELOCITY_INERTIA,
    cognitive_weight=VELOCITY_TOWARDS_BEST_POSITION,
    social_weight=VELOCITY_TOWARDS_GLOBAL_BEST,
    min_house_distance=MIN_HOUSE_DISTANCE,
    max_velocity=MAX_VELOCITY,
)
visualizer = MultiWarehouseVisualizer(env)
visualizer.animate_environment(interval=30)
