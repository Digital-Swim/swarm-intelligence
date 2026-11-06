
# Store / Houses and Warehouse Environment Variables

# Positions of Store and Warehouse are fixed in 2D space
# We need to position n Warehose in the space, such that they are optimally distributed, 
# Conditions 
# Warehoouse should not be close to each other and houses to avoid traffic around the house 
# Warehhouse should not be too far from the store to avoid long delivery times

import numpy as np


class WarehouseEnv:
    def __init__(self, num_of_swarm_agents, store_positions, house_positions, world_size):
        self.num_of_swarm_agents = num_of_swarm_agents
        self.agent_velocities = np.zeros((num_of_swarm_agents, 2))  # Initialize velocities for each agent
        self.agent_scores = np.zeros(num_of_swarm_agents)  # Initialize scores for each agent
        self.store_positions = store_positions
        self.house_positions = house_positions
        self.world_size = world_size
        self.initialize()


    def _update_agent_scores(self):
        scores = []
        for agent_pos in self.swarm_agent_positions:
            scores.append(self.get_score_at_position(agent_pos))
        self.agent_scores = np.array(scores)


    def step(self, dt: float = 1.0):
       
        # Update the positions of the swarm agents
        self._update_warehouse_positions(dt)
        
        # Calculate scores for the new positions
        self._update_agent_scores()
        
        return self.agent_scores
     
    def get_score_at_position(self, position):
        
        # Worst case scenario score is getting the distance from the nearest house and the farthest store
        # S = Distance of agent from nearest house - distannce from farthest store
        # The higher the score, the better the position of the agent
                
        # Calculate distance from nearest store
        distances_to_stores = np.linalg.norm(self.store_positions - position, axis=1)
        farthest_store_distance = np.max(distances_to_stores)

        # Calculate distance from farthest house
        distances_to_houses = np.linalg.norm(self.house_positions - position, axis=1)
        nearest_house_distance = np.min(distances_to_houses)

        # Calculate score
        score = nearest_house_distance - farthest_store_distance

        return score

    def initialize(self):
        # Randomly initialize warehouse positions within the world size
        positions = np.random.rand(self.num_of_swarm_agents, 2) * self.world_size
        self.swarm_agent_positions = positions
        
        velocities = np.random.rand(self.num_of_swarm_agents, 2) * 2 - 1  # Random velocities in range [-1, 1]
        self.agent_velocities = velocities
        
        self._update_agent_scores()  # Calculate initial scores for the agents


    def get_swarm_agent_positions(self):
        return self.swarm_agent_positions

    def find_best_position(self):
        return self.swarm_agent_positions[np.argmax(self.agent_scores)]

    def _update_warehouse_positions(self, dt:float = 1.0):
        
        # Depend on 3 vectors 
        # Intertia of the agent to keep moving in the same direction
        # Moving towards best position based on score
        # Random exploration to avoid local minima and explore new positions

        velocity_inertia = .7
        velocity_towards_best_position = 3
        velocity_random_exploration = 2
        
        # calculate net velocity for each agent
        for i in range(self.num_of_swarm_agents):
            # Inertia component
            inertia_component = velocity_inertia * self.agent_velocities[i]

            # Best position component
            best_position = self.find_best_position()
            direction_to_best = best_position - self.swarm_agent_positions[i]
            direction_to_best /= (np.linalg.norm(direction_to_best) + 1e-8)  # Normalize
            best_position_component = velocity_towards_best_position * direction_to_best

            # Random exploration component
            random_direction = np.random.rand(2) * 2 - 1  # Random direction in range [-1, 1]
            random_direction /= (np.linalg.norm(random_direction) + 1e-8)  # Normalize
            random_exploration_component = velocity_random_exploration * random_direction

            # Update velocity and position
            self.agent_velocities[i] = inertia_component + best_position_component + random_exploration_component
            new_position = self.swarm_agent_positions[i] + self.agent_velocities[i] * dt

            # Ensure the new position is within the world boundaries
            new_position = np.clip(new_position, 0, self.world_size)

            self.swarm_agent_positions[i] = new_position

