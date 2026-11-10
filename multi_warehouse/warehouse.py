
import numpy as np

from entities import AgentGroup, House, Store
from multi_warehouse import config


class WareHouseEnvironment:

    def __init__(
        self,
        dimension=2,
        world_size =100 ,
        no_of_agents = 100,
        no_of_houses = 20,
        no_of_stores = 10,
        no_of_warehouses = 3,
        inertia_weight=0.5,
        cognitive_weight=1.5,
        social_weight=1.5,
        min_house_distance=10.0,
        max_velocity=5.0,
    ):


        self.dimension = dimension
        self.world_size = world_size

        self.no_of_agents = no_of_agents
        self.no_of_houses = no_of_houses
        self.no_of_stores = no_of_stores
        self.no_of_warehouses = no_of_warehouses

        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        self.min_house_distance = min_house_distance
        self.max_velocity = max_velocity

        self.no_of_agents_per_warehouse = (
            no_of_agents + no_of_warehouses - 1
        ) // no_of_warehouses

        # Initialize houses
        self.houses = [
            House(dimension, world_size)
            for _ in range(no_of_houses)
        ]

        # Initialize stores
        self.stores = [
            Store(dimension, world_size)
            for _ in range(no_of_stores)
        ]

        # Initialize warehouse groups
        self.groups = [
            AgentGroup(
                self.no_of_agents_per_warehouse,
                dimension,
                world_size
            )
            for _ in range(no_of_warehouses)
        ]

        self.agents = [
            agent
            for group in self.groups
            for agent in group.agents
        ]

        self.iteration = 0

        self.evaluate()

    # --------------------------------------------------
    # POSITION PROPERTIES
    # --------------------------------------------------

    @property
    def agent_positions(self):
        return np.array([
            agent.position for agent in self.agents
        ]).reshape(-1, self.dimension)

    @property
    def agent_colors(self):
        return [
            group.color
            for group in self.groups
            for agent in group.agents
        ]
            
    @property
    def store_positions(self):

        return np.array([
            store.position for store in self.stores
        ]).reshape(-1, self.dimension)


    @property
    def house_positions(self):

        return np.array([
            house.position for house in self.houses
        ]).reshape(-1, self.dimension)


    @property
    def agent_scores(self):
        return np.array([
            agent.score for agent in self.agents
        ])

    @property
    def best_group_positions(self):
        return np.array([
           group.best_position for group in self.groups
        ]).reshape(-1, self.dimension)
   

    def get_swarm_agent_positions(self):
        return self.agent_positions

    # --------------------------------------------------
    # FITNESS FUNCTION
    # --------------------------------------------------

    # Env fitness score
    def get_score(self, group_index, agent_index):

        group = self.groups[group_index]
        agent = group.agents[agent_index]

        position = agent.position

        score = 0.0

        # Objective 1: Minimize distance to stores
        if self.no_of_stores > 0:

            distances = np.linalg.norm(
                self.store_positions - position,
                axis=1
            )

            score -= np.max(distances)

        # Objective 2: Maximize distance between warehouses
        for index, other_group in enumerate(self.groups):

            if index == group_index:
                continue

            if not other_group.agents:
                continue

            distances = np.linalg.norm(
                np.array([
                    a.position
                    for a in other_group.agents
                ]) - position,
                axis=1
            )

            min_distance = np.min(distances)
            max_distance = np.max(distances)

            score += min_distance

            if max_distance > config.MAX_WAREHOUSE_DISTANCE:
                score -= config.WAREHOUSE_DISTANCE_PENALTY * (
                    max_distance - config.MAX_WAREHOUSE_DISTANCE
                ) ** 2

        # Objective 3: Maximize distance from houses
        if self.no_of_houses > 0:

            distances = np.linalg.norm(
                self.house_positions - position,
                axis=1
            )

            nearest_house = np.min(distances)

            score += nearest_house

            # Additional penalty for entering residential buffer
            penalty = np.sum(
                np.maximum(
                    0,
                    self.min_house_distance - distances
                ) ** 2
            )

            score -= penalty

        # Objective to maintin distance between the boundary
        score = score - self.calculate_boundary_penalty(position)
        
        return float(score)

    def calculate_boundary_penalty(self, position):
        """
        Penalize positions that are too close to the environment boundaries.
        """
        margin = 10
        world_size = self.world_size
        penalty_weight = 0.8

        penalty = 0.0

        for coordinate in position:
            if coordinate < margin:
                penalty += (margin - coordinate) ** 2

            elif coordinate > world_size - margin:
                penalty += (coordinate - (world_size - margin)) ** 2

        return penalty * penalty_weight

    # Score for heatmap, 
    def get_score_at_position(self, position):

        position = np.asarray(position, dtype=float)

        score = 0.0

        # Store objective
        if self.no_of_stores > 0:

            distances = np.linalg.norm(
                self.store_positions - position,
                axis=1
            )

            score -= np.max(distances)

        # Distance from other warehouse groups
        for group in self.groups:

            if not group.agents:
                continue

            distances = np.linalg.norm(
                np.array([
                    agent.position
                    for agent in group.agents
                ]) - position,
                axis=1
            )

            score += np.min(distances)

        # House objective
        if self.no_of_houses > 0:

            distances = np.linalg.norm(
                self.house_positions - position,
                axis=1
            )

            score += np.min(distances)

            score -= np.sum(
                np.maximum(
                    0,
                    self.min_house_distance - distances
                ) ** 2
            )

        return float(score)

    # --------------------------------------------------
    # INITIAL EVALUATION
    # --------------------------------------------------

    def evaluate(self):

        for group_index, group in enumerate(self.groups):

            for agent_index, agent in enumerate(group.agents):

                agent.score = self.get_score(
                    group_index,
                    agent_index
                )

                # Update personal best
                if agent.score > agent.best_position_score:

                    agent.best_position = agent.position.copy() # type:ignore

                    agent.best_position_score = agent.score

                # Update group best
                if agent.score > group.best_position_score:

                    group.best_position = agent.position.copy() # type:ignore

                    group.best_position_score = agent.score

    # --------------------------------------------------
    # PSO UPDATE
    # --------------------------------------------------

    def step(self, dt=1.0):

        """
        Execute ONE PSO iteration.

        dt is accepted for compatibility with the visualizer.
        PSO iterations are discrete, so dt does not directly
        scale the velocity update.
        """

        self.evaluate()

        for group in self.groups:

            for agent in group.agents:

                r1, r2 = np.random.rand(2)

                cognitive_component = (
                    self.cognitive_weight
                    * r1
                    * (agent.best_position - agent.position)
                )

                social_component = (
                    self.social_weight
                    * r2
                    * (group.best_position - agent.position) # type:ignore
                )

                agent.velocity = (
                    self.inertia_weight * agent.velocity
                    + cognitive_component
                    + social_component
                )

                # Prevent excessively large movements
                speed = np.linalg.norm(agent.velocity)

                if speed > self.max_velocity:

                    agent.velocity = (
                        agent.velocity / speed
                    ) * self.max_velocity

                # Update position
                agent.position += agent.velocity

                # Enforce world boundaries
                agent.position = np.clip(
                    agent.position,
                    0,
                    self.world_size
                )

        self.iteration += 1

        # Evaluate new positions
        self.evaluate()

    # --------------------------------------------------
    # RUN COMPLETE OPTIMIZATION
    # --------------------------------------------------

    def optimize(self, iterations=100):

        for _ in range(iterations):
            self.step()

        return self.agent_positions
