from turtle import position
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



import numpy as np


class Particle:

    def __init__(self, dimension, world_size):

        self.dimension = dimension
        self.world_size = world_size
        
        # Initialize position within environment
        self.position = np.random.uniform(
            0, world_size, size=dimension
        )

        # Initialize velocity
        self.velocity = np.random.uniform(
            -1, 1, size=dimension
        )

        self.best_position = self.position.copy() # type:ignore
        self.best_position_score = float('-inf')


class Agent(Particle):

    def __init__(self, dimension, world_size):
        super().__init__(dimension, world_size)
        self.score = float('-inf')


class AgentGroup:

    def __init__(self, no_of_agents, dimension, world_size):
        self.agents = [
            Agent(dimension, world_size)
            for _ in range(no_of_agents)
        ]
        self.color = np.random.rand(3)
        self.best_position = None
        self.best_position_score = float('-inf')


class House:
    def __init__(self, dimension, world_size):
        self.position = np.random.uniform(
            0, world_size, size=dimension
        )

class Store:
    def __init__(self, dimension, world_size):
        self.position = np.random.uniform(
            0, world_size, size=dimension
        )

