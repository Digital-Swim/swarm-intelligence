# Warehouse Swarm Optimization

This module models a warehouse placement problem as a continuous swarm optimization task. A population of agents moves through a 2D world and searches for locations that maximize a custom score based on the relative positions of stores and houses.

## Overview

The environment contains:
- a set of store locations
- a set of house locations
- a rectangular world boundary
- a swarm of agents that explore the world and update their positions over time

Each agent evaluates its current position using a score function. The swarm gradually converges toward promising positions by combining:
- inertia from the previous velocity
- attraction toward the current best-known position
- random exploration to avoid settling too early

## Core idea

The system treats each warehouse candidate as a particle in a swarm. The score of a candidate position is computed as:

`score = nearest_house_distance - farthest_store_distance`

where:
- `nearest_house_distance` is the distance from the candidate to the closest house
- `farthest_store_distance` is the distance from the candidate to the farthest store

This reward function is designed to favor positions that are relatively far from houses while remaining reasonably close to the store network.

## Environment model

The environment is defined in `warehouse/env.py`.

Simulation defaults are centralized in `warehouse/config.py`. Change `NUM_STORES`,
`NUM_HOUSES`, and `NUM_SWARM_AGENTS` to adjust the scenario size. The same file
contains `WORLD_SIZE`, `POSITION_RANGE` for generated store and house coordinates,
and the three velocity coefficients used by the swarm.

Key components:
- `WarehouseEnv.__init__`: initializes the swarm, velocities, and score array
- `initialize()`: places agents randomly inside the world and computes initial scores
- `get_score_at_position()`: evaluates a single point in the environment
- `find_best_position()`: selects the highest-scoring agent position
- `_update_warehouse_positions()`: updates each agent's velocity and position
- `step()`: advances the simulation by one time step

## Agent update rule

Each agent updates its velocity using a weighted combination of three terms:

1. Inertia
   - keeps the agent moving in its previous direction
   - `VELOCITY_INERTIA * previous_velocity` (configured value: `1`)

2. Attraction to global best
   - moves toward the best-scoring position found so far
   - normalized direction to the best agent
   - `VELOCITY_TOWARDS_BEST_POSITION * direction_to_best` (configured value: `3`)

3. Random exploration
   - injects a new random direction to escape local minima
   - `VELOCITY_RANDOM_EXPLORATION * random_direction` (configured value: `5`)

The new position is then computed as:

`new_position = current_position + velocity * dt`

and clamped to the world bounds.

## Simulation flow

1. Initialize random agent positions and random velocities.
2. Compute each agent's score from the current position.
3. Select the best-scoring agent as the global target.
4. Update each agent's velocity with inertia + attraction + exploration.
5. Move the agents forward in time.
6. Repeat until the visualization loop ends.

## Visualization

The `warehouse/visual.py` module renders the simulation as a live animated environment.

It shows:
- swarm agents as blue points
- stores as green points
- houses as red points
- the best current location as a gold star
- a continuous score heatmap background

The animation runs in `main.py` and can be started by executing:

```bash
python warehouse/main.py
```

## Example usage

```python
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
visualizer.animate_environment()
```

## Interpretation

This approach is a lightweight swarm-intelligence optimization method. It does not require a strict mathematical solver; instead, it uses repeated local updates and random exploration to gradually discover high-scoring warehouse locations. The result is a dynamic search process that visually reveals where the swarm prefers to situate warehouse candidates.

## Notes

- The algorithm is fully continuous in 2D space.
- The world size and number of agents are configurable.
- The simulation is designed to be easy to inspect visually and easy to extend for other optimization scenarios.
