# Multi-Warehouse Swarm Simulation

This module simulates multiple warehouse-location candidates in a shared 2D
world. Each warehouse is represented by a group of particle-swarm agents.
Agents search for positions that score well relative to stores, houses, and
the other warehouse groups, and a live Matplotlib view displays the search.

## Requirements

The simulation uses Python, NumPy, and Matplotlib. From the repository root,
install the Python dependencies if they are not already available:

```bash
python -m pip install numpy matplotlib
```

## Run the visualization

Run the module from the repository root:

```bash
python -m multi_warehouse.main
```

The animation runs continuously until its window is closed. It displays the
stores, houses, swarm agents, each group's best-known position, and a score
heatmap.

## How it works

The environment randomly places stores, houses, and agents in a square world
from `0` to `WORLD_SIZE` on each axis. Agents are divided into warehouse
groups, with each group tracking its own best-scoring position. On each
iteration, an agent's velocity combines:

- **Inertia** — its previous velocity.
- **Cognitive attraction** — movement toward its own best-known position.
- **Social attraction** — movement toward its group's best-known position.

The velocity is capped by `MAX_VELOCITY`, then the agent moves and its
coordinates are clamped to the world bounds. The process uses random
initialization and random coefficients, so results vary between runs.

The optimization score for an agent rewards:

- Greater distance from the nearest house, with an additional penalty for
  entering the configured residential buffer.
- Greater separation from other groups' agents.
- Shorter distance to stores, measured by penalizing the distance to the
  farthest store.
- Staying away from the world boundary, using a fixed boundary penalty.

The heatmap is provided as a visual guide; it is calculated by
`get_score_at_position` and is not identical to the full per-agent score
calculation.

## Configuration

Edit `multi_warehouse/config.py` to change the defaults used by the runnable
simulation:

| Setting | Default | Description |
| --- | ---: | --- |
| `DIMENSION` | `2` | Number of spatial dimensions. |
| `WORLD_SIZE` | `100` | Upper coordinate bound for each dimension. |
| `NUM_STORES` | `10` | Number of randomly placed stores. |
| `NUM_HOUSES` | `50` | Number of randomly placed houses. |
| `NUM_WAREHOUSES` | `4` | Number of agent groups (warehouse candidates). |
| `NUM_SWARM_AGENTS` | `200` | Agents requested by the simulation. |
| `VELOCITY_INERTIA` | `0.5` | Weight for the previous velocity. |
| `VELOCITY_TOWARDS_BEST_POSITION` | `2` | Cognitive weight. |
| `VELOCITY_TOWARDS_GLOBAL_BEST` | `1.5` | Group-best/social weight. |
| `MIN_HOUSE_DISTANCE` | `5.0` | Residential buffer used by the score. |
| `MAX_VELOCITY` | `5.0` | Maximum magnitude of an agent's velocity. |

Agents are allocated evenly among groups by rounding up the requested number
per group. Consequently, the actual population can be slightly larger than
`NUM_SWARM_AGENTS` when it is not divisible by `NUM_WAREHOUSES`.

## Python API

Create and advance an environment without starting the animation:

```python
from multi_warehouse.warehouse import WareHouseEnvironment

env = WareHouseEnvironment(
    dimension=2,
    world_size=100,
    no_of_agents=200,
    no_of_houses=50,
    no_of_stores=10,
    no_of_warehouses=4,
    min_house_distance=5.0,
)

# Advance the swarm by 100 discrete iterations.
positions = env.optimize(iterations=100)

# Inspect each group's best-known position and score.
for group in env.groups:
    print(group.best_position, group.best_position_score)
```

`optimize()` returns the agents' current positions as a NumPy array; it does
not return only the best positions. Useful environment members include:

- `env.step()` — advance by one iteration.
- `env.evaluate()` — score agents and update their personal and group bests.
- `env.get_score(group_index, agent_index)` — score a specific agent.
- `env.get_score_at_position(position)` — calculate the heatmap score at a
  point.
- `env.agent_positions`, `env.store_positions`, and `env.house_positions` —
  position arrays.
- `env.best_group_positions` — best-known position for each group.

To display an existing environment, pass it to
`MultiWarehouseVisualizer` from `multi_warehouse.visual` and call
`animate_environment()`.
