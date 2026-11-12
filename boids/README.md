# Swarm Playground

A minimal Pygame + NumPy visual testbed. It contains **no Boids or swarm
steering implementation**. The algorithm hook is `behavior.py`.

## Install and run

```bash
python -m pip install -r requirements.txt
python main.py
```

## Controls

- `Space`: pause / resume
- `R`: reset particles to random positions and headings
- `C`: clear accumulated trails
- `+` / `-`: add or remove 100 particles
- Mouse left button: passed to your algorithm as `mouse_mode=+1` (attract)
- Mouse right button: passed as `mouse_mode=-1` (repel)
- `Esc`: quit

Mouse attraction/repulsion is **not implemented** in the starter. The mouse
state is simply made available to your algorithm.

## Implement your algorithm

Edit `behavior.py`, specifically `update_velocities(...)`.

It receives:
- `positions`: `(N, 2)` positions in pixels
- `velocities`: `(N, 2)` velocities in pixels/second
- `dt`: frame delta in seconds
- `mouse_position`: `(x, y)`
- `mouse_mode`: `0`, `+1`, or `-1`
- `bounds`: `(width, height)`

Return an `(N, 2)` velocity array. `simulation.py` then integrates positions
using `position += velocity * dt`.

Keep rendering and simulation infrastructure separate from your behavior
implementation. You can replace the starter dynamics without changing
`renderer.py` or `main.py`.

## Visual tuning

Change `config.py`:
- `initial_particles`
- `trail_fade_alpha` (smaller values produce longer trails)
- `particle_radius`
- `initial_speed_min` / `initial_speed_max`

The starter uses wrap-around screen boundaries. Change that in
`simulation.py` if your experiment needs walls, bounce, or another boundary
condition.
