from dataclasses import dataclass

@dataclass
class Configuration:
    width: int = 1100
    height: int = 750
    initial_particles: int = 50
    initial_obstacles: int = 21
    initial_targets: int = 0

    # Bounds are ((x_min, x_max), (y_min, y_max)); None uses the full canvas.
    particle_spawn_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    obstacle_spawn_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    target_spawn_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None

    # Fixed locations are followed by random placements up to the configured count.
    initial_particle_positions: tuple[tuple[float, float], ...] | None = None
    initial_obstacle_positions: tuple[tuple[float, float], ...] | None = None
    initial_target_positions: tuple[tuple[float, float], ...] | None = None

    fps_limit: int = 120

    # Visual settings
    background: tuple = (5, 8, 16)
    trail_fade_alpha: int = 22  # lower = longer trails
    particle_radius: int = 2
    target_radius: int = 5
    min_particle_brightness: int = 100
    max_particle_brightness: int = 245

    # Initial state only; implement your own dynamics in behavior.py
    initial_speed_min: float = 35.0
    initial_speed_max: float = 85.0
