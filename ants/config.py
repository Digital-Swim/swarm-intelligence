from dataclasses import dataclass
from visual.default import Configuration

@dataclass
class Config(Configuration):

    width: int = 1100
    height: int = 750

    initial_particles: int = 31
    initial_obstacles: int = 0
    initial_targets: int = 1

    # Initial state only; implement your own dynamics in behavior.py
    initial_speed_min: float = 35.0
    initial_speed_max: float = 85.0

    particle_spawn_bounds: tuple[tuple[float, float], tuple[float, float]] | None = ((400, 600), (250, 350))
    target_spawn_bounds: tuple[tuple[float, float], tuple[float, float]] | None = ((400, 600), (250, 350))
    
    # Initial state only; implement your own dynamics in behavior.py
    initial_speed_min: float = 20.0
    initial_speed_max: float = 30.0
    
    min_particle_brightness: int = 255
    max_particle_brightness: int = 255
