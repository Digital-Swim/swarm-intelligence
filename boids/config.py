from dataclasses import dataclass

@dataclass
class Config:
    width: int = 1100
    height: int = 750
    initial_particles: int = 200
    fps_limit: int = 120

    # Visual settings
    background: tuple = (5, 8, 16)
    trail_fade_alpha: int = 22  # lower = longer trails
    particle_radius: int = 2
    min_particle_brightness: int = 100
    max_particle_brightness: int = 245

    # Initial state only; implement your own dynamics in behavior.py
    initial_speed_min: float = 35.0
    initial_speed_max: float = 85.0
