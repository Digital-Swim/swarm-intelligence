from dataclasses import dataclass
from visual.default import Configuration

@dataclass
class Config(Configuration):
    initial_particles: int = 50
    # Initial state only; implement your own dynamics in behavior.py
    initial_speed_min: float = 35.0
    initial_speed_max: float = 85.0
