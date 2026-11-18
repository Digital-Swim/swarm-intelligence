from typing import Callable
import numpy as np
from visual.default import Configuration

class Simulation:
    def __init__(self, config : Configuration):
        self.cfg = config
        self.reset()

    @property
    def count(self):
        return len(self.positions)

    def _create_initial_positions(self, count, spawn_bounds, fixed_positions):
        if spawn_bounds is None:
            x_bounds = (0, self.cfg.width)
            y_bounds = (0, self.cfg.height)
        else:
            x_bounds, y_bounds = spawn_bounds

        if x_bounds[0] > x_bounds[1] or y_bounds[0] > y_bounds[1]:
            raise ValueError("Spawn bounds must be ordered from minimum to maximum.")

        if fixed_positions is None:
            fixed = np.empty((0, 2), dtype=np.float32)
        else:
            fixed = np.asarray(fixed_positions, dtype=np.float32)
            if fixed.size == 0:
                fixed = np.empty((0, 2), dtype=np.float32)
            elif fixed.ndim != 2 or fixed.shape[1] != 2:
                raise ValueError("Initial positions must contain (x, y) pairs.")

        if len(fixed) > count:
            raise ValueError("Initial positions cannot exceed the configured count.")

        random_count = count - len(fixed)
        random = np.column_stack((
            np.random.uniform(x_bounds[0], x_bounds[1], random_count),
            np.random.uniform(y_bounds[0], y_bounds[1], random_count),
        )).astype(np.float32)
        return np.concatenate((fixed, random))

    def reset(self):
        self.positions = self._create_initial_positions(
            self.cfg.initial_particles,
            self.cfg.particle_spawn_bounds,
            self.cfg.initial_particle_positions,
        )
        self.obstacles = self._create_initial_positions(
            self.cfg.initial_obstacles,
            self.cfg.obstacle_spawn_bounds,
            self.cfg.initial_obstacle_positions,
        )
        self.targets = self._create_initial_positions(
            self.cfg.initial_targets,
            self.cfg.target_spawn_bounds,
            self.cfg.initial_target_positions,
        )
        angles = np.random.uniform(0, 2 * np.pi, self.cfg.initial_particles)
        speeds = np.random.uniform(
            self.cfg.initial_speed_min,
            self.cfg.initial_speed_max,
            self.cfg.initial_particles,
        )
        self.velocities = np.column_stack((
            np.cos(angles) * speeds,
            np.sin(angles) * speeds,
        )).astype(np.float32)

    def add_particles(self, amount):
        amount = max(0, int(amount))
        if amount == 0:
            return
        p = np.column_stack((
            np.random.uniform(0, self.cfg.width, amount),
            np.random.uniform(0, self.cfg.height, amount),
        )).astype(np.float32)
        angles = np.random.uniform(0, 2 * np.pi, amount)
        speeds = np.random.uniform(
            self.cfg.initial_speed_min,
            self.cfg.initial_speed_max,
            amount,
        )
        v = np.column_stack((np.cos(angles) * speeds,
                             np.sin(angles) * speeds)).astype(np.float32)
        self.positions = np.concatenate((self.positions, p))
        self.velocities = np.concatenate((self.velocities, v))

    def remove_particles(self, amount):
        keep = max(0, self.count - int(amount))
        self.positions = self.positions[:keep]
        self.velocities = self.velocities[:keep]

    def step(self, dt, update_velocities:Callable, mouse_position=(0, 0), mouse_mode=0):
        # The behavior module owns all steering/dynamics.
        self.velocities = update_velocities(
            positions=self.positions,
            velocities=self.velocities,
            obstacles=self.obstacles,
            targets=self.targets,
            dt=dt,
            mouse_position=np.asarray(mouse_position, dtype=np.float32),
            mouse_mode=mouse_mode,  # 0 none, +1 attract, -1 repel
            bounds=(self.cfg.width, self.cfg.height),
        )

        # Integrate the velocities returned by your behavior module.
        self.positions += self.velocities * dt

        # Wrap-around boundary condition; replace if desired.
        self.positions[:, 0] %= self.cfg.width
        self.positions[:, 1] %= self.cfg.height
