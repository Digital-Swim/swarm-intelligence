import numpy as np
from  boids.behavior import update_velocities

class Simulation:
    def __init__(self, config):
        self.cfg = config
        self.reset()

    @property
    def count(self):
        return len(self.positions)

    def reset(self):
        n = self.cfg.initial_particles
        self.positions = np.column_stack((
            np.random.uniform(0, self.cfg.width, n),
            np.random.uniform(0, self.cfg.height, n),
        )).astype(np.float32)

        angles = np.random.uniform(0, 2 * np.pi, n)
        speeds = np.random.uniform(
            self.cfg.initial_speed_min,
            self.cfg.initial_speed_max,
            n,
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

    def step(self, dt, mouse_position=(0, 0), mouse_mode=0):
        # The behavior module owns all steering/dynamics.
        self.velocities = update_velocities(
            positions=self.positions,
            velocities=self.velocities,
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
