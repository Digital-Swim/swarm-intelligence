import numpy as np
import pygame

class Renderer:
    def __init__(self, config):
        self.cfg = config
        self.trails = pygame.Surface(
            (config.width, config.height),
            flags=pygame.SRCALPHA,
        )

    def clear(self):
        self.trails.fill((0, 0, 0, 0))

    def draw(self, screen, positions, velocities, obstacles):
        screen.fill(self.cfg.background)

        # Fade the previous frame instead of clearing it completely.
        fade = pygame.Surface(
            (self.cfg.width, self.cfg.height),
            flags=pygame.SRCALPHA,
        )
        fade.fill((0, 0, 0, self.cfg.trail_fade_alpha))
        self.trails.blit(fade, (0, 0))

        if len(positions):
            speeds = np.linalg.norm(velocities, axis=1)
            denom = max(float(speeds.max()), 1.0)
            intensity = np.clip(speeds / denom, 0.0, 1.0)

            for (x, y), t in zip(positions, intensity):
                b = int(
                    self.cfg.min_particle_brightness
                    + t * (self.cfg.max_particle_brightness
                           - self.cfg.min_particle_brightness)
                )
                color = (int(b * 0.55), int(b * 0.82), b, 220)
                pygame.draw.circle(
                    self.trails,
                    color,
                    (int(x), int(y)),
                    self.cfg.particle_radius,
                )

        screen.blit(self.trails, (0, 0))
        
        if len(obstacles):
            color = (255, 255, 255)
            for o in obstacles:
                # Supports obstacles as (x, y) or (x, y, radius)
                ox, oy = o[0], o[1]
                #radius = o[2] if len(o) > 2 else getattr(self.cfg, 'obstacle_radius', self.cfg.particle_radius * 2)
                
                pygame.draw.circle(
                    screen,
                    color,
                    (int(ox), int(oy)),
                    int(5),
                )

