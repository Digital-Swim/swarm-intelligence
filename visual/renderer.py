import numpy as np
import pygame
from visual.default import Configuration

class Renderer:
    def __init__(self, config: Configuration):
        self.cfg: Configuration = config 
        self.trails = pygame.Surface(
            (config.width, config.height),
            flags=pygame.SRCALPHA,
        )
        self.grid_surface = pygame.Surface(
            (config.width, config.height),
            flags=pygame.SRCALPHA,
        )

    def clear(self):
        self.trails.fill((0, 0, 0, 0))

    def draw(self, screen, positions, velocities, obstacles, targets=None, grid=None):
        screen.fill(self.cfg.background)

        if grid is not None:
            grid_width = min(grid.shape[0], self.cfg.width)
            grid_height = min(grid.shape[1], self.cfg.height)
            visible_grid = np.maximum(grid[:grid_width, :grid_height], 0)
            
            self.grid_surface.fill((255, 255, 255, 0))
            alpha = pygame.surfarray.pixels_alpha(self.grid_surface)
            alpha[:grid_width, :grid_height] = np.clip(
                visible_grid * 255 / 50,
                0,
                255,
            ).astype(np.uint8)
            del alpha

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
        
        if grid is not None:
            screen.blit(self.grid_surface, (0, 0))
        
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

        if targets is not None:
            for target in targets:
                pygame.draw.circle(
                    screen,
                    (0, 255, 0),
                    (int(target[0]), int(target[1])),
                    self.cfg.target_radius,
                )

