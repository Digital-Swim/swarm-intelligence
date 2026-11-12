
import pygame
import numpy as np

pygame.init()

W, H = 1100, 750
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Boids - Swarm Simulation")
clock = pygame.time.Clock()

N = 350

# Simulation parameters
MAX_SPEED = 3.2
MAX_FORCE = 0.055
PERCEPTION = 65

SEPARATION_WEIGHT = 1.8
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 0.8

# Initial particle positions and velocities
pos = np.random.rand(N, 2) * [W, H]
vel = np.random.randn(N, 2)
vel = vel / np.linalg.norm(vel, axis=1)[:, None] * MAX_SPEED

trail = pygame.Surface((W, H), pygame.SRCALPHA)
running = True


def limit(v, maximum):
    mag = np.linalg.norm(v)
    if mag > maximum:
        return v / mag * maximum
    return v


def steer(desired, velocity):
    if np.linalg.norm(desired) < 1e-6:
        return np.zeros(2)

    desired = desired / np.linalg.norm(desired) * MAX_SPEED
    return limit(desired - velocity, MAX_FORCE)


while running:
    dt = clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fade previous particle trails
    trail.fill((0, 0, 0, 24))
    screen.fill((5, 7, 15))

    new_vel = vel.copy()

    for i in range(N):
        diff = pos - pos[i]
        dist = np.linalg.norm(diff, axis=1)

        neighbors = (dist < PERCEPTION) & (dist > 0)
        close = (dist < 24) & (dist > 0)

        separation = np.zeros(2)
        alignment = np.zeros(2)
        cohesion = np.zeros(2)

        if np.any(close):
            separation = np.sum(
                -diff[close] / (dist[close, None] ** 2 + 1),
                axis=0
            )
            separation = steer(separation, vel[i])

        if np.any(neighbors):
            alignment = np.mean(vel[neighbors], axis=0)
            alignment = steer(alignment, vel[i])

            center = np.mean(pos[neighbors], axis=0)
            cohesion = steer(center - pos[i], vel[i])

        acceleration = (
            separation * SEPARATION_WEIGHT
            + alignment * ALIGNMENT_WEIGHT
            + cohesion * COHESION_WEIGHT
        )

        new_vel[i] += acceleration
        new_vel[i] = limit(new_vel[i], MAX_SPEED)

    vel = new_vel
    pos += vel

    # Wrap screen boundaries
    pos[:, 0] %= W
    pos[:, 1] %= H

    # Render boids as directional triangles
    for p, v in zip(pos, vel):
        angle = np.arctan2(v[1], v[0])
        direction = np.array([np.cos(angle), np.sin(angle)])
        side = np.array([-direction[1], direction[0]])

        points = [
            p + direction * 7,
            p - direction * 5 + side * 3,
            p - direction * 5 - side * 3
        ]

        pygame.draw.polygon(
            trail,
            (80, 190, 255, 180),
            points
        )

    screen.blit(trail, (0, 0))
    pygame.display.flip()

pygame.quit()
