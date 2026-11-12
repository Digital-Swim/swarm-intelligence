"""
YOUR ALGORITHM LIVES HERE.

Contract:
    update_velocities(...) -> ndarray shaped (N, 2), dtype float32 preferred.

Inputs:
    positions:      (N, 2) particle positions in pixels
    velocities:     (N, 2) current velocities in pixels/second
    dt:             elapsed frame time in seconds
    mouse_position: (2,) mouse coordinates
    mouse_mode:     0 = none, +1 = attract, -1 = repel
    bounds:         (width, height)

Return a NEW velocity array (or a modified copy). Do not update positions here;
simulation.py performs position integration after this function returns.

The starter implementation deliberately applies no steering. Particles simply
continue at their initial velocities, giving you a baseline to replace.
"""
import numpy as np


NEIGHBOUR_MAX_DISTANCE = 300
CLOSE_MAX_DISTANCE = 50
MAX_SPEED = 100
MAX_FORCE = 10

SEPARATION_WEIGHT = 1.8
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 0.8

def limit(v, maximum):
    mag = np.linalg.norm(v)
    
    if mag > maximum:
        return (v / mag) * maximum

    return v

def steer(desired, velocity):
    
    if np.linalg.norm(desired) < 1e-6:
        return np.zeros(2)
    
    desired = desired / np.linalg.norm(desired) * MAX_SPEED
    return limit(desired - velocity, MAX_FORCE)
    
def update_velocities(
    positions,
    velocities,
    dt,
    mouse_position,
    mouse_mode,
    bounds,
):
    # TODO: implement your swarm / Boids algorithm here.
    
    N = len(positions)
    new_vel = velocities.copy()

    for i in range(N):
        diff = positions - positions[i]   # position difference from each point to the ith point 
        distance = np.linalg.norm(diff, axis=1) # lenght magniture row by row 
        
        neighbours = (distance < NEIGHBOUR_MAX_DISTANCE) & (distance > 0)
        close = (distance < CLOSE_MAX_DISTANCE) & (distance > 0)
        
        
        separation = np.zeros(2)
        if np.any(close):
            separation = np.sum(-diff[close]/(distance[close, None] ** 2 + 1), axis = 0)
            separation = steer(separation, MAX_SPEED)
        
        # Objective - Cohesion and Alignment 
        # Moving towards the center of neighbours and towards the avg veloclity of the group 
        
        cohesion = np.zeros(2)
        alignment = np.zeros(2)
        if np.any(neighbours):
            # Finding center point of neighbours 
            center = np.mean(positions[neighbours], axis=0)
            cohesion = steer(center - positions[i], velocities[i])

            alignment = np.mean(velocities[neighbours], axis=0)
            alignment = steer(alignment, velocities[i])
        
        accleration = cohesion * COHESION_WEIGHT + alignment * ALIGNMENT_WEIGHT + separation * SEPARATION_WEIGHT
        
        new_vel[i] += accleration
        new_vel[i] = limit(new_vel[i], MAX_SPEED)

    return new_vel
