import numpy as np

NEIGHBOUR_MAX_DISTANCE = 300
CLOSE_MAX_DISTANCE = 50
OBSTACLE_LOOKAHEAD = 52  # How far ahead to look for obstacles
OBSTACLE_RADIUS = 10      # Lateral clearance threshold (width of the avoidance corridor)
MAX_SPEED = 211
MAX_FORCE = 10

SEPARATION_WEIGHT = 0.7
ALIGNMENT_WEIGHT = 0.9
COHESION_WEIGHT = 0.9
OBSTACLE_WEIGHT = 5.0     # Strong steering weight for smooth side-stepping

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
    obstacles,
    dt,
    mouse_position,
    mouse_mode,
    bounds,
    targets=None,
):
    N = len(positions)
    new_vel = velocities.copy()

    for i in range(N):
        pos = positions[i]
        vel = velocities[i]
        vel_mag = np.linalg.norm(vel)

        # 1. Flock Separation, Cohesion, and Alignment
        diff = positions - pos
        distance = np.linalg.norm(diff, axis=1)

        neighbours = (distance < NEIGHBOUR_MAX_DISTANCE) & (distance > 0)
        close = (distance < CLOSE_MAX_DISTANCE) & (distance > 0)

        separation = np.zeros(2)
        if np.any(close):
            separation = np.sum(-diff[close] / (distance[close, None] ** 2 + 1), axis=0)
            separation = steer(separation, vel)

        cohesion = np.zeros(2)
        alignment = np.zeros(2)
        if np.any(neighbours):
            center = np.mean(positions[neighbours], axis=0)
            cohesion = steer(center - pos, vel)

            alignment = np.mean(velocities[neighbours], axis=0)
            alignment = steer(alignment, vel)

        # 2. Sideways / Tangential Obstacle Avoidance
        obstacle_force = np.zeros(2)
        if obstacles is not None and len(obstacles) > 0 and vel_mag > 1e-5:
            vel_dir = vel / vel_mag
            obs_diff = obstacles - pos  # Vector from particle to each obstacle
            
            # Project obstacle vector onto velocity vector (distance along our path)
            forward_proj = np.sum(obs_diff * vel_dir, axis=1)
            
            # Calculate lateral vector (perpendicular distance vector from our path to the obstacle)
            lateral_vec = obs_diff - forward_proj[:, None] * vel_dir[None, :]
            lateral_dist = np.linalg.norm(lateral_vec, axis=1)
            
            # Threat condition: Obstacle is ahead (0 < forward_proj < lookahead) 
            # AND close to our trajectory laterally (< radius)
            threats = (forward_proj > 0) & (forward_proj < OBSTACLE_LOOKAHEAD) & (lateral_dist < OBSTACLE_RADIUS)
            
            if np.any(threats):
                # Steer away from the obstacle center *perpendicular* to velocity (-lateral_vec)
                escape_vectors = -lateral_vec[threats]
                
                # Weight closer lateral threats more strongly
                weights = 1.0 / (lateral_dist[threats] + 1e-3)
                sum_escape = np.sum(escape_vectors * weights[:, None], axis=0)
                
                obstacle_force = steer(sum_escape, vel)

        # 3. Mouse Interaction (Attract / Repel)
        mouse_force = np.zeros(2)
        if mouse_mode != 0:
            mouse_diff = mouse_position - pos
            mouse_dist = np.linalg.norm(mouse_diff)
            if mouse_dist > 0:
                desired_mouse = mouse_diff if mouse_mode > 0 else -mouse_diff
                mouse_force = steer(desired_mouse, vel)

        # 4. Combine all forces
        acceleration = (
            cohesion * COHESION_WEIGHT +
            alignment * ALIGNMENT_WEIGHT +
            separation * SEPARATION_WEIGHT +
            obstacle_force * OBSTACLE_WEIGHT
        )
        
        if mouse_mode != 0:
            acceleration += mouse_force * 2.0

        new_vel[i] += acceleration
        new_vel[i] = limit(new_vel[i], MAX_SPEED)

    return new_vel