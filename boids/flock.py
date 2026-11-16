"""Boids flocking with per-bird speed variation and organic randomness.

Drop-in replacement: update_velocities(...) keeps the original signature and
still returns the new (N, 2) velocity array. Positions are integrated by the
caller. `bounds` is assumed to be (width, height) with the origin at (0, 0);
pass None to disable wall steering (e.g. if your world wraps around).
"""
import numpy as np

# --- Perception ---------------------------------------------------------
NEIGHBOUR_MAX_DISTANCE = 300
CLOSE_MAX_DISTANCE = 50
OBSTACLE_LOOKAHEAD = 52
OBSTACLE_RADIUS = 21
MOUSE_RADIUS = 350
BOUNDARY_MARGIN = 120

# --- Speeds (px/s) and accelerations (px/s^2) ----------------------------
MAX_SPEED = 211          # hard ceiling for any bird
MIN_SPEED = 60           # birds can't hover
CRUISE_SPEED = 140       # mean preferred speed across the flock
CRUISE_SPREAD = 28       # per-bird std-dev of preferred speed
MAX_ACCEL = 600          # old MAX_FORCE=10 per frame at ~60 fps
STEER_CAP = 2.0          # cap on the summed (unitless) steering demand

# --- Rule weights (each rule returns a magnitude in [0, 1]) --------------
SEPARATION_WEIGHT = 1.8
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 0.8
OBSTACLE_WEIGHT = 4.0
BOUNDARY_WEIGHT = 3.5
MOUSE_WEIGHT = 2.0
WANDER_WEIGHT = 0.5
SPEED_WEIGHT = 0.6

# --- Randomness / speed variation ---------------------------------------
WANDER_RETURN = 2.0      # pull of the wander angle back toward straight ahead
WANDER_SIGMA = 2.0       # random-walk strength of the wander angle (rad/sqrt(s))
SPEED_VARIATION = 0.35   # +/- fraction of cruise speed that oscillates over time
TURN_SLOWDOWN = 0.25     # fraction of speed shed in the sharpest turns


def clip_norm(v, maximum):
    """Row-wise: scale vectors so their length is at most `maximum`."""
    mag = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.minimum(1.0, maximum / np.maximum(mag, 1e-12))
    return v * scale


class Flock:
    """Holds the per-bird traits and random state that make birds individual."""

    def __init__(self, n, seed=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.cruise = np.clip(r.normal(CRUISE_SPEED, CRUISE_SPREAD, n),
                              MIN_SPEED + 20, MAX_SPEED * 0.8)
        self.max_speed = np.minimum(self.cruise * 1.4, MAX_SPEED)
        self.phase = r.uniform(0, 2 * np.pi, n)          # speed-oscillation phase
        self.omega = r.uniform(0.5, 1.5, n)              # speed-oscillation rate (rad/s)
        self.wander = r.normal(0, 0.5, n)                # current wander angle
        self.side = r.choice([-1.0, 1.0], n)             # preferred dodge side
        ang = r.uniform(0, 2 * np.pi, n)
        self.fallback_dir = np.column_stack((np.cos(ang), np.sin(ang)))
        self.t = 0.0

    def step(self, positions, velocities, obstacles, dt, mouse_position,
             mouse_mode, bounds):
        rng, N = self.rng, self.n
        self.t += dt
        pos = np.asarray(positions, float)
        vel = np.asarray(velocities, float)

        speed = np.linalg.norm(vel, axis=1)
        heading = np.where((speed > 1e-6)[:, None],
                           vel / np.maximum(speed, 1e-9)[:, None],
                           self.fallback_dir)
        perp = np.column_stack((-heading[:, 1], heading[:, 0]))

        # ---- 1. Flocking (all pairs at once) ---------------------------
        # tiny jitter so exactly coincident birds still push apart
        pj = pos + rng.normal(0, 1e-3, pos.shape)
        diff = pj[None, :, :] - pj[:, None, :]           # diff[i, j] = p_j - p_i
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)                   # exclude self by index
        neigh = dist < NEIGHBOUR_MAX_DISTANCE
        close = dist < CLOSE_MAX_DISTANCE

        # Separation: stronger the closer the neighbour (magnitude is kept)
        w = np.where(close, 1.0 - dist / CLOSE_MAX_DISTANCE, 0.0)
        unit = diff / np.maximum(dist, 1e-9)[..., None]
        sep = clip_norm(-(unit * w[..., None]).sum(axis=1), 1.0)

        count = neigh.sum(axis=1)
        has = (count > 0)[:, None]
        cnt = np.maximum(count, 1)[:, None]
        nf = neigh.astype(float)

        center = (nf @ pos) / cnt
        coh = clip_norm((center - pos) / (NEIGHBOUR_MAX_DISTANCE / 3), 1.0) * has

        # Alignment matches neighbours' velocity *including speed*
        avg_vel = (nf @ vel) / cnt
        ali = clip_norm((avg_vel - vel) / (MAX_SPEED * 0.25), 1.0) * has

        # ---- 2. Wander: smooth random steering (Ornstein-Uhlenbeck) -----
        self.wander += (-WANDER_RETURN * self.wander * dt
                        + WANDER_SIGMA * np.sqrt(dt) * rng.standard_normal(N))
        self.wander = np.clip(self.wander, -1.2, 1.2)
        wander = perp * np.sin(self.wander)[:, None]

        # ---- 3. Obstacle avoidance --------------------------------------
        obs = np.zeros((N, 2))
        if obstacles is not None and len(obstacles) > 0:
            o = np.asarray(obstacles, float)
            od = o[None, :, :] - pos[:, None, :]                 # (N, M, 2)
            fwd = np.einsum('nmk,nk->nm', od, heading)
            lat = od - fwd[..., None] * heading[:, None, :]
            lat_d = np.linalg.norm(lat, axis=2)

            threat = (fwd > 0) & (fwd < OBSTACLE_LOOKAHEAD) & (lat_d < OBSTACLE_RADIUS)
            away = -lat / np.maximum(lat_d, 1e-9)[..., None]
            # head-on: lateral offset ~ 0 gives no direction, so pick a side
            dodge = (self.side[:, None] * perp)[:, None, :]
            away = np.where((lat_d < 1e-3)[..., None], dodge, away)
            s = np.where(threat,
                         (1 - fwd / OBSTACLE_LOOKAHEAD) * (0.5 + 0.5 * (1 - lat_d / OBSTACLE_RADIUS)),
                         0.0)
            obs = (away * s[..., None]).sum(axis=1)

            # already touching / alongside: plain radial push
            d = np.linalg.norm(od, axis=2)
            near = (d < OBSTACLE_RADIUS) & ~threat
            radial = -od / np.maximum(d, 1e-9)[..., None]
            obs += (radial * np.where(near, 1 - d / OBSTACLE_RADIUS, 0.0)[..., None]).sum(axis=1)
            obs = clip_norm(obs, 1.0)

        # ---- 4. Walls ----------------------------------------------------
        bnd = np.zeros((N, 2))
        if bounds is not None:
            W, H = bounds[0], bounds[1]
            m = BOUNDARY_MARGIN
            bnd[:, 0] = (np.clip((m - pos[:, 0]) / m, 0, 1)
                         - np.clip((pos[:, 0] - (W - m)) / m, 0, 1))
            bnd[:, 1] = (np.clip((m - pos[:, 1]) / m, 0, 1)
                         - np.clip((pos[:, 1] - (H - m)) / m, 0, 1))

        # ---- 5. Mouse (range-limited, falls off with distance) ----------
        mouse = np.zeros((N, 2))
        flee = np.zeros(N, bool)
        if mouse_mode != 0 and mouse_position is not None:
            md = np.asarray(mouse_position, float) - pos
            d = np.linalg.norm(md, axis=1)
            inr = (d > 1e-6) & (d < MOUSE_RADIUS)
            f = (1 - d / MOUSE_RADIUS) * np.sign(mouse_mode)
            mouse = np.where(inr[:, None], md / np.maximum(d, 1e-9)[:, None] * f[:, None], 0.0)
            flee = inr & (mouse_mode < 0)

        # ---- 6. Combine steering ----------------------------------------
        steer = (coh * COHESION_WEIGHT + ali * ALIGNMENT_WEIGHT
                 + sep * SEPARATION_WEIGHT + wander * WANDER_WEIGHT
                 + bnd * BOUNDARY_WEIGHT + mouse * MOUSE_WEIGHT)
        steer = steer + obs * OBSTACLE_WEIGHT
        turn = np.clip(np.abs(np.sum(steer * perp, axis=1)) / 2.0, 0, 1)
        steer = clip_norm(steer, STEER_CAP)

        # ---- 7. Individual speed: cruise + slow oscillation + gusts ------
        gust = 1 + 0.08 * np.sin(0.3 * self.t)           # flock-wide breathing
        target = self.cruise * gust * (1 + SPEED_VARIATION
                                       * np.sin(self.phase + self.omega * self.t))
        target *= 1 - TURN_SLOWDOWN * turn               # slow down in sharp turns
        target = np.where(flee, self.max_speed, target)  # burst away from the mouse
        speed_acc = np.clip((target - speed) / (MAX_SPEED * 0.25), -1, 1)

        accel = steer * MAX_ACCEL + heading * (speed_acc * SPEED_WEIGHT * MAX_ACCEL)[:, None]

        # ---- 8. Integrate (dt-scaled) and clamp speed --------------------
        new_vel = vel + accel * dt
        s = np.linalg.norm(new_vel, axis=1)
        clamped = np.clip(s, MIN_SPEED, self.max_speed)
        new_vel = np.where((s < 1e-6)[:, None],
                           self.fallback_dir * MIN_SPEED,
                           new_vel * (clamped / np.maximum(s, 1e-9))[:, None])
        return new_vel


_state = None


def update_velocities(positions, velocities, obstacles, dt,
                      mouse_position, mouse_mode, bounds):
    """Drop-in wrapper with the original signature (keeps a module-level Flock)."""
    global _state
    n = len(positions)
    if _state is None or _state.n != n:
        _state = Flock(n)
    return _state.step(positions, velocities, obstacles, dt,
                       mouse_position, mouse_mode, bounds)