from typing import Protocol
import numpy as np
from numpy.typing import NDArray

class SwarmBehavior(Protocol):
    def update_velocities(
        self,
        positions: NDArray[np.float32],
        velocities: NDArray[np.float32],
        obstacles: NDArray[np.float32],
        dt: float,
        mouse_position: NDArray[np.float32],
        mouse_mode: int,
        bounds: tuple[int, int],
        targets: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        ...