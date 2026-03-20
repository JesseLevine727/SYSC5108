from __future__ import annotations

import numpy as np


def expert_action(observation: np.ndarray) -> int:
    """A simple controller that passes pipes reliably enough to bootstrap a policy."""
    gap_offset = float(observation[3])
    velocity = float(observation[1])
    if gap_offset < -0.01 and velocity > -0.30:
        return 1
    return 0

