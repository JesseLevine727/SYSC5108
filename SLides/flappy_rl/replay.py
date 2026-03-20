from __future__ import annotations

from collections import deque
import random

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.random = random.Random(seed)
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = self.random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

