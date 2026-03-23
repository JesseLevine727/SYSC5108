from __future__ import annotations

from collections import deque

import numpy as np

from flappy_rl.environment import StepResult


class FrameStackWrapper:
    def __init__(self, env, num_frames: int) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        self.env = env
        self.num_frames = num_frames
        self.base_observation_size = env.observation_size
        self._frames: deque[np.ndarray] = deque(maxlen=num_frames)

    @property
    def observation_size(self) -> int:
        return self.base_observation_size * self.num_frames

    @property
    def action_size(self) -> int:
        return self.env.action_size

    def reset(self, seed: int | None = None):
        observation, info = self.env.reset(seed=seed)
        self._frames.clear()
        for _ in range(self.num_frames):
            self._frames.append(observation.copy())
        return self._stacked_observation(), info

    def step(self, action: int) -> StepResult:
        step = self.env.step(action)
        self._frames.append(step.observation.copy())
        return StepResult(
            observation=self._stacked_observation(),
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            info=step.info,
        )

    def render(self) -> None:
        self.env.render()

    def render_text(self, width: int = 40, height: int = 20) -> str:
        return self.env.render_text(width=width, height=height)

    def close(self) -> None:
        self.env.close()

    def _stacked_observation(self) -> np.ndarray:
        return np.concatenate(list(self._frames), dtype=np.float32)

