from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, float | int | bool]


class CartPoleEnv:
    action_size = 2
    observation_size = 4

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 500,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        tau: float = 0.02,
    ) -> None:
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masscart + self.masspole
        self.length = length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau = tau
        self.theta_threshold_radians = 12.0 * 2.0 * math.pi / 360.0
        self.x_threshold = 2.4
        self.max_steps = max_steps
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(self.observation_size, dtype=np.float32)
        self.steps = 0
        self.episode_return = 0.0
        self.done = False

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        self.episode_return = 0.0
        self.done = False
        return self.state.copy(), self._build_info(terminated=False, truncated=False)

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("step() called after episode end; call reset() before stepping again.")
        if action not in (0, 1):
            raise ValueError(f"Invalid action {action}; expected 0 or 1.")

        x, x_dot, theta, theta_dot = (float(value) for value in self.state)
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.asarray((x, x_dot, theta, theta_dot), dtype=np.float32)

        self.steps += 1
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = self.steps >= self.max_steps and not terminated
        reward = 1.0
        self.episode_return += reward
        self.done = terminated or truncated

        return StepResult(
            observation=self.state.copy(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=self._build_info(terminated=terminated, truncated=truncated),
        )

    def render_text(self, width: int = 41) -> str:
        normalized_x = (float(self.state[0]) / self.x_threshold + 1.0) * 0.5
        cart_index = int(np.clip(round(normalized_x * (width - 1)), 0, width - 1))
        track = ["-"] * width
        track[cart_index] = "C"
        angle_deg = math.degrees(float(self.state[2]))
        return (
            f"|{''.join(track)}|\n"
            f"x={self.state[0]:+0.3f} x_dot={self.state[1]:+0.3f} "
            f"theta={angle_deg:+0.2f}deg theta_dot={self.state[3]:+0.3f} "
            f"steps={self.steps} return={self.episode_return:0.1f}"
        )

    def close(self) -> None:
        return None

    def _build_info(self, terminated: bool, truncated: bool) -> dict[str, float | int | bool]:
        return {
            "steps": self.steps,
            "episode_return": float(self.episode_return),
            "terminated": terminated,
            "truncated": truncated,
            "x": float(self.state[0]),
            "theta_radians": float(self.state[2]),
        }
