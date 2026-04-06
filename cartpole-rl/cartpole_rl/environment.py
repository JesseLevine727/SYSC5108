from __future__ import annotations

from dataclasses import dataclass
import math
import time

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
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._window_closed = False
        self._screen_width = 800
        self._screen_height = 500

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)

        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        self.episode_return = 0.0
        self.done = False
        self._window_closed = False
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

    def render_human(self, delay: float = 0.03, action: int | None = None) -> bool:
        pygame = self._load_pygame()
        self._ensure_pygame_surface()
        if self._window_closed:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._window_closed = True
                return False

        screen = self._screen
        assert screen is not None
        screen.fill((245, 247, 250))

        track_y = int(self._screen_height * 0.74)
        pygame.draw.line(screen, (70, 70, 70), (80, track_y), (self._screen_width - 80, track_y), 4)

        cart_x = self._cart_screen_x()
        cart_width = 92
        cart_height = 32
        cart_rect = pygame.Rect(0, 0, cart_width, cart_height)
        cart_rect.center = (cart_x, track_y - 18)
        pygame.draw.rect(screen, (49, 87, 187), cart_rect, border_radius=8)

        wheel_radius = 9
        for wheel_x in (cart_rect.left + 20, cart_rect.right - 20):
            pygame.draw.circle(screen, (45, 45, 45), (wheel_x, track_y - 1), wheel_radius)

        pole_base = (cart_rect.centerx, cart_rect.top + 4)
        pole_length_px = 220
        theta = float(self.state[2])
        pole_tip = (
            pole_base[0] + pole_length_px * math.sin(theta),
            pole_base[1] - pole_length_px * math.cos(theta),
        )
        pygame.draw.line(screen, (176, 61, 61), pole_base, pole_tip, 10)
        pygame.draw.circle(screen, (176, 61, 61), (int(pole_tip[0]), int(pole_tip[1])), 14)
        pygame.draw.circle(screen, (35, 35, 35), pole_base, 7)

        info_lines = [
            f"steps: {self.steps}",
            f"return: {self.episode_return:0.1f}",
            f"action: {action if action is not None else '-'}",
            f"x: {self.state[0]:+0.3f}",
            f"x_dot: {self.state[1]:+0.3f}",
            f"theta_deg: {math.degrees(theta):+0.2f}",
            f"theta_dot: {self.state[3]:+0.3f}",
        ]
        self._draw_info_panel(info_lines)

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(max(1, int(round(1.0 / max(delay, 1e-3)))))
        else:
            time.sleep(delay)
        return not self._window_closed

    def close(self) -> None:
        if self._pygame is not None:
            if self._screen is not None:
                self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._window_closed = False
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

    def _load_pygame(self):
        if self._pygame is not None:
            return self._pygame
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for --render human. Install it with `./.venv/bin/pip install pygame`."
            ) from exc
        self._pygame = pygame
        return pygame

    def _ensure_pygame_surface(self) -> None:
        if self._screen is not None:
            return
        pygame = self._load_pygame()
        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode((self._screen_width, self._screen_height))
        pygame.display.set_caption("CartPole Demo")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 22)

    def _cart_screen_x(self) -> int:
        usable_width = self._screen_width - 160
        normalized_x = (float(self.state[0]) + self.x_threshold) / (2.0 * self.x_threshold)
        normalized_x = float(np.clip(normalized_x, 0.0, 1.0))
        return int(80 + usable_width * normalized_x)

    def _draw_info_panel(self, lines: list[str]) -> None:
        if self._screen is None or self._font is None:
            return
        panel = self._screen
        text_x = 24
        text_y = 20
        for line in lines:
            surface = self._font.render(line, True, (30, 30, 30))
            panel.blit(surface, (text_x, text_y))
            text_y += 30
