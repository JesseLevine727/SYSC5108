from __future__ import annotations

from dataclasses import dataclass
import os
import random

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, float | int]


@dataclass
class Pipe:
    x: float
    gap_y: float
    passed: bool = False


class FlappyBirdEnv:
    """A pygame-based Flappy Bird environment with vector observations."""

    action_meanings = {
        0: "glide",
        1: "flap",
    }

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 4_000,
        render_mode: str | None = None,
        frame_rate: int = 60,
    ) -> None:
        self.random = random.Random(seed)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.frame_rate = frame_rate

        self.screen_width = 288
        self.screen_height = 512
        self.floor_y = 460
        self.bird_x = 72
        self.bird_radius = 12
        self.pipe_width = 52
        self.pipe_gap = 150
        self.pipe_spacing = 180
        self.pipe_speed = 3.2

        self.gravity = 0.45
        self.flap_velocity = -7.5
        self.max_velocity = 10.0
        self.min_gap_y = 120
        self.max_gap_y = 330

        self.bird_y = self.screen_height / 2.0
        self.velocity = 0.0
        self.score = 0
        self.steps = 0
        self.pipes: list[Pipe] = []

        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._pygame_ready = False

    @property
    def observation_size(self) -> int:
        return 6

    @property
    def action_size(self) -> int:
        return 2

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, float | int]]:
        if seed is not None:
            self.random.seed(seed)

        self.bird_y = self.screen_height / 2.0
        self.velocity = 0.0
        self.score = 0
        self.steps = 0
        self.pipes = [
            Pipe(x=self.screen_width + 80.0, gap_y=self.screen_height / 2.0),
            Pipe(x=self.screen_width + 80.0 + self.pipe_spacing, gap_y=self._sample_gap_y()),
            Pipe(x=self.screen_width + 80.0 + (2 * self.pipe_spacing), gap_y=self._sample_gap_y()),
        ]
        return self._observation(), self._info()

    def step(self, action: int) -> StepResult:
        if action not in (0, 1):
            raise ValueError(f"Invalid action {action}")

        self.steps += 1
        next_pipe = self._next_pipe()
        previous_distance = abs(self.bird_y - next_pipe.gap_y)

        if action == 1:
            self.velocity = self.flap_velocity

        self.velocity = min(self.velocity + self.gravity, self.max_velocity)
        self.bird_y += self.velocity

        reward = 0.1
        terminated = False
        truncated = self.steps >= self.max_steps

        for pipe in self.pipes:
            pipe.x -= self.pipe_speed

            if not pipe.passed and (pipe.x + self.pipe_width) < self.bird_x:
                pipe.passed = True
                self.score += 1
                reward += 5.0

        if self._collision():
            reward = -10.0
            terminated = True
        else:
            next_pipe = self._next_pipe()
            current_distance = abs(self.bird_y - next_pipe.gap_y)
            reward += 0.10 * (previous_distance - current_distance) / self.pipe_gap
            reward += max(0.0, 0.02 - (current_distance / self.pipe_gap) * 0.02)

        self._recycle_pipes()
        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=self._info(),
        )

    def render(self) -> None:
        if self.render_mode != "human":
            return

        self._ensure_pygame()
        assert self._screen is not None
        assert self._font is not None
        assert self._clock is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        self._screen.fill((118, 200, 240))

        for pipe in self.pipes:
            top_height = int(pipe.gap_y - (self.pipe_gap / 2.0))
            bottom_y = int(pipe.gap_y + (self.pipe_gap / 2.0))
            pygame.draw.rect(self._screen, (73, 175, 72), pygame.Rect(int(pipe.x), 0, self.pipe_width, top_height))
            pygame.draw.rect(
                self._screen,
                (73, 175, 72),
                pygame.Rect(int(pipe.x), bottom_y, self.pipe_width, self.floor_y - bottom_y),
            )

        pygame.draw.rect(
            self._screen,
            (214, 182, 86),
            pygame.Rect(0, self.floor_y, self.screen_width, self.screen_height - self.floor_y),
        )
        pygame.draw.circle(self._screen, (248, 231, 28), (self.bird_x, int(self.bird_y)), self.bird_radius)
        pygame.draw.circle(self._screen, (32, 32, 32), (self.bird_x + 4, int(self.bird_y) - 4), 2)

        score_text = self._font.render(f"Score: {self.score}", True, (255, 255, 255))
        self._screen.blit(score_text, (12, 12))
        pygame.display.flip()
        self._clock.tick(self.frame_rate)

    def close(self) -> None:
        if self._pygame_ready:
            pygame.display.quit()
            pygame.quit()
        self._screen = None
        self._clock = None
        self._font = None
        self._pygame_ready = False

    def clone_state(self) -> dict[str, object]:
        return {
            "random_state": self.random.getstate(),
            "bird_y": self.bird_y,
            "velocity": self.velocity,
            "score": self.score,
            "steps": self.steps,
            "pipes": [(pipe.x, pipe.gap_y, pipe.passed) for pipe in self.pipes],
        }

    def restore_state(self, snapshot: dict[str, object]) -> tuple[np.ndarray, dict[str, float | int]]:
        self.random.setstate(snapshot["random_state"])  # type: ignore[arg-type]
        self.bird_y = float(snapshot["bird_y"])  # type: ignore[arg-type]
        self.velocity = float(snapshot["velocity"])  # type: ignore[arg-type]
        self.score = int(snapshot["score"])  # type: ignore[arg-type]
        self.steps = int(snapshot["steps"])  # type: ignore[arg-type]
        self.pipes = [
            Pipe(x=float(pipe_x), gap_y=float(gap_y), passed=bool(passed))
            for pipe_x, gap_y, passed in snapshot["pipes"]  # type: ignore[misc]
        ]
        return self._observation(), self._info()

    def render_text(self, width: int = 40, height: int = 20) -> str:
        grid = [[" " for _ in range(width)] for _ in range(height)]

        bird_col = max(0, min(width - 1, int((self.bird_x / self.screen_width) * (width - 1))))
        bird_row = max(0, min(height - 1, int((self.bird_y / self.floor_y) * (height - 1))))
        grid[bird_row][bird_col] = "@"

        for pipe in self.pipes:
            pipe_col = max(0, min(width - 1, int((pipe.x / self.screen_width) * (width - 1))))
            gap_top = int(((pipe.gap_y - (self.pipe_gap / 2.0)) / self.floor_y) * (height - 1))
            gap_bottom = int(((pipe.gap_y + (self.pipe_gap / 2.0)) / self.floor_y) * (height - 1))
            for row in range(height):
                if row < gap_top or row > gap_bottom:
                    grid[row][pipe_col] = "|"

        floor_row = max(0, min(height - 1, int((self.floor_y / self.screen_height) * (height - 1))))
        for col in range(width):
            grid[floor_row][col] = "="

        return "\n".join("".join(row) for row in grid)

    def _sample_gap_y(self) -> float:
        return self.random.uniform(self.min_gap_y, self.max_gap_y)

    def _next_pipe(self) -> Pipe:
        return self._upcoming_pipes()[0]

    def _upcoming_pipes(self) -> list[Pipe]:
        candidates = [pipe for pipe in self.pipes if (pipe.x + self.pipe_width) >= self.bird_x]
        if not candidates:
            return sorted(self.pipes, key=lambda pipe: pipe.x)
        return sorted(candidates, key=lambda pipe: pipe.x)

    def _primary_and_secondary_pipe(self) -> tuple[Pipe, Pipe]:
        upcoming = self._upcoming_pipes()
        primary = upcoming[0]
        if len(upcoming) >= 2:
            return primary, upcoming[1]

        synthetic_secondary = Pipe(
            x=primary.x + self.pipe_spacing,
            gap_y=primary.gap_y,
            passed=False,
        )
        return primary, synthetic_secondary

    def _recycle_pipes(self) -> None:
        farthest_x = max(pipe.x for pipe in self.pipes)
        for pipe in self.pipes:
            if pipe.x + self.pipe_width < 0:
                pipe.x = farthest_x + self.pipe_spacing
                pipe.gap_y = self._sample_gap_y()
                pipe.passed = False
                farthest_x = pipe.x

    def _collision(self) -> bool:
        if (self.bird_y - self.bird_radius) <= 0 or (self.bird_y + self.bird_radius) >= self.floor_y:
            return True

        bird_rect = pygame.Rect(
            int(self.bird_x - self.bird_radius),
            int(self.bird_y - self.bird_radius),
            self.bird_radius * 2,
            self.bird_radius * 2,
        )

        for pipe in self.pipes:
            top_rect = pygame.Rect(int(pipe.x), 0, self.pipe_width, int(pipe.gap_y - (self.pipe_gap / 2.0)))
            bottom_y = int(pipe.gap_y + (self.pipe_gap / 2.0))
            bottom_rect = pygame.Rect(int(pipe.x), bottom_y, self.pipe_width, self.floor_y - bottom_y)
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                return True
        return False

    def _observation(self) -> np.ndarray:
        next_pipe, second_pipe = self._primary_and_secondary_pipe()
        obs = np.array(
            [
                self.bird_y / self.floor_y,
                np.clip(self.velocity / self.max_velocity, -1.0, 1.0),
                np.clip((next_pipe.x - self.bird_x) / self.screen_width, -1.0, 1.0),
                np.clip((next_pipe.gap_y - self.bird_y) / self.floor_y, -1.0, 1.0),
                np.clip((second_pipe.x - self.bird_x) / self.screen_width, -1.0, 1.0),
                np.clip((second_pipe.gap_y - self.bird_y) / self.floor_y, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _info(self) -> dict[str, float | int]:
        next_pipe, second_pipe = self._primary_and_secondary_pipe()
        return {
            "score": self.score,
            "steps": self.steps,
            "bird_y": self.bird_y,
            "velocity": self.velocity,
            "next_pipe_x": next_pipe.x,
            "next_gap_y": next_pipe.gap_y,
            "second_pipe_x": second_pipe.x,
            "second_gap_y": second_pipe.gap_y,
        }

    def _ensure_pygame(self) -> None:
        if self._pygame_ready:
            return

        if not os.environ.get("DISPLAY") and self.render_mode == "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        pygame.display.set_caption("Flappy Bird RL")
        self._screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("arial", 24)
        self._pygame_ready = True
