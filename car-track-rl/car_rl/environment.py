from __future__ import annotations

import math
import os
import random

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    import gym
    from gym import spaces

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame


class CarTrackEnv(gym.Env):
    """A lightweight Gymnasium-compatible top-down car driving task."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    action_meanings = {
        0: "coast",
        1: "accelerate",
        2: "brake",
        3: "left",
        4: "right",
        5: "accelerate_left",
        6: "accelerate_right",
    }

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 1_500,
        render_mode: str | None = None,
        frame_rate: int = 60,
        randomize_start: bool = True,
    ) -> None:
        super().__init__()
        self.random = random.Random(seed)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.frame_rate = frame_rate
        self.randomize_start = randomize_start

        self.track_base_radius = 165.0
        self.track_half_width = 26.0
        self.track_length = self._estimate_track_length()
        self.world_extent = self.track_base_radius + self.track_half_width + 70.0

        self.dt = 0.11
        self.max_speed = 22.0
        self.acceleration = 10.0
        self.brake_deceleration = 12.0
        self.drag = 0.28
        self.steering_rate = 2.7
        self.ray_length = 84.0
        self.ray_step = 6.0
        self.ray_angles = (-80.0, -40.0, 0.0, 40.0, 80.0)

        self.reward_progress_gain = 1.8
        self.reward_heading_gain = 0.10
        self.reward_center_penalty = 0.14
        self.reward_steer_penalty = 0.015
        self.reward_idle_penalty = 0.035
        self.reward_off_track_penalty = 5.0
        self.lap_bonus = 2.5

        self.action_space = spaces.Discrete(len(self.action_meanings))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5 + len(self.ray_angles),),
            dtype=np.float32,
        )

        self.position = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.speed = 0.0
        self.steps = 0
        self.laps_completed = 0
        self.last_theta = 0.0
        self.last_progress_distance = 0.0
        self.total_progress = 0.0

        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._pygame_ready = False
        self._render_thetas = np.linspace(0.0, 2.0 * math.pi, 240, endpoint=False, dtype=np.float32)

    @property
    def observation_size(self) -> int:
        return int(self.observation_space.shape[0])

    @property
    def action_size(self) -> int:
        return int(self.action_space.n)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        del options
        if seed is not None:
            self.random.seed(seed)

        start_theta = self.random.uniform(0.0, 2.0 * math.pi) if self.randomize_start else 0.0
        lateral_offset = self.random.uniform(-0.12, 0.12) * self.track_half_width
        heading_jitter = self.random.uniform(-0.05, 0.05)

        center = self._centerline_point(start_theta)
        normal = self._normal_vector(start_theta)
        tangent = self._tangent_vector(start_theta)

        self.position = (center + (normal * lateral_offset)).astype(np.float32)
        self.heading = self._wrap_angle(math.atan2(float(tangent[1]), float(tangent[0])) + heading_jitter)
        self.speed = 0.0
        self.steps = 0
        self.laps_completed = 0
        self.last_theta = start_theta
        self.last_progress_distance = 0.0
        self.total_progress = 0.0

        observation = self._observation(
            theta=start_theta,
            lateral_error=self._lateral_error(self.position, theta=start_theta),
            heading_error=self._heading_error(start_theta),
            progress_distance=0.0,
        )
        return observation, self._info(theta=start_theta, lap_complete=False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, float | int]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        throttle, brake, steer = self._decode_action(action)
        speed_norm = self.speed / self.max_speed

        self.speed = float(
            np.clip(
                self.speed
                + (throttle * self.acceleration * self.dt)
                - (brake * self.brake_deceleration * self.dt)
                - (self.drag * self.speed * self.dt),
                0.0,
                self.max_speed,
            )
        )
        steer_scale = 0.30 + (0.70 * speed_norm)
        self.heading = self._wrap_angle(self.heading + (steer * self.steering_rate * self.dt * steer_scale))
        heading_vector = np.asarray([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32)
        self.position = self.position + (heading_vector * self.speed * self.dt)
        self.steps += 1

        theta = self._position_theta(self.position)
        delta_theta = self._wrapped_angle(theta - self.last_theta)
        progress_distance = delta_theta * self._arc_scale(theta)
        lap_complete = delta_theta > 0.0 and self.last_theta > (1.5 * math.pi) and theta < (0.5 * math.pi)
        if lap_complete:
            self.laps_completed += 1

        self.last_theta = theta
        self.last_progress_distance = progress_distance
        self.total_progress += progress_distance

        lateral_error = self._lateral_error(self.position, theta=theta)
        heading_error = self._heading_error(theta)
        terminated = abs(lateral_error) > self.track_half_width
        truncated = self.steps >= self.max_steps

        reward = self._compute_reward(
            progress_distance=progress_distance,
            lateral_error=lateral_error,
            heading_error=heading_error,
            steer=steer,
            throttle=throttle,
            terminated=terminated,
        )
        if lap_complete:
            reward += self.lap_bonus

        observation = self._observation(
            theta=theta,
            lateral_error=lateral_error,
            heading_error=heading_error,
            progress_distance=progress_distance,
        )
        info = self._info(theta=theta, lateral_error=lateral_error, heading_error=heading_error, lap_complete=lap_complete)
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return

        self._ensure_pygame()
        assert self._screen is not None
        assert self._clock is not None
        assert self._font is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        self._screen.fill((74, 125, 72))

        outer_points = [
            self._world_to_screen(self._centerline_point(theta) + (self._normal_vector(theta) * self.track_half_width))
            for theta in self._render_thetas
        ]
        inner_points = [
            self._world_to_screen(self._centerline_point(theta) - (self._normal_vector(theta) * self.track_half_width))
            for theta in self._render_thetas
        ]
        centerline_points = [self._world_to_screen(self._centerline_point(theta)) for theta in self._render_thetas]

        pygame.draw.polygon(self._screen, (58, 58, 58), outer_points)
        pygame.draw.polygon(self._screen, (74, 125, 72), inner_points)
        pygame.draw.lines(self._screen, (230, 230, 230), True, centerline_points, 2)

        car_points = self._car_polygon()
        pygame.draw.polygon(self._screen, (196, 67, 67), car_points)
        pygame.draw.polygon(self._screen, (245, 229, 99), car_points, 2)

        hud_lines = [
            f"Speed: {self.speed:0.1f}",
            f"Laps: {self.laps_completed}",
            f"Progress: {self.total_progress / self.track_length:0.2f}",
            f"Steps: {self.steps}",
        ]
        for index, line in enumerate(hud_lines):
            text = self._font.render(line, True, (255, 255, 255))
            self._screen.blit(text, (14, 14 + (index * 24)))

        pygame.display.flip()
        self._clock.tick(self.frame_rate)

    def render_text(self) -> str:
        lateral_error = self._lateral_error(self.position)
        heading_error = math.degrees(self._heading_error(self._position_theta(self.position)))
        return (
            f"speed={self.speed:0.2f} laps={self.laps_completed} "
            f"progress={self.total_progress / self.track_length:0.2f} "
            f"offset={lateral_error:0.2f} heading_error_deg={heading_error:0.1f}"
        )

    def close(self) -> None:
        if self._pygame_ready:
            pygame.display.quit()
            pygame.quit()
        self._screen = None
        self._clock = None
        self._font = None
        self._pygame_ready = False

    def _decode_action(self, action: int) -> tuple[float, float, float]:
        if action == 0:
            return 0.0, 0.0, 0.0
        if action == 1:
            return 1.0, 0.0, 0.0
        if action == 2:
            return 0.0, 1.0, 0.0
        if action == 3:
            return 0.0, 0.0, -1.0
        if action == 4:
            return 0.0, 0.0, 1.0
        if action == 5:
            return 1.0, 0.0, -1.0
        if action == 6:
            return 1.0, 0.0, 1.0
        raise ValueError(f"Invalid action {action}")

    def _compute_reward(
        self,
        progress_distance: float,
        lateral_error: float,
        heading_error: float,
        steer: float,
        throttle: float,
        terminated: bool,
    ) -> float:
        progress_norm = float(
            np.clip(progress_distance / max(self.max_speed * self.dt, 1e-6), -1.0, 1.0)
        )
        center_penalty = self.reward_center_penalty * abs(lateral_error / self.track_half_width)
        heading_bonus = self.reward_heading_gain * max(0.0, math.cos(heading_error))
        steer_penalty = self.reward_steer_penalty * abs(steer) * (0.25 + (self.speed / self.max_speed))
        idle_penalty = self.reward_idle_penalty if self.speed < 0.75 and throttle <= 0.0 else 0.0

        reward = (self.reward_progress_gain * progress_norm) + heading_bonus - center_penalty - steer_penalty - idle_penalty
        if terminated:
            reward -= self.reward_off_track_penalty
        return reward

    def _observation(
        self,
        *,
        theta: float | None = None,
        lateral_error: float | None = None,
        heading_error: float | None = None,
        progress_distance: float | None = None,
    ) -> np.ndarray:
        current_theta = self._position_theta(self.position) if theta is None else theta
        current_lateral_error = self._lateral_error(self.position, theta=current_theta) if lateral_error is None else lateral_error
        current_heading_error = self._heading_error(current_theta) if heading_error is None else heading_error
        current_progress = self.last_progress_distance if progress_distance is None else progress_distance

        ray_distances = [
            self._ray_distance(math.radians(relative_angle)) / self.ray_length for relative_angle in self.ray_angles
        ]
        observation = np.asarray(
            [
                self.speed / self.max_speed,
                float(np.clip(current_lateral_error / self.track_half_width, -1.0, 1.0)),
                math.sin(current_heading_error),
                math.cos(current_heading_error),
                float(np.clip(current_progress / max(self.max_speed * self.dt, 1e-6), -1.0, 1.0)),
                *ray_distances,
            ],
            dtype=np.float32,
        )
        return observation

    def _info(
        self,
        *,
        theta: float | None = None,
        lateral_error: float | None = None,
        heading_error: float | None = None,
        lap_complete: bool,
    ) -> dict[str, float | int]:
        current_theta = self._position_theta(self.position) if theta is None else theta
        current_lateral_error = self._lateral_error(self.position, theta=current_theta) if lateral_error is None else lateral_error
        current_heading_error = self._heading_error(current_theta) if heading_error is None else heading_error
        return {
            "steps": self.steps,
            "laps": self.laps_completed,
            "progress": float(self.total_progress / self.track_length),
            "speed": float(self.speed),
            "lateral_error": float(current_lateral_error),
            "heading_error_deg": float(math.degrees(current_heading_error)),
            "lap_complete": int(lap_complete),
        }

    def _track_radius(self, theta: float) -> float:
        return self.track_base_radius + (22.0 * math.sin((2.0 * theta) + 0.45)) + (14.0 * math.cos((3.0 * theta) - 0.35))

    def _track_radius_derivative(self, theta: float) -> float:
        return (44.0 * math.cos((2.0 * theta) + 0.45)) - (42.0 * math.sin((3.0 * theta) - 0.35))

    def _centerline_point(self, theta: float) -> np.ndarray:
        radius = self._track_radius(theta)
        return np.asarray([radius * math.cos(theta), radius * math.sin(theta)], dtype=np.float32)

    def _tangent_vector(self, theta: float) -> np.ndarray:
        radius = self._track_radius(theta)
        radius_derivative = self._track_radius_derivative(theta)
        derivative = np.asarray(
            [
                (radius_derivative * math.cos(theta)) - (radius * math.sin(theta)),
                (radius_derivative * math.sin(theta)) + (radius * math.cos(theta)),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(derivative))
        return derivative / max(norm, 1e-6)

    def _normal_vector(self, theta: float) -> np.ndarray:
        tangent = self._tangent_vector(theta)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
        norm = float(np.linalg.norm(normal))
        return normal / max(norm, 1e-6)

    def _arc_scale(self, theta: float) -> float:
        radius = self._track_radius(theta)
        radius_derivative = self._track_radius_derivative(theta)
        return float(math.sqrt((radius * radius) + (radius_derivative * radius_derivative)))

    def _estimate_track_length(self) -> float:
        thetas = np.linspace(0.0, 2.0 * math.pi, 1_024, endpoint=False, dtype=np.float32)
        arc_scales = np.asarray([self._arc_scale(float(theta)) for theta in thetas], dtype=np.float32)
        delta = (2.0 * math.pi) / len(thetas)
        return float(np.sum(arc_scales) * delta)

    def _position_theta(self, position: np.ndarray) -> float:
        theta = math.atan2(float(position[1]), float(position[0]))
        if theta < 0.0:
            theta += 2.0 * math.pi
        return theta

    def _lateral_error(self, position: np.ndarray, *, theta: float | None = None) -> float:
        current_theta = self._position_theta(position) if theta is None else theta
        return float(np.linalg.norm(position) - self._track_radius(current_theta))

    def _heading_error(self, theta: float) -> float:
        tangent = self._tangent_vector(theta)
        target_heading = math.atan2(float(tangent[1]), float(tangent[0]))
        return self._wrap_angle(target_heading - self.heading)

    def _ray_distance(self, relative_angle_radians: float) -> float:
        direction = np.asarray(
            [math.cos(self.heading + relative_angle_radians), math.sin(self.heading + relative_angle_radians)],
            dtype=np.float32,
        )
        distance = 0.0
        while distance < self.ray_length:
            distance += self.ray_step
            sample = self.position + (direction * distance)
            if abs(self._lateral_error(sample)) > self.track_half_width:
                return float(distance)
        return self.ray_length

    def _car_polygon(self) -> list[tuple[int, int]]:
        heading = np.asarray([math.cos(self.heading), math.sin(self.heading)], dtype=np.float32)
        lateral = np.asarray([-heading[1], heading[0]], dtype=np.float32)
        front = self.position + (heading * 10.0)
        rear_left = self.position - (heading * 7.0) + (lateral * 5.5)
        rear_right = self.position - (heading * 7.0) - (lateral * 5.5)
        return [
            self._world_to_screen(front),
            self._world_to_screen(rear_left),
            self._world_to_screen(rear_right),
        ]

    def _world_to_screen(self, point: np.ndarray) -> tuple[int, int]:
        width = 800
        height = 800
        scale = min(width, height) / (2.0 * self.world_extent)
        x = int((point[0] * scale) + (width / 2.0))
        y = int((height / 2.0) - (point[1] * scale))
        return x, y

    def _ensure_pygame(self) -> None:
        if self._pygame_ready:
            return
        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Car Track RL")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 28)
        self._pygame_ready = True

    def _wrap_angle(self, angle: float) -> float:
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        while angle > math.pi:
            angle -= 2.0 * math.pi
        return angle

    def _wrapped_angle(self, angle: float) -> float:
        if angle > math.pi:
            angle -= 2.0 * math.pi
        elif angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
