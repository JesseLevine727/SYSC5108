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
        domain_randomization_scale: float = 1.0,
        track_pool: str = "train",
    ) -> None:
        super().__init__()
        self.random = random.Random(seed)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.frame_rate = frame_rate
        self.randomize_start = randomize_start
        self.domain_randomization_scale = domain_randomization_scale
        self.track_pool = track_pool

        self.base_track_half_width = 26.0
        self.track_half_width = self.base_track_half_width
        self.track_family = "radial"
        self.track_generator = "radial"
        self.generator_params: dict[str, float | list[tuple[float, int, float]]] = {}

        self.base_dt = 0.11
        self.base_max_speed = 22.0
        self.base_acceleration = 10.0
        self.base_brake_deceleration = 12.0
        self.base_drag = 0.28
        self.base_steering_rate = 2.7
        self.dt = self.base_dt
        self.max_speed = self.base_max_speed
        self.acceleration = self.base_acceleration
        self.brake_deceleration = self.base_brake_deceleration
        self.drag = self.base_drag
        self.steering_rate = self.base_steering_rate

        self.ray_length = 84.0
        self.ray_step = 6.0
        self.ray_angles = (-80.0, -40.0, 0.0, 40.0, 80.0)
        self.max_sensor_noise = 0.035
        self.sensor_noise = 0.0

        self.reward_progress_gain = 1.8
        self.reward_heading_gain = 0.10
        self.reward_center_penalty = 0.14
        self.reward_steer_penalty = 0.015
        self.reward_idle_penalty = 0.035
        self.reward_off_track_penalty = 5.0
        self.lap_bonus = 2.5

        self.track_sample_count = 512
        self.track_points = np.zeros((self.track_sample_count, 2), dtype=np.float32)
        self.track_tangents = np.zeros((self.track_sample_count, 2), dtype=np.float32)
        self.track_normals = np.zeros((self.track_sample_count, 2), dtype=np.float32)
        self.track_progress = np.zeros(self.track_sample_count, dtype=np.float32)
        self.track_length = 1.0
        self.world_extent = 300.0

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
        self.last_progress_distance = 0.0
        self.total_progress = 0.0
        self.last_track_index = 0
        self.last_track_progress = 0.0

        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._pygame_ready = False
        self._render_outer = np.zeros((self.track_sample_count, 2), dtype=np.float32)
        self._render_inner = np.zeros((self.track_sample_count, 2), dtype=np.float32)
        self._resample_domain()

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

        self._resample_domain()
        start_index = self.random.randrange(self.track_sample_count) if self.randomize_start else 0
        lateral_span = 0.10 + (0.20 * self.domain_randomization_scale)
        heading_span = 0.05 + (0.12 * self.domain_randomization_scale)
        lateral_offset = self.random.uniform(-lateral_span, lateral_span) * self.track_half_width
        heading_jitter = self.random.uniform(-heading_span, heading_span)

        center = self.track_points[start_index]
        normal = self.track_normals[start_index]
        tangent = self.track_tangents[start_index]

        self.position = (center + (normal * lateral_offset)).astype(np.float32)
        self.heading = self._wrap_angle(math.atan2(float(tangent[1]), float(tangent[0])) + heading_jitter)
        self.speed = 0.0
        self.steps = 0
        self.laps_completed = 0
        self.last_progress_distance = 0.0
        self.total_progress = 0.0
        self.last_track_index = start_index
        self.last_track_progress = float(self.track_progress[start_index])

        lateral_error = self._signed_lateral_error(self.position, guess_index=start_index)
        heading_error = self._heading_error(start_index)
        observation = self._observation(
            track_index=start_index,
            lateral_error=lateral_error,
            heading_error=heading_error,
            progress_distance=0.0,
        )
        return observation, self._info(track_index=start_index, lap_complete=False)

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

        track_index = self._nearest_track_index(self.position, guess_index=self.last_track_index)
        current_progress = float(self.track_progress[track_index])
        raw_progress_delta = current_progress - self.last_track_progress
        lap_complete = raw_progress_delta < (-0.5 * self.track_length)
        if raw_progress_delta < (-0.5 * self.track_length):
            progress_distance = raw_progress_delta + self.track_length
        elif raw_progress_delta > (0.5 * self.track_length):
            progress_distance = raw_progress_delta - self.track_length
        else:
            progress_distance = raw_progress_delta

        if lap_complete and progress_distance > 0.0:
            self.laps_completed += 1

        self.last_track_index = track_index
        self.last_track_progress = current_progress
        self.last_progress_distance = progress_distance
        self.total_progress += progress_distance

        lateral_error = self._signed_lateral_error(self.position, guess_index=track_index)
        heading_error = self._heading_error(track_index)
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
            track_index=track_index,
            lateral_error=lateral_error,
            heading_error=heading_error,
            progress_distance=progress_distance,
        )
        info = self._info(
            track_index=track_index,
            lateral_error=lateral_error,
            heading_error=heading_error,
            lap_complete=lap_complete,
        )
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

        outer_points = [self._world_to_screen(point) for point in self._render_outer]
        inner_points = [self._world_to_screen(point) for point in self._render_inner]
        centerline_points = [self._world_to_screen(point) for point in self.track_points]

        pygame.draw.polygon(self._screen, (58, 58, 58), outer_points)
        pygame.draw.polygon(self._screen, (74, 125, 72), inner_points)
        pygame.draw.lines(self._screen, (230, 230, 230), True, centerline_points, 2)

        car_points = self._car_polygon()
        pygame.draw.polygon(self._screen, (196, 67, 67), car_points)
        pygame.draw.polygon(self._screen, (245, 229, 99), car_points, 2)

        hud_lines = [
            f"Family: {self.track_family}",
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
        lateral_error = self._signed_lateral_error(self.position, guess_index=self.last_track_index)
        heading_error = math.degrees(self._heading_error(self.last_track_index))
        return (
            f"family={self.track_family} speed={self.speed:0.2f} laps={self.laps_completed} "
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

    def set_domain_randomization(self, scale: float) -> None:
        self.domain_randomization_scale = max(0.0, min(1.5, float(scale)))

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
        track_index: int,
        lateral_error: float | None = None,
        heading_error: float | None = None,
        progress_distance: float | None = None,
    ) -> np.ndarray:
        current_lateral_error = (
            self._signed_lateral_error(self.position, guess_index=track_index) if lateral_error is None else lateral_error
        )
        current_heading_error = self._heading_error(track_index) if heading_error is None else heading_error
        current_progress = self.last_progress_distance if progress_distance is None else progress_distance

        ray_distances = [
            self._ray_distance(math.radians(relative_angle), guess_index=track_index) / self.ray_length
            for relative_angle in self.ray_angles
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
        if self.sensor_noise > 0.0:
            noise = np.asarray(
                [self.random.gauss(0.0, self.sensor_noise) for _ in range(observation.shape[0])],
                dtype=np.float32,
            )
            observation = np.clip(observation + noise, -1.0, 1.0)
        return observation

    def _info(
        self,
        *,
        track_index: int,
        lateral_error: float | None = None,
        heading_error: float | None = None,
        lap_complete: bool,
    ) -> dict[str, float | int | str]:
        current_lateral_error = (
            self._signed_lateral_error(self.position, guess_index=track_index) if lateral_error is None else lateral_error
        )
        current_heading_error = self._heading_error(track_index) if heading_error is None else heading_error
        return {
            "steps": self.steps,
            "laps": self.laps_completed,
            "progress": float(self.total_progress / self.track_length),
            "speed": float(self.speed),
            "lateral_error": float(current_lateral_error),
            "heading_error_deg": float(math.degrees(current_heading_error)),
            "lap_complete": int(lap_complete),
            "track_family": self.track_family,
            "track_width": float(self.track_half_width * 2.0),
            "track_generator": self.track_generator,
            "track_pool": self.track_pool,
        }

    def _resample_domain(self) -> None:
        scale = self.domain_randomization_scale
        generator = self.random.choice(self._available_generators())
        self.track_generator = generator

        self.track_half_width = float(
            np.clip(
                self.base_track_half_width + self.random.uniform(-5.0, 5.0) * scale,
                20.0,
                34.0,
            )
        )
        self.max_speed = self.base_max_speed * (1.0 + self.random.uniform(-0.14, 0.14) * scale)
        self.acceleration = self.base_acceleration * (1.0 + self.random.uniform(-0.16, 0.16) * scale)
        self.brake_deceleration = self.base_brake_deceleration * (1.0 + self.random.uniform(-0.16, 0.16) * scale)
        self.drag = self.base_drag * (1.0 + self.random.uniform(-0.25, 0.25) * scale)
        self.steering_rate = self.base_steering_rate * (1.0 + self.random.uniform(-0.18, 0.18) * scale)
        self.sensor_noise = self.max_sensor_noise * scale

        if generator == "radial":
            family_templates = {
                "radial_oval": [(22.0, 2, 0.45), (14.0, 3, -0.35), (6.0, 5, 1.15)],
                "radial_technical": [(18.0, 3, 0.10), (12.0, 5, -0.70), (8.0, 7, 0.40)],
                "radial_chicane": [(16.0, 2, -0.20), (18.0, 4, 0.90), (7.0, 6, -1.10)],
                "radial_kidney": [(25.0, 1, 0.35), (10.0, 3, -0.45), (6.0, 5, 0.85)],
            }
            self.track_family = self.random.choice(list(family_templates))
            template = family_templates[self.track_family]
            self.generator_params = {
                "base_radius": 165.0 + self.random.uniform(-18.0, 18.0) * scale,
                "harmonics": [
                    (
                        amplitude * (1.0 + self.random.uniform(-0.40, 0.40) * scale),
                        frequency,
                        phase + (self.random.uniform(-0.9, 0.9) * scale),
                    )
                    for amplitude, frequency, phase in template
                ],
            }
        elif generator == "ellipse":
            self.track_family = self.random.choice(["ellipse_classic", "ellipse_offset", "ellipse_fast"])
            self.generator_params = {
                "a": 185.0 + self.random.uniform(-24.0, 26.0) * scale,
                "b": 128.0 + self.random.uniform(-22.0, 22.0) * scale,
                "x3": self.random.uniform(0.0, 22.0) * scale,
                "y2": self.random.uniform(-18.0, 18.0) * scale,
                "phase_x": self.random.uniform(-0.8, 0.8) * scale,
                "phase_y": self.random.uniform(-0.8, 0.8) * scale,
            }
        elif generator == "peanut":
            self.track_family = self.random.choice(["peanut_balanced", "peanut_twist", "peanut_long"])
            self.generator_params = {
                "a": 150.0 + self.random.uniform(-20.0, 18.0) * scale,
                "b": 32.0 + self.random.uniform(-10.0, 14.0) * scale,
                "c": 138.0 + self.random.uniform(-18.0, 20.0) * scale,
                "d": self.random.uniform(-18.0, 18.0) * scale,
                "phase": self.random.uniform(-0.9, 0.9) * scale,
            }
        elif generator == "stadium":
            self.track_family = self.random.choice(["stadium_soft", "stadium_boxy", "stadium_squircle"])
            self.generator_params = {
                "a": 178.0 + self.random.uniform(-18.0, 22.0) * scale,
                "b": 138.0 + self.random.uniform(-18.0, 20.0) * scale,
                "sharpness": 1.2 + (self.random.uniform(0.0, 2.0) * max(scale, 0.1)),
                "x3": self.random.uniform(-12.0, 12.0) * scale,
                "y4": self.random.uniform(-10.0, 10.0) * scale,
            }
        else:
            templates = self._handcrafted_templates()
            family = self.random.choice(list(templates))
            self.track_family = family
            self.generator_params = {
                "control_points": templates[family],
                "scale_x": 1.0 + self.random.uniform(-0.16, 0.18) * scale,
                "scale_y": 1.0 + self.random.uniform(-0.16, 0.18) * scale,
                "rotation": self.random.uniform(-0.45, 0.45) * scale,
                "samples_per_segment": 36,
            }

        self._build_track_cache()

    def _build_track_cache(self) -> None:
        if self.track_generator == "handcrafted":
            points = self._build_handcrafted_points()
        else:
            params = np.linspace(0.0, 2.0 * math.pi, self.track_sample_count, endpoint=False, dtype=np.float32)
            points = np.asarray([self._centerline_point(float(param)) for param in params], dtype=np.float32)
        forward = np.roll(points, -1, axis=0)
        backward = np.roll(points, 1, axis=0)
        tangent_raw = forward - backward
        tangent_norm = np.linalg.norm(tangent_raw, axis=1, keepdims=True)
        tangents = tangent_raw / np.maximum(tangent_norm, 1e-6)
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0])).astype(np.float32)

        segment_vectors = forward - points
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        progress = np.zeros(self.track_sample_count, dtype=np.float32)
        if self.track_sample_count > 1:
            progress[1:] = np.cumsum(segment_lengths[:-1], dtype=np.float32)

        self.track_points = points
        self.track_tangents = tangents.astype(np.float32)
        self.track_normals = normals
        self.track_progress = progress
        self.track_length = float(np.sum(segment_lengths))

        max_extent = float(np.max(np.abs(points)))
        self.world_extent = max_extent + self.track_half_width + 90.0
        self._render_outer = self.track_points + (self.track_normals * self.track_half_width)
        self._render_inner = self.track_points - (self.track_normals * self.track_half_width)

    def _build_handcrafted_points(self) -> np.ndarray:
        control_points = np.asarray(self.generator_params["control_points"], dtype=np.float32)
        scale_x = float(self.generator_params["scale_x"])
        scale_y = float(self.generator_params["scale_y"])
        rotation = float(self.generator_params["rotation"])
        samples_per_segment = int(self.generator_params["samples_per_segment"])

        scaled = control_points.copy()
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y

        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        rotation_matrix = np.asarray([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        rotated = scaled @ rotation_matrix.T

        sampled_points: list[np.ndarray] = []
        point_count = len(rotated)
        for index in range(point_count):
            p0 = rotated[(index - 1) % point_count]
            p1 = rotated[index]
            p2 = rotated[(index + 1) % point_count]
            p3 = rotated[(index + 2) % point_count]
            for step in range(samples_per_segment):
                t = step / samples_per_segment
                t2 = t * t
                t3 = t2 * t
                point = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + ((2.0 * p0) - (5.0 * p1) + (4.0 * p2) - p3) * t2
                    + (-p0 + (3.0 * p1) - (3.0 * p2) + p3) * t3
                )
                sampled_points.append(point.astype(np.float32))

        points = np.asarray(sampled_points, dtype=np.float32)
        if points.shape[0] != self.track_sample_count:
            indices = np.linspace(0, points.shape[0] - 1, self.track_sample_count, dtype=np.int32)
            points = points[indices]
        return points

    def _centerline_point(self, param: float) -> np.ndarray:
        if self.track_generator == "radial":
            base_radius = float(self.generator_params["base_radius"])
            harmonics = self.generator_params["harmonics"]  # type: ignore[assignment]
            radius = base_radius
            for amplitude, frequency, phase in harmonics:  # type: ignore[misc]
                radius += amplitude * math.sin((frequency * param) + phase)
            return np.asarray([radius * math.cos(param), radius * math.sin(param)], dtype=np.float32)

        if self.track_generator == "ellipse":
            a = float(self.generator_params["a"])
            b = float(self.generator_params["b"])
            x3 = float(self.generator_params["x3"])
            y2 = float(self.generator_params["y2"])
            phase_x = float(self.generator_params["phase_x"])
            phase_y = float(self.generator_params["phase_y"])
            return np.asarray(
                [
                    (a * math.cos(param)) + (x3 * math.cos((3.0 * param) + phase_x)),
                    (b * math.sin(param)) + (y2 * math.sin((2.0 * param) + phase_y)),
                ],
                dtype=np.float32,
            )

        if self.track_generator == "peanut":
            a = float(self.generator_params["a"])
            b = float(self.generator_params["b"])
            c = float(self.generator_params["c"])
            d = float(self.generator_params["d"])
            phase = float(self.generator_params["phase"])
            x_radius = a + (b * math.cos((2.0 * param) + phase))
            y_radius = c + (d * math.sin((2.0 * param) - phase))
            return np.asarray(
                [
                    x_radius * math.cos(param),
                    y_radius * math.sin(param),
                ],
                dtype=np.float32,
            )

        a = float(self.generator_params["a"])
        b = float(self.generator_params["b"])
        sharpness = float(self.generator_params["sharpness"])
        x3 = float(self.generator_params["x3"])
        y4 = float(self.generator_params["y4"])
        denom = max(math.tanh(sharpness), 1e-6)
        x = a * math.tanh(sharpness * math.cos(param)) / denom
        y = b * math.tanh(sharpness * math.sin(param)) / denom
        return np.asarray(
            [
                x + (x3 * math.cos(3.0 * param)),
                y + (y4 * math.sin(4.0 * param)),
            ],
            dtype=np.float32,
        )

    def _available_generators(self) -> list[str]:
        if self.track_pool == "procedural":
            return ["radial", "ellipse", "peanut", "stadium"]
        if self.track_pool == "holdout":
            return ["handcrafted"]
        if self.track_pool == "all":
            return ["radial", "ellipse", "peanut", "stadium", "handcrafted"]
        return ["radial", "ellipse", "peanut", "stadium", "handcrafted"]

    def _handcrafted_templates(self) -> dict[str, list[tuple[float, float]]]:
        train_templates = {
            "handcrafted_train_switchback": [
                (-190.0, -20.0),
                (-145.0, -170.0),
                (0.0, -205.0),
                (155.0, -130.0),
                (205.0, -5.0),
                (165.0, 125.0),
                (20.0, 205.0),
                (-135.0, 155.0),
                (-215.0, 45.0),
            ],
            "handcrafted_train_box": [
                (-205.0, -85.0),
                (-145.0, -185.0),
                (10.0, -205.0),
                (160.0, -165.0),
                (215.0, -40.0),
                (195.0, 115.0),
                (75.0, 205.0),
                (-85.0, 190.0),
                (-205.0, 95.0),
            ],
        }
        holdout_templates = {
            "handcrafted_holdout_serpentine": [
                (-220.0, -60.0),
                (-150.0, -190.0),
                (-10.0, -150.0),
                (120.0, -210.0),
                (225.0, -90.0),
                (175.0, 60.0),
                (45.0, 165.0),
                (-90.0, 135.0),
                (-185.0, 210.0),
                (-235.0, 45.0),
            ],
            "handcrafted_holdout_clover": [
                (-150.0, -10.0),
                (-110.0, -140.0),
                (-10.0, -200.0),
                (95.0, -140.0),
                (150.0, -5.0),
                (110.0, 140.0),
                (5.0, 210.0),
                (-110.0, 150.0),
            ],
        }
        if self.track_pool == "holdout":
            return holdout_templates
        if self.track_pool == "all":
            return {**train_templates, **holdout_templates}
        return train_templates

    def _nearest_track_index(self, position: np.ndarray, guess_index: int | None = None) -> int:
        if guess_index is None:
            deltas = self.track_points - position
            distances = np.sum(deltas * deltas, axis=1)
            return int(np.argmin(distances))

        window_radius = 72
        offsets = np.arange(-window_radius, window_radius + 1)
        indices = (guess_index + offsets) % self.track_sample_count
        deltas = self.track_points[indices] - position
        distances = np.sum(deltas * deltas, axis=1)
        return int(indices[int(np.argmin(distances))])

    def _signed_lateral_error(self, position: np.ndarray, guess_index: int | None = None) -> float:
        track_index = self._nearest_track_index(position, guess_index=guess_index)
        delta = position - self.track_points[track_index]
        return float(np.dot(delta, self.track_normals[track_index]))

    def _heading_error(self, track_index: int) -> float:
        tangent = self.track_tangents[track_index]
        target_heading = math.atan2(float(tangent[1]), float(tangent[0]))
        return self._wrap_angle(target_heading - self.heading)

    def _ray_distance(self, relative_angle_radians: float, guess_index: int) -> float:
        direction = np.asarray(
            [math.cos(self.heading + relative_angle_radians), math.sin(self.heading + relative_angle_radians)],
            dtype=np.float32,
        )
        distance = 0.0
        ray_guess = guess_index
        while distance < self.ray_length:
            distance += self.ray_step
            sample = self.position + (direction * distance)
            ray_guess = self._nearest_track_index(sample, guess_index=ray_guess)
            delta = sample - self.track_points[ray_guess]
            lateral_error = abs(float(np.dot(delta, self.track_normals[ray_guess])))
            if lateral_error > self.track_half_width:
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
