from __future__ import annotations

import numpy as np

from flappy_rl.environment import FlappyBirdEnv


def predicted_gap_error_at_crossing(observation: np.ndarray, env: FlappyBirdEnv) -> float:
    bird_y = float(observation[0]) * env.floor_y
    velocity = float(observation[1]) * env.max_velocity
    dx = float(observation[2]) * env.screen_width
    gap_offset = float(observation[3]) * env.floor_y
    next_gap_y = bird_y + gap_offset

    travel_time = max(0.0, dx / max(env.pipe_speed, 1e-6))
    predicted_y = bird_y + (velocity * travel_time) + (0.5 * env.gravity * travel_time * travel_time)
    return float(np.clip((predicted_y - next_gap_y) / env.floor_y, -1.0, 1.0))


def build_ppo_state(
    observation: np.ndarray,
    env: FlappyBirdEnv,
    use_predicted_gap_error: bool,
) -> np.ndarray:
    base_state = np.asarray(observation[:4], dtype=np.float32)
    if not use_predicted_gap_error:
        return base_state

    predicted_error = predicted_gap_error_at_crossing(observation, env)
    return np.asarray([base_state[0], base_state[1], base_state[2], base_state[3], predicted_error], dtype=np.float32)
