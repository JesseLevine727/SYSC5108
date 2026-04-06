from __future__ import annotations

import argparse
import time

import pygame

from car_rl.environment import CarTrackEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play the car-track environment manually.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=1_500)
    parser.add_argument("--delay", type=float, default=0.02)
    return parser.parse_args()


def keyboard_action() -> int:
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    if up and left:
        return 5
    if up and right:
        return 6
    if up:
        return 1
    if down:
        return 2
    if left:
        return 3
    if right:
        return 4
    return 0


def main() -> None:
    args = parse_args()
    env = CarTrackEnv(seed=args.seed, max_steps=args.max_steps, render_mode="human", randomize_start=False)
    observation, info = env.reset(seed=args.seed)
    del observation, info

    while True:
        action = keyboard_action()
        _, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(args.delay)

        if terminated or truncated:
            end_reason = "truncated" if truncated and not terminated else "off_track"
            print(
                f"reward={reward:0.2f} laps={int(info['laps'])} progress={info['progress']:0.3f} "
                f"steps={int(info['steps'])} end={end_reason}"
            )
            env.reset(seed=args.seed)


if __name__ == "__main__":
    main()
