from __future__ import annotations

import argparse

import pygame

from flappy_rl.environment import FlappyBirdEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play the pygame Flappy Bird environment manually.")
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = FlappyBirdEnv(seed=args.seed, max_steps=args.max_steps, render_mode="human")
    _, _ = env.reset()

    try:
        running = True
        while running:
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    action = 1

            step = env.step(action)
            env.render()

            if step.terminated or step.truncated:
                print(f"Game over. score={step.info['score']}")
                _, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
