from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from flappy_rl.environment import FlappyBirdEnv
from flappy_rl.model import DQN
from flappy_rl.policy import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Flappy Bird DQN agent.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--render", choices=("none", "text", "human"), default="none")
    parser.add_argument("--delay", type=float, default=0.03)
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    render_mode = "human" if args.render == "human" else None
    env = FlappyBirdEnv(seed=args.seed, max_steps=args.max_steps, render_mode=render_mode)
    algorithm = checkpoint.get("algorithm", "dqn")
    q_network: DQN | None = None
    policy_network: ActorCritic | None = None

    if "policy_network" in checkpoint or algorithm == "ppo":
        policy_network = ActorCritic(env.observation_size, env.action_size).to(device)
        policy_network.load_state_dict(checkpoint["policy_network"])
        policy_network.eval()
    else:
        q_network = DQN(env.observation_size, env.action_size).to(device)
        q_network.load_state_dict(checkpoint["q_network"])
        q_network.eval()

    scores = []
    rewards = []
    end_reasons = []

    if device.type == "cuda":
        print(f"Evaluating on GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Evaluating on CPU")

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0

        for _ in range(args.max_steps):
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                if policy_network is not None:
                    logits, _ = policy_network(state_tensor)
                    action = int(torch.argmax(logits, dim=1).item())
                else:
                    assert q_network is not None
                    action = int(torch.argmax(q_network(state_tensor), dim=1).item())

            step = env.step(action)
            state = step.observation
            episode_reward += step.reward

            if args.render == "human":
                env.render()
                time.sleep(args.delay)
            elif args.render == "text":
                print("\033[2J\033[H", end="")
                print(env.render_text())
                print(f"episode={episode} score={step.info['score']} reward={episode_reward:0.2f}")
                time.sleep(args.delay)

            if step.terminated or step.truncated:
                break

        scores.append(float(step.info["score"]))
        rewards.append(episode_reward)
        end_reason = "truncated" if step.truncated and not step.terminated else "crashed"
        end_reasons.append(end_reason)
        print(
            f"episode={episode} score={int(step.info['score'])} "
            f"reward={episode_reward:0.2f} steps={int(step.info['steps'])} end={end_reason}"
        )

    truncated_count = sum(reason == "truncated" for reason in end_reasons)
    crashed_count = sum(reason == "crashed" for reason in end_reasons)
    print(
        f"mean_score={np.mean(scores):0.2f} mean_reward={np.mean(rewards):0.2f} "
        f"crashed={crashed_count} truncated={truncated_count}"
    )
    env.close()


if __name__ == "__main__":
    main()
