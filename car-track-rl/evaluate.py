from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from car_rl.normalization import RunningMeanStd
from car_rl.policy import ActorCritic, infer_actor_critic_hidden_dims, normalize_actor_critic_state_dict
from train_ppo import make_env, normalize_observation, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO car-track agent.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/ppo_best_model.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--render", choices=("none", "human", "text"), default="none")
    parser.add_argument("--delay", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    checkpoint_config = checkpoint.get("config", {})

    frame_stack = int(checkpoint_config.get("frame_stack", 1))
    randomize_start = bool(checkpoint_config.get("randomize_start", True))
    render_mode = "human" if args.render == "human" else None

    env = make_env(
        seed=args.seed,
        max_steps=args.max_steps,
        frame_stack=frame_stack,
        render_mode=render_mode,
        randomize_start=randomize_start,
    )
    policy_state_dict = normalize_actor_critic_state_dict(checkpoint["policy_network"])
    hidden_dims = infer_actor_critic_hidden_dims(policy_state_dict)
    expected_state_dim = int(policy_state_dict["backbone.linear_0.weight"].shape[1])
    policy_network = ActorCritic(expected_state_dim, env.action_size, hidden_dims=hidden_dims).to(device)
    policy_network.load_state_dict(policy_state_dict)
    policy_network.eval()
    observation_norm_state = checkpoint.get("observation_normalization")
    observation_stats = None
    if observation_norm_state is not None:
        observation_stats = RunningMeanStd(shape=(expected_state_dim,))
        observation_stats.load_state_dict(observation_norm_state)

    rewards = []
    laps = []
    progress = []
    end_reasons = []

    if device.type == "cuda":
        print(f"Evaluating on GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Evaluating on CPU")

    for episode in range(1, args.episodes + 1):
        observation, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        while True:
            with torch.no_grad():
                model_input = normalize_observation(observation, observation_stats, checkpoint_config)
                state = torch.as_tensor(model_input[:expected_state_dim], dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = policy_network(state)
                action = int(torch.argmax(logits, dim=1).item())

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if args.render == "human":
                env.render()
                time.sleep(args.delay)
            elif args.render == "text":
                print("\033[2J\033[H", end="")
                print(env.render_text())
                time.sleep(args.delay)

            if terminated or truncated:
                end_reason = "truncated" if truncated and not terminated else "off_track"
                rewards.append(episode_reward)
                laps.append(float(info["laps"]))
                progress.append(float(info["progress"]))
                end_reasons.append(end_reason)
                print(
                    f"episode={episode} reward={episode_reward:7.2f} "
                    f"laps={int(info['laps'])} progress={info['progress']:0.3f} "
                    f"steps={int(info['steps'])} end={end_reason}"
                )
                break

    off_track_count = sum(reason == "off_track" for reason in end_reasons)
    print(
        f"mean_reward={np.mean(rewards):7.2f} mean_laps={np.mean(laps):0.2f} "
        f"mean_progress={np.mean(progress):0.3f} off_track={off_track_count}"
    )
    env.close()


if __name__ == "__main__":
    main()
