from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from car_rl.normalization import RunningMeanStd
from car_rl.policy import ActorCritic, infer_actor_critic_hidden_dims, normalize_actor_critic_state_dict
from train_ppo import make_env, normalize_observation, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a PPO car-track policy across multiple randomization regimes.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/ppo_solved_model.pt")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def evaluate_suite(
    *,
    label: str,
    randomization_scale: float,
    track_pool: str,
    checkpoint: dict[str, object],
    device: torch.device,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    checkpoint_config = checkpoint.get("config", {})
    frame_stack = int(checkpoint_config.get("frame_stack", 1))
    randomize_start = bool(checkpoint_config.get("randomize_start", True))

    env = make_env(
        seed=seed,
        max_steps=max_steps,
        frame_stack=frame_stack,
        render_mode=None,
        randomize_start=randomize_start,
        domain_randomization_scale=randomization_scale,
        track_pool=track_pool,
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
    off_track = 0

    for episode_index in range(episodes):
        observation, _ = env.reset(seed=seed + episode_index)
        episode_reward = 0.0
        while True:
            with torch.no_grad():
                model_input = normalize_observation(observation, observation_stats, checkpoint_config)
                state = torch.as_tensor(model_input[:expected_state_dim], dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = policy_network(state)
                action = int(torch.argmax(logits, dim=1).item())

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                rewards.append(float(episode_reward))
                laps.append(float(info["laps"]))
                progress.append(float(info["progress"]))
                off_track += int(terminated and not truncated)
                break

    env.close()
    return {
        "label": label,
        "randomization_scale": randomization_scale,
        "track_pool": track_pool,
        "mean_reward": float(np.mean(rewards)),
        "mean_laps": float(np.mean(laps)),
        "mean_progress": float(np.mean(progress)),
        "off_track_rate": float(off_track / max(1, episodes)),
    }


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)

    suites = [
        ("procedural_baseline", 0.0, "procedural"),
        ("mixed_generalization", 1.0, "all"),
        ("holdout_handcrafted", 1.0, "holdout"),
        ("holdout_stress", 1.25, "holdout"),
    ]
    results = [
        evaluate_suite(
            label=label,
            randomization_scale=scale,
            track_pool=track_pool,
            checkpoint=checkpoint,
            device=device,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed + (index * 10_000),
        )
        for index, (label, scale, track_pool) in enumerate(suites)
    ]

    for result in results:
        print(
            f"suite={result['label']} rand_scale={result['randomization_scale']:0.2f} "
            f"track_pool={result['track_pool']} "
            f"mean_reward={result['mean_reward']:7.2f} mean_laps={result['mean_laps']:0.2f} "
            f"mean_progress={result['mean_progress']:0.3f} off_track_rate={result['off_track_rate']:0.2f}"
        )


if __name__ == "__main__":
    main()
