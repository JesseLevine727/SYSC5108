from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from flappy_rl.environment import FlappyBirdEnv
from flappy_rl.policy import ActorCritic


@dataclass
class PPOConfig:
    updates: int = 120
    num_envs: int = 32
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    max_steps: int = 20_000
    eval_every: int = 5
    eval_episodes: int = 20
    seed: int = 7
    checkpoint_dir: str = "artifacts"
    device_name: str = "auto"
    reward_alignment_gain: float = 1.2
    reward_center_bonus: float = 2.0
    reward_alive_bonus: float = 0.03


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train a pure RL Flappy Bird agent with PPO.")
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--reward-alignment-gain", type=float, default=1.2)
    parser.add_argument("--reward-center-bonus", type=float, default=2.0)
    parser.add_argument("--reward-alive-bonus", type=float, default=0.03)
    args = parser.parse_args()
    return PPOConfig(
        updates=args.updates,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        learning_rate=args.learning_rate,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
        reward_alignment_gain=args.reward_alignment_gain,
        reward_center_bonus=args.reward_center_bonus,
        reward_alive_bonus=args.reward_alive_bonus,
    )


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(seed: int, max_steps: int) -> FlappyBirdEnv:
    return FlappyBirdEnv(seed=seed, max_steps=max_steps)


def shape_reward(
    previous_observation: np.ndarray,
    next_observation: np.ndarray,
    env_reward: float,
    config: PPOConfig,
) -> float:
    alignment_gain = abs(float(previous_observation[3])) - abs(float(next_observation[3]))
    centered_bonus = max(0.0, 0.03 - abs(float(next_observation[3]))) * config.reward_center_bonus
    return env_reward + (config.reward_alignment_gain * alignment_gain) + centered_bonus + config.reward_alive_bonus


def evaluate_policy(
    model: ActorCritic,
    device: torch.device,
    config: PPOConfig,
    episodes: int | None = None,
) -> tuple[float, list[int]]:
    env = make_env(config.seed + 50_000, config.max_steps)
    scores: list[int] = []
    episode_count = episodes if episodes is not None else config.eval_episodes

    for episode in range(episode_count):
        observation, _ = env.reset(seed=config.seed + 60_000 + episode)
        while True:
            with torch.no_grad():
                state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(state)
                action = int(torch.argmax(logits, dim=1).item())
            step = env.step(action)
            observation = step.observation
            if step.terminated or step.truncated:
                scores.append(int(step.info["score"]))
                break

    return float(np.mean(scores)), scores


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    device: torch.device,
    update: int,
    total_steps: int,
    best_average_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": "ppo",
            "policy_network": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "device": str(device),
            "update": update,
            "total_steps": total_steps,
            "best_average_score": best_average_score,
        },
        path,
    )


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    device = pick_device(config.device_name)

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")

    envs = [make_env(config.seed + i, config.max_steps) for i in range(config.num_envs)]
    observations = np.stack([env.reset(seed=config.seed + i)[0] for i, env in enumerate(envs)])

    model = ActorCritic(state_dim=4, action_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_dir = Path(config.checkpoint_dir)

    best_average_score = float("-inf")
    total_steps = 0
    started_at = time.time()

    for update in range(1, config.updates + 1):
        obs_buf = np.zeros((config.rollout_steps, config.num_envs, 4), dtype=np.float32)
        actions_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int64)
        logprob_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        rewards_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        dones_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        values_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)

        for step_idx in range(config.rollout_steps):
            obs_buf[step_idx] = observations
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, values = model(obs_tensor)
                distribution = Categorical(logits=logits)
                actions = distribution.sample()
                logprobs = distribution.log_prob(actions)

            actions_np = actions.cpu().numpy()
            logprob_buf[step_idx] = logprobs.cpu().numpy()
            values_buf[step_idx] = values.cpu().numpy()
            actions_buf[step_idx] = actions_np

            next_observations = np.zeros_like(observations)
            for env_index, env in enumerate(envs):
                previous_observation = observations[env_index]
                step = env.step(int(actions_np[env_index]))
                next_observation = step.observation
                rewards_buf[step_idx, env_index] = shape_reward(
                    previous_observation,
                    next_observation,
                    step.reward,
                    config,
                )
                done = step.terminated or step.truncated
                dones_buf[step_idx, env_index] = float(done)
                total_steps += 1

                if done:
                    next_observations[env_index], _ = env.reset()
                else:
                    next_observations[env_index] = next_observation

            observations = next_observations

        with torch.no_grad():
            next_values = model(torch.as_tensor(observations, dtype=torch.float32, device=device))[1].cpu().numpy()

        advantages = np.zeros_like(rewards_buf)
        last_gae = np.zeros(config.num_envs, dtype=np.float32)
        for step_idx in reversed(range(config.rollout_steps)):
            if step_idx == config.rollout_steps - 1:
                next_non_terminal = 1.0 - dones_buf[step_idx]
                next_value = next_values
            else:
                next_non_terminal = 1.0 - dones_buf[step_idx + 1]
                next_value = values_buf[step_idx + 1]

            delta = rewards_buf[step_idx] + config.gamma * next_value * next_non_terminal - values_buf[step_idx]
            last_gae = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae
            advantages[step_idx] = last_gae

        returns = advantages + values_buf

        batch_observations = torch.as_tensor(obs_buf.reshape(-1, 4), dtype=torch.float32, device=device)
        batch_actions = torch.as_tensor(actions_buf.reshape(-1), dtype=torch.long, device=device)
        batch_logprobs = torch.as_tensor(logprob_buf.reshape(-1), dtype=torch.float32, device=device)
        batch_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        batch_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

        batch_indices = np.arange(batch_observations.shape[0])
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        batch_count = 0

        for _ in range(config.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, batch_observations.shape[0], config.minibatch_size):
                batch = batch_indices[start : start + config.minibatch_size]
                logits, values = model(batch_observations[batch])
                distribution = Categorical(logits=logits)
                new_logprobs = distribution.log_prob(batch_actions[batch])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_logprobs - batch_logprobs[batch])

                unclipped = ratio * batch_advantages[batch]
                clipped = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * batch_advantages[batch]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, batch_returns[batch])
                loss = policy_loss + (config.value_coef * value_loss) - (config.entropy_coef * entropy)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                batch_count += 1

        if update % config.eval_every == 0 or update == 1 or update == config.updates:
            average_score, sample_scores = evaluate_policy(model, device, config)
            elapsed = time.time() - started_at
            mean_policy_loss = total_policy_loss / max(1, batch_count)
            mean_value_loss = total_value_loss / max(1, batch_count)
            mean_entropy = total_entropy / max(1, batch_count)
            print(
                f"update={update:4d} steps={total_steps:7d} "
                f"score={average_score:6.2f} best={max(best_average_score, average_score):6.2f} "
                f"policy_loss={mean_policy_loss:0.4f} value_loss={mean_value_loss:0.4f} "
                f"entropy={mean_entropy:0.4f} sample={sample_scores[:5]} elapsed={elapsed:6.1f}s"
            )

            if average_score > best_average_score:
                best_average_score = average_score
                save_checkpoint(
                    checkpoint_dir / "ppo_best_model.pt",
                    model,
                    optimizer,
                    config,
                    device,
                    update,
                    total_steps,
                    best_average_score,
                )

        if update % max(1, config.eval_every * 2) == 0 or update == config.updates:
            save_checkpoint(
                checkpoint_dir / "ppo_last_model.pt",
                model,
                optimizer,
                config,
                device,
                update,
                total_steps,
                best_average_score,
            )

    print(f"PPO training finished. Best eval mean score: {best_average_score:0.2f}")


if __name__ == "__main__":
    main()
