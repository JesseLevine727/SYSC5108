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

from cartpole_rl.environment import CartPoleEnv
from cartpole_rl.policy import ActorCritic


@dataclass
class PPOConfig:
    updates: int = 80
    num_envs: int = 16
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    max_steps: int = 500
    eval_every: int = 5
    eval_episodes: int = 20
    seed: int = 7
    checkpoint_dir: str = "artifacts"
    device_name: str = "auto"
    hidden_dims: tuple[int, ...] = (128, 128)
    learning_rate_final_scale: float = 0.2
    entropy_coef_final_scale: float = 0.1
    init_from_checkpoint: str | None = None


PROJECT_DIR = Path(__file__).resolve().parent


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims must contain at least one integer")
    return dims


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train a PPO CartPole agent.")
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(128, 128))
    parser.add_argument("--learning-rate-final-scale", type=float, default=0.2)
    parser.add_argument("--entropy-coef-final-scale", type=float, default=0.1)
    parser.add_argument("--init-from-checkpoint", type=str, default=None)
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
        hidden_dims=args.hidden_dims,
        learning_rate_final_scale=args.learning_rate_final_scale,
        entropy_coef_final_scale=args.entropy_coef_final_scale,
        init_from_checkpoint=args.init_from_checkpoint or None,
    )


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def annealed_value(initial: float, final_scale: float, progress: float) -> float:
    return initial * (1.0 - progress * (1.0 - final_scale))


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    device: torch.device,
    update: int,
    total_steps: int,
    best_metrics: dict[str, float],
) -> None:
    payload = {
        "policy_network": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "device": str(device),
        "update": update,
        "total_steps": total_steps,
        "best_metrics": best_metrics,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, dict[str, float]]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["policy_network"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    update = int(checkpoint.get("update", 0))
    total_steps = int(checkpoint.get("total_steps", 0))
    best_metrics = dict(checkpoint.get("best_metrics", {"mean_return": 0.0, "solved_rate": 0.0}))
    return update, total_steps, best_metrics


def evaluate_policy(
    model: ActorCritic,
    device: torch.device,
    config: PPOConfig,
    episodes: int,
    seed_base: int,
) -> dict[str, float | list[float]]:
    returns: list[float] = []
    lengths: list[int] = []

    model.eval()
    for episode_idx in range(episodes):
        env = CartPoleEnv(seed=seed_base + episode_idx, max_steps=config.max_steps)
        observation, _ = env.reset(seed=seed_base + episode_idx)
        episode_return = 0.0

        for _ in range(config.max_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(obs_tensor)
                action = int(torch.argmax(logits, dim=1).item())

            step = env.step(action)
            observation = step.observation
            episode_return += step.reward
            if step.terminated or step.truncated:
                returns.append(episode_return)
                lengths.append(int(step.info["steps"]))
                break

        env.close()

    solved_rate = float(np.mean([value >= config.max_steps for value in returns])) if returns else 0.0
    return {
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "median_return": float(np.median(returns)) if returns else 0.0,
        "min_return": float(np.min(returns)) if returns else 0.0,
        "max_return": float(np.max(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "solved_rate": solved_rate,
        "returns": returns,
    }


def main() -> None:
    config = parse_args()
    device = pick_device(config.device_name)
    set_seed(config.seed)

    state_dim = CartPoleEnv.observation_size
    action_dim = CartPoleEnv.action_size
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_DIR / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = ActorCritic(state_dim, action_dim, hidden_dims=config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_metrics: dict[str, float] = {"mean_return": 0.0, "solved_rate": 0.0}
    best_selection_key = (0.0, 0.0)
    start_update = 1
    total_steps = 0

    if config.init_from_checkpoint:
        checkpoint_path = Path(config.init_from_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_DIR / checkpoint_path
        if checkpoint_path.exists():
            loaded_update, total_steps, best_metrics = load_checkpoint(checkpoint_path, model, optimizer, device)
            best_selection_key = (
                float(best_metrics.get("solved_rate", 0.0)),
                float(best_metrics.get("mean_return", 0.0)),
            )
            start_update = loaded_update + 1
            print(f"Resumed from {checkpoint_path} at update={loaded_update} steps={total_steps}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    envs = [CartPoleEnv(seed=config.seed + env_idx, max_steps=config.max_steps) for env_idx in range(config.num_envs)]
    observations = np.stack(
        [env.reset(seed=config.seed + env_idx)[0] for env_idx, env in enumerate(envs)],
        axis=0,
    ).astype(np.float32)
    episode_returns = np.zeros(config.num_envs, dtype=np.float32)
    recent_completed_returns: list[float] = []
    started_at = time.time()

    if device.type == "cuda":
        print(f"Training on GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Training on CPU")

    for update in range(start_update, config.updates + 1):
        progress = (update - 1) / max(1, config.updates - 1)
        current_lr = annealed_value(config.learning_rate, config.learning_rate_final_scale, progress)
        current_entropy_coef = annealed_value(config.entropy_coef, config.entropy_coef_final_scale, progress)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        obs_buf = np.zeros((config.rollout_steps, config.num_envs, state_dim), dtype=np.float32)
        actions_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int64)
        logprob_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        rewards_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        dones_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        values_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)

        model.train()
        for step_idx in range(config.rollout_steps):
            obs_buf[step_idx] = observations
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, values = model(obs_tensor)
                distribution = Categorical(logits=logits)
                actions = distribution.sample()
                logprobs = distribution.log_prob(actions)

            actions_np = actions.cpu().numpy()
            logprobs_np = logprobs.cpu().numpy()
            values_np = values.cpu().numpy()

            actions_buf[step_idx] = actions_np
            logprob_buf[step_idx] = logprobs_np
            values_buf[step_idx] = values_np

            next_observations = np.zeros_like(observations)
            for env_idx, env in enumerate(envs):
                step = env.step(int(actions_np[env_idx]))
                rewards_buf[step_idx, env_idx] = step.reward
                dones_buf[step_idx, env_idx] = float(step.terminated or step.truncated)
                total_steps += 1
                episode_returns[env_idx] += step.reward

                if step.terminated or step.truncated:
                    recent_completed_returns.append(float(episode_returns[env_idx]))
                    next_observation, _ = env.reset(seed=config.seed + total_steps + env_idx)
                    next_observations[env_idx] = next_observation
                    episode_returns[env_idx] = 0.0
                else:
                    next_observations[env_idx] = step.observation

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
                next_non_terminal = 1.0 - dones_buf[step_idx]
                next_value = values_buf[step_idx + 1]

            delta = rewards_buf[step_idx] + config.gamma * next_value * next_non_terminal - values_buf[step_idx]
            last_gae = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae
            advantages[step_idx] = last_gae

        returns = advantages + values_buf

        batch_observations = torch.as_tensor(obs_buf.reshape(-1, state_dim), dtype=torch.float32, device=device)
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
                loss = policy_loss + config.value_coef * value_loss - current_entropy_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                batch_count += 1

        should_evaluate = update == 1 or update % config.eval_every == 0 or update == config.updates
        if should_evaluate:
            eval_metrics = evaluate_policy(
                model,
                device,
                config,
                episodes=config.eval_episodes,
                seed_base=config.seed + 50_000 + update * config.eval_episodes,
            )
            selection_key = (
                float(eval_metrics["solved_rate"]),
                float(eval_metrics["mean_return"]),
            )
            mean_policy_loss = total_policy_loss / max(1, batch_count)
            mean_value_loss = total_value_loss / max(1, batch_count)
            mean_entropy = total_entropy / max(1, batch_count)
            elapsed = time.time() - started_at
            recent_train_mean = float(np.mean(recent_completed_returns[-50:])) if recent_completed_returns else 0.0
            print(
                f"update={update:4d} steps={total_steps:7d} "
                f"train_recent_mean={recent_train_mean:7.2f} "
                f"eval_mean={float(eval_metrics['mean_return']):7.2f} "
                f"eval_median={float(eval_metrics['median_return']):7.2f} "
                f"eval_solved={float(eval_metrics['solved_rate']):0.2f} "
                f"min={float(eval_metrics['min_return']):6.1f} max={float(eval_metrics['max_return']):6.1f} "
                f"policy_loss={mean_policy_loss:0.4f} value_loss={mean_value_loss:0.4f} "
                f"entropy={mean_entropy:0.4f} ent_coef={current_entropy_coef:0.5f} lr={current_lr:0.6f} "
                f"sample_returns={eval_metrics['returns'][:5]} elapsed={elapsed:6.1f}s"
            )

            if selection_key > best_selection_key:
                best_selection_key = selection_key
                best_metrics = {
                    "mean_return": float(eval_metrics["mean_return"]),
                    "median_return": float(eval_metrics["median_return"]),
                    "min_return": float(eval_metrics["min_return"]),
                    "max_return": float(eval_metrics["max_return"]),
                    "mean_length": float(eval_metrics["mean_length"]),
                    "solved_rate": float(eval_metrics["solved_rate"]),
                }
                save_checkpoint(
                    checkpoint_dir / "ppo_best_model.pt",
                    model,
                    optimizer,
                    config,
                    device,
                    update,
                    total_steps,
                    best_metrics,
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
                best_metrics,
            )

    print(
        "PPO training finished. "
        f"best_mean_return={best_metrics['mean_return']:0.2f} "
        f"best_solved_rate={best_metrics['solved_rate']:0.2f}"
    )


if __name__ == "__main__":
    main()
