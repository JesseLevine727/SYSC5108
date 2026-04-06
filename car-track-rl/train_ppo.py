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

from car_rl.environment import CarTrackEnv
from car_rl.normalization import RunningMeanStd
from car_rl.policy import ActorCritic, infer_actor_critic_hidden_dims, normalize_actor_critic_state_dict
from car_rl.wrappers import FrameStackWrapper


@dataclass
class PPOConfig:
    updates: int = 180
    num_envs: int = 24
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    max_steps: int = 1_500
    eval_every: int = 10
    eval_episodes: int = 10
    benchmark_episodes: int = 5
    seed: int = 7
    checkpoint_dir: str = "artifacts"
    device_name: str = "auto"
    hidden_dims: tuple[int, ...] = (256, 256)
    learning_rate_final_scale: float = 0.15
    entropy_coef_final_scale: float = 0.10
    frame_stack: int = 1
    randomize_start: bool = True
    train_track_pool: str = "curriculum"
    evaluation_track_pool: str = "all"
    selection_track_pool: str = "holdout"
    min_domain_randomization_scale: float = 0.10
    max_domain_randomization_scale: float = 1.00
    domain_randomization_ramp_fraction: float = 0.35
    evaluation_domain_randomization_scale: float = 1.00
    selection_domain_randomization_scale: float = 1.00
    selection_episodes: int = 12
    curriculum_switch_fraction: float = 0.30
    normalize_observations: bool = True
    observation_clip: float = 5.0
    solved_progress_threshold: float = 3.20
    solved_off_track_threshold: float = 0.05
    solved_streak: int = 2
    init_from_checkpoint: str | None = None


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims must contain at least one integer")
    return dims


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train a PPO car-driving agent on a simple closed track.")
    parser.add_argument("--updates", type=int, default=180)
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-clip-coef", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=1_500)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--benchmark-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(256, 256))
    parser.add_argument("--learning-rate-final-scale", type=float, default=0.15)
    parser.add_argument("--entropy-coef-final-scale", type=float, default=0.10)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--disable-randomize-start", action="store_true")
    parser.add_argument("--train-track-pool", choices=("curriculum", "train", "procedural", "all"), default="curriculum")
    parser.add_argument("--evaluation-track-pool", choices=("all", "holdout", "procedural", "train"), default="all")
    parser.add_argument("--selection-track-pool", choices=("all", "holdout", "procedural", "train"), default="holdout")
    parser.add_argument("--min-domain-randomization-scale", type=float, default=0.10)
    parser.add_argument("--max-domain-randomization-scale", type=float, default=1.00)
    parser.add_argument("--domain-randomization-ramp-fraction", type=float, default=0.35)
    parser.add_argument("--evaluation-domain-randomization-scale", type=float, default=1.00)
    parser.add_argument("--selection-domain-randomization-scale", type=float, default=1.00)
    parser.add_argument("--selection-episodes", type=int, default=12)
    parser.add_argument("--curriculum-switch-fraction", type=float, default=0.30)
    parser.add_argument("--disable-observation-normalization", action="store_true")
    parser.add_argument("--observation-clip", type=float, default=5.0)
    parser.add_argument("--solved-progress-threshold", type=float, default=3.20)
    parser.add_argument("--solved-off-track-threshold", type=float, default=0.05)
    parser.add_argument("--solved-streak", type=int, default=2)
    parser.add_argument("--init-from-checkpoint", type=str, default="")
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
        value_clip_coef=args.value_clip_coef,
        learning_rate=args.learning_rate,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        benchmark_episodes=args.benchmark_episodes,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
        hidden_dims=args.hidden_dims,
        learning_rate_final_scale=args.learning_rate_final_scale,
        entropy_coef_final_scale=args.entropy_coef_final_scale,
        frame_stack=args.frame_stack,
        randomize_start=not args.disable_randomize_start,
        train_track_pool=args.train_track_pool,
        evaluation_track_pool=args.evaluation_track_pool,
        selection_track_pool=args.selection_track_pool,
        min_domain_randomization_scale=args.min_domain_randomization_scale,
        max_domain_randomization_scale=args.max_domain_randomization_scale,
        domain_randomization_ramp_fraction=args.domain_randomization_ramp_fraction,
        evaluation_domain_randomization_scale=args.evaluation_domain_randomization_scale,
        selection_domain_randomization_scale=args.selection_domain_randomization_scale,
        selection_episodes=args.selection_episodes,
        curriculum_switch_fraction=args.curriculum_switch_fraction,
        normalize_observations=not args.disable_observation_normalization,
        observation_clip=args.observation_clip,
        solved_progress_threshold=args.solved_progress_threshold,
        solved_off_track_threshold=args.solved_off_track_threshold,
        solved_streak=args.solved_streak,
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(
    seed: int,
    max_steps: int,
    frame_stack: int,
    render_mode: str | None = None,
    randomize_start: bool = True,
    domain_randomization_scale: float = 1.0,
    track_pool: str = "train",
) -> CarTrackEnv | FrameStackWrapper:
    env = CarTrackEnv(
        seed=seed,
        max_steps=max_steps,
        render_mode=render_mode,
        randomize_start=randomize_start,
        domain_randomization_scale=domain_randomization_scale,
        track_pool=track_pool,
    )
    if frame_stack > 1:
        return FrameStackWrapper(env, num_frames=frame_stack)
    return env


def base_env(env: CarTrackEnv | FrameStackWrapper) -> CarTrackEnv:
    return env.env if isinstance(env, FrameStackWrapper) else env


def current_domain_randomization_scale(config: PPOConfig, progress_fraction: float) -> float:
    ramp_fraction = max(config.domain_randomization_ramp_fraction, 1e-6)
    ramp_progress = min(1.0, max(0.0, progress_fraction / ramp_fraction))
    scale_span = config.max_domain_randomization_scale - config.min_domain_randomization_scale
    return config.min_domain_randomization_scale + (scale_span * ramp_progress)


def current_train_track_pool(config: PPOConfig, progress_fraction: float) -> str:
    if config.train_track_pool != "curriculum":
        return config.train_track_pool
    return "procedural" if progress_fraction < config.curriculum_switch_fraction else "train"


def evaluate_policy(
    model: ActorCritic,
    device: torch.device,
    config: PPOConfig,
    state_dim: int,
    episodes: int,
    seed_base: int,
    observation_stats: RunningMeanStd | None,
    track_pool: str | None = None,
    domain_randomization_scale: float | None = None,
) -> dict[str, float | list[float]]:
    env = make_env(
        config.seed + seed_base,
        config.max_steps,
        frame_stack=config.frame_stack,
        randomize_start=config.randomize_start,
        domain_randomization_scale=(
            config.evaluation_domain_randomization_scale
            if domain_randomization_scale is None
            else domain_randomization_scale
        ),
        track_pool=config.evaluation_track_pool if track_pool is None else track_pool,
    )
    rewards: list[float] = []
    laps: list[float] = []
    progress: list[float] = []
    end_reasons: list[str] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=config.seed + seed_base + episode)
        episode_reward = 0.0
        while True:
            with torch.no_grad():
                model_input = normalize_observation(observation, observation_stats, config)
                state = torch.as_tensor(model_input[:state_dim], dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(state)
                action = int(torch.argmax(logits, dim=1).item())

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                rewards.append(float(episode_reward))
                laps.append(float(info["laps"]))
                progress.append(float(info["progress"]))
                end_reasons.append("truncated" if truncated and not terminated else "off_track")
                break

    env.close()
    rewards_np = np.asarray(rewards, dtype=np.float32)
    laps_np = np.asarray(laps, dtype=np.float32)
    progress_np = np.asarray(progress, dtype=np.float32)
    off_track_count = sum(reason == "off_track" for reason in end_reasons)
    return {
        "mean_reward": float(np.mean(rewards_np)),
        "median_reward": float(np.median(rewards_np)),
        "p25_reward": float(np.percentile(rewards_np, 25)),
        "mean_laps": float(np.mean(laps_np)),
        "mean_progress": float(np.mean(progress_np)),
        "off_track_rate": float(off_track_count / max(1, episodes)),
        "rewards": rewards,
        "progress": progress,
    }


def selection_key(
    validation_metrics: dict[str, float | list[float]],
    benchmark_metrics: dict[str, float | list[float]],
    selection_metrics: dict[str, float | list[float]],
) -> tuple[float, float, float]:
    return (
        -max(
            float(validation_metrics["off_track_rate"]),
            float(benchmark_metrics["off_track_rate"]),
            float(selection_metrics["off_track_rate"]),
        ),
        min(
            float(validation_metrics["mean_progress"]),
            float(benchmark_metrics["mean_progress"]),
            float(selection_metrics["mean_progress"]),
        ),
        min(
            float(validation_metrics["mean_reward"]),
            float(benchmark_metrics["mean_reward"]),
            float(selection_metrics["mean_reward"]),
        ),
    )


def is_solved(
    validation_metrics: dict[str, float | list[float]],
    benchmark_metrics: dict[str, float | list[float]],
    selection_metrics: dict[str, float | list[float]],
    config: PPOConfig,
) -> bool:
    return (
        float(validation_metrics["mean_progress"]) >= config.solved_progress_threshold
        and float(benchmark_metrics["mean_progress"]) >= config.solved_progress_threshold
        and float(selection_metrics["mean_progress"]) >= config.solved_progress_threshold
        and float(validation_metrics["off_track_rate"]) <= config.solved_off_track_threshold
        and float(benchmark_metrics["off_track_rate"]) <= config.solved_off_track_threshold
        and float(selection_metrics["off_track_rate"]) <= config.solved_off_track_threshold
    )


def config_value(config: PPOConfig | dict[str, object], key: str, default: object) -> object:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_observation(
    observation: np.ndarray,
    observation_stats: RunningMeanStd | None,
    config: PPOConfig | dict[str, object],
) -> np.ndarray:
    normalize_obs = bool(config_value(config, "normalize_observations", True))
    observation_clip = float(config_value(config, "observation_clip", 5.0))
    if not normalize_obs or observation_stats is None:
        return np.asarray(observation, dtype=np.float32)
    return observation_stats.normalize(observation, clip=observation_clip)


def initialize_from_checkpoint(model: ActorCritic, checkpoint_path: str) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "policy_network" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain a PPO policy_network.")

    policy_state_dict = normalize_actor_critic_state_dict(checkpoint["policy_network"])
    hidden_dims = infer_actor_critic_hidden_dims(policy_state_dict)
    expected_state_dim = int(policy_state_dict["backbone.linear_0.weight"].shape[1])
    actual_state_dim = int(model.backbone.linear_0.weight.shape[1])

    if hidden_dims != tuple(layer.out_features for layer in model.backbone if isinstance(layer, nn.Linear)):
        raise ValueError(f"Checkpoint hidden dims {hidden_dims} do not match model hidden dims.")
    if expected_state_dim != actual_state_dim:
        raise ValueError(f"Checkpoint state dim {expected_state_dim} does not match model state dim {actual_state_dim}.")

    model.load_state_dict(policy_state_dict)
    return int(checkpoint.get("total_steps", 0))


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    device: torch.device,
    update: int,
    total_steps: int,
    best_metrics: dict[str, dict[str, float]],
    observation_stats: RunningMeanStd | None,
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
            "best_metrics": best_metrics,
            "observation_normalization": observation_stats.state_dict() if observation_stats is not None else None,
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

    envs = [
        make_env(
            config.seed + env_index,
            config.max_steps,
            frame_stack=config.frame_stack,
            randomize_start=config.randomize_start,
            domain_randomization_scale=config.min_domain_randomization_scale,
            track_pool=config.train_track_pool,
        )
        for env_index in range(config.num_envs)
    ]
    observations = np.stack(
        [env.reset(seed=config.seed + env_index)[0] for env_index, env in enumerate(envs)],
        axis=0,
    )
    state_dim = observations.shape[1]
    action_dim = envs[0].action_size
    observation_stats = RunningMeanStd(shape=(state_dim,)) if config.normalize_observations else None
    if observation_stats is not None:
        observation_stats.update(observations)

    model = ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_dims=config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_dir = Path(config.checkpoint_dir)

    best_selection = (float("-inf"), float("-inf"), float("-inf"))
    best_metrics = {
        "validation": {
            "mean_reward": float("-inf"),
            "mean_progress": float("-inf"),
            "off_track_rate": float("inf"),
        },
        "benchmark": {
            "mean_reward": float("-inf"),
            "mean_progress": float("-inf"),
            "off_track_rate": float("inf"),
        },
        "selection": {
            "mean_reward": float("-inf"),
            "mean_progress": float("-inf"),
            "off_track_rate": float("inf"),
        },
    }
    total_steps = 0
    started_at = time.time()
    reset_counters = [0 for _ in envs]
    solved_streak = 0

    if config.init_from_checkpoint:
        total_steps = initialize_from_checkpoint(model, config.init_from_checkpoint)
        print(f"Initialized PPO from checkpoint {config.init_from_checkpoint}.")

    for update in range(1, config.updates + 1):
        progress_fraction = (update - 1) / max(1, config.updates - 1)
        domain_randomization_scale = current_domain_randomization_scale(config, progress_fraction)
        train_track_pool = current_train_track_pool(config, progress_fraction)
        current_lr = config.learning_rate * (1.0 - progress_fraction * (1.0 - config.learning_rate_final_scale))
        current_entropy_coef = config.entropy_coef * (
            1.0 - progress_fraction * (1.0 - config.entropy_coef_final_scale)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        for env in envs:
            base_env(env).set_domain_randomization(domain_randomization_scale)
            base_env(env).set_track_pool(train_track_pool)

        obs_buf = np.zeros((config.rollout_steps, config.num_envs, state_dim), dtype=np.float32)
        actions_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int64)
        logprob_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        rewards_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        dones_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        values_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        normalized_observations = normalize_observation(observations, observation_stats, config)

        for step_idx in range(config.rollout_steps):
            obs_buf[step_idx] = normalized_observations
            obs_tensor = torch.as_tensor(normalized_observations, dtype=torch.float32, device=device)
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
                next_observation, reward, terminated, truncated, _ = env.step(int(actions_np[env_index]))
                rewards_buf[step_idx, env_index] = reward
                done = terminated or truncated
                dones_buf[step_idx, env_index] = float(done)
                total_steps += 1

                if done:
                    reset_counters[env_index] += 1
                    next_seed = (
                        config.seed
                        + (update * 10_000)
                        + (env_index * 1_000)
                        + reset_counters[env_index]
                    )
                    next_observations[env_index] = env.reset(seed=next_seed)[0]
                else:
                    next_observations[env_index] = next_observation

            if observation_stats is not None:
                observation_stats.update(next_observations)
            observations = next_observations
            normalized_observations = normalize_observation(observations, observation_stats, config)

        with torch.no_grad():
            next_values = model(
                torch.as_tensor(normalized_observations, dtype=torch.float32, device=device)
            )[1].cpu().numpy()

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
        batch_values = torch.as_tensor(values_buf.reshape(-1), dtype=torch.float32, device=device)
        batch_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        batch_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

        batch_indices = np.arange(batch_observations.shape[0])
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        batch_count = 0
        early_stop_for_kl = False

        for _ in range(config.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, batch_observations.shape[0], config.minibatch_size):
                batch = batch_indices[start : start + config.minibatch_size]
                logits, values = model(batch_observations[batch])
                distribution = Categorical(logits=logits)
                new_logprobs = distribution.log_prob(batch_actions[batch])
                entropy = distribution.entropy().mean()
                log_ratio = new_logprobs - batch_logprobs[batch]
                ratio = torch.exp(log_ratio)

                unclipped = ratio * batch_advantages[batch]
                clipped = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * batch_advantages[batch]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_delta = values - batch_values[batch]
                value_pred_clipped = batch_values[batch] + torch.clamp(
                    value_delta,
                    -config.value_clip_coef,
                    config.value_clip_coef,
                )
                unclipped_value_loss = torch.square(values - batch_returns[batch])
                clipped_value_loss = torch.square(value_pred_clipped - batch_returns[batch])
                value_loss = 0.5 * torch.max(unclipped_value_loss, clipped_value_loss).mean()
                loss = policy_loss + (config.value_coef * value_loss) - (current_entropy_coef * entropy)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                approx_kl = float((((ratio - 1.0) - log_ratio).mean()).item())

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_approx_kl += approx_kl
                batch_count += 1
                if config.target_kl > 0.0 and approx_kl > (1.5 * config.target_kl):
                    early_stop_for_kl = True
                    break
            if early_stop_for_kl:
                break

        if update % config.eval_every == 0 or update == 1 or update == config.updates:
            validation_metrics = evaluate_policy(
                model,
                device,
                config,
                state_dim=state_dim,
                episodes=config.eval_episodes,
                seed_base=50_000,
                observation_stats=observation_stats,
            )
            benchmark_metrics = evaluate_policy(
                model,
                device,
                config,
                state_dim=state_dim,
                episodes=config.benchmark_episodes,
                seed_base=90_000,
                observation_stats=observation_stats,
            )
            selection_metrics = evaluate_policy(
                model,
                device,
                config,
                state_dim=state_dim,
                episodes=config.selection_episodes,
                seed_base=130_000,
                observation_stats=observation_stats,
                track_pool=config.selection_track_pool,
                domain_randomization_scale=config.selection_domain_randomization_scale,
            )
            elapsed = time.time() - started_at
            mean_policy_loss = total_policy_loss / max(1, batch_count)
            mean_value_loss = total_value_loss / max(1, batch_count)
            mean_entropy = total_entropy / max(1, batch_count)
            mean_approx_kl = total_approx_kl / max(1, batch_count)
            current_selection = selection_key(validation_metrics, benchmark_metrics, selection_metrics)
            solved_now = is_solved(validation_metrics, benchmark_metrics, selection_metrics, config)
            solved_streak = solved_streak + 1 if solved_now else 0

            print(
                f"update={update:4d} steps={total_steps:7d} "
                f"val_progress={validation_metrics['mean_progress']:0.3f} "
                f"val_reward={validation_metrics['mean_reward']:7.2f} "
                f"val_offtrack={validation_metrics['off_track_rate']:0.2f} "
                f"bench_progress={benchmark_metrics['mean_progress']:0.3f} "
                f"bench_reward={benchmark_metrics['mean_reward']:7.2f} "
                f"bench_offtrack={benchmark_metrics['off_track_rate']:0.2f} "
                f"sel_progress={selection_metrics['mean_progress']:0.3f} "
                f"sel_reward={selection_metrics['mean_reward']:7.2f} "
                f"sel_offtrack={selection_metrics['off_track_rate']:0.2f} "
                f"policy_loss={mean_policy_loss:0.4f} value_loss={mean_value_loss:0.4f} "
                f"entropy={mean_entropy:0.4f} approx_kl={mean_approx_kl:0.5f} "
                f"train_pool={train_track_pool} rand_scale={domain_randomization_scale:0.2f} "
                f"ent_coef={current_entropy_coef:0.5f} lr={current_lr:0.6f} "
                f"kl_stop={int(early_stop_for_kl)} solved_streak={solved_streak} elapsed={elapsed:6.1f}s"
            )

            if current_selection > best_selection:
                best_selection = current_selection
                best_metrics = {
                    "validation": {
                        "mean_reward": float(validation_metrics["mean_reward"]),
                        "mean_progress": float(validation_metrics["mean_progress"]),
                        "off_track_rate": float(validation_metrics["off_track_rate"]),
                    },
                    "benchmark": {
                        "mean_reward": float(benchmark_metrics["mean_reward"]),
                        "mean_progress": float(benchmark_metrics["mean_progress"]),
                        "off_track_rate": float(benchmark_metrics["off_track_rate"]),
                    },
                    "selection": {
                        "mean_reward": float(selection_metrics["mean_reward"]),
                        "mean_progress": float(selection_metrics["mean_progress"]),
                        "off_track_rate": float(selection_metrics["off_track_rate"]),
                    },
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
                    observation_stats,
                )

            if solved_now and solved_streak >= config.solved_streak:
                save_checkpoint(
                    checkpoint_dir / "ppo_solved_model.pt",
                    model,
                    optimizer,
                    config,
                    device,
                    update,
                    total_steps,
                    best_metrics,
                    observation_stats,
                )
                save_checkpoint(
                    checkpoint_dir / "ppo_last_model.pt",
                    model,
                    optimizer,
                    config,
                    device,
                    update,
                    total_steps,
                    best_metrics,
                    observation_stats,
                )
                print(
                    "Solved threshold reached. "
                    f"progress>={config.solved_progress_threshold:0.2f} "
                    f"off_track<={config.solved_off_track_threshold:0.2f}"
                )
                break

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
                observation_stats,
            )

    for env in envs:
        env.close()

    print(
        "PPO training finished. "
        f"val_progress={best_metrics['validation']['mean_progress']:0.3f} "
        f"bench_progress={best_metrics['benchmark']['mean_progress']:0.3f} "
        f"sel_progress={best_metrics['selection']['mean_progress']:0.3f} "
        f"val_reward={best_metrics['validation']['mean_reward']:0.2f}"
    )


if __name__ == "__main__":
    main()
