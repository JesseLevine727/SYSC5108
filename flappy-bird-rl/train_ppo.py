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
from flappy_rl.ppo_features import build_ppo_state
from flappy_rl.policy import ActorCritic, infer_actor_critic_hidden_dims, normalize_actor_critic_state_dict
from flappy_rl.wrappers import FrameStackWrapper


@dataclass
class PPOConfig:
    updates: int = 220
    num_envs: int = 32
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    learning_rate: float = 2e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.008
    max_grad_norm: float = 1.0
    max_steps: int = 100_000
    eval_every: int = 8
    eval_episodes: int = 40
    benchmark_episodes: int = 20
    seed: int = 7
    checkpoint_dir: str = "artifacts"
    device_name: str = "auto"
    reward_alignment_gain: float = 1.2
    reward_center_bonus: float = 2.0
    reward_alive_bonus: float = 0.03
    reward_second_pipe_gain: float = 0.0
    reward_second_pipe_center_bonus: float = 0.0
    reward_velocity_penalty: float = 0.03
    reward_flap_penalty: float = 0.003
    hidden_dims: tuple[int, ...] = (128, 128)
    learning_rate_final_scale: float = 0.05
    entropy_coef_final_scale: float = 0.1
    frame_stack: int = 1
    observation_dim: int = 4
    use_predicted_gap_error: bool = False
    train_randomization: bool = True
    train_pipe_gap_delta: float = 5.0
    train_pipe_speed_delta: float = 0.08
    train_gravity_delta: float = 0.012
    train_flap_velocity_delta: float = 0.12
    train_randomization_ramp_fraction: float = 0.35
    init_from_checkpoint: str | None = "artifacts/ppo_best_model.pt"
    hard_seed_replay_prob: float = 0.0
    hard_seed_pool_size: int = 8
    validation_seed_base: int = 60_000
    benchmark_seed_base: int = 90_000


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims must contain at least one integer")
    return dims


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train a pure RL Flappy Bird agent with PPO.")
    parser.add_argument("--updates", type=int, default=220)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.008)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--benchmark-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--reward-alignment-gain", type=float, default=1.2)
    parser.add_argument("--reward-center-bonus", type=float, default=2.0)
    parser.add_argument("--reward-alive-bonus", type=float, default=0.03)
    parser.add_argument("--reward-second-pipe-gain", type=float, default=0.0)
    parser.add_argument("--reward-second-pipe-center-bonus", type=float, default=0.0)
    parser.add_argument("--reward-velocity-penalty", type=float, default=0.03)
    parser.add_argument("--reward-flap-penalty", type=float, default=0.003)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(128, 128))
    parser.add_argument("--learning-rate-final-scale", type=float, default=0.05)
    parser.add_argument("--entropy-coef-final-scale", type=float, default=0.1)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--observation-dim", type=int, default=4)
    parser.add_argument("--use-predicted-gap-error", action="store_true")
    parser.add_argument("--disable-train-randomization", action="store_true")
    parser.add_argument("--train-pipe-gap-delta", type=float, default=5.0)
    parser.add_argument("--train-pipe-speed-delta", type=float, default=0.08)
    parser.add_argument("--train-gravity-delta", type=float, default=0.012)
    parser.add_argument("--train-flap-velocity-delta", type=float, default=0.12)
    parser.add_argument("--train-randomization-ramp-fraction", type=float, default=0.35)
    parser.add_argument("--init-from-checkpoint", type=str, default="artifacts/ppo_best_model.pt")
    parser.add_argument("--hard-seed-replay-prob", type=float, default=0.0)
    parser.add_argument("--hard-seed-pool-size", type=int, default=8)
    parser.add_argument("--validation-seed-base", type=int, default=60_000)
    parser.add_argument("--benchmark-seed-base", type=int, default=90_000)
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
        benchmark_episodes=args.benchmark_episodes,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
        reward_alignment_gain=args.reward_alignment_gain,
        reward_center_bonus=args.reward_center_bonus,
        reward_alive_bonus=args.reward_alive_bonus,
        reward_second_pipe_gain=args.reward_second_pipe_gain,
        reward_second_pipe_center_bonus=args.reward_second_pipe_center_bonus,
        reward_velocity_penalty=args.reward_velocity_penalty,
        reward_flap_penalty=args.reward_flap_penalty,
        hidden_dims=args.hidden_dims,
        learning_rate_final_scale=args.learning_rate_final_scale,
        entropy_coef_final_scale=args.entropy_coef_final_scale,
        frame_stack=args.frame_stack,
        observation_dim=args.observation_dim,
        use_predicted_gap_error=args.use_predicted_gap_error,
        train_randomization=not args.disable_train_randomization,
        train_pipe_gap_delta=args.train_pipe_gap_delta,
        train_pipe_speed_delta=args.train_pipe_speed_delta,
        train_gravity_delta=args.train_gravity_delta,
        train_flap_velocity_delta=args.train_flap_velocity_delta,
        train_randomization_ramp_fraction=args.train_randomization_ramp_fraction,
        init_from_checkpoint=args.init_from_checkpoint or None,
        hard_seed_replay_prob=args.hard_seed_replay_prob,
        hard_seed_pool_size=args.hard_seed_pool_size,
        validation_seed_base=args.validation_seed_base,
        benchmark_seed_base=args.benchmark_seed_base,
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


def make_env(seed: int, max_steps: int, frame_stack: int) -> FlappyBirdEnv | FrameStackWrapper:
    env = FlappyBirdEnv(seed=seed, max_steps=max_steps)
    if frame_stack > 1:
        return FrameStackWrapper(env, num_frames=frame_stack)
    return env


def base_env(env: FlappyBirdEnv | FrameStackWrapper) -> FlappyBirdEnv:
    return env.env if isinstance(env, FrameStackWrapper) else env


def apply_training_randomization(
    env: FlappyBirdEnv | FrameStackWrapper,
    rng: random.Random,
    config: PPOConfig,
    update_progress: float,
) -> None:
    game_env = base_env(env)
    if not config.train_randomization:
        game_env.pipe_gap = 150
        game_env.pipe_speed = 3.2
        game_env.gravity = 0.45
        game_env.flap_velocity = -7.5
        return

    ramp_fraction = max(config.train_randomization_ramp_fraction, 1e-6)
    randomization_scale = min(1.0, max(0.0, update_progress / ramp_fraction))

    game_env.pipe_gap = 150 + rng.uniform(
        -config.train_pipe_gap_delta * randomization_scale,
        config.train_pipe_gap_delta * randomization_scale,
    )
    game_env.pipe_speed = 3.2 + rng.uniform(
        -config.train_pipe_speed_delta * randomization_scale,
        config.train_pipe_speed_delta * randomization_scale,
    )
    game_env.gravity = 0.45 + rng.uniform(
        -config.train_gravity_delta * randomization_scale,
        config.train_gravity_delta * randomization_scale,
    )
    game_env.flap_velocity = -7.5 + rng.uniform(
        -config.train_flap_velocity_delta * randomization_scale,
        config.train_flap_velocity_delta * randomization_scale,
    )


def choose_training_seed(
    config: PPOConfig,
    training_rng: random.Random,
    hard_seeds: list[int],
    fallback_seed: int,
) -> int:
    if hard_seeds and training_rng.random() < config.hard_seed_replay_prob:
        return int(training_rng.choice(hard_seeds))
    return fallback_seed


def shape_reward(
    previous_observation: np.ndarray,
    next_observation: np.ndarray,
    env_reward: float,
    action: int,
    config: PPOConfig,
) -> float:
    next_pipe_distance = float(np.clip(next_observation[2], 0.0, 1.0))
    next_pipe_progress = 1.0 - next_pipe_distance
    second_pipe_weight = float(np.clip((next_pipe_progress - 0.35) / 0.45, 0.0, 1.0))

    next_alignment_gain = abs(float(previous_observation[3])) - abs(float(next_observation[3]))
    second_alignment_gain = 0.0
    if previous_observation.shape[0] >= 6 and next_observation.shape[0] >= 6:
        second_alignment_gain = abs(float(previous_observation[5])) - abs(float(next_observation[5]))

    next_center_bonus = max(0.0, 0.03 - abs(float(next_observation[3]))) * config.reward_center_bonus
    second_center_bonus = (
        max(0.0, 0.04 - abs(float(next_observation[5]))) * config.reward_second_pipe_center_bonus * second_pipe_weight
        if next_observation.shape[0] >= 6
        else 0.0
    )

    velocity_penalty = config.reward_velocity_penalty * abs(float(next_observation[1]))
    flap_penalty = config.reward_flap_penalty if action == 1 else 0.0

    return (
        env_reward
        + (config.reward_alignment_gain * next_alignment_gain)
        + (config.reward_second_pipe_gain * second_pipe_weight * second_alignment_gain)
        + next_center_bonus
        + second_center_bonus
        + config.reward_alive_bonus
        - velocity_penalty
        - flap_penalty
    )


def evaluate_policy(
    model: ActorCritic,
    device: torch.device,
    config: PPOConfig,
    state_dim: int,
    episodes: int | None = None,
    seed_base: int | None = None,
) -> dict[str, float | list[int]]:
    env = make_env(config.seed + 50_000, config.max_steps, frame_stack=config.frame_stack)
    scores: list[int] = []
    seeds: list[int] = []
    end_reasons: list[str] = []
    episode_count = episodes if episodes is not None else config.eval_episodes
    expected_state_dim = min(state_dim, env.observation_size)
    episode_seed_base = config.validation_seed_base if seed_base is None else seed_base

    for episode in range(episode_count):
        episode_seed = config.seed + episode_seed_base + episode
        observation, _ = env.reset(seed=episode_seed)
        state_observation = build_ppo_state(
            observation,
            base_env(env),
            use_predicted_gap_error=config.use_predicted_gap_error,
        )
        while True:
            with torch.no_grad():
                state = torch.as_tensor(
                    state_observation[:expected_state_dim], dtype=torch.float32, device=device
                ).unsqueeze(0)
                logits, _ = model(state)
                action = int(torch.argmax(logits, dim=1).item())
            step = env.step(action)
            observation = step.observation
            state_observation = build_ppo_state(
                observation,
                base_env(env),
                use_predicted_gap_error=config.use_predicted_gap_error,
            )
            if step.terminated or step.truncated:
                scores.append(int(step.info["score"]))
                seeds.append(episode_seed)
                end_reasons.append("truncated" if step.truncated and not step.terminated else "crashed")
                break

    scores_np = np.asarray(scores, dtype=np.float32)
    truncated_count = sum(reason == "truncated" for reason in end_reasons)
    return {
        "mean": float(np.mean(scores_np)),
        "median": float(np.median(scores_np)),
        "p05": float(np.percentile(scores_np, 5)),
        "p10": float(np.percentile(scores_np, 10)),
        "p25": float(np.percentile(scores_np, 25)),
        "min": float(np.min(scores_np)),
        "max": float(np.max(scores_np)),
        "scores": scores,
        "seeds": seeds,
        "end_reasons": end_reasons,
        "truncated_count": float(truncated_count),
        "truncated_rate": float(truncated_count / max(1, episode_count)),
    }


def dual_selection_key(
    validation_metrics: dict[str, float | list[int]],
    benchmark_metrics: dict[str, float | list[int]],
) -> tuple[float, float, float, float, float, float]:
    return (
        min(float(validation_metrics["p05"]), float(benchmark_metrics["p05"])),
        min(float(validation_metrics["p10"]), float(benchmark_metrics["p10"])),
        min(float(validation_metrics["p25"]), float(benchmark_metrics["p25"])),
        min(float(validation_metrics["truncated_rate"]), float(benchmark_metrics["truncated_rate"])),
        0.5 * (float(validation_metrics["median"]) + float(benchmark_metrics["median"])),
        0.5 * (float(validation_metrics["mean"]) + float(benchmark_metrics["mean"])),
    )


def initialize_from_checkpoint(model: ActorCritic, checkpoint_path: str) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "policy_network" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain a PPO policy_network.")

    policy_state_dict = normalize_actor_critic_state_dict(checkpoint["policy_network"])
    hidden_dims = infer_actor_critic_hidden_dims(policy_state_dict)
    expected_state_dim = int(policy_state_dict["backbone.linear_0.weight"].shape[1])
    actual_state_dim = int(model.backbone.linear_0.weight.shape[1])

    if hidden_dims != tuple(layer.out_features for layer in model.backbone if isinstance(layer, nn.Linear)):
        raise ValueError(f"Checkpoint hidden dims {hidden_dims} do not match model hidden dims.")
    if expected_state_dim == actual_state_dim:
        model.load_state_dict(policy_state_dict)
        return int(checkpoint.get("total_steps", 0))

    if expected_state_dim == 4 and actual_state_dim == 5:
        current_state_dict = model.state_dict()
        for key, tensor in policy_state_dict.items():
            if key == "backbone.linear_0.weight":
                expanded = current_state_dict[key].clone()
                expanded.zero_()
                expanded[:, :expected_state_dim] = tensor
                current_state_dict[key] = expanded
            else:
                current_state_dict[key] = tensor
        model.load_state_dict(current_state_dict, strict=True)
        return int(checkpoint.get("total_steps", 0))

    raise ValueError(f"Checkpoint state dim {expected_state_dim} does not match model state dim {actual_state_dim}.")


def update_hard_seed_pool(
    scores: list[int],
    seeds: list[int],
    pool_size: int,
) -> list[int]:
    if pool_size <= 0:
        return []
    ranked = sorted(zip(scores, seeds), key=lambda item: (item[0], item[1]))
    selected = [seed for _, seed in ranked[:pool_size]]
    return selected


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
        },
        path,
    )


def main() -> None:
    config = parse_args()
    if config.use_predicted_gap_error and config.observation_dim < 5:
        config.observation_dim = 5
    seed_everything(config.seed)
    device = pick_device(config.device_name)

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")

    envs = [make_env(config.seed + i, config.max_steps, frame_stack=config.frame_stack) for i in range(config.num_envs)]
    training_rng = random.Random(config.seed + 10_000)
    initial_observations = []
    for env_index, env in enumerate(envs):
        apply_training_randomization(env, training_rng, config, update_progress=0.0)
        initial_observations.append(
            build_ppo_state(
                env.reset(seed=config.seed + env_index)[0],
                base_env(env),
                use_predicted_gap_error=config.use_predicted_gap_error,
            )
        )
    observations = np.stack(initial_observations)
    full_state_dim = observations.shape[1]
    state_dim = min(config.observation_dim, full_state_dim)
    base_state_dim = base_env(envs[0]).observation_size

    model = ActorCritic(state_dim=state_dim, action_dim=2, hidden_dims=config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_dir = Path(config.checkpoint_dir)

    best_selection_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_metrics = {
        "validation": {
            "mean": float("-inf"),
            "median": float("-inf"),
            "p25": float("-inf"),
            "p10": float("-inf"),
            "p05": float("-inf"),
            "truncated_rate": float("-inf"),
        },
        "benchmark": {
            "mean": float("-inf"),
            "median": float("-inf"),
            "p25": float("-inf"),
            "p10": float("-inf"),
            "p05": float("-inf"),
            "truncated_rate": float("-inf"),
        },
    }
    total_steps = 0
    started_at = time.time()
    hard_seed_pool: list[int] = []
    reset_counters = [0 for _ in envs]

    if config.init_from_checkpoint:
        total_steps = initialize_from_checkpoint(model, config.init_from_checkpoint)
        print(f"Initialized PPO from checkpoint {config.init_from_checkpoint}.")

    for update in range(1, config.updates + 1):
        progress = (update - 1) / max(1, config.updates - 1)
        current_lr = config.learning_rate * (
            1.0 - progress * (1.0 - config.learning_rate_final_scale)
        )
        current_entropy_coef = config.entropy_coef * (
            1.0 - progress * (1.0 - config.entropy_coef_final_scale)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        obs_buf = np.zeros((config.rollout_steps, config.num_envs, state_dim), dtype=np.float32)
        actions_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int64)
        logprob_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        rewards_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        dones_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        values_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)

        for step_idx in range(config.rollout_steps):
            obs_buf[step_idx] = observations[:, :state_dim]
            obs_tensor = torch.as_tensor(observations[:, :state_dim], dtype=torch.float32, device=device)
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
                previous_frame = previous_observation[-base_state_dim:]
                next_frame = next_observation[-base_state_dim:]
                rewards_buf[step_idx, env_index] = shape_reward(
                    previous_frame,
                    next_frame,
                    step.reward,
                    int(actions_np[env_index]),
                    config,
                )
                done = step.terminated or step.truncated
                dones_buf[step_idx, env_index] = float(done)
                total_steps += 1

                if done:
                    apply_training_randomization(env, training_rng, config, update_progress=progress)
                    reset_counters[env_index] += 1
                    fallback_seed = (
                        config.seed
                        + (update * 10_000)
                        + (env_index * 1_000)
                        + reset_counters[env_index]
                    )
                    next_seed = choose_training_seed(config, training_rng, hard_seed_pool, fallback_seed)
                    next_observations[env_index] = build_ppo_state(
                        env.reset(seed=next_seed)[0],
                        base_env(env),
                        use_predicted_gap_error=config.use_predicted_gap_error,
                    )
                else:
                    next_observations[env_index] = build_ppo_state(
                        next_observation,
                        base_env(env),
                        use_predicted_gap_error=config.use_predicted_gap_error,
                    )

            observations = next_observations

        with torch.no_grad():
            next_values = model(
                torch.as_tensor(observations[:, :state_dim], dtype=torch.float32, device=device)
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
                loss = policy_loss + (config.value_coef * value_loss) - (current_entropy_coef * entropy)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                batch_count += 1

        if update % config.eval_every == 0 or update == 1 or update == config.updates:
            validation_metrics = evaluate_policy(
                model,
                device,
                config,
                state_dim=state_dim,
                episodes=config.eval_episodes,
                seed_base=config.validation_seed_base,
            )
            benchmark_metrics = evaluate_policy(
                model,
                device,
                config,
                state_dim=state_dim,
                episodes=config.benchmark_episodes,
                seed_base=config.benchmark_seed_base,
            )
            hard_seed_pool = (
                update_hard_seed_pool(validation_metrics["scores"], validation_metrics["seeds"], config.hard_seed_pool_size)
                if config.hard_seed_replay_prob > 0.0
                else []
            )
            elapsed = time.time() - started_at
            mean_policy_loss = total_policy_loss / max(1, batch_count)
            mean_value_loss = total_value_loss / max(1, batch_count)
            mean_entropy = total_entropy / max(1, batch_count)
            selection_key = dual_selection_key(validation_metrics, benchmark_metrics)
            print(
                f"update={update:4d} steps={total_steps:7d} "
                f"val_mean={validation_metrics['mean']:6.2f} val_p05={validation_metrics['p05']:6.2f} "
                f"val_p10={validation_metrics['p10']:6.2f} val_p25={validation_metrics['p25']:6.2f} "
                f"val_trunc={validation_metrics['truncated_rate']:0.2f} "
                f"bench_mean={benchmark_metrics['mean']:6.2f} bench_p05={benchmark_metrics['p05']:6.2f} "
                f"bench_p10={benchmark_metrics['p10']:6.2f} bench_p25={benchmark_metrics['p25']:6.2f} "
                f"bench_trunc={benchmark_metrics['truncated_rate']:0.2f} "
                f"best_floor_p05={max(best_selection_key[0], selection_key[0]):6.2f} "
                f"policy_loss={mean_policy_loss:0.4f} value_loss={mean_value_loss:0.4f} "
                f"entropy={mean_entropy:0.4f} ent_coef={current_entropy_coef:0.5f} lr={current_lr:0.6f} "
                f"hard_seeds={hard_seed_pool[:5]} val_sample={validation_metrics['scores'][:5]} "
                f"bench_sample={benchmark_metrics['scores'][:5]} elapsed={elapsed:6.1f}s"
            )

            if selection_key > best_selection_key:
                best_selection_key = selection_key
                best_metrics = {
                    "validation": {
                        "mean": validation_metrics["mean"],
                        "median": validation_metrics["median"],
                        "p05": validation_metrics["p05"],
                        "p10": validation_metrics["p10"],
                        "p25": validation_metrics["p25"],
                        "truncated_rate": validation_metrics["truncated_rate"],
                    },
                    "benchmark": {
                        "mean": benchmark_metrics["mean"],
                        "median": benchmark_metrics["median"],
                        "p05": benchmark_metrics["p05"],
                        "p10": benchmark_metrics["p10"],
                        "p25": benchmark_metrics["p25"],
                        "truncated_rate": benchmark_metrics["truncated_rate"],
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
        f"val_p05={best_metrics['validation']['p05']:0.2f} "
        f"val_p10={best_metrics['validation']['p10']:0.2f} "
        f"bench_p05={best_metrics['benchmark']['p05']:0.2f} "
        f"bench_p10={best_metrics['benchmark']['p10']:0.2f}"
    )


if __name__ == "__main__":
    main()
