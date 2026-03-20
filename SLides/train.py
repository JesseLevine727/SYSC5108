from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from flappy_rl.environment import FlappyBirdEnv
from flappy_rl.model import DQN
from flappy_rl.replay import ReplayBuffer


@dataclass
class TrainConfig:
    episodes: int = 800
    max_steps: int = 2_000
    batch_size: int = 256
    buffer_size: int = 100_000
    min_replay_size: int = 5_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 60_000
    target_update_every: int = 1_000
    train_every: int = 4
    seed: int = 7
    checkpoint_dir: str = "artifacts"
    hidden_device: str = "auto"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Flappy Bird.")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--min-replay-size", type=int, default=5_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=60_000)
    parser.add_argument("--target-update-every", type=int, default=1_000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    return TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_replay_size=args.min_replay_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_every=args.target_update_every,
        train_every=args.train_every,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        hidden_device=args.device,
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


def epsilon_for_step(step: int, config: TrainConfig) -> float:
    progress = min(1.0, step / max(1, config.epsilon_decay_steps))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def select_action(
    state: np.ndarray,
    epsilon: float,
    q_network: DQN,
    device: torch.device,
    action_size: int,
) -> int:
    if random.random() < epsilon:
        return random.randrange(action_size)

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def train_step(
    buffer: ReplayBuffer,
    q_network: DQN,
    target_network: DQN,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
) -> float:
    states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = q_network(states_t).gather(1, actions_t)
    with torch.no_grad():
        next_q_values = target_network(next_states_t).max(dim=1, keepdim=True).values
        targets = rewards_t + (1.0 - dones_t) * config.gamma * next_q_values

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def save_checkpoint(
    path: Path,
    q_network: DQN,
    target_network: DQN,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
    total_steps: int,
    episode: int,
    best_average_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q_network": q_network.state_dict(),
            "target_network": target_network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "device": str(device),
            "total_steps": total_steps,
            "episode": episode,
            "best_average_score": best_average_score,
        },
        path,
    )


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    device = pick_device(config.hidden_device)

    env = FlappyBirdEnv(seed=config.seed, max_steps=config.max_steps)
    q_network = DQN(env.observation_size, env.action_size).to(device)
    target_network = DQN(env.observation_size, env.action_size).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.buffer_size, seed=config.seed)

    checkpoint_dir = Path(config.checkpoint_dir)
    rolling_scores: deque[float] = deque(maxlen=50)
    best_average_score = float("-inf")
    total_steps = 0
    last_loss = float("nan")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")

    started_at = time.time()
    for episode in range(1, config.episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(config.max_steps):
            epsilon = epsilon_for_step(total_steps, config)
            action = select_action(state, epsilon, q_network, device, env.action_size)
            step = env.step(action)
            done = step.terminated or step.truncated

            replay_buffer.add(state, action, step.reward, step.observation, done)
            state = step.observation
            episode_reward += step.reward
            total_steps += 1

            if len(replay_buffer) >= config.min_replay_size and total_steps % config.train_every == 0:
                last_loss = train_step(replay_buffer, q_network, target_network, optimizer, config, device)

            if total_steps % config.target_update_every == 0:
                target_network.load_state_dict(q_network.state_dict())

            if done:
                break

        rolling_scores.append(float(step.info["score"]))
        average_score = float(np.mean(rolling_scores))
        if average_score > best_average_score:
            best_average_score = average_score
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                q_network,
                target_network,
                optimizer,
                config,
                device,
                total_steps,
                episode,
                best_average_score,
            )

        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - started_at
            print(
                f"episode={episode:4d} steps={total_steps:6d} "
                f"score={step.info['score']:3d} avg50={average_score:5.2f} "
                f"reward={episode_reward:7.2f} epsilon={epsilon:0.3f} "
                f"loss={last_loss:0.4f} elapsed={elapsed:6.1f}s"
            )

    save_checkpoint(
        checkpoint_dir / "last_model.pt",
        q_network,
        target_network,
        optimizer,
        config,
        device,
        total_steps,
        config.episodes,
        best_average_score,
    )
    print(f"Training finished. Best avg50 score: {best_average_score:0.2f}")


if __name__ == "__main__":
    main()

