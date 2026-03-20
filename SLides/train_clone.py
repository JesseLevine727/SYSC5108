from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from flappy_rl.environment import FlappyBirdEnv
from flappy_rl.expert import expert_action
from flappy_rl.model import DQN


@dataclass
class CloneConfig:
    expert_episodes: int = 250
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-3
    max_steps: int = 4_000
    eval_every: int = 5
    eval_episodes: int = 20
    seed: int = 7
    checkpoint_dir: str = "artifacts"
    device_name: str = "auto"


def parse_args() -> CloneConfig:
    parser = argparse.ArgumentParser(description="Train a Flappy Bird policy by cloning a strong controller.")
    parser.add_argument("--expert-episodes", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=4_000)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    return CloneConfig(
        expert_episodes=args.expert_episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
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


def collect_expert_data(config: CloneConfig) -> tuple[np.ndarray, np.ndarray]:
    env = FlappyBirdEnv(seed=config.seed, max_steps=config.max_steps)
    states: list[np.ndarray] = []
    actions: list[int] = []

    for episode in range(config.expert_episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        while True:
            action = expert_action(observation)
            states.append(observation.copy())
            actions.append(action)
            step = env.step(action)
            observation = step.observation
            if step.terminated or step.truncated:
                break

    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def evaluate_policy(model: DQN, device: torch.device, config: CloneConfig) -> tuple[float, list[int]]:
    env = FlappyBirdEnv(seed=config.seed + 10_000, max_steps=config.max_steps)
    scores: list[int] = []

    for episode in range(config.eval_episodes):
        observation, _ = env.reset(seed=config.seed + 20_000 + episode)
        while True:
            with torch.no_grad():
                state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(model(state), dim=1).item())
            step = env.step(action)
            observation = step.observation
            if step.terminated or step.truncated:
                scores.append(int(step.info["score"]))
                break

    return float(np.mean(scores)), scores


def save_checkpoint(
    path: Path,
    model: DQN,
    optimizer: torch.optim.Optimizer,
    config: CloneConfig,
    device: torch.device,
    epoch: int,
    best_average_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": "expert_clone",
            "q_network": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "device": str(device),
            "epoch": epoch,
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

    states_np, actions_np = collect_expert_data(config)
    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.long, device=device)

    model = DQN(state_dim=4, action_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    checkpoint_dir = Path(config.checkpoint_dir)
    best_average_score = float("-inf")

    print(f"Collected {states.shape[0]} expert states from {config.expert_episodes} episodes.")

    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(states.shape[0], device=device)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, states.shape[0], config.batch_size):
            batch_indices = permutation[start : start + config.batch_size]
            logits = model(states[batch_indices])
            loss = nn.functional.cross_entropy(logits, actions[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1

        average_loss = epoch_loss / max(1, batches)

        if epoch % config.eval_every == 0 or epoch == 1 or epoch == config.epochs:
            average_score, sample_scores = evaluate_policy(model, device, config)
            print(
                f"epoch={epoch:3d} loss={average_loss:0.4f} "
                f"eval_mean_score={average_score:0.2f} sample_scores={sample_scores[:5]}"
            )

            if average_score > best_average_score:
                best_average_score = average_score
                save_checkpoint(
                    checkpoint_dir / "best_model.pt",
                    model,
                    optimizer,
                    config,
                    device,
                    epoch,
                    best_average_score,
                )
                save_checkpoint(
                    checkpoint_dir / "expert_clone.pt",
                    model,
                    optimizer,
                    config,
                    device,
                    epoch,
                    best_average_score,
                )

    save_checkpoint(
        checkpoint_dir / "last_clone.pt",
        model,
        optimizer,
        config,
        device,
        config.epochs,
        best_average_score,
    )
    print(f"Clone training finished. Best eval mean score: {best_average_score:0.2f}")


if __name__ == "__main__":
    main()

