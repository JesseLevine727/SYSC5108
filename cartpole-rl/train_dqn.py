from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from cartpole_rl.dqn import QNetwork
from cartpole_rl.environment import CartPoleEnv


@dataclass
class DQNConfig:
    episodes: int = 1000
    max_steps: int = 500
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 100_000
    warmup_steps: int = 1_000
    target_update_every: int = 100
    train_every: int = 1
    gradient_clip: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 10_000
    eval_every_episodes: int = 25
    eval_episodes: int = 20
    solve_mean_return: float = 475.0
    solve_rate: float = 0.90
    seed: int = 7
    checkpoint_dir: str = "dqn_artifacts"
    device_name: str = "auto"
    hidden_dims: tuple[int, ...] = (128, 128)


PROJECT_DIR = Path(__file__).resolve().parent


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[batch],
            self.actions[batch],
            self.rewards[batch],
            self.next_states[batch],
            self.dones[batch],
        )


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims must contain at least one integer")
    return dims


def parse_args() -> DQNConfig:
    parser = argparse.ArgumentParser(description="Train a DQN CartPole agent.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--target-update-every", type=int, default=100)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10_000)
    parser.add_argument("--eval-every-episodes", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--solve-mean-return", type=float, default=475.0)
    parser.add_argument("--solve-rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="dqn_artifacts")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(128, 128))
    args = parser.parse_args()
    return DQNConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        target_update_every=args.target_update_every,
        train_every=args.train_every,
        gradient_clip=args.gradient_clip,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        eval_every_episodes=args.eval_every_episodes,
        eval_episodes=args.eval_episodes,
        solve_mean_return=args.solve_mean_return,
        solve_rate=args.solve_rate,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        device_name=args.device,
        hidden_dims=args.hidden_dims,
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


def epsilon_by_step(config: DQNConfig, step: int) -> float:
    progress = min(1.0, step / max(1, config.epsilon_decay_steps))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def save_checkpoint(
    path: Path,
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    config: DQNConfig,
    device: torch.device,
    episode: int,
    total_steps: int,
    best_metrics: dict[str, float],
) -> None:
    payload = {
        "q_network": q_network.state_dict(),
        "target_network": target_network.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "device": str(device),
        "episode": episode,
        "total_steps": total_steps,
        "best_metrics": best_metrics,
    }
    torch.save(payload, path)


def evaluate_policy(
    q_network: QNetwork,
    device: torch.device,
    config: DQNConfig,
    episodes: int,
    seed_base: int,
) -> dict[str, float | list[float]]:
    returns: list[float] = []
    lengths: list[int] = []
    q_network.eval()

    for episode_idx in range(episodes):
        env = CartPoleEnv(seed=seed_base + episode_idx, max_steps=config.max_steps)
        state, _ = env.reset(seed=seed_base + episode_idx)
        episode_return = 0.0
        for _ in range(config.max_steps):
            with torch.no_grad():
                q_values = q_network(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(torch.argmax(q_values, dim=1).item())
            step = env.step(action)
            state = step.observation
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

    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_DIR / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = CartPoleEnv(seed=config.seed, max_steps=config.max_steps)
    state_dim = env.observation_size
    action_dim = env.action_size
    q_network = QNetwork(state_dim, action_dim, hidden_dims=config.hidden_dims).to(device)
    target_network = QNetwork(state_dim, action_dim, hidden_dims=config.hidden_dims).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_capacity, state_dim)
    loss_fn = nn.SmoothL1Loss()

    total_steps = 0
    total_updates = 0
    best_metrics: dict[str, float] = {"mean_return": 0.0, "solved_rate": 0.0}
    best_selection_key = (0.0, 0.0)
    recent_returns: list[float] = []
    started_at = time.time()

    if device.type == "cuda":
        print(f"Training on GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Training on CPU")

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        episode_return = 0.0
        episode_losses: list[float] = []

        for _ in range(config.max_steps):
            epsilon = epsilon_by_step(config, total_steps)
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q_values = q_network(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = int(torch.argmax(q_values, dim=1).item())

            step = env.step(action)
            replay_buffer.add(state, action, step.reward, step.observation, step.terminated or step.truncated)
            state = step.observation
            episode_return += step.reward
            total_steps += 1

            if replay_buffer.size >= config.warmup_steps and total_steps % config.train_every == 0:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                    config.batch_size
                )
                states_tensor = torch.as_tensor(batch_states, dtype=torch.float32, device=device)
                actions_tensor = torch.as_tensor(batch_actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards_tensor = torch.as_tensor(batch_rewards, dtype=torch.float32, device=device)
                next_states_tensor = torch.as_tensor(batch_next_states, dtype=torch.float32, device=device)
                dones_tensor = torch.as_tensor(batch_dones, dtype=torch.float32, device=device)

                q_values = q_network(states_tensor).gather(1, actions_tensor).squeeze(1)
                with torch.no_grad():
                    next_actions = torch.argmax(q_network(next_states_tensor), dim=1, keepdim=True)
                    next_q_values = target_network(next_states_tensor).gather(1, next_actions).squeeze(1)
                    targets = rewards_tensor + config.gamma * next_q_values * (1.0 - dones_tensor)

                loss = loss_fn(q_values, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), config.gradient_clip)
                optimizer.step()
                episode_losses.append(float(loss.item()))
                total_updates += 1

                if total_updates % config.target_update_every == 0:
                    target_network.load_state_dict(q_network.state_dict())

            if step.terminated or step.truncated:
                break

        recent_returns.append(float(episode_return))

        should_evaluate = episode == 1 or episode % config.eval_every_episodes == 0 or episode == config.episodes
        if should_evaluate:
            eval_metrics = evaluate_policy(
                q_network,
                device,
                config,
                episodes=config.eval_episodes,
                seed_base=config.seed + 100_000 + episode * config.eval_episodes,
            )
            selection_key = (float(eval_metrics["solved_rate"]), float(eval_metrics["mean_return"]))
            recent_mean = float(np.mean(recent_returns[-25:]))
            mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            elapsed = time.time() - started_at
            print(
                f"episode={episode:4d} steps={total_steps:7d} "
                f"train_recent_mean={recent_mean:7.2f} "
                f"eval_mean={float(eval_metrics['mean_return']):7.2f} "
                f"eval_median={float(eval_metrics['median_return']):7.2f} "
                f"eval_solved={float(eval_metrics['solved_rate']):0.2f} "
                f"epsilon={epsilon_by_step(config, total_steps):0.3f} "
                f"loss={mean_loss:0.4f} sample_returns={eval_metrics['returns'][:5]} "
                f"elapsed={elapsed:6.1f}s"
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
                    checkpoint_dir / "dqn_best_model.pt",
                    q_network,
                    target_network,
                    optimizer,
                    config,
                    device,
                    episode,
                    total_steps,
                    best_metrics,
                )

            if (
                float(eval_metrics["mean_return"]) >= config.solve_mean_return
                and float(eval_metrics["solved_rate"]) >= config.solve_rate
            ):
                save_checkpoint(
                    checkpoint_dir / "dqn_last_model.pt",
                    q_network,
                    target_network,
                    optimizer,
                    config,
                    device,
                    episode,
                    total_steps,
                    best_metrics,
                )
                print(
                    "Solved CartPole. "
                    f"mean_return={float(eval_metrics['mean_return']):0.2f} "
                    f"solved_rate={float(eval_metrics['solved_rate']):0.2f}"
                )
                env.close()
                return

        if episode % max(1, config.eval_every_episodes * 2) == 0 or episode == config.episodes:
            save_checkpoint(
                checkpoint_dir / "dqn_last_model.pt",
                q_network,
                target_network,
                optimizer,
                config,
                device,
                episode,
                total_steps,
                best_metrics,
            )

    env.close()
    print(
        "DQN training finished. "
        f"best_mean_return={best_metrics['mean_return']:0.2f} "
        f"best_solved_rate={best_metrics['solved_rate']:0.2f}"
    )


if __name__ == "__main__":
    main()
