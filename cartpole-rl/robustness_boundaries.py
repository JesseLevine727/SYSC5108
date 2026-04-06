from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cartpole_rl.dqn import QNetwork, infer_q_hidden_dims
from cartpole_rl.environment import CartPoleEnv


PROJECT_DIR = Path(__file__).resolve().parent

BASE_ENV_PARAMS = {
    "gravity": 9.8,
    "masscart": 1.0,
    "masspole": 0.1,
    "length": 0.5,
    "force_mag": 10.0,
    "tau": 0.02,
}

SEARCH_LIMITS = {
    "gravity": (0.05, 6.0),
    "masscart": (0.05, 10.0),
    "masspole": (0.02, 10.0),
    "length": (0.05, 6.0),
    "force_mag": (0.05, 6.0),
    "tau": (0.10, 5.0),
}


@dataclass
class EvalSummary:
    mean_return: float
    median_return: float
    min_return: float
    p05_return: float
    solved_rate: float
    terminated_rate: float
    truncated_rate: float

    @property
    def strict_pass(self) -> bool:
        return self.terminated_rate == 0.0 and self.truncated_rate == 1.0

    def practical_pass(self, cap: int) -> bool:
        return self.mean_return >= 0.95 * cap and self.solved_rate >= 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find robustness boundaries for the CartPole DQN policy.")
    parser.add_argument("--checkpoint", type=str, default="dqn_gpu_tuned/dqn_best_model.pt")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--binary-steps", type=int, default=10)
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_network(checkpoint_path: Path, device: torch.device) -> tuple[QNetwork, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_dims = infer_q_hidden_dims(checkpoint["q_network"])
    q_network = QNetwork(CartPoleEnv.observation_size, CartPoleEnv.action_size, hidden_dims=hidden_dims).to(device)
    q_network.load_state_dict(checkpoint["q_network"])
    q_network.eval()
    return q_network, checkpoint


def evaluate(
    q_network: QNetwork,
    device: torch.device,
    episodes: int,
    seed_base: int,
    max_steps: int,
    env_kwargs: dict[str, float],
) -> EvalSummary:
    returns: list[float] = []
    terminated = 0
    truncated = 0
    for episode_idx in range(episodes):
        env = CartPoleEnv(seed=seed_base + episode_idx, max_steps=max_steps, **env_kwargs)
        state, _ = env.reset(seed=seed_base + episode_idx)
        episode_return = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                q_values = q_network(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(torch.argmax(q_values, dim=1).item())
            step = env.step(action)
            state = step.observation
            episode_return += step.reward
            if step.terminated or step.truncated:
                returns.append(episode_return)
                terminated += int(step.terminated)
                truncated += int(step.truncated)
                break
        env.close()

    returns_np = np.asarray(returns, dtype=np.float32)
    return EvalSummary(
        mean_return=float(np.mean(returns_np)),
        median_return=float(np.median(returns_np)),
        min_return=float(np.min(returns_np)),
        p05_return=float(np.percentile(returns_np, 5)),
        solved_rate=float(np.mean(returns_np >= max_steps)),
        terminated_rate=float(terminated / len(returns)),
        truncated_rate=float(truncated / len(returns)),
    )


def env_kwargs_for(param_name: str, value: float) -> dict[str, float]:
    env_kwargs = dict(BASE_ENV_PARAMS)
    env_kwargs[param_name] = value
    return env_kwargs


def check_scale(
    q_network: QNetwork,
    device: torch.device,
    param_name: str,
    scale: float,
    episodes: int,
    max_steps: int,
    seed_base: int,
) -> EvalSummary:
    value = BASE_ENV_PARAMS[param_name] * scale
    return evaluate(
        q_network,
        device,
        episodes=episodes,
        seed_base=seed_base,
        max_steps=max_steps,
        env_kwargs=env_kwargs_for(param_name, value),
    )


def geometric_candidates(direction: str, limit: float) -> list[float]:
    if direction == "up":
        candidates = [1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        return [candidate for candidate in candidates if candidate <= limit]
    candidates = [0.95, 0.90, 0.80, 0.67, 0.50, 0.40, 0.33, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02]
    return [candidate for candidate in candidates if candidate >= limit]


def find_boundary(
    q_network: QNetwork,
    device: torch.device,
    param_name: str,
    direction: str,
    episodes: int,
    max_steps: int,
    binary_steps: int,
    criterion_name: str,
) -> dict[str, float | str | bool]:
    lower_limit, upper_limit = SEARCH_LIMITS[param_name]
    limit = upper_limit if direction == "up" else lower_limit
    candidates = geometric_candidates(direction, limit)
    seed_base = 700_000 if criterion_name == "strict" else 800_000

    def passes(summary: EvalSummary) -> bool:
        if criterion_name == "strict":
            return summary.strict_pass
        return summary.practical_pass(max_steps)

    last_pass_scale = 1.0
    last_pass_summary = check_scale(q_network, device, param_name, 1.0, episodes, max_steps, seed_base)
    first_fail_scale: float | None = None
    first_fail_summary: EvalSummary | None = None

    for candidate in candidates:
        summary = check_scale(q_network, device, param_name, candidate, episodes, max_steps, seed_base)
        if passes(summary):
            last_pass_scale = candidate
            last_pass_summary = summary
        else:
            first_fail_scale = candidate
            first_fail_summary = summary
            break

    if first_fail_scale is None:
        return {
            "direction": direction,
            "criterion": criterion_name,
            "bounded": False,
            "last_pass_scale": float(last_pass_scale),
            "last_pass_value": BASE_ENV_PARAMS[param_name] * last_pass_scale,
            "last_pass_mean_return": last_pass_summary.mean_return,
            "last_pass_solved_rate": last_pass_summary.solved_rate,
        }

    left = min(last_pass_scale, first_fail_scale)
    right = max(last_pass_scale, first_fail_scale)
    best_scale = last_pass_scale
    best_summary = last_pass_summary
    worst_scale = first_fail_scale
    worst_summary = first_fail_summary

    for _ in range(binary_steps):
        midpoint = (left + right) / 2.0
        summary = check_scale(q_network, device, param_name, midpoint, episodes, max_steps, seed_base)
        if passes(summary):
            best_scale = midpoint
            best_summary = summary
            if direction == "up":
                left = midpoint
            else:
                right = midpoint
        else:
            worst_scale = midpoint
            worst_summary = summary
            if direction == "up":
                right = midpoint
            else:
                left = midpoint

    return {
        "direction": direction,
        "criterion": criterion_name,
        "bounded": True,
        "last_pass_scale": float(best_scale),
        "last_pass_value": BASE_ENV_PARAMS[param_name] * best_scale,
        "last_pass_mean_return": best_summary.mean_return,
        "last_pass_solved_rate": best_summary.solved_rate,
        "first_fail_scale": float(worst_scale),
        "first_fail_value": BASE_ENV_PARAMS[param_name] * worst_scale,
        "first_fail_mean_return": worst_summary.mean_return,
        "first_fail_solved_rate": worst_summary.solved_rate,
    }


def print_result(param_name: str, result: dict[str, float | str | bool]) -> None:
    direction = str(result["direction"])
    criterion = str(result["criterion"])
    prefix = f"{param_name} {direction} {criterion}"
    if bool(result["bounded"]):
        print(
            f"{prefix}: pass<=scale {float(result['last_pass_scale']):.4f} "
            f"(value {float(result['last_pass_value']):.6f}), "
            f"fail>=scale {float(result['first_fail_scale']):.4f} "
            f"(value {float(result['first_fail_value']):.6f}), "
            f"pass_mean={float(result['last_pass_mean_return']):.2f}, "
            f"fail_mean={float(result['first_fail_mean_return']):.2f}"
        )
    else:
        print(
            f"{prefix}: no failure found up to tested limit; "
            f"pass_scale={float(result['last_pass_scale']):.4f} "
            f"(value {float(result['last_pass_value']):.6f}), "
            f"pass_mean={float(result['last_pass_mean_return']):.2f}"
        )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    device = pick_device(args.device)
    q_network, checkpoint = load_network(checkpoint_path, device)
    checkpoint_config = checkpoint.get("config", {})
    max_steps = int(args.max_steps or checkpoint_config.get("max_steps", 500))

    print("CartPole Robustness Boundaries")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print(f"episodes_per_check: {args.episodes}")
    print(f"max_steps: {max_steps}")

    baseline = check_scale(q_network, device, "gravity", 1.0, args.episodes, max_steps, 700_000)
    print(
        f"baseline: mean={baseline.mean_return:.2f} min={baseline.min_return:.2f} "
        f"solved_rate={baseline.solved_rate:.2f} strict_pass={baseline.strict_pass}"
    )

    for param_name in BASE_ENV_PARAMS:
        print(f"\n## {param_name}")
        for criterion_name in ("strict", "practical"):
            for direction in ("down", "up"):
                result = find_boundary(
                    q_network,
                    device,
                    param_name,
                    direction,
                    episodes=args.episodes,
                    max_steps=max_steps,
                    binary_steps=args.binary_steps,
                    criterion_name=criterion_name,
                )
                print_result(param_name, result)


if __name__ == "__main__":
    main()
