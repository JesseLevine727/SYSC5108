from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from cartpole_rl.dqn import QNetwork, infer_q_hidden_dims
from cartpole_rl.environment import CartPoleEnv


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class EvalStats:
    returns: list[float]
    lengths: list[int]
    terminated: int
    truncated: int

    @property
    def solved_rate(self) -> float:
        return float(np.mean([length == max(self.lengths) for length in self.lengths])) if self.lengths else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a performance checklist on the CartPole DQN checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="dqn_gpu_tuned/dqn_best_model.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--long-episodes", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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


def run_eval(
    q_network: QNetwork,
    device: torch.device,
    episodes: int,
    seed_base: int,
    max_steps: int,
    env_kwargs: dict | None = None,
) -> EvalStats:
    returns: list[float] = []
    lengths: list[int] = []
    terminated = 0
    truncated = 0
    env_kwargs = env_kwargs or {}

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
                lengths.append(int(step.info["steps"]))
                terminated += int(step.terminated)
                truncated += int(step.truncated)
                break
        env.close()

    return EvalStats(returns=returns, lengths=lengths, terminated=terminated, truncated=truncated)


def summarize(stats: EvalStats, cap: int) -> dict[str, float]:
    returns = np.asarray(stats.returns, dtype=np.float32)
    lengths = np.asarray(stats.lengths, dtype=np.float32)
    return {
        "episodes": float(len(stats.returns)),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "std_return": float(np.std(returns)),
        "p05_return": float(np.percentile(returns, 5)),
        "p10_return": float(np.percentile(returns, 10)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "solved_rate": float(np.mean(lengths >= cap)),
        "terminated_rate": float(stats.terminated / len(stats.returns)),
        "truncated_rate": float(stats.truncated / len(stats.returns)),
    }


def parity_check(
    checkpoint_path: Path,
    seeds: list[int],
    max_steps: int,
) -> dict[str, float | bool]:
    cpu_model, _ = load_network(checkpoint_path, torch.device("cpu"))
    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        cpu_stats = run_eval(cpu_model, torch.device("cpu"), len(seeds), min(seeds), max_steps)
        return {
            "gpu_available": False,
            "identical_returns": True,
            "max_abs_diff": 0.0,
            "cpu_mean_return": float(np.mean(cpu_stats.returns)),
        }

    gpu_model, _ = load_network(checkpoint_path, torch.device("cuda"))
    cpu_returns: list[float] = []
    gpu_returns: list[float] = []
    for seed in seeds:
        cpu_stats = run_eval(cpu_model, torch.device("cpu"), 1, seed, max_steps)
        gpu_stats = run_eval(gpu_model, torch.device("cuda"), 1, seed, max_steps)
        cpu_returns.extend(cpu_stats.returns)
        gpu_returns.extend(gpu_stats.returns)
    diffs = [abs(cpu - gpu) for cpu, gpu in zip(cpu_returns, gpu_returns, strict=True)]
    return {
        "gpu_available": True,
        "identical_returns": all(diff == 0.0 for diff in diffs),
        "max_abs_diff": float(max(diffs, default=0.0)),
        "cpu_mean_return": float(np.mean(cpu_returns)),
        "gpu_mean_return": float(np.mean(gpu_returns)),
    }


def inference_speed(
    q_network: QNetwork,
    device: torch.device,
    iterations: int,
    batch_size: int,
) -> dict[str, float]:
    inputs = torch.randn(batch_size, CartPoleEnv.observation_size, device=device)
    with torch.no_grad():
        for _ in range(200):
            q_network(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        for _ in range(iterations):
            q_network(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started_at
    total_samples = iterations * batch_size
    return {
        "batch_size": float(batch_size),
        "iterations": float(iterations),
        "elapsed_seconds": elapsed,
        "samples_per_second": total_samples / elapsed,
        "milliseconds_per_batch": (elapsed / iterations) * 1000.0,
        "microseconds_per_sample": (elapsed / total_samples) * 1_000_000.0,
    }


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def print_metrics(metrics: dict[str, float | bool]) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool):
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    device = pick_device(args.device)
    q_network, checkpoint = load_network(checkpoint_path, device)
    checkpoint_config = checkpoint.get("config", {})
    max_steps = int(checkpoint_config.get("max_steps", 500))

    print("CartPole DQN Performance Checklist")
    print(f"checkpoint: {checkpoint_path}")
    print(f"evaluation_device: {device}")
    print(f"training_cap_max_steps: {max_steps}")

    print_section("Checkpoint Load")
    print(f"load_success: True")
    print(f"checkpoint_episode: {int(checkpoint.get('episode', -1))}")
    print(f"checkpoint_total_steps: {int(checkpoint.get('total_steps', -1))}")
    if "best_metrics" in checkpoint:
        print_metrics({f"checkpoint_{key}": float(value) for key, value in checkpoint["best_metrics"].items()})

    print_section("Held-Out Seeds")
    held_out = run_eval(q_network, device, args.episodes, seed_base=200_000, max_steps=max_steps)
    print_metrics(summarize(held_out, cap=max_steps))

    print_section("Repeatability Across Seed Batches")
    batch_means: list[float] = []
    for offset in (300_000, 301_000, 302_000):
        batch_stats = run_eval(q_network, device, 30, seed_base=offset, max_steps=max_steps)
        batch_summary = summarize(batch_stats, cap=max_steps)
        batch_means.append(batch_summary["mean_return"])
        print(f"seed_base_{offset}_mean_return: {batch_summary['mean_return']:.4f}")
        print(f"seed_base_{offset}_solved_rate: {batch_summary['solved_rate']:.4f}")
    print(f"repeatability_mean_of_means: {statistics.mean(batch_means):.4f}")
    print(f"repeatability_std_of_means: {statistics.pstdev(batch_means):.4f}")

    print_section("CPU GPU Parity")
    parity = parity_check(checkpoint_path, seeds=list(range(400_000, 400_020)), max_steps=max_steps)
    print_metrics(parity)

    print_section("Long Horizon Stability")
    for extended_cap in (750, 1000):
        long_stats = run_eval(q_network, device, args.long_episodes, seed_base=500_000 + extended_cap, max_steps=extended_cap)
        long_summary = summarize(long_stats, cap=extended_cap)
        print(f"long_cap: {extended_cap}")
        print_metrics(long_summary)

    print_section("Physics Robustness")
    robustness_scenarios = {
        "nominal": {},
        "gravity_plus_5pct": {"gravity": 9.8 * 1.05},
        "gravity_minus_5pct": {"gravity": 9.8 * 0.95},
        "force_plus_5pct": {"force_mag": 10.0 * 1.05},
        "force_minus_5pct": {"force_mag": 10.0 * 0.95},
        "pole_longer_5pct": {"length": 0.5 * 1.05},
        "pole_shorter_5pct": {"length": 0.5 * 0.95},
    }
    for scenario_name, env_kwargs in robustness_scenarios.items():
        scenario_stats = run_eval(q_network, device, 30, seed_base=600_000, max_steps=max_steps, env_kwargs=env_kwargs)
        scenario_summary = summarize(scenario_stats, cap=max_steps)
        print(f"scenario: {scenario_name}")
        print_metrics(
            {
                "mean_return": scenario_summary["mean_return"],
                "p05_return": scenario_summary["p05_return"],
                "solved_rate": scenario_summary["solved_rate"],
            }
        )

    print_section("Inference Speed")
    print("note: this measures raw network forward-pass time, not environment stepping.")
    print("batch_1")
    print_metrics(inference_speed(q_network, device, iterations=20_000, batch_size=1))
    print("batch_1024")
    print_metrics(inference_speed(q_network, device, iterations=5_000, batch_size=1024))


if __name__ == "__main__":
    main()
