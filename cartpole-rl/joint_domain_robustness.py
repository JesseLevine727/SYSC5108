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

# One-at-a-time ranges measured previously for this solved DQN checkpoint.
STRICT_BOUNDS = {
    "gravity": (0.4883, 1.6221),
    "masscart": (0.2950, 2.4355),
    "masspole": (0.0200, 10.0000),
    "length": (0.6056, 1.3031),
    "force_mag": (0.4223, 5.5000),
    "tau": (0.1000, 1.4953),
}

PRACTICAL_BOUNDS = {
    "gravity": (0.4727, 2.2578),
    "masscart": (0.2459, 2.4980),
    "masspole": (0.0200, 10.0000),
    "length": (0.5823, 1.3148),
    "force_mag": (0.4129, 5.7188),
    "tau": (0.1000, 1.5762),
}


@dataclass
class DomainEval:
    mean_return: float
    median_return: float
    min_return: float
    solved_rate: float
    terminated_rate: float
    scales: dict[str, float]

    @property
    def strict_pass(self) -> bool:
        return self.terminated_rate == 0.0 and self.solved_rate == 1.0

    def practical_pass(self, cap: int) -> bool:
        return self.mean_return >= 0.95 * cap and self.solved_rate >= 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan joint random-domain robustness for the CartPole DQN policy.")
    parser.add_argument("--checkpoint", type=str, default="dqn_gpu_tuned/dqn_best_model.pt")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--domains", type=int, default=40)
    parser.add_argument("--episodes-per-domain", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
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


def sample_scales_from_unit(
    bounds: dict[str, tuple[float, float]],
    alpha: float,
    unit_samples: dict[str, float],
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for param_name, (lower_bound, upper_bound) in bounds.items():
        if alpha <= 0.0:
            scales[param_name] = 1.0
            continue
        lower = lower_bound**alpha
        upper = upper_bound**alpha
        sampled_log_scale = np.log(lower) + unit_samples[param_name] * (np.log(upper) - np.log(lower))
        scales[param_name] = float(np.exp(sampled_log_scale))
    return scales


def evaluate_domain(
    q_network: QNetwork,
    device: torch.device,
    scales: dict[str, float],
    episodes: int,
    seed_base: int,
    max_steps: int,
) -> DomainEval:
    env_kwargs = {param_name: BASE_ENV_PARAMS[param_name] * scale for param_name, scale in scales.items()}
    returns: list[float] = []
    terminated = 0
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
                break
        env.close()

    returns_np = np.asarray(returns, dtype=np.float32)
    return DomainEval(
        mean_return=float(np.mean(returns_np)),
        median_return=float(np.median(returns_np)),
        min_return=float(np.min(returns_np)),
        solved_rate=float(np.mean(returns_np >= max_steps)),
        terminated_rate=float(terminated / len(returns)),
        scales=scales,
    )


def summarize_domain_batch(
    q_network: QNetwork,
    device: torch.device,
    bounds: dict[str, tuple[float, float]],
    alpha: float,
    domain_unit_samples: list[dict[str, float]],
    episodes_per_domain: int,
    max_steps: int,
) -> tuple[dict[str, float], list[DomainEval]]:
    evaluations: list[DomainEval] = []
    for domain_idx, unit_samples in enumerate(domain_unit_samples):
        scales = sample_scales_from_unit(bounds, alpha, unit_samples)
        domain_eval = evaluate_domain(
            q_network,
            device,
            scales=scales,
            episodes=episodes_per_domain,
            seed_base=900_000 + domain_idx * 100,
            max_steps=max_steps,
        )
        evaluations.append(domain_eval)

    domain_means = np.asarray([evaluation.mean_return for evaluation in evaluations], dtype=np.float32)
    strict_pass_rate = float(np.mean([evaluation.strict_pass for evaluation in evaluations]))
    practical_pass_rate = float(np.mean([evaluation.practical_pass(max_steps) for evaluation in evaluations]))
    summary = {
        "alpha": alpha,
        "domains": float(len(domain_unit_samples)),
        "episodes_per_domain": float(episodes_per_domain),
        "mean_of_domain_means": float(np.mean(domain_means)),
        "median_of_domain_means": float(np.median(domain_means)),
        "p10_domain_mean": float(np.percentile(domain_means, 10)),
        "min_domain_mean": float(np.min(domain_means)),
        "strict_domain_pass_rate": strict_pass_rate,
        "practical_domain_pass_rate": practical_pass_rate,
    }
    return summary, evaluations


def print_summary(box_name: str, summary: dict[str, float]) -> None:
    print(
        f"{box_name} alpha={summary['alpha']:.2f} "
        f"mean_of_domain_means={summary['mean_of_domain_means']:.2f} "
        f"p10_domain_mean={summary['p10_domain_mean']:.2f} "
        f"min_domain_mean={summary['min_domain_mean']:.2f} "
        f"strict_pass_rate={summary['strict_domain_pass_rate']:.2f} "
        f"practical_pass_rate={summary['practical_domain_pass_rate']:.2f}"
    )


def print_failure_examples(box_name: str, evaluations: list[DomainEval], max_steps: int, limit: int = 3) -> None:
    ordered = sorted(evaluations, key=lambda evaluation: (evaluation.mean_return, evaluation.min_return))
    print(f"{box_name} worst_domains:")
    for index, evaluation in enumerate(ordered[:limit], start=1):
        scale_text = ", ".join(f"{name}={value:.3f}x" for name, value in evaluation.scales.items())
        print(
            f"  {index}. mean_return={evaluation.mean_return:.2f} "
            f"min_return={evaluation.min_return:.2f} solved_rate={evaluation.solved_rate:.2f} "
            f"strict={evaluation.strict_pass} practical={evaluation.practical_pass(max_steps)} "
            f"scales: {scale_text}"
        )


def first_alpha_where(rows: list[dict[str, float]], key: str, threshold: float, less_than: bool) -> float | None:
    for row in rows:
        value = row[key]
        if less_than and value < threshold:
            return float(row["alpha"])
        if not less_than and value >= threshold:
            return float(row["alpha"])
    return None


def last_alpha_where(rows: list[dict[str, float]], key: str, threshold: float) -> float | None:
    passing = [float(row["alpha"]) for row in rows if row[key] >= threshold]
    return passing[-1] if passing else None


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    device = pick_device(args.device)
    q_network, checkpoint = load_network(checkpoint_path, device)
    checkpoint_config = checkpoint.get("config", {})
    max_steps = int(args.max_steps or checkpoint_config.get("max_steps", 500))
    alphas = [round(value, 2) for value in np.linspace(0.0, 1.0, 11)]

    print("CartPole Joint Random-Domain Robustness")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print(f"domains_per_alpha: {args.domains}")
    print(f"episodes_per_domain: {args.episodes_per_domain}")
    print(f"max_steps: {max_steps}")

    all_results: dict[str, list[dict[str, float]]] = {}
    saved_batches: dict[tuple[str, float], list[DomainEval]] = {}
    rng = np.random.default_rng(args.seed)
    domain_unit_samples = [
        {param_name: float(rng.uniform()) for param_name in BASE_ENV_PARAMS}
        for _ in range(args.domains)
    ]

    for box_name, bounds in (("strict_box", STRICT_BOUNDS), ("practical_box", PRACTICAL_BOUNDS)):
        print(f"\n== {box_name} ==")
        rows: list[dict[str, float]] = []
        for alpha in alphas:
            summary, evaluations = summarize_domain_batch(
                q_network,
                device,
                bounds=bounds,
                alpha=alpha,
                domain_unit_samples=domain_unit_samples,
                episodes_per_domain=args.episodes_per_domain,
                max_steps=max_steps,
            )
            rows.append(summary)
            saved_batches[(box_name, alpha)] = evaluations
            print_summary(box_name, summary)
        all_results[box_name] = rows

        first_strict_drop = first_alpha_where(rows, "strict_domain_pass_rate", 1.0, less_than=True)
        last_strict_95 = last_alpha_where(rows, "strict_domain_pass_rate", 0.95)
        first_practical_drop = first_alpha_where(rows, "practical_domain_pass_rate", 1.0, less_than=True)
        last_practical_95 = last_alpha_where(rows, "practical_domain_pass_rate", 0.95)
        first_practical_major_fail = first_alpha_where(rows, "practical_domain_pass_rate", 0.50, less_than=True)

        print(
            f"{box_name} boundary_summary: "
            f"first_strict_drop={first_strict_drop} "
            f"last_strict_at_or_above_0.95={last_strict_95} "
            f"first_practical_drop={first_practical_drop} "
            f"last_practical_at_or_above_0.95={last_practical_95} "
            f"first_practical_below_0.50={first_practical_major_fail}"
        )

        interesting_alphas = sorted({value for value in [first_practical_drop, 1.0] if value is not None})
        for alpha in interesting_alphas:
            print_failure_examples(box_name, saved_batches[(box_name, alpha)], max_steps=max_steps)


if __name__ == "__main__":
    main()
