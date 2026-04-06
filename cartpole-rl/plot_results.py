from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_DIR / "plots"


EXPERIMENTS = [
    {
        "label": "PPO Smoke\nCPU",
        "family": "PPO",
        "mean_return": 99.38,
        "solved_rate": 0.00,
        "notes": "longer CPU test",
    },
    {
        "label": "PPO Default\nGPU",
        "family": "PPO",
        "mean_return": 465.95,
        "solved_rate": 0.70,
        "notes": "best validation",
    },
    {
        "label": "PPO Tuned\nGPU",
        "family": "PPO",
        "mean_return": 455.10,
        "solved_rate": 0.75,
        "notes": "reduced LR/entropy",
    },
    {
        "label": "DQN Initial\nGPU",
        "family": "DQN",
        "mean_return": 197.80,
        "solved_rate": 0.00,
        "notes": "first DQN pass",
    },
    {
        "label": "DQN Tuned\nGPU",
        "family": "DQN",
        "mean_return": 500.00,
        "solved_rate": 1.00,
        "notes": "final solved model",
    },
]

BOUNDARY_DATA = {
    "strict": {
        "gravity": (0.4883, 1.6221),
        "masscart": (0.2950, 2.4355),
        "masspole": (0.0200, 10.0000),
        "length": (0.6056, 1.3031),
        "force_mag": (0.4223, 5.5000),
        "tau": (0.1000, 1.4953),
    },
    "practical": {
        "gravity": (0.4727, 2.2578),
        "masscart": (0.2459, 2.4980),
        "masspole": (0.0200, 10.0000),
        "length": (0.5823, 1.3148),
        "force_mag": (0.4129, 5.7188),
        "tau": (0.1000, 1.5762),
    },
}

JOINT_DOMAIN_DATA = {
    "Strict Box": {
        0.6: 0.9750,
        0.7: 0.9500,
        0.8: 0.9000,
        1.0: 0.8375,
    },
    "Practical Box": {
        0.6: 0.9625,
        0.7: 0.8625,
        0.8: 0.8125,
        1.0: 0.7750,
    },
}

INFERENCE_SPEED = {
    "Batch 1": 0.0304,
    "Batch 1024": 0.0252,
}

LONG_HORIZON = {
    "500-step cap": 1.00,
    "750-step cap": 1.00,
    "1000-step cap": 1.00,
}

COLORS = {
    "PPO": "#c25b36",
    "DQN": "#2d6a9f",
    "strict": "#23395d",
    "practical": "#d17a22",
    "strict_box": "#215968",
    "practical_box": "#9b4d16",
}


def style_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f6f3ee",
            "axes.facecolor": "#fffdf9",
            "axes.edgecolor": "#3a3a3a",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "font.size": 11,
            "grid.color": "#d9d4cc",
            "grid.alpha": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "savefig.facecolor": "#f6f3ee",
            "savefig.bbox": "tight",
        }
    )


def ensure_output_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_experiment_comparison() -> Path:
    labels = [item["label"] for item in EXPERIMENTS]
    mean_returns = [item["mean_return"] for item in EXPERIMENTS]
    solved_rates = [item["solved_rate"] for item in EXPERIMENTS]
    bar_colors = [COLORS[item["family"]] for item in EXPERIMENTS]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, mean_returns, color=bar_colors, width=0.68, edgecolor="#1f1f1f", linewidth=0.8)
    ax.set_title("CartPole Experiment Outcomes")
    ax.set_ylabel("Best Mean Return")
    ax.set_ylim(0, 540)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(500, color="#2f2f2f", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(len(labels) - 0.35, 507, "Solved Cap", ha="right", va="bottom", fontsize=10, color="#2f2f2f")

    ax2 = ax.twinx()
    ax2.plot(x, solved_rates, color="#111111", marker="o", linewidth=2.2)
    ax2.set_ylabel("Solved Rate")
    ax2.set_ylim(0, 1.05)

    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 9,
            f"{mean_returns[idx]:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202020",
        )
        ax2.text(
            x[idx],
            solved_rates[idx] + 0.04,
            f"{solved_rates[idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111111",
        )

    output_path = PLOTS_DIR / "experiment_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_boundary_ranges() -> Path:
    params = list(BOUNDARY_DATA["strict"].keys())
    y_positions = np.arange(len(params))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for idx, param in enumerate(params):
        strict_low, strict_high = BOUNDARY_DATA["strict"][param]
        practical_low, practical_high = BOUNDARY_DATA["practical"][param]
        ax.hlines(y_positions[idx] + 0.14, strict_low, strict_high, color=COLORS["strict"], linewidth=6)
        ax.hlines(y_positions[idx] - 0.14, practical_low, practical_high, color=COLORS["practical"], linewidth=6)
        ax.plot([1.0], [y_positions[idx] + 0.14], marker="|", markersize=16, color="#111111", markeredgewidth=2)
        ax.plot([1.0], [y_positions[idx] - 0.14], marker="|", markersize=16, color="#111111", markeredgewidth=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(params)
    ax.set_xscale("log")
    ax.set_xlim(0.018, 12.5)
    ax.set_xlabel("Allowed Scale Relative to Nominal (log scale)")
    ax.set_title("One-at-a-Time Robustness Boundaries")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=COLORS["strict"], linewidth=6, label="Strict"),
            plt.Line2D([0], [0], color=COLORS["practical"], linewidth=6, label="Practical"),
            plt.Line2D([0], [0], color="#111111", marker="|", markersize=14, linewidth=0, label="Nominal"),
        ],
        loc="lower right",
        frameon=False,
    )

    output_path = PLOTS_DIR / "robustness_boundaries.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_joint_domain_robustness() -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for label, series in JOINT_DOMAIN_DATA.items():
        xs = sorted(series.keys())
        ys = [series[value] for value in xs]
        color = COLORS["strict_box"] if "Strict" in label else COLORS["practical_box"]
        ax.plot(xs, ys, marker="o", linewidth=2.6, markersize=7, color=color, label=label)
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, y + 0.018, f"{y:.3f}", ha="center", va="bottom", fontsize=9, color=color)

    ax.set_title("Joint Random-Domain Pass Rate")
    ax.set_xlabel("Alpha (Expansion Toward Full Joint Box)")
    ax.set_ylabel("Domain Pass Rate")
    ax.set_xlim(0.55, 1.02)
    ax.set_ylim(0.72, 1.02)
    ax.legend(frameon=False, loc="lower left")
    ax.axvspan(0.7, 1.0, color="#d8c6ab", alpha=0.22)
    ax.text(0.705, 0.735, "failure region starts\naround alpha ≈ 0.7", color="#5a4835", fontsize=10)

    output_path = PLOTS_DIR / "joint_domain_robustness.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_validation_and_speed() -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.5))

    horizon_labels = list(LONG_HORIZON.keys())
    horizon_values = list(LONG_HORIZON.values())
    ax1.bar(horizon_labels, horizon_values, color=["#406882", "#4f7e4f", "#9c6a27"], edgecolor="#222222")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Solved Rate")
    ax1.set_title("Long-Horizon Validation")
    for idx, value in enumerate(horizon_values):
        ax1.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=10)

    speed_labels = list(INFERENCE_SPEED.keys())
    speed_values = list(INFERENCE_SPEED.values())
    ax2.bar(speed_labels, speed_values, color=["#a44a3f", "#355c7d"], edgecolor="#222222")
    ax2.set_ylabel("Milliseconds per Forward Pass")
    ax2.set_title("GPU Inference Speed")
    for idx, value in enumerate(speed_values):
        ax2.text(idx, value + 0.001, f"{value:.4f} ms", ha="center", va="bottom", fontsize=10)

    output_path = PLOTS_DIR / "validation_and_speed.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def build_dashboard(plot_paths: list[Path]) -> Path:
    fig = plt.figure(figsize=(13.5, 10))
    fig.suptitle("CartPole RL Results Dashboard", fontsize=18, fontweight="bold", y=0.98)

    grid = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.12)
    for idx, path in enumerate(plot_paths):
        axis = fig.add_subplot(grid[idx // 2, idx % 2])
        image = plt.imread(path)
        axis.imshow(image)
        axis.axis("off")

    output_path = PLOTS_DIR / "results_dashboard.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    style_matplotlib()
    ensure_output_dir()

    plot_paths = [
        plot_experiment_comparison(),
        plot_boundary_ranges(),
        plot_joint_domain_robustness(),
        plot_validation_and_speed(),
    ]
    dashboard_path = build_dashboard(plot_paths)

    print("Generated plots:")
    for path in plot_paths + [dashboard_path]:
        print(path)


if __name__ == "__main__":
    main()
