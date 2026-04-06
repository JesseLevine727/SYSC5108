from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "experiment_results.json"
PLOTS_DIR = ROOT / "plots"

COLORS = {
    "blue": "#2563eb",
    "green": "#059669",
    "orange": "#d97706",
    "red": "#dc2626",
    "slate": "#334155",
    "gray": "#94a3b8",
    "grid": "#e2e8f0",
    "bg": "#f8fafc",
    "text": "#0f172a",
}


def load_results() -> dict[str, object]:
    return json.loads(RESULTS_PATH.read_text())


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}" />',
    ]


def svg_footer(lines: list[str]) -> str:
    return "\n".join([*lines, "</svg>"])


def add_text(lines: list[str], x: float, y: float, text: str, *, size: int = 16, fill: str | None = None, anchor: str = "start", weight: str = "400") -> None:
    lines.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill or COLORS["text"]}" text-anchor="{anchor}">{escape(text)}</text>'
    )


def add_multiline_text(lines: list[str], x: float, y: float, text: str, *, size: int = 12, fill: str | None = None, anchor: str = "middle", line_gap: int = 14) -> None:
    for index, part in enumerate(text.split("\n")):
        add_text(lines, x, y + (index * line_gap), part, size=size, fill=fill, anchor=anchor)


def add_rect(lines: list[str], x: float, y: float, width: float, height: float, *, fill: str, rx: int = 4, stroke: str | None = None, stroke_width: int = 1) -> None:
    stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
    lines.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{rx}" fill="{fill}"{stroke_attr} />'
    )


def add_line(lines: list[str], x1: float, y1: float, x2: float, y2: float, *, stroke: str, stroke_width: int = 2, dash: str | None = None) -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    lines.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr} />'
    )


def add_circle(lines: list[str], cx: float, cy: float, r: float, *, fill: str, stroke: str | None = None, stroke_width: int = 1) -> None:
    stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
    lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{stroke_attr} />')


def add_polyline(lines: list[str], points: Iterable[tuple[float, float]], *, stroke: str, stroke_width: int = 3) -> None:
    points_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    lines.append(
        f'<polyline points="{points_text}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round" />'
    )


def add_axes(lines: list[str], x: float, y: float, width: float, height: float, y_max: float, y_ticks: int) -> None:
    add_line(lines, x, y, x, y + height, stroke=COLORS["slate"], stroke_width=2)
    add_line(lines, x, y + height, x + width, y + height, stroke=COLORS["slate"], stroke_width=2)
    for tick_index in range(y_ticks + 1):
        tick_value = y_max * tick_index / y_ticks
        tick_y = y + height - (height * tick_index / y_ticks)
        add_line(lines, x, tick_y, x + width, tick_y, stroke=COLORS["grid"], stroke_width=1)
        add_text(lines, x - 10, tick_y + 4, f"{tick_value:.1f}", size=11, fill=COLORS["slate"], anchor="end")


def scale_y(value: float, chart_top: float, chart_height: float, y_max: float) -> float:
    return chart_top + chart_height - (value / y_max) * chart_height


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def plot_training_summary(results: dict[str, object]) -> None:
    runs = results["runs"]  # type: ignore[index]
    width, height = 960, 560
    chart_left, chart_top, chart_width, chart_height = 90, 100, 800, 320
    y_max = 3.8

    lines = svg_header(width, height)
    add_text(lines, 40, 48, "Training Summary Across Runs", size=28, weight="700")
    add_text(lines, 40, 74, "Best benchmark progress vs final benchmark progress", size=15, fill=COLORS["slate"])
    add_axes(lines, chart_left, chart_top, chart_width, chart_height, y_max=y_max, y_ticks=4)

    bar_slot = chart_width / len(runs)
    bar_width = 26
    best_points: list[tuple[float, float]] = []
    final_points: list[tuple[float, float]] = []

    for index, run in enumerate(runs):
        center_x = chart_left + (index + 0.5) * bar_slot
        best_value = float(run["best_progress"])
        final_value = float(run["final_progress"])
        best_y = scale_y(best_value, chart_top, chart_height, y_max)
        final_y = scale_y(final_value, chart_top, chart_height, y_max)

        add_rect(lines, center_x - 36, best_y, bar_width, chart_top + chart_height - best_y, fill=COLORS["blue"])
        add_rect(lines, center_x + 10, final_y, bar_width, chart_top + chart_height - final_y, fill=COLORS["orange"])
        add_text(lines, center_x - 23, best_y - 8, f"{best_value:.2f}", size=11, anchor="middle", fill=COLORS["blue"])
        add_text(lines, center_x + 23, final_y - 8, f"{final_value:.2f}", size=11, anchor="middle", fill=COLORS["orange"])
        add_multiline_text(lines, center_x, chart_top + chart_height + 28, str(run["label"]), size=12)
        best_points.append((center_x - 23, best_y))
        final_points.append((center_x + 23, final_y))

    add_polyline(lines, best_points, stroke=COLORS["blue"], stroke_width=2)
    add_polyline(lines, final_points, stroke=COLORS["orange"], stroke_width=2)

    legend_y = 460
    add_rect(lines, 90, legend_y, 18, 18, fill=COLORS["blue"])
    add_text(lines, 116, legend_y + 14, "Best benchmark progress", size=13)
    add_rect(lines, 300, legend_y, 18, 18, fill=COLORS["orange"])
    add_text(lines, 326, legend_y + 14, "Final benchmark progress", size=13)

    add_text(lines, 90, 515, "Runs 1 and 5 show why checkpoint selection matters: both learned strong policies before late-run regression.", size=13, fill=COLORS["slate"])
    (PLOTS_DIR / "training_summary.svg").write_text(svg_footer(lines))


def plot_steps_vs_progress(results: dict[str, object]) -> None:
    runs = results["runs"]  # type: ignore[index]
    width, height = 960, 560
    chart_left, chart_top, chart_width, chart_height = 110, 90, 760, 350
    x_max = max(float(run["best_steps"]) for run in runs) * 1.05
    y_max = 3.8

    lines = svg_header(width, height)
    add_text(lines, 40, 48, "Sample Efficiency", size=28, weight="700")
    add_text(lines, 40, 74, "Best benchmark progress against environment steps", size=15, fill=COLORS["slate"])
    add_axes(lines, chart_left, chart_top, chart_width, chart_height, y_max=y_max, y_ticks=4)

    for tick_index in range(5):
        x_value = x_max * tick_index / 4
        x = chart_left + chart_width * tick_index / 4
        add_line(lines, x, chart_top, x, chart_top + chart_height, stroke=COLORS["grid"], stroke_width=1)
        add_text(lines, x, chart_top + chart_height + 24, f"{int(x_value/1000)}k", size=11, fill=COLORS["slate"], anchor="middle")

    points: list[tuple[float, float]] = []
    for run in runs:
        x = chart_left + (float(run["best_steps"]) / x_max) * chart_width
        y = scale_y(float(run["best_progress"]), chart_top, chart_height, y_max)
        points.append((x, y))

    add_polyline(lines, points, stroke=COLORS["green"], stroke_width=3)
    for run, (x, y) in zip(runs, points):
        add_circle(lines, x, y, 6, fill=COLORS["green"], stroke="white", stroke_width=2)
        add_text(lines, x + 10, y - 10, str(run["id"]).upper(), size=12, fill=COLORS["green"])

    add_text(lines, 110, 500, "Lower and further left is better on sample cost. Run 4 was the fastest robust multi-generator solution.", size=13, fill=COLORS["slate"])
    (PLOTS_DIR / "sample_efficiency.svg").write_text(svg_footer(lines))


def plot_transfer_benchmark(results: dict[str, object]) -> None:
    suites = results["run5_transfer"]  # type: ignore[index]
    width, height = 980, 560
    chart_left, chart_top, chart_width, chart_height = 90, 100, 820, 320
    y_max = 3.8

    lines = svg_header(width, height)
    add_text(lines, 40, 48, "Run 5 Transfer Benchmark", size=28, weight="700")
    add_text(lines, 40, 74, "Best checkpoint from train-only pool evaluated on seen, mixed, and unseen holdout tracks", size=15, fill=COLORS["slate"])
    add_axes(lines, chart_left, chart_top, chart_width, chart_height, y_max=y_max, y_ticks=4)

    bar_slot = chart_width / len(suites)
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    for index, (suite, color) in enumerate(zip(suites, colors)):
        center_x = chart_left + (index + 0.5) * bar_slot
        value = float(suite["progress"])
        reward = float(suite["reward"])
        off_track = float(suite["off_track_rate"])
        top_y = scale_y(value, chart_top, chart_height, y_max)
        add_rect(lines, center_x - 38, top_y, 76, chart_top + chart_height - top_y, fill=color)
        add_text(lines, center_x, top_y - 8, f"{value:.2f}", size=11, anchor="middle", fill=color)
        add_multiline_text(lines, center_x, chart_top + chart_height + 28, str(suite["suite"]), size=12)
        add_text(lines, center_x, 470, f"reward {reward:.0f}", size=11, anchor="middle", fill=COLORS["slate"])
        add_text(lines, center_x, 488, f"off-track {off_track:.2f}", size=11, anchor="middle", fill=COLORS["slate"])

    add_line(lines, chart_left, scale_y(3.0, chart_top, chart_height, y_max), chart_left + chart_width, scale_y(3.0, chart_top, chart_height, y_max), stroke=COLORS["gray"], stroke_width=2, dash="6 4")
    add_text(lines, 905, scale_y(3.0, chart_top, chart_height, y_max) - 6, "3.0 progress", size=11, fill=COLORS["slate"], anchor="end")
    add_text(lines, 90, 528, "Holdout performance is lower than seen/mixed distributions, but the best checkpoint still keeps off-track failures at zero.", size=13, fill=COLORS["slate"])
    (PLOTS_DIR / "run5_transfer_benchmark.svg").write_text(svg_footer(lines))


def plot_run13_transfer_benchmark(results: dict[str, object]) -> None:
    suites = results["run13_transfer"]  # type: ignore[index]
    width, height = 980, 560
    chart_left, chart_top, chart_width, chart_height = 90, 100, 820, 320
    y_max = 5.0

    lines = svg_header(width, height)
    add_text(lines, 40, 48, "Run 13 From-Scratch Racing Benchmark", size=28, weight="700")
    add_text(lines, 40, 74, "Curriculum-trained racing dynamics checkpoint evaluated on procedural, mixed, and unseen holdout tracks", size=15, fill=COLORS["slate"])
    add_axes(lines, chart_left, chart_top, chart_width, chart_height, y_max=y_max, y_ticks=5)

    bar_slot = chart_width / len(suites)
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    for index, (suite, color) in enumerate(zip(suites, colors)):
        center_x = chart_left + (index + 0.5) * bar_slot
        value = float(suite["progress"])
        reward = float(suite["reward"])
        off_track = float(suite["off_track_rate"])
        top_y = scale_y(value, chart_top, chart_height, y_max)
        add_rect(lines, center_x - 38, top_y, 76, chart_top + chart_height - top_y, fill=color)
        add_text(lines, center_x, top_y - 8, f"{value:.2f}", size=11, anchor="middle", fill=color)
        add_multiline_text(lines, center_x, chart_top + chart_height + 28, str(suite["suite"]), size=12)
        add_text(lines, center_x, 470, f"reward {reward:.0f}", size=11, anchor="middle", fill=COLORS["slate"])
        add_text(lines, center_x, 488, f"off-track {off_track:.2f}", size=11, anchor="middle", fill=COLORS["slate"])

    add_line(lines, chart_left, scale_y(3.0, chart_top, chart_height, y_max), chart_left + chart_width, scale_y(3.0, chart_top, chart_height, y_max), stroke=COLORS["gray"], stroke_width=2, dash="6 4")
    add_text(lines, 905, scale_y(3.0, chart_top, chart_height, y_max) - 6, "3.0 progress", size=11, fill=COLORS["slate"], anchor="end")
    add_text(lines, 90, 528, "Run 13 is the first checkpoint that combines fast racing dynamics with zero off-track failures across all evaluated holdout suites.", size=13, fill=COLORS["slate"])
    (PLOTS_DIR / "run13_transfer_benchmark.svg").write_text(svg_footer(lines))


def plot_generalization_benchmark(results: dict[str, object]) -> None:
    suites = results["run4_benchmark"]  # type: ignore[index]
    width, height = 960, 560
    chart_left, chart_top, chart_width, chart_height = 90, 100, 800, 320
    y_max = 3.8

    lines = svg_header(width, height)
    add_text(lines, 40, 48, "Run 4 Multi-Generator Benchmark", size=28, weight="700")
    add_text(lines, 40, 74, "Solved checkpoint performance across baseline, full randomization, and stress", size=15, fill=COLORS["slate"])
    add_axes(lines, chart_left, chart_top, chart_width, chart_height, y_max=y_max, y_ticks=4)

    bar_slot = chart_width / len(suites)
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    for index, (suite, color) in enumerate(zip(suites, colors)):
        center_x = chart_left + (index + 0.5) * bar_slot
        value = float(suite["progress"])
        reward = float(suite["reward"])
        top_y = scale_y(value, chart_top, chart_height, y_max)
        add_rect(lines, center_x - 40, top_y, 80, chart_top + chart_height - top_y, fill=color)
        add_text(lines, center_x, top_y - 8, f"{value:.2f}", size=11, anchor="middle", fill=color)
        add_multiline_text(lines, center_x, chart_top + chart_height + 28, str(suite["suite"]), size=12)
        add_text(lines, center_x, 468, f"reward {reward:.0f}", size=11, anchor="middle", fill=COLORS["slate"])

    add_text(lines, 90, 522, "Run 4 shows the strongest broad procedural robustness: all three suites stayed above 3.49 mean progress.", size=13, fill=COLORS["slate"])
    (PLOTS_DIR / "run4_generalization_benchmark.svg").write_text(svg_footer(lines))


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    plot_training_summary(results)
    plot_steps_vs_progress(results)
    plot_generalization_benchmark(results)
    plot_transfer_benchmark(results)
    plot_run13_transfer_benchmark(results)
    print("Generated:", *(str(path.name) for path in sorted(PLOTS_DIR.glob("*.svg"))))


if __name__ == "__main__":
    main()
