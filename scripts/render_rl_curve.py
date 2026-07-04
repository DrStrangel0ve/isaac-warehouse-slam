from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "episode": float(row["episode"]),
                    "reward": float(row["total_reward"]),
                    "success": float(row["success"] == "True"),
                    "distance": float(row["crate_goal_distance"]),
                }
            )
    return rows


def rolling(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        sample = values[start : index + 1]
        out.append(sum(sample) / len(sample))
    return out


def scale_points(
    xs: list[float],
    ys: list[float],
    bounds: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    left, top, right, bottom = bounds
    x_min = min(xs)
    x_max = max(xs)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-6)
    points: list[tuple[int, int]] = []
    for x, y in zip(xs, ys, strict=True):
        px = left + int((x - x_min) / x_span * (right - left))
        py = bottom - int((y - y_min) / y_span * (bottom - top))
        points.append((px, py))
    return points


def draw_chart(rows: list[dict[str, float]], output: Path) -> None:
    width, height = 1100, 640
    margin_left, margin_top, margin_right, margin_bottom = 84, 86, 44, 100
    plot = (margin_left, margin_top, width - margin_right, height - margin_bottom)
    image = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    episodes = [row["episode"] for row in rows]
    rewards = rolling([row["reward"] for row in rows], 15)
    successes = rolling([row["success"] for row in rows], 25)
    distances = rolling([row["distance"] for row in rows], 15)

    reward_min = min(rewards)
    reward_max = max(rewards)
    distance_min = min(distances)
    distance_max = max(distances)

    draw.rectangle(plot, fill="#ffffff", outline="#c9ced8", width=2)
    draw.text((margin_left, 30), "RL crate-pushing training", fill="#111827", font=font)
    draw.text((margin_left, 52), "Rolling reward, success rate, and final crate-goal error", fill="#4b5563", font=font)

    for i in range(6):
        y = plot[3] - int(i / 5 * (plot[3] - plot[1]))
        draw.line((plot[0], y, plot[2], y), fill="#e5e7eb")

    reward_points = scale_points(episodes, rewards, plot, reward_min, reward_max)
    success_points = scale_points(episodes, successes, plot, 0.0, 1.0)
    distance_points = scale_points(episodes, distances, plot, distance_min, distance_max)

    if len(reward_points) > 1:
        draw.line(reward_points, fill="#2563eb", width=3)
        draw.line(success_points, fill="#16a34a", width=3)
        draw.line(distance_points, fill="#dc2626", width=3)

    draw.text((plot[0], height - 70), "episode", fill="#374151", font=font)
    draw.text((plot[0], height - 48), f"reward range: {reward_min:.1f} to {reward_max:.1f}", fill="#2563eb", font=font)
    draw.text((plot[0] + 270, height - 48), "success rate: 0.0 to 1.0", fill="#16a34a", font=font)
    draw.text(
        (plot[0] + 540, height - 48),
        f"crate-goal error: {distance_min:.2f}m to {distance_max:.2f}m",
        fill="#dc2626",
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_training_curve.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_training_curve.png")
    args = parser.parse_args()
    rows = load_rows(args.curve)
    if not rows:
        raise SystemExit(f"No rows found in {args.curve}")
    draw_chart(rows, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
