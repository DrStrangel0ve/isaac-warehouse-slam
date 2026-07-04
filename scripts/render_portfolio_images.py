from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_slam.grid_map import OccupancyGridMap  # noqa: E402
from warehouse_slam.mission import WarehouseMissionPlanner  # noqa: E402
from warehouse_slam.rl import QTablePushPolicy  # noqa: E402
from warehouse_slam.sim2d import Rect, Warehouse2DSim  # noqa: E402
from warehouse_slam.slam import OccupancyGridSlam  # noqa: E402
from warehouse_slam.types import Pose2D  # noqa: E402


@dataclass(frozen=True)
class Snapshot:
    step: int
    mode: str
    robot: Pose2D
    crate_xy: tuple[float, float]
    trace: tuple[tuple[float, float], ...]


def collect_snapshots() -> tuple[Warehouse2DSim, list[Snapshot]]:
    sim = Warehouse2DSim()
    grid = OccupancyGridMap(width_m=11.0, height_m=8.5, resolution_m=0.08)
    slam = OccupancyGridSlam(initial_pose=Pose2D(sim.pose.x, sim.pose.y, sim.pose.yaw), grid=grid)
    policy_path = PROJECT_ROOT / "artifacts" / "rl_push_q_table.json"
    policy = QTablePushPolicy.load(policy_path) if policy_path.exists() else None
    mission = WarehouseMissionPlanner(
        goal_zone_xy=(sim.goal_zone.cx, sim.goal_zone.cy),
        rl_push_policy=policy,
    )
    command = mission.update(slam, [], bumper_active=False)
    trace: list[tuple[float, float]] = [(sim.pose.x, sim.pose.y)]
    snapshots: list[Snapshot] = [
        Snapshot(0, mission.status.mode, sim.pose, (sim.crate.cx, sim.crate.cy), tuple(trace))
    ]
    wanted_steps = {30, 67}
    for step in range(1, 500):
        scan, imu, detections, bumper = sim.step(command, 0.1)
        odom = sim.last_executed_command
        slam.step(odom.linear_mps * 1.05, odom.angular_rps * 1.02, 0.1, scan, imu)
        interaction_enabled = step >= 40
        command = mission.update(
            slam,
            detections if interaction_enabled else [],
            bumper_active=bumper if interaction_enabled else False,
        )
        trace.append((sim.pose.x, sim.pose.y))
        if step in wanted_steps or sim.crate_in_goal():
            snapshots.append(
                Snapshot(
                    step,
                    mission.status.mode,
                    sim.pose,
                    (sim.crate.cx, sim.crate.cy),
                    tuple(trace),
                )
            )
        if sim.crate_in_goal():
            break
    return sim, snapshots


def world_to_px(
    x: float,
    y: float,
    bounds: Rect,
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, right, bottom = rect
    px = left + int((x - bounds.xmin) / bounds.w * (right - left))
    py = bottom - int((y - bounds.ymin) / bounds.h * (bottom - top))
    return px, py


def draw_world(
    draw: ImageDraw.ImageDraw,
    sim: Warehouse2DSim,
    snapshot: Snapshot,
    rect: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    title: str,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=18, fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.text((left + 20, top + 18), title, fill="#111827", font=font)
    arena = (left + 24, top + 54, right - 24, bottom - 24)
    draw.rectangle(arena, fill="#eef2f7", outline="#94a3b8", width=2)

    for obstacle in sim.static_obstacles:
        draw_rect(draw, obstacle, sim.bounds, arena, "#555f6f")
    draw_rect(draw, sim.goal_zone, sim.bounds, arena, "#78d48b", outline="#15803d")
    crate = Rect(snapshot.crate_xy[0], snapshot.crate_xy[1], sim.crate.w, sim.crate.h)
    draw_rect(draw, crate, sim.bounds, arena, "#f97316", outline="#9a3412")

    if len(snapshot.trace) > 1:
        points = [world_to_px(x, y, sim.bounds, arena) for x, y in snapshot.trace]
        draw.line(points, fill="#2563eb", width=4)

    robot_px = world_to_px(snapshot.robot.x, snapshot.robot.y, sim.bounds, arena)
    draw_lidar_fan(draw, snapshot.robot, sim.bounds, arena)
    draw_robot(draw, robot_px, snapshot.robot.yaw)
    draw.text((left + 20, bottom - 22), f"step {snapshot.step} | {snapshot.mode}", fill="#334155", font=font)


def draw_rect(
    draw: ImageDraw.ImageDraw,
    item: Rect,
    bounds: Rect,
    canvas: tuple[int, int, int, int],
    fill: str,
    outline: str | None = None,
) -> None:
    x0, y0 = world_to_px(item.xmin, item.ymax, bounds, canvas)
    x1, y1 = world_to_px(item.xmax, item.ymin, bounds, canvas)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=4, fill=fill, outline=outline)


def draw_lidar_fan(
    draw: ImageDraw.ImageDraw,
    pose: Pose2D,
    bounds: Rect,
    canvas: tuple[int, int, int, int],
) -> None:
    origin = world_to_px(pose.x, pose.y, bounds, canvas)
    for angle in (-1.15, -0.55, 0.0, 0.55, 1.15):
        end = world_to_px(
            pose.x + math.cos(pose.yaw + angle) * 1.25,
            pose.y + math.sin(pose.yaw + angle) * 1.25,
            bounds,
            canvas,
        )
        draw.line((origin, end), fill="#93c5fd", width=1)


def draw_robot(draw: ImageDraw.ImageDraw, center: tuple[int, int], yaw: float) -> None:
    cx, cy = center
    points = []
    for angle, radius in ((0.0, 16), (2.45, 11), (-2.45, 11)):
        points.append((cx + math.cos(yaw + angle) * radius, cy - math.sin(yaw + angle) * radius))
    draw.polygon(points, fill="#1d4ed8", outline="#172554")
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill="#eff6ff")


def image_card(
    canvas: Image.Image,
    source: Path,
    box: tuple[int, int, int, int],
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((box[0] + 18, box[1] + 14), title, fill="#111827", font=font)
    if not source.exists():
        return
    image = Image.open(source).convert("RGB")
    image.thumbnail((box[2] - box[0] - 36, box[3] - box[1] - 58))
    x = box[0] + (box[2] - box[0] - image.width) // 2
    y = box[1] + 46 + (box[3] - box[1] - 58 - image.height) // 2
    canvas.paste(image, (x, y))


def metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    value: str,
    caption: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=16, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((box[0] + 18, box[1] + 14), title, fill="#475569", font=font)
    draw.text((box[0] + 18, box[1] + 43), value, fill="#0f172a", font=font)
    draw.text((box[0] + 18, box[1] + 73), caption, fill="#64748b", font=font)


def main() -> int:
    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sim, snapshots = collect_snapshots()
    font = ImageFont.load_default()

    storyboard = Image.new("RGB", (1500, 760), "#e8edf5")
    draw = ImageDraw.Draw(storyboard)
    draw.text((42, 30), "Isaac Warehouse SLAM + RL crate interaction", fill="#0f172a", font=font)
    draw.text((42, 52), "A real autonomy loop: map, detect, plan, then hand off local contact control to a learned policy.", fill="#475569", font=font)
    panel_w = 456
    for index, snapshot in enumerate(snapshots[:3]):
        draw_world(
            draw,
            sim,
            snapshot,
            (42 + index * (panel_w + 24), 92, 42 + index * (panel_w + 24) + panel_w, 710),
            font,
            ["scene setup", "crate acquisition", "learned push"][min(index, 2)],
        )
    storyboard.save(artifact_dir / "warehouse_storyboard.png")

    hero = Image.new("RGB", (1500, 900), "#e8edf5")
    hero_draw = ImageDraw.Draw(hero)
    hero_draw.text((48, 32), "Isaac Warehouse SLAM + RL", fill="#0f172a", font=font)
    hero_draw.text((48, 54), "Multi-sensor mapping, frontier/A* navigation, and learned local crate pushing.", fill="#475569", font=font)
    draw_world(hero_draw, sim, snapshots[-1], (48, 98, 930, 812), font, "mission rollout")
    image_card(hero, artifact_dir / "offline_slam_map.png", (960, 98, 1452, 336), "occupancy map", font)
    image_card(hero, artifact_dir / "rl_training_curve.png", (960, 360, 1452, 608), "RL training curve", font)

    metrics = json.loads((artifact_dir / "rl_training_metrics.json").read_text(encoding="utf-8"))
    metric_card(
        hero_draw,
        (960, 634, 1194, 812),
        "policy success",
        f"{metrics['success_rate'] * 100:.1f}%",
        "held-out local starts",
        font,
    )
    metric_card(
        hero_draw,
        (1218, 634, 1452, 812),
        "Q states",
        str(metrics["q_states"]),
        "learned table entries",
        font,
    )
    hero.save(artifact_dir / "portfolio_hero.png")
    print(artifact_dir / "warehouse_storyboard.png")
    print(artifact_dir / "portfolio_hero.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
