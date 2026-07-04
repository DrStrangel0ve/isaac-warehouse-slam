from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_slam.grid_map import OccupancyGridMap  # noqa: E402
from warehouse_slam.mission import WarehouseMissionPlanner  # noqa: E402
from warehouse_slam.rl import QTablePushPolicy  # noqa: E402
from warehouse_slam.sim2d import Warehouse2DSim  # noqa: E402
from warehouse_slam.slam import OccupancyGridSlam  # noqa: E402
from warehouse_slam.types import Pose2D  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--min-interaction-steps", type=int, default=40)
    parser.add_argument("--odom-linear-scale", type=float, default=1.05)
    parser.add_argument("--odom-angular-scale", type=float, default=1.02)
    parser.add_argument("--rl-policy", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    sim = Warehouse2DSim()
    grid = OccupancyGridMap(width_m=11.0, height_m=8.5, resolution_m=0.08)
    slam = OccupancyGridSlam(initial_pose=Pose2D(sim.pose.x, sim.pose.y, sim.pose.yaw), grid=grid)
    rl_policy = QTablePushPolicy.load(args.rl_policy) if args.rl_policy is not None else None
    mission = WarehouseMissionPlanner(
        goal_zone_xy=(sim.goal_zone.cx, sim.goal_zone.cy),
        rl_push_policy=rl_policy,
    )

    command = mission.update(slam, [], bumper_active=False)
    telemetry_rows: list[dict[str, str]] = []
    accepted_scan_matches = 0

    for step in range(args.steps):
        scan, imu, detections, bumper = sim.step(command, args.dt)
        odom = sim.last_executed_command
        pose = slam.step(
            odom.linear_mps * args.odom_linear_scale,
            odom.angular_rps * args.odom_angular_scale,
            args.dt,
            scan,
            imu,
        )
        if slam.last_match and slam.last_match.accepted:
            accepted_scan_matches += 1
        interaction_enabled = step >= args.min_interaction_steps
        command = mission.update(
            slam,
            detections if interaction_enabled else [],
            bumper_active=bumper if interaction_enabled else False,
        )

        if step % 10 == 0:
            telemetry_rows.append(
                {
                    "step": str(step),
                    "mode": mission.status.mode,
                    "estimated_x": f"{pose.x:.3f}",
                    "estimated_y": f"{pose.y:.3f}",
                    "true_x": f"{sim.pose.x:.3f}",
                    "true_y": f"{sim.pose.y:.3f}",
                    "crate_x": f"{sim.crate.cx:.3f}",
                    "crate_y": f"{sim.crate.cy:.3f}",
                    "explored_fraction": f"{slam.grid.explored_fraction():.4f}",
                    "detections": str(len(detections)),
                    "scan_match_accepted": str(bool(slam.last_match and slam.last_match.accepted)),
                    "scan_match_score": f"{slam.last_match.score:.4f}" if slam.last_match else "0.0000",
                }
            )

        if sim.crate_in_goal():
            print(f"Crate reached the goal at step {step}.")
            break

    map_path = args.artifact_dir / "offline_slam_map.png"
    telemetry_path = args.artifact_dir / "offline_telemetry.csv"
    metrics_path = args.artifact_dir / "offline_metrics.json"
    slam.grid.to_image(map_path, robot_pose=slam.pose)
    with telemetry_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(telemetry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(telemetry_rows)
    pose_error = slam.pose.distance_to((sim.pose.x, sim.pose.y))
    crate_goal_error = ((sim.crate.cx - sim.goal_zone.cx) ** 2 + (sim.crate.cy - sim.goal_zone.cy) ** 2) ** 0.5
    metrics = {
        "mode": mission.status.mode,
        "success": sim.crate_in_goal(),
        "pose_error_m": round(pose_error, 4),
        "crate_goal_error_m": round(crate_goal_error, 4),
        "explored_fraction": round(slam.grid.explored_fraction(), 4),
        "accepted_scan_matches": accepted_scan_matches,
        "steps_recorded": len(telemetry_rows),
        "min_interaction_steps": args.min_interaction_steps,
        "odom_linear_scale": args.odom_linear_scale,
        "odom_angular_scale": args.odom_angular_scale,
        "rl_policy": str(args.rl_policy) if args.rl_policy is not None else None,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"mode={mission.status.mode}")
    print(f"estimated_pose=({slam.pose.x:.2f}, {slam.pose.y:.2f}, {slam.pose.yaw:.2f})")
    print(f"true_pose=({sim.pose.x:.2f}, {sim.pose.y:.2f}, {sim.pose.yaw:.2f})")
    print(f"crate=({sim.crate.cx:.2f}, {sim.crate.cy:.2f}) goal=({sim.goal_zone.cx:.2f}, {sim.goal_zone.cy:.2f})")
    print(f"explored_fraction={slam.grid.explored_fraction():.3f}")
    print(f"accepted_scan_matches={accepted_scan_matches}")
    print(f"map={map_path}")
    print(f"telemetry={telemetry_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
