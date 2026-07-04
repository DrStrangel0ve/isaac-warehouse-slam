from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_slam.control import DifferentialCommand  # noqa: E402
from warehouse_slam.grid_map import OccupancyGridMap  # noqa: E402
from warehouse_slam.mission import WarehouseMissionPlanner  # noqa: E402
from warehouse_slam.rl import QTablePushPolicy  # noqa: E402
from warehouse_slam.sim2d import Rect, Warehouse2DSim  # noqa: E402
from warehouse_slam.slam import OccupancyGridSlam  # noqa: E402
from warehouse_slam.types import CameraDetection, ImuReading, LidarScan, Pose2D, wrap_angle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--headed", "--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=False)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--save-map-every", type=int, default=180)
    parser.add_argument("--rl-policy", type=Path, default=None)
    args = parser.parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "width": args.window_width,
            "height": args.window_height,
            "renderer": "RayTracedLighting",
            "multi_gpu": False,
        }
    )

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
        DifferentialController,
    )
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot
    from isaacsim.sensors.camera import Camera
    from isaacsim.storage.native import get_assets_root_path

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    template = Warehouse2DSim()
    static_rects = template.static_obstacles
    for rect in static_rects:
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/Warehouse/{rect.label}",
                name=rect.label,
                position=np.array([rect.cx, rect.cy, 0.35]),
                scale=np.array([rect.w, rect.h, 0.7]),
                color=np.array([0.42, 0.44, 0.48]),
            )
        )

    goal_zone = world.scene.add(
        VisualCuboid(
            prim_path="/World/GoalZone",
            name="goal_zone",
            position=np.array([template.goal_zone.cx, template.goal_zone.cy, 0.015]),
            scale=np.array([template.goal_zone.w, template.goal_zone.h, 0.03]),
            color=np.array([0.05, 0.7, 0.25]),
        )
    )
    _ = goal_zone

    crate = world.scene.add(
        DynamicCuboid(
            prim_path="/World/SupplyCrate",
            name="supply_crate",
            position=np.array([template.crate.cx, template.crate.cy, 0.28]),
            scale=np.array([template.crate.w, template.crate.h, 0.56]),
            color=np.array([1.0, 0.38, 0.05]),
            mass=4.0,
        )
    )

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root.")
    jetbot_usd = assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
    add_reference_to_stage(usd_path=jetbot_usd, prim_path="/World/ExplorerBot")
    robot = world.scene.add(
        WheeledRobot(
            prim_path="/World/ExplorerBot",
            name="explorer_bot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=False,
            position=np.array([template.pose.x, template.pose.y, 0.03]),
            orientation=euler_angles_to_quat(np.array([0.0, 0.0, math.degrees(template.pose.yaw)]), degrees=True),
        )
    )
    bumper_probe = world.scene.add(
        DynamicCuboid(
            prim_path="/World/FrontBumperProbe",
            name="front_bumper_probe",
            position=np.array([template.pose.x + 0.28, template.pose.y, 0.11]),
            orientation=euler_angles_to_quat(np.array([0.0, 0.0, math.degrees(template.pose.yaw)]), degrees=True),
            scale=np.array([0.08, 0.34, 0.12]),
            color=np.array([1.0, 0.92, 0.05]),
            mass=0.05,
        )
    )

    rgb_camera = Camera(
        prim_path="/World/ExplorerBot/front_rgbd_camera",
        name="front_rgbd_camera",
        position=np.array([0.18, 0.0, 0.18]),
        orientation=euler_angles_to_quat(np.array([0.0, 90.0, 0.0]), degrees=True),
        resolution=(640, 360),
        frequency=30,
    )
    rgb_camera.initialize()
    _try_enable_depth(rgb_camera)
    _try_create_rtx_lidar()
    physics_sensors = _try_create_physics_sensors(contact_parent_path="/World/FrontBumperProbe")
    physics_sensors["bumper_probe"] = bumper_probe

    differential = DifferentialController(
        name="warehouse_diff_controller",
        wheel_radius=0.0325,
        wheel_base=0.1125,
    )
    grid = OccupancyGridMap(width_m=11.0, height_m=8.5, resolution_m=0.08)
    slam = OccupancyGridSlam(initial_pose=template.pose, grid=grid)
    rl_policy = QTablePushPolicy.load(args.rl_policy) if args.rl_policy is not None else None
    mission = WarehouseMissionPlanner(
        goal_zone_xy=(template.goal_zone.cx, template.goal_zone.cy),
        rl_push_policy=rl_policy,
    )

    world.reset()
    for _ in range(30):
        world.step(render=True)

    command = DifferentialCommand(0.0, 0.0)
    previous_robot_pose = _read_robot_pose(robot)
    telemetry_rows: list[dict[str, str]] = []
    for step in range(args.steps):
        robot_pose = _read_robot_pose(robot)
        _sync_bumper_probe(physics_sensors.get("bumper_probe"), robot_pose)
        odom = _pose_delta_to_command(previous_robot_pose, robot_pose, args.dt)
        crate_pose = _read_crate_pose(crate)
        scan = _analytic_scan(robot_pose, static_rects, crate_pose, template.max_range_m, template.angles)
        detections = _crate_detection_from_camera_model(robot_pose, crate_pose)
        imu = _read_imu_sensor(physics_sensors.get("imu"), fallback_yaw_rate=odom.angular_rps)
        bumper = _read_contact_sensor(
            physics_sensors.get("contact"),
            fallback=robot_pose.distance_to((crate_pose.cx, crate_pose.cy)) < 0.55,
        )

        slam.step(odom.linear_mps, odom.angular_rps, args.dt, scan, imu)
        command = mission.update(slam, detections, bumper_active=bumper)
        robot.apply_wheel_actions(differential.forward(command=[command.linear_mps, command.angular_rps]))
        world.step(render=True)
        previous_robot_pose = robot_pose

        if step % 30 == 0:
            telemetry_rows.append(
                {
                    "step": str(step),
                    "mode": mission.status.mode,
                    "slam_x": f"{slam.pose.x:.3f}",
                    "slam_y": f"{slam.pose.y:.3f}",
                    "truth_x": f"{robot_pose.x:.3f}",
                    "truth_y": f"{robot_pose.y:.3f}",
                    "crate_x": f"{crate_pose.cx:.3f}",
                    "crate_y": f"{crate_pose.cy:.3f}",
                    "explored_fraction": f"{slam.grid.explored_fraction():.4f}",
                    "detections": str(len(detections)),
                    "scan_match_accepted": str(bool(slam.last_match and slam.last_match.accepted)),
                    "scan_match_score": f"{slam.last_match.score:.4f}" if slam.last_match else "0.0000",
                }
            )
            print(
                f"step={step:04d} mode={mission.status.mode} "
                f"slam=({slam.pose.x:+.2f},{slam.pose.y:+.2f}) "
                f"crate=({crate_pose.cx:+.2f},{crate_pose.cy:+.2f}) "
                f"cmd=({command.linear_mps:.2f},{command.angular_rps:+.2f}) "
                f"match={bool(slam.last_match and slam.last_match.accepted)}"
            )

        if args.save_map_every > 0 and step > 0 and step % args.save_map_every == 0:
            slam.grid.to_image(args.artifact_dir / f"isaac_map_step_{step:04d}.png", robot_pose=slam.pose)

    robot.apply_wheel_actions(differential.forward(command=[0.0, 0.0]))
    for _ in range(8):
        world.step(render=True)

    slam.grid.to_image(args.artifact_dir / "isaac_final_map.png", robot_pose=slam.pose)
    telemetry_path = args.artifact_dir / "isaac_telemetry.csv"
    if telemetry_rows:
        with telemetry_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(telemetry_rows[0].keys()))
            writer.writeheader()
            writer.writerows(telemetry_rows)
    print(f"Saved map and telemetry under {args.artifact_dir}")
    simulation_app.close()


def _try_enable_depth(camera) -> None:
    for method in ("add_distance_to_image_plane_to_frame", "add_distance_to_camera_to_frame"):
        if hasattr(camera, method):
            try:
                getattr(camera, method)()
                print(f"Enabled depth annotator via Camera.{method}().")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"Depth annotator setup skipped: {exc}")


def _try_create_rtx_lidar() -> None:
    try:
        from isaacsim.sensors.experimental.rtx import Lidar

        Lidar.create(
            path="/World/ExplorerBot/rtx_lidar",
            config="Example_Rotary",
            translations=np.array([0.0, 0.0, 0.28]),
            orientations=np.array([1.0, 0.0, 0.0, 0.0]),
            attributes={"omni:sensor:Core:scanRateBaseHz": 10},
        )
        print("Created RTX LiDAR prim at /World/ExplorerBot/rtx_lidar.")
    except Exception as exc:  # noqa: BLE001
        print(f"RTX LiDAR creation skipped; analytic scan bridge remains active: {exc}")


def _try_create_physics_sensors(contact_parent_path: str) -> dict[str, object]:
    sensors: dict[str, object] = {}
    try:
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor, IMU, IMUSensor
    except Exception as exc:  # noqa: BLE001
        print(f"Physics sensor imports skipped; analytic IMU/contact bridge remains active: {exc}")
        return sensors

    try:
        imu_prim = IMU.create(
            path="/World/ExplorerBot/imu_sensor",
            translations=np.array([0.0, 0.0, 0.25]),
            orientations=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        sensors["imu"] = IMUSensor(imu_prim)
        print("Created experimental physics IMU sensor.")
    except Exception as exc:  # noqa: BLE001
        print(f"Physics IMU creation skipped; analytic IMU bridge remains active: {exc}")

    try:
        contact_prim = Contact.create(
            path=f"{contact_parent_path}/front_bumper_contact",
            translations=np.array([0.0, 0.0, 0.0]),
            orientations=np.array([1.0, 0.0, 0.0, 0.0]),
            min_threshold=0.01,
            max_threshold=100000.0,
            radius=0.18,
        )
        sensors["contact"] = ContactSensor(contact_prim)
        print("Created experimental front contact sensor.")
    except Exception as exc:  # noqa: BLE001
        print(f"Physics contact sensor creation skipped; analytic contact bridge remains active: {exc}")
    return sensors


def _read_imu_sensor(sensor: object | None, fallback_yaw_rate: float) -> ImuReading:
    if sensor is None:
        return ImuReading(yaw_rate_rps=fallback_yaw_rate, accel_x_mps2=0.0, accel_y_mps2=0.0)
    for method in ("get_data", "get_current_frame", "get_sensor_reading", "get_current_reading"):
        if not hasattr(sensor, method):
            continue
        try:
            reading = getattr(sensor, method)()
            return _coerce_imu_reading(reading, fallback_yaw_rate)
        except Exception:  # noqa: BLE001
            continue
    return ImuReading(yaw_rate_rps=fallback_yaw_rate, accel_x_mps2=0.0, accel_y_mps2=0.0)


def _read_contact_sensor(sensor: object | None, fallback: bool) -> bool:
    if sensor is None:
        return fallback
    for method in ("get_data", "get_current_frame", "get_sensor_reading", "get_current_reading"):
        if not hasattr(sensor, method):
            continue
        try:
            reading = getattr(sensor, method)()
            return _coerce_contact_reading(reading, fallback)
        except Exception:  # noqa: BLE001
            continue
    return fallback


def _sync_bumper_probe(probe: object | None, pose: Pose2D) -> None:
    if probe is None or not hasattr(probe, "set_world_pose"):
        return
    try:
        position = np.array(
            [
                pose.x + math.cos(pose.yaw) * 0.28,
                pose.y + math.sin(pose.yaw) * 0.28,
                0.11,
            ]
        )
        getattr(probe, "set_world_pose")(position=position, orientation=_yaw_to_quat(pose.yaw))
    except Exception:  # noqa: BLE001
        return


def _coerce_imu_reading(reading: object, fallback_yaw_rate: float) -> ImuReading:
    if isinstance(reading, dict):
        angular = reading.get("angular_velocity") or reading.get("angularVelocity")
        linear = reading.get("linear_acceleration") or reading.get("linearAcceleration")
    else:
        angular = getattr(reading, "angular_velocity", None) or getattr(reading, "angularVelocity", None)
        linear = getattr(reading, "linear_acceleration", None) or getattr(reading, "linearAcceleration", None)
        if angular is None and hasattr(reading, "angular_velocity_z"):
            angular = [
                getattr(reading, "angular_velocity_x", 0.0),
                getattr(reading, "angular_velocity_y", 0.0),
                getattr(reading, "angular_velocity_z", fallback_yaw_rate),
            ]
        if linear is None and hasattr(reading, "linear_acceleration_x"):
            linear = [
                getattr(reading, "linear_acceleration_x", 0.0),
                getattr(reading, "linear_acceleration_y", 0.0),
                getattr(reading, "linear_acceleration_z", 0.0),
            ]
    yaw_rate = _component_or_default(angular, 2, fallback_yaw_rate)
    accel_x = _component_or_default(linear, 0, 0.0)
    accel_y = _component_or_default(linear, 1, 0.0)
    return ImuReading(yaw_rate_rps=yaw_rate, accel_x_mps2=accel_x, accel_y_mps2=accel_y)


def _coerce_contact_reading(reading: object, fallback: bool) -> bool:
    if isinstance(reading, dict):
        for key in ("in_contact", "is_contacting", "contacts", "force", "net_force"):
            if key in reading:
                return _truthy_contact_value(reading[key], fallback)
    for attr in ("in_contact", "is_contacting", "contacts", "force", "net_force"):
        if hasattr(reading, attr):
            return _truthy_contact_value(getattr(reading, attr), fallback)
    return fallback


def _component_or_default(value: object, index: int, default: float) -> float:
    if value is None:
        return default
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[index]) if arr.size > index else default
    except Exception:  # noqa: BLE001
        return default


def _truthy_contact_value(value: object, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    try:
        arr = np.asarray(value, dtype=float)
        return bool(np.any(np.abs(arr) > 1e-4))
    except Exception:  # noqa: BLE001
        return bool(value)


def _read_robot_pose(robot) -> Pose2D:
    position, orientation = robot.get_world_pose()
    yaw = _quat_to_yaw(orientation)
    return Pose2D(float(position[0]), float(position[1]), yaw)


def _read_crate_pose(crate) -> Rect:
    position, _ = crate.get_world_pose()
    return Rect(float(position[0]), float(position[1]), 0.55, 0.55, "supply_crate", movable=True)


def _analytic_scan(
    pose: Pose2D,
    static_rects: list[Rect],
    crate: Rect,
    max_range_m: float,
    angles: np.ndarray,
) -> LidarScan:
    rects = [*static_rects, crate]
    ranges = np.full_like(angles, max_range_m, dtype=np.float32)
    step = 0.035
    for index, rel_angle in enumerate(angles):
        theta = pose.yaw + float(rel_angle)
        distance = 0.0
        while distance < max_range_m:
            px = pose.x + math.cos(theta) * distance
            py = pose.y + math.sin(theta) * distance
            if any(rect.contains(px, py) for rect in rects):
                break
            if not (-5.25 <= px <= 5.25 and -4.0 <= py <= 4.0):
                break
            distance += step
        ranges[index] = min(distance, max_range_m)
    return LidarScan(ranges, angles.copy(), max_range_m)


def _crate_detection_from_camera_model(pose: Pose2D, crate: Rect) -> list[CameraDetection]:
    dx = crate.cx - pose.x
    dy = crate.cy - pose.y
    distance = math.hypot(dx, dy)
    bearing = wrap_angle(math.atan2(dy, dx) - pose.yaw)
    if distance > 4.5 or abs(bearing) > math.radians(55):
        return []
    confidence = max(0.25, 1.0 - distance / 4.5) * (1.0 - abs(bearing) / math.radians(55))
    return [CameraDetection("supply_crate", distance, bearing, confidence)]


def _quat_to_yaw(quat) -> float:
    q = np.asarray(quat, dtype=float).reshape(-1)
    if q.size != 4:
        return 0.0
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def _pose_delta_to_command(previous: Pose2D, current: Pose2D, dt: float) -> DifferentialCommand:
    dx = current.x - previous.x
    dy = current.y - previous.y
    forward = dx * math.cos(previous.yaw) + dy * math.sin(previous.yaw)
    yaw_rate = wrap_angle(current.yaw - previous.yaw) / max(dt, 1e-6)
    return DifferentialCommand(forward / max(dt, 1e-6), yaw_rate)


if __name__ == "__main__":
    main()
