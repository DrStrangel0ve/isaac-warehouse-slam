from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np

from .control import DifferentialCommand
from .types import CameraDetection, ImuReading, LidarScan, Pose2D, wrap_angle


@dataclass
class Rect:
    cx: float
    cy: float
    w: float
    h: float
    label: str = "obstacle"
    movable: bool = False

    @property
    def xmin(self) -> float:
        return self.cx - self.w / 2.0

    @property
    def xmax(self) -> float:
        return self.cx + self.w / 2.0

    @property
    def ymin(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def ymax(self) -> float:
        return self.cy + self.h / 2.0

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.xmin - margin <= x <= self.xmax + margin
            and self.ymin - margin <= y <= self.ymax + margin
        )


class Warehouse2DSim:
    """Small deterministic warehouse used for fast SLAM validation."""

    def __init__(self, seed: int = 4) -> None:
        self.rng = np.random.default_rng(seed)
        self.bounds = Rect(0.0, 0.0, 10.5, 8.0, "bounds")
        self.robot_radius_m = 0.22
        self.pose = Pose2D(-4.25, -2.85, 0.15)
        self.last_command = DifferentialCommand(0.0, 0.0)
        self.last_executed_command = DifferentialCommand(0.0, 0.0)
        self.goal_zone = Rect(-2.55, -1.75, 0.8, 0.8, "goal")
        self.crate = Rect(-3.2, -1.95, 0.55, 0.55, "supply_crate", movable=True)
        self.static_obstacles = [
            Rect(0.0, -3.6, 10.5, 0.18, "south_wall"),
            Rect(0.0, 3.6, 10.5, 0.18, "north_wall"),
            Rect(-5.15, 0.0, 0.18, 8.0, "west_wall"),
            Rect(5.15, 0.0, 0.18, 8.0, "east_wall"),
            Rect(-2.2, -0.65, 0.38, 3.3, "shelf_a"),
            Rect(0.15, 1.0, 0.38, 3.8, "shelf_b"),
            Rect(2.45, -0.05, 0.38, 3.2, "shelf_c"),
            Rect(-3.65, 2.05, 1.15, 0.38, "pallet_stack"),
            Rect(3.55, 1.15, 1.05, 0.38, "workbench"),
        ]
        self.angles = np.deg2rad(np.linspace(-135.0, 135.0, 181)).astype(np.float32)
        self.max_range_m = 5.5

    def all_obstacles(self) -> list[Rect]:
        return [*self.static_obstacles, self.crate]

    def step(self, command: DifferentialCommand, dt: float = 0.1) -> tuple[LidarScan, ImuReading, list[CameraDetection], bool]:
        previous_command = self.last_command
        bumper = self.step_dynamics(command, dt)

        scan = self.lidar_scan(self.pose)
        imu = ImuReading(
            yaw_rate_rps=self.last_executed_command.angular_rps + float(self.rng.normal(0.0, 0.01)),
            accel_x_mps2=(self.last_executed_command.linear_mps - previous_command.linear_mps) / max(dt, 1e-3),
            accel_y_mps2=0.0,
        )
        detections = self.camera_detections(self.pose)
        return scan, imu, detections, bumper

    def step_dynamics(self, command: DifferentialCommand, dt: float = 0.1) -> bool:
        """Advance the warehouse physics surrogate without generating sensor frames."""

        self.last_command = command
        old_pose = self.pose
        contact_before = self.front_bumper_contacting_crate(old_pose)
        if command.linear_mps > 0.05 and contact_before:
            self._push_crate(command.linear_mps * dt, old_pose.yaw)

        desired = self.pose.moved(command.linear_mps, command.angular_rps, dt)
        collided = self._collides(desired.x, desired.y)
        if not collided:
            self.pose = desired
        elif command.linear_mps > 0.05 and contact_before:
            desired_after_push = old_pose.moved(command.linear_mps * 0.45, command.angular_rps, dt)
            if not self._collides(desired_after_push.x, desired_after_push.y):
                self.pose = desired_after_push
        else:
            self.pose = Pose2D(old_pose.x, old_pose.y, wrap_angle(old_pose.yaw + command.angular_rps * dt))

        traveled = old_pose.distance_to((self.pose.x, self.pose.y))
        heading_delta = wrap_angle(self.pose.yaw - old_pose.yaw)
        self.last_executed_command = DifferentialCommand(
            linear_mps=traveled / max(dt, 1e-3),
            angular_rps=heading_delta / max(dt, 1e-3),
        )
        return self.front_bumper_contacting_crate(self.pose)

    def lidar_scan(self, pose: Pose2D) -> LidarScan:
        ranges = np.full_like(self.angles, self.max_range_m, dtype=np.float32)
        for index, rel_angle in enumerate(self.angles):
            theta = pose.yaw + float(rel_angle)
            ranges[index] = self._raycast(pose.x, pose.y, theta)
        ranges += self.rng.normal(0.0, 0.015, size=ranges.shape).astype(np.float32)
        ranges = np.clip(ranges, 0.03, self.max_range_m)
        return LidarScan(ranges, self.angles.copy(), self.max_range_m)

    def camera_detections(self, pose: Pose2D) -> list[CameraDetection]:
        dx = self.crate.cx - pose.x
        dy = self.crate.cy - pose.y
        distance = hypot(dx, dy)
        bearing = wrap_angle(atan2(dy, dx) - pose.yaw)
        if distance > 4.5 or abs(bearing) > pi / 3.0:
            return []
        confidence = max(0.2, 1.0 - distance / 4.5) * (1.0 - abs(bearing) / (pi / 3.0))
        return [CameraDetection("supply_crate", distance, bearing, float(confidence))]

    def crate_in_goal(self) -> bool:
        return self.goal_zone.contains(self.crate.cx, self.crate.cy)

    def crate_goal_distance(self) -> float:
        return hypot(self.crate.cx - self.goal_zone.cx, self.crate.cy - self.goal_zone.cy)

    def front_bumper_contacting_crate(self, pose: Pose2D | None = None) -> bool:
        return self._front_contacting_crate(pose or self.pose)

    def _raycast(self, x: float, y: float, theta: float) -> float:
        step = 0.035
        distance = 0.0
        while distance < self.max_range_m:
            px = x + distance * cos(theta)
            py = y + distance * sin(theta)
            if not self.bounds.contains(px, py):
                return distance
            if any(rect.contains(px, py) for rect in self.all_obstacles()):
                return distance
            distance += step
        return self.max_range_m

    def _collides(self, x: float, y: float) -> bool:
        if not self.bounds.contains(x, y, margin=-self.robot_radius_m):
            return True
        return any(rect.contains(x, y, margin=self.robot_radius_m) for rect in self.all_obstacles())

    def _front_contacting_crate(self, pose: Pose2D) -> bool:
        nose_x = pose.x + cos(pose.yaw) * (self.robot_radius_m + 0.12)
        nose_y = pose.y + sin(pose.yaw) * (self.robot_radius_m + 0.12)
        return self.crate.contains(nose_x, nose_y, margin=0.12)

    def _push_crate(self, distance: float, yaw: float) -> None:
        new_cx = self.crate.cx + cos(yaw) * distance
        new_cy = self.crate.cy + sin(yaw) * distance
        original = (self.crate.cx, self.crate.cy)
        self.crate.cx = new_cx
        self.crate.cy = new_cy
        if any(
            rect is not self.crate and rect.contains(self.crate.cx, self.crate.cy, margin=0.32)
            for rect in self.static_obstacles
        ):
            self.crate.cx, self.crate.cy = original
