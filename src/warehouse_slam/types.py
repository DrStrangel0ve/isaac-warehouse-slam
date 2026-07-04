from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def moved(self, linear_mps: float, angular_rps: float, dt: float) -> "Pose2D":
        if abs(angular_rps) < 1e-6:
            dx = linear_mps * dt * cos(self.yaw)
            dy = linear_mps * dt * sin(self.yaw)
            return Pose2D(self.x + dx, self.y + dy, wrap_angle(self.yaw))

        radius = linear_mps / angular_rps
        new_yaw = self.yaw + angular_rps * dt
        dx = radius * (sin(new_yaw) - sin(self.yaw))
        dy = -radius * (cos(new_yaw) - cos(self.yaw))
        return Pose2D(self.x + dx, self.y + dy, wrap_angle(new_yaw))

    def distance_to(self, xy: tuple[float, float]) -> float:
        return hypot(float(xy[0]) - self.x, float(xy[1]) - self.y)

    def heading_error_to(self, xy: tuple[float, float]) -> float:
        return wrap_angle(atan2(float(xy[1]) - self.y, float(xy[0]) - self.x) - self.yaw)

    def transform_points(self, points_xy: np.ndarray) -> np.ndarray:
        c = cos(self.yaw)
        s = sin(self.yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        return points_xy @ rot.T + np.array([self.x, self.y], dtype=np.float32)


@dataclass(frozen=True)
class LidarScan:
    ranges_m: np.ndarray
    angles_rad: np.ndarray
    max_range_m: float

    def valid_points_robot_frame(self) -> np.ndarray:
        ranges = np.asarray(self.ranges_m, dtype=np.float32)
        angles = np.asarray(self.angles_rad, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges > 0.02)
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        return np.stack([x, y], axis=1).astype(np.float32)


@dataclass(frozen=True)
class ImuReading:
    yaw_rate_rps: float
    accel_x_mps2: float
    accel_y_mps2: float


@dataclass(frozen=True)
class CameraDetection:
    label: str
    range_m: float
    bearing_rad: float
    confidence: float


def wrap_angle(angle: float) -> float:
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle

