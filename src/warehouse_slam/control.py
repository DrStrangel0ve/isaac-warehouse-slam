from __future__ import annotations

from dataclasses import dataclass

from .types import Pose2D


@dataclass(frozen=True)
class DifferentialCommand:
    linear_mps: float
    angular_rps: float


@dataclass
class WaypointController:
    max_linear_mps: float = 0.55
    max_angular_rps: float = 1.35
    linear_gain: float = 0.9
    angular_gain: float = 1.8
    waypoint_tolerance_m: float = 0.18

    def command(self, pose: Pose2D, waypoint_xy: tuple[float, float]) -> DifferentialCommand:
        distance = pose.distance_to(waypoint_xy)
        heading_error = pose.heading_error_to(waypoint_xy)
        if distance < self.waypoint_tolerance_m:
            return DifferentialCommand(0.0, 0.0)
        angular = _clamp(self.angular_gain * heading_error, -self.max_angular_rps, self.max_angular_rps)
        linear = _clamp(self.linear_gain * distance, 0.0, self.max_linear_mps)
        if abs(heading_error) > 0.65:
            linear *= 0.2
        elif abs(heading_error) > 0.35:
            linear *= 0.55
        return DifferentialCommand(linear, angular)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))

