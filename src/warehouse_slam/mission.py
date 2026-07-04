from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin
from typing import Protocol

from .control import DifferentialCommand, WaypointController
from .planning import AStarPlanner
from .slam import OccupancyGridSlam
from .types import CameraDetection, Pose2D


class LocalPushPolicy(Protocol):
    def command(
        self,
        pose: Pose2D,
        crate_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        bumper_active: bool,
    ) -> DifferentialCommand: ...


@dataclass
class MissionStatus:
    mode: str = "explore"
    active_goal: tuple[float, float] | None = None
    crate_estimate: tuple[float, float] | None = None
    path: list[tuple[float, float]] = field(default_factory=list)
    waypoints_reached: int = 0


class WarehouseMissionPlanner:
    """Frontier exploration until the crate is seen, then push it toward the goal."""

    def __init__(
        self,
        goal_zone_xy: tuple[float, float] = (4.25, 2.65),
        planner: AStarPlanner | None = None,
        controller: WaypointController | None = None,
        rl_push_policy: LocalPushPolicy | None = None,
        rl_activation_distance_m: float = 1.15,
    ) -> None:
        self.goal_zone_xy = goal_zone_xy
        self.planner = planner or AStarPlanner()
        self.controller = controller or WaypointController()
        self.rl_push_policy = rl_push_policy
        self.rl_activation_distance_m = rl_activation_distance_m
        self.status = MissionStatus()

    def update(
        self,
        slam: OccupancyGridSlam,
        detections: list[CameraDetection],
        bumper_active: bool = False,
    ) -> DifferentialCommand:
        pose = slam.pose
        self._update_crate_estimate(pose, detections)
        if self.status.crate_estimate is not None:
            self.status.mode = "push_crate" if bumper_active else "approach_crate"

        if self.status.mode in {"approach_crate", "push_crate"}:
            return self._crate_command(pose, bumper_active)
        return self._explore_command(slam)

    def _update_crate_estimate(self, pose: Pose2D, detections: list[CameraDetection]) -> None:
        for detection in detections:
            if detection.label != "supply_crate" or detection.confidence < 0.15:
                continue
            world_bearing = pose.yaw + detection.bearing_rad
            estimate = (
                pose.x + cos(world_bearing) * detection.range_m,
                pose.y + sin(world_bearing) * detection.range_m,
            )
            if self.status.crate_estimate is None:
                self.status.crate_estimate = estimate
            else:
                old = self.status.crate_estimate
                self.status.crate_estimate = (old[0] * 0.65 + estimate[0] * 0.35, old[1] * 0.65 + estimate[1] * 0.35)

    def _crate_command(self, pose: Pose2D, bumper_active: bool = False) -> DifferentialCommand:
        assert self.status.crate_estimate is not None
        crate = self.status.crate_estimate
        goal = self.goal_zone_xy
        if self.rl_push_policy is not None and pose.distance_to(crate) < self.rl_activation_distance_m:
            self.status.mode = "rl_push_crate" if bumper_active else "rl_approach_crate"
            self.status.active_goal = goal
            self.status.path = [crate, goal]
            return self.rl_push_policy.command(pose, crate, goal, bumper_active=bumper_active)

        push_dx = goal[0] - crate[0]
        push_dy = goal[1] - crate[1]
        norm = max((push_dx**2 + push_dy**2) ** 0.5, 1e-6)
        ux = push_dx / norm
        uy = push_dy / norm
        approach = (crate[0] - 0.65 * ux, crate[1] - 0.65 * uy)
        pose_from_crate = (pose.x - crate[0], pose.y - crate[1])
        lateral_error = abs(pose_from_crate[0] * -uy + pose_from_crate[1] * ux)
        behind_crate = pose_from_crate[0] * ux + pose_from_crate[1] * uy < 0.1
        if pose.distance_to(crate) < 0.82 and lateral_error < 0.22 and behind_crate:
            self.status.mode = "push_crate"
            heading_error = pose.heading_error_to(goal)
            angular = max(-1.0, min(1.0, 1.6 * heading_error))
            linear = 0.34 if abs(heading_error) < 0.55 else 0.06
            self.status.active_goal = goal
            self.status.path = [goal]
            return DifferentialCommand(linear, angular)

        target = crate if pose.distance_to(approach) < 0.35 else approach
        command = self.controller.command(pose, target)
        if pose.distance_to(crate) < 0.55:
            self.status.mode = "push_crate"
            command = DifferentialCommand(min(0.32, max(0.16, command.linear_mps + 0.12)), command.angular_rps)
        self.status.active_goal = target
        self.status.path = [target]
        return command

    def _explore_command(self, slam: OccupancyGridSlam) -> DifferentialCommand:
        pose = slam.pose
        if not self.status.path or pose.distance_to(self.status.path[0]) < self.controller.waypoint_tolerance_m:
            if self.status.path:
                self.status.path.pop(0)
                self.status.waypoints_reached += 1
            if not self.status.path:
                goal = self.planner.choose_frontier_goal(slam.grid, (pose.x, pose.y))
                if goal is None:
                    return DifferentialCommand(0.0, 0.45)
                planned = self.planner.plan(slam.grid, (pose.x, pose.y), goal)
                if planned is None or len(planned.cells) < 2:
                    self.status.active_goal = goal
                    self.status.path = [goal]
                else:
                    world_path = planned.as_world_points(slam.grid)
                    world_path = [
                        point
                        for point in world_path[1:]
                        if pose.distance_to(point) > self.controller.waypoint_tolerance_m * 1.25
                    ]
                    self.status.active_goal = goal
                    self.status.path = world_path[:18:3] or [goal]

        if not self.status.path:
            return DifferentialCommand(0.0, 0.35)
        return self.controller.command(pose, self.status.path[0])
