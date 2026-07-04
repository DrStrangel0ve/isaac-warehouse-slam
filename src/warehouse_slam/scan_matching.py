from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid_map import OccupancyGridMap
from .types import LidarScan, Pose2D, wrap_angle


@dataclass(frozen=True)
class ScanMatchResult:
    pose: Pose2D
    score: float
    candidates: int
    accepted: bool


@dataclass
class CorrelativeScanMatcher:
    """Brute-force local scan matcher against occupied cells in the map."""

    xy_window_m: float = 0.18
    yaw_window_rad: float = 0.16
    xy_step_m: float = 0.06
    yaw_step_rad: float = 0.04
    min_known_fraction: float = 0.025
    min_score_delta: float = 0.15

    def match(
        self,
        grid: OccupancyGridMap,
        predicted_pose: Pose2D,
        scan: LidarScan,
    ) -> ScanMatchResult:
        if grid.known_mask().mean() < self.min_known_fraction:
            return ScanMatchResult(predicted_pose, score=0.0, candidates=0, accepted=False)

        hit_points = _hit_points_robot_frame(scan)
        if len(hit_points) < 6:
            return ScanMatchResult(predicted_pose, score=0.0, candidates=0, accepted=False)

        base_score = self._score_pose(grid, predicted_pose, hit_points)
        best_pose = predicted_pose
        best_score = base_score
        candidates = 0
        xy_offsets = np.arange(-self.xy_window_m, self.xy_window_m + 1e-6, self.xy_step_m)
        yaw_offsets = np.arange(-self.yaw_window_rad, self.yaw_window_rad + 1e-6, self.yaw_step_rad)

        for dx in xy_offsets:
            for dy in xy_offsets:
                for dyaw in yaw_offsets:
                    candidates += 1
                    candidate = Pose2D(
                        predicted_pose.x + float(dx),
                        predicted_pose.y + float(dy),
                        wrap_angle(predicted_pose.yaw + float(dyaw)),
                    )
                    score = self._score_pose(grid, candidate, hit_points)
                    if score > best_score:
                        best_score = score
                        best_pose = candidate

        accepted = best_score >= base_score + self.min_score_delta
        return ScanMatchResult(
            pose=best_pose if accepted else predicted_pose,
            score=best_score,
            candidates=candidates,
            accepted=accepted,
        )

    def _score_pose(
        self,
        grid: OccupancyGridMap,
        pose: Pose2D,
        points_robot_frame: np.ndarray,
    ) -> float:
        points_world = pose.transform_points(points_robot_frame)
        score = 0.0
        used = 0
        for x, y in points_world:
            cell = grid.world_to_grid(float(x), float(y))
            if cell is None:
                continue
            local = _local_occupied_score(grid.log_odds, cell.x, cell.y)
            score += local
            used += 1
        if used == 0:
            return -1e9
        return score / used


def _hit_points_robot_frame(scan: LidarScan) -> np.ndarray:
    ranges = np.asarray(scan.ranges_m, dtype=np.float32)
    angles = np.asarray(scan.angles_rad, dtype=np.float32)
    hit = np.isfinite(ranges) & (ranges > 0.03) & (ranges < scan.max_range_m * 0.985)
    if not hit.any():
        return np.empty((0, 2), dtype=np.float32)
    x = ranges[hit] * np.cos(angles[hit])
    y = ranges[hit] * np.sin(angles[hit])
    return np.stack([x, y], axis=1).astype(np.float32)


def _local_occupied_score(log_odds: np.ndarray, x: int, y: int) -> float:
    y0 = max(0, y - 1)
    y1 = min(log_odds.shape[0], y + 2)
    x0 = max(0, x - 1)
    x1 = min(log_odds.shape[1], x + 2)
    return float(log_odds[y0:y1, x0:x1].max())

