from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, sin
from pathlib import Path

import numpy as np
from PIL import Image

from .types import LidarScan, Pose2D


@dataclass(frozen=True)
class GridCell:
    x: int
    y: int


class OccupancyGridMap:
    """Log-odds 2D occupancy map built from planar range scans."""

    def __init__(
        self,
        width_m: float = 12.0,
        height_m: float = 10.0,
        resolution_m: float = 0.08,
        origin_xy: tuple[float, float] | None = None,
    ) -> None:
        self.resolution_m = float(resolution_m)
        self.width = int(round(width_m / resolution_m))
        self.height = int(round(height_m / resolution_m))
        if origin_xy is None:
            origin_xy = (-width_m / 2.0, -height_m / 2.0)
        self.origin_x = float(origin_xy[0])
        self.origin_y = float(origin_xy[1])
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.min_log_odds = -4.0
        self.max_log_odds = 4.0
        self.free_update = -0.35
        self.occupied_update = 0.9

    def world_to_grid(self, x: float, y: float) -> GridCell | None:
        gx = int((x - self.origin_x) / self.resolution_m)
        gy = int((y - self.origin_y) / self.resolution_m)
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return GridCell(gx, gy)
        return None

    def grid_to_world(self, cell: GridCell | tuple[int, int]) -> tuple[float, float]:
        gx, gy = (cell.x, cell.y) if isinstance(cell, GridCell) else cell
        return (
            self.origin_x + (gx + 0.5) * self.resolution_m,
            self.origin_y + (gy + 0.5) * self.resolution_m,
        )

    def update_with_scan(self, pose: Pose2D, scan: LidarScan) -> None:
        robot_cell = self.world_to_grid(pose.x, pose.y)
        if robot_cell is None:
            return

        for raw_range, beam_angle in zip(scan.ranges_m, scan.angles_rad):
            if not np.isfinite(raw_range) or raw_range <= 0.0:
                continue
            hit = float(raw_range) < scan.max_range_m * 0.985
            clipped = min(float(raw_range), scan.max_range_m)
            theta = pose.yaw + float(beam_angle)
            end_x = pose.x + clipped * cos(theta)
            end_y = pose.y + clipped * sin(theta)
            end_cell = self.world_to_grid(end_x, end_y)
            if end_cell is None:
                continue

            ray = list(_bresenham(robot_cell.x, robot_cell.y, end_cell.x, end_cell.y))
            for gx, gy in ray[:-1]:
                self.log_odds[gy, gx] += self.free_update
            if hit:
                self.log_odds[end_cell.y, end_cell.x] += self.occupied_update

        np.clip(self.log_odds, self.min_log_odds, self.max_log_odds, out=self.log_odds)

    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def known_mask(self) -> np.ndarray:
        return np.abs(self.log_odds) > 0.15

    def free_mask(self) -> np.ndarray:
        return self.log_odds < -0.25

    def occupied_mask(self) -> np.ndarray:
        return self.log_odds > 0.35

    def inflated_obstacles(self, radius_cells: int = 2) -> np.ndarray:
        occupied = self.occupied_mask()
        if radius_cells <= 0:
            return occupied
        inflated = occupied.copy()
        ys, xs = np.nonzero(occupied)
        for y, x in zip(ys, xs):
            y0 = max(0, y - radius_cells)
            y1 = min(self.height, y + radius_cells + 1)
            x0 = max(0, x - radius_cells)
            x1 = min(self.width, x + radius_cells + 1)
            inflated[y0:y1, x0:x1] = True
        return inflated

    def frontier_cells(self, min_cluster_size: int = 4) -> list[GridCell]:
        free = self.free_mask()
        unknown = ~self.known_mask()
        frontier = np.zeros_like(free, dtype=bool)
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if not free[y, x]:
                    continue
                neighborhood = unknown[y - 1 : y + 2, x - 1 : x + 2]
                frontier[y, x] = bool(neighborhood.any())

        visited = np.zeros_like(frontier, dtype=bool)
        centroids: list[GridCell] = []
        for y, x in zip(*np.nonzero(frontier)):
            if visited[y, x]:
                continue
            cluster = _flood_fill(frontier, visited, int(x), int(y))
            if len(cluster) < min_cluster_size:
                continue
            mean_x = int(round(sum(c[0] for c in cluster) / len(cluster)))
            mean_y = int(round(sum(c[1] for c in cluster) / len(cluster)))
            centroids.append(GridCell(mean_x, mean_y))
        return centroids

    def explored_fraction(self) -> float:
        return float(self.known_mask().mean())

    def to_image(self, path: str | Path, robot_pose: Pose2D | None = None) -> None:
        prob = self.probability()
        image = np.full((self.height, self.width, 3), 160, dtype=np.uint8)
        image[prob < 0.42] = (245, 245, 245)
        image[prob > 0.58] = (30, 30, 30)
        frontiers = self.frontier_cells(min_cluster_size=2)
        for cell in frontiers:
            if 0 <= cell.x < self.width and 0 <= cell.y < self.height:
                image[cell.y, cell.x] = (35, 140, 255)
        if robot_pose is not None:
            cell = self.world_to_grid(robot_pose.x, robot_pose.y)
            if cell is not None:
                _draw_disc(image, cell.x, cell.y, radius=3, color=(255, 60, 60))
        image = np.flipud(image)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(path)


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _flood_fill(mask: np.ndarray, visited: np.ndarray, start_x: int, start_y: int) -> list[tuple[int, int]]:
    q: deque[tuple[int, int]] = deque([(start_x, start_y)])
    visited[start_y, start_x] = True
    cluster: list[tuple[int, int]] = []
    while q:
        x, y = q.popleft()
        cluster.append((x, y))
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0]:
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((nx, ny))
    return cluster


def _draw_disc(image: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    y0 = max(0, cy - radius)
    y1 = min(image.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(image.shape[1], cx + radius + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                image[y, x] = color

