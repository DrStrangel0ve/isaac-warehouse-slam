from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import hypot

import numpy as np

from .grid_map import GridCell, OccupancyGridMap


@dataclass(frozen=True)
class PlannedPath:
    cells: list[GridCell]

    def as_world_points(self, grid: OccupancyGridMap) -> list[tuple[float, float]]:
        return [grid.grid_to_world(cell) for cell in self.cells]


class AStarPlanner:
    def __init__(self, obstacle_inflation_cells: int = 3) -> None:
        self.obstacle_inflation_cells = obstacle_inflation_cells

    def plan(
        self,
        grid: OccupancyGridMap,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> PlannedPath | None:
        start = grid.world_to_grid(*start_xy)
        goal = grid.world_to_grid(*goal_xy)
        if start is None or goal is None:
            return None

        blocked = grid.inflated_obstacles(self.obstacle_inflation_cells)
        blocked[start.y, start.x] = False
        blocked[goal.y, goal.x] = False
        known_free = grid.free_mask() | _disc_mask(grid.height, grid.width, start.x, start.y, 5)

        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, (start.x, start.y))]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {(start.x, start.y): None}
        cost_so_far: dict[tuple[int, int], float] = {(start.x, start.y): 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == (goal.x, goal.y):
                return PlannedPath(_reconstruct(came_from, current))

            for nx, ny, step_cost in _neighbors(current[0], current[1], grid.width, grid.height):
                if blocked[ny, nx]:
                    continue
                if not known_free[ny, nx] and (nx, ny) != (goal.x, goal.y):
                    continue
                new_cost = cost_so_far[current] + step_cost
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + _heuristic((nx, ny), (goal.x, goal.y))
                    heapq.heappush(frontier, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current
        return None

    def choose_frontier_goal(
        self,
        grid: OccupancyGridMap,
        robot_xy: tuple[float, float],
    ) -> tuple[float, float] | None:
        frontiers = grid.frontier_cells()
        if not frontiers:
            return None
        robot_cell = grid.world_to_grid(*robot_xy)
        if robot_cell is None:
            return None
        frontiers.sort(
            key=lambda c: hypot(c.x - robot_cell.x, c.y - robot_cell.y)
            - 0.08 * _unknown_neighbor_count(grid, c),
        )
        return grid.grid_to_world(frontiers[0])


def _neighbors(x: int, y: int, width: int, height: int):
    for dx, dy, cost in (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, 1.414),
        (1, -1, 1.414),
        (-1, 1, 1.414),
        (-1, -1, 1.414),
    ):
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny, cost


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    current: tuple[int, int],
) -> list[GridCell]:
    cells: list[GridCell] = []
    while current is not None:
        cells.append(GridCell(current[0], current[1]))
        current = came_from[current]
    cells.reverse()
    return cells


def _disc_mask(height: int, width: int, cx: int, cy: int, radius: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                mask[y, x] = True
    return mask


def _unknown_neighbor_count(grid: OccupancyGridMap, cell: GridCell) -> int:
    known = grid.known_mask()
    y0 = max(0, cell.y - 3)
    y1 = min(grid.height, cell.y + 4)
    x0 = max(0, cell.x - 3)
    x1 = min(grid.width, cell.x + 4)
    return int((~known[y0:y1, x0:x1]).sum())

