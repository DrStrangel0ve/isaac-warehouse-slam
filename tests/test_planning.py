import numpy as np

from warehouse_slam.grid_map import OccupancyGridMap
from warehouse_slam.planning import AStarPlanner


def test_a_star_plans_around_known_obstacle():
    grid = OccupancyGridMap(width_m=4.0, height_m=4.0, resolution_m=0.2)
    grid.log_odds[:, :] = -1.0
    wall_x = grid.width // 2
    grid.log_odds[:, wall_x] = 2.0
    grid.log_odds[grid.height // 2, wall_x] = -1.0

    planner = AStarPlanner(obstacle_inflation_cells=0)
    path = planner.plan(grid, (-1.4, 0.0), (1.4, 0.0))

    assert path is not None
    assert len(path.cells) > 2
    assert any(cell.x == wall_x and cell.y == grid.height // 2 for cell in path.cells)


def test_choose_frontier_goal_returns_known_frontier():
    grid = OccupancyGridMap(width_m=4.0, height_m=4.0, resolution_m=0.2)
    grid.log_odds[:, :] = 0.0
    grid.log_odds[7:13, 7:13] = -1.0
    planner = AStarPlanner()

    goal = planner.choose_frontier_goal(grid, (0.0, 0.0))

    assert goal is not None
    assert np.isfinite(goal[0])

