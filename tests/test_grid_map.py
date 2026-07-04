import numpy as np

from warehouse_slam.grid_map import OccupancyGridMap
from warehouse_slam.types import LidarScan, Pose2D


def test_scan_update_marks_free_and_occupied_cells():
    grid = OccupancyGridMap(width_m=4.0, height_m=4.0, resolution_m=0.1)
    pose = Pose2D(0.0, 0.0, 0.0)
    scan = LidarScan(
        ranges_m=np.array([1.0], dtype=np.float32),
        angles_rad=np.array([0.0], dtype=np.float32),
        max_range_m=3.0,
    )

    grid.update_with_scan(pose, scan)

    hit_cell = grid.world_to_grid(1.0, 0.0)
    free_cell = grid.world_to_grid(0.5, 0.0)
    assert hit_cell is not None
    assert free_cell is not None
    assert grid.log_odds[hit_cell.y, hit_cell.x] > 0.0
    assert grid.log_odds[free_cell.y, free_cell.x] < 0.0


def test_frontiers_exist_after_partial_scan():
    grid = OccupancyGridMap(width_m=4.0, height_m=4.0, resolution_m=0.1)
    pose = Pose2D(0.0, 0.0, 0.0)
    angles = np.deg2rad(np.linspace(-60, 60, 9)).astype(np.float32)
    scan = LidarScan(
        ranges_m=np.full(angles.shape, 1.5, dtype=np.float32),
        angles_rad=angles,
        max_range_m=3.0,
    )

    grid.update_with_scan(pose, scan)

    assert grid.explored_fraction() > 0.0
    assert grid.frontier_cells(min_cluster_size=1)

