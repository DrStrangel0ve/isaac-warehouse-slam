from warehouse_slam.grid_map import OccupancyGridMap
from warehouse_slam.scan_matching import CorrelativeScanMatcher
from warehouse_slam.sim2d import Warehouse2DSim
from warehouse_slam.types import Pose2D


def test_correlative_scan_matcher_reduces_pose_bias():
    sim = Warehouse2DSim(seed=9)
    grid = OccupancyGridMap(width_m=11.0, height_m=8.5, resolution_m=0.08)
    true_pose = Pose2D(-4.05, -2.72, 0.28)
    scan = sim.lidar_scan(true_pose)
    grid.update_with_scan(true_pose, scan)

    biased_pose = Pose2D(true_pose.x + 0.12, true_pose.y - 0.06, true_pose.yaw + 0.08)
    matcher = CorrelativeScanMatcher(min_known_fraction=0.001, min_score_delta=0.01)
    result = matcher.match(grid, biased_pose, scan)

    assert result.accepted
    assert result.pose.distance_to((true_pose.x, true_pose.y)) < biased_pose.distance_to(
        (true_pose.x, true_pose.y)
    )

