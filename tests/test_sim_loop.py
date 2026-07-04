from warehouse_slam.control import DifferentialCommand
from warehouse_slam.grid_map import OccupancyGridMap
from warehouse_slam.mission import WarehouseMissionPlanner
from warehouse_slam.sim2d import Warehouse2DSim
from warehouse_slam.slam import OccupancyGridSlam
from warehouse_slam.types import Pose2D


def test_offline_slam_loop_builds_map_and_moves_robot():
    sim = Warehouse2DSim(seed=2)
    slam = OccupancyGridSlam(
        initial_pose=Pose2D(sim.pose.x, sim.pose.y, sim.pose.yaw),
        grid=OccupancyGridMap(width_m=11.0, height_m=8.5, resolution_m=0.1),
    )
    mission = WarehouseMissionPlanner(goal_zone_xy=(sim.goal_zone.cx, sim.goal_zone.cy))
    command = DifferentialCommand(0.0, 0.0)

    for _ in range(80):
        scan, imu, detections, bumper = sim.step(command, 0.1)
        odom = sim.last_executed_command
        slam.step(odom.linear_mps, odom.angular_rps, 0.1, scan, imu)
        command = mission.update(slam, detections, bumper)

    assert slam.grid.explored_fraction() > 0.01
    assert sim.pose.distance_to((-4.25, -2.85)) > 0.1
    assert mission.status.mode in {"explore", "approach_crate", "push_crate"}
