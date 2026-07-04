from __future__ import annotations

from dataclasses import dataclass, field

from .grid_map import OccupancyGridMap
from .scan_matching import CorrelativeScanMatcher, ScanMatchResult
from .types import ImuReading, LidarScan, Pose2D, wrap_angle


@dataclass
class OccupancyGridSlam:
    """SLAM front-end: odometry/IMU prediction, LiDAR scan matching, occupancy mapping."""

    initial_pose: Pose2D = Pose2D(0.0, 0.0, 0.0)
    grid: OccupancyGridMap = field(default_factory=OccupancyGridMap)
    imu_yaw_blend: float = 0.25
    scan_matcher: CorrelativeScanMatcher | None = field(default_factory=CorrelativeScanMatcher)

    def __post_init__(self) -> None:
        self.pose = self.initial_pose
        self.path: list[Pose2D] = [self.pose]
        self.last_match: ScanMatchResult | None = None

    def predict(
        self,
        linear_mps: float,
        angular_rps: float,
        dt: float,
        imu: ImuReading | None = None,
    ) -> Pose2D:
        yaw_rate = angular_rps
        if imu is not None:
            yaw_rate = (1.0 - self.imu_yaw_blend) * angular_rps + self.imu_yaw_blend * imu.yaw_rate_rps
        self.pose = self.pose.moved(linear_mps, yaw_rate, dt)
        self.pose = Pose2D(self.pose.x, self.pose.y, wrap_angle(self.pose.yaw))
        self.path.append(self.pose)
        return self.pose

    def correct(self, scan: LidarScan) -> Pose2D:
        if self.scan_matcher is None:
            return self.pose
        self.last_match = self.scan_matcher.match(self.grid, self.pose, scan)
        self.pose = self.last_match.pose
        self.path[-1] = self.pose
        return self.pose

    def update(self, scan: LidarScan) -> None:
        self.grid.update_with_scan(self.pose, scan)

    def step(
        self,
        linear_mps: float,
        angular_rps: float,
        dt: float,
        scan: LidarScan,
        imu: ImuReading | None = None,
    ) -> Pose2D:
        pose = self.predict(linear_mps, angular_rps, dt, imu)
        pose = self.correct(scan)
        self.update(scan)
        return pose


DeadReckoningSlam = OccupancyGridSlam
