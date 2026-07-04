# Portfolio Brief

## Project

Isaac Warehouse SLAM + RL Crate Pushing

## One-Sentence Pitch

A differential-drive robot in Isaac Sim fuses LiDAR, camera, IMU, and bumper signals to map a warehouse, navigate with frontier/A* planning, and use a learned Q-learning policy for local crate pushing.

## Why It Matters

This project combines robotics fundamentals and applied AI in one reproducible system:

- Mapping: log-odds occupancy grid from range scans.
- Localization: odometry and IMU prediction corrected by local correlative scan matching.
- Planning: frontier goal selection and A* path planning.
- Interaction: crate approach and pushing through contact-rich simulation.
- Learning: a Q-learning policy controls local push behavior near the crate.
- Evaluation: metrics, tests, maps, telemetry, and Isaac smoke runs.

## Key Results

- 96.7% greedy RL evaluation success over 30 held-out randomized local starts.
- 157 visited Q states after 300 training episodes.
- Offline SLAM + RL mission reached the crate goal at step 67.
- Isaac Sim headed smoke run created RGB/depth camera, RTX LiDAR, IMU/contact sensor bridges, loaded the learned policy, ran 120 simulator steps, and saved map/telemetry artifacts.
- Python test suite covers map logic, planning, scan matching, simulation loop, and RL policy behavior.

## Resume Bullets

- Built an Isaac Sim warehouse autonomy stack where a differential-drive robot fuses LiDAR, RGB/depth camera detections, IMU yaw, and bumper contact to perform occupancy-grid SLAM, frontier exploration, A* navigation, and learned local crate pushing.
- Trained and evaluated a Q-learning crate-interaction policy with 157 discrete Q states, reaching 96.7% held-out success across randomized starts and integrating the policy into the same mission loop used by the Isaac Sim robot.

## Interview Talking Points

- Why the core autonomy stack is pure Python and simulator-independent.
- How scan matching corrects odometry drift.
- Why the learned policy is scoped to local interaction instead of replacing the whole autonomy stack.
- How Isaac sensor APIs are bridged with analytic fallbacks for reproducibility.
- What the next step would be: train visual perception from Isaac synthetic RGB-D/segmentation data and replace the analytic crate detector.
