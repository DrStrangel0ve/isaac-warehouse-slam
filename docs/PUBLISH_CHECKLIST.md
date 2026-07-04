# Public Release Checklist

## Before Publishing

- `python -m pytest -q`
- `python -m ruff check .`
- `python scripts\check_compute.py --kaggle --json-out artifacts\compute_status.json`
- `python scripts\render_rl_curve.py`
- `C:\iw\Scripts\python.exe scripts\run_offline_slam_demo.py --steps 500 --rl-policy artifacts\rl_push_q_table.json`

## Nice To Have

- Headed Isaac demo clip.
- GitHub social preview image.
- GitHub repo topics:
  - `isaac-sim`
  - `robotics`
  - `slam`
  - `reinforcement-learning`
  - `path-planning`
  - `rtx-lidar`
  - `portfolio-project`

## Repo Description

Isaac Sim warehouse autonomy demo with multi-sensor SLAM, frontier/A* navigation, and learned Q-learning crate pushing.

## Release Notes

Initial public release includes:

- Pure-Python SLAM, planning, simulation, and RL modules.
- Isaac Sim 6 runner with RGB/depth camera, RTX LiDAR prim, IMU/contact bridges, JetBot, movable crate, and goal zone.
- Offline deterministic fallback.
- RL policy artifacts, training metrics, maps, and telemetry.
- Tests and reproducible validation commands.
