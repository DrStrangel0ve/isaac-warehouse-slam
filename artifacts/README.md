# Artifacts

These files are intentionally tracked because they let a reviewer understand the project quickly without rerunning Isaac Sim.

- `offline_slam_map.png` - deterministic offline SLAM occupancy map.
- `offline_metrics.json` - offline SLAM + RL success metrics.
- `offline_telemetry.csv` - offline telemetry sampled during the mission.
- `rl_push_q_table.json` - learned Q-table policy used for local crate pushing.
- `rl_training_curve.csv` - episode-level RL training history.
- `rl_training_curve.png` - rendered training curve for README/demo use.
- `rl_training_metrics.json` - held-out greedy policy evaluation metrics.
- `isaac_final_map.png` - final Isaac Sim map artifact from a simulator smoke run.
- `isaac_telemetry.csv` - telemetry from the Isaac Sim smoke run.
- `portfolio_hero.png` - visual README hero combining rollout, map, learning curve, and metrics.
- `warehouse_storyboard.png` - top-down mission storyboard from setup to learned crate push.

Local-only files such as `compute_status.json` and transient maps such as `isaac_map_step_*.png` are ignored by default. Regenerate maps with `scripts/run_isaac_warehouse_slam.py --save-map-every`.
