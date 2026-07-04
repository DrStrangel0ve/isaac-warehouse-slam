# Demo Capture Checklist

Use this when recording a short demo for GitHub, a portfolio page, or an internship application.

## Recommended Clip

Target length: 45 to 75 seconds.

Show:

1. Isaac Sim warehouse scene with JetBot, crate, goal zone, and obstacles.
2. Robot exploring and updating map artifacts.
3. Transition into crate approach or RL crate push mode.
4. Final artifact view: map image, training curve, and metrics table.

## Headed Isaac Run

```powershell
cd C:\Users\arnav\Documents\Codex\2026-06-15\that-is-internship-legit-to-make\outputs\isaac-warehouse-slam
$env:OMNI_KIT_ACCEPT_EULA="YES"
C:\iw\Scripts\python.exe scripts\run_isaac_warehouse_slam.py --headed --steps 300 --save-map-every 150 --rl-policy artifacts\rl_push_q_table.json
```

If the headed run is too heavy, capture the headless artifact path instead:

```powershell
$env:OMNI_KIT_ACCEPT_EULA="YES"
C:\iw\Scripts\python.exe scripts\run_isaac_warehouse_slam.py --headless --steps 100 --save-map-every 0 --rl-policy artifacts\rl_push_q_table.json
```

## What To Say In The Demo

One tight version:

> This is an Isaac Sim warehouse autonomy demo. The robot uses LiDAR, RGB/depth camera detections, IMU yaw, and bumper contact to build a 2D occupancy map, plan with frontiers and A*, then switch to a learned Q-table policy for local crate pushing. The pure-Python stack is tested offline, and the same mission stack runs in Isaac Sim with sensor bridges and artifact logging.

## Files To Show

- `artifacts/offline_slam_map.png`
- `artifacts/rl_training_curve.png`
- `artifacts/rl_training_metrics.json`
- `artifacts/isaac_final_map.png`
