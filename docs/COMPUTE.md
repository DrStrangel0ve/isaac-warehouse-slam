# Compute Workflow

The default compute lane is the local RTX 3080 Ti. Kaggle is optional and should be used for perception-model training or sweeps, not interactive Isaac Sim.

## Check Local And Kaggle Status

```powershell
python scripts\check_compute.py --kaggle --json-out artifacts\compute_status.json
```

The script checks:

- `nvidia-smi` GPU utilization, memory use, and visible compute processes.
- Kaggle CLI/package availability.
- Kaggle API reachability through the local credential file.
- Recent Kaggle kernels and their statuses when `--kaggle` is supplied.

## Decide Where Work Should Run

- Isaac Sim scene runs: local RTX 3080 Ti.
- Offline SLAM tests: local CPU.
- Tabular Q-learning policy: local CPU is enough.
- Neural perception training from Isaac data: local RTX 3080 Ti first, Kaggle if the run is long or the local GPU is busy.
- Kaggle notebooks: verify current accelerator availability in the Kaggle notebook Settings pane before counting on GPU time.

## Keep Compute Busy Usefully

Good active-work commands for this repo:

```powershell
C:\iw\Scripts\python.exe scripts\train_rl_push_policy.py --episodes 300 --eval-episodes 30 --seed 7 --max-steps 80
C:\iw\Scripts\python.exe scripts\run_offline_slam_demo.py --steps 500 --rl-policy artifacts\rl_push_q_table.json
$env:OMNI_KIT_ACCEPT_EULA="YES"; C:\iw\Scripts\python.exe scripts\run_isaac_warehouse_slam.py --headless --steps 100 --save-map-every 0 --rl-policy artifacts\rl_push_q_table.json
```

For headed demo capture:

```powershell
$env:OMNI_KIT_ACCEPT_EULA="YES"; C:\iw\Scripts\python.exe scripts\run_isaac_warehouse_slam.py --headed --steps 300 --save-map-every 150 --rl-policy artifacts\rl_push_q_table.json
```
