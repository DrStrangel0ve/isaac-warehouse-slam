from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_slam.rl import evaluate_policy, train_push_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    policy, history = train_push_policy(
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    metrics = evaluate_policy(
        policy,
        episodes=args.eval_episodes,
        seed=args.seed + 1000,
        max_steps=args.max_steps,
    )
    metrics["training_episodes"] = args.episodes
    metrics["q_states"] = len(policy.q_table)

    q_path = args.artifact_dir / "rl_push_q_table.json"
    curve_path = args.artifact_dir / "rl_training_curve.csv"
    metrics_path = args.artifact_dir / "rl_training_metrics.json"
    policy.save(q_path)
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    recent = history[-min(25, len(history)) :]
    recent_success_rate = sum(int(row["success"]) for row in recent) / max(len(recent), 1)
    print(f"trained_episodes={args.episodes}")
    print(f"q_states={len(policy.q_table)}")
    print(f"recent_train_success_rate={recent_success_rate:.3f}")
    print(f"eval_success_rate={metrics['success_rate']:.3f}")
    print(f"mean_final_crate_goal_error_m={metrics['mean_final_crate_goal_error_m']:.3f}")
    print(f"q_table={q_path}")
    print(f"curve={curve_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
