from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .control import DifferentialCommand
from .sim2d import Warehouse2DSim
from .types import Pose2D, wrap_angle


@dataclass(frozen=True)
class PushAction:
    name: str
    linear_mps: float
    angular_rps: float

    def command(self) -> DifferentialCommand:
        return DifferentialCommand(self.linear_mps, self.angular_rps)


DEFAULT_PUSH_ACTIONS: tuple[PushAction, ...] = (
    PushAction("turn_left", 0.0, 1.15),
    PushAction("turn_right", 0.0, -1.15),
    PushAction("forward", 0.38, 0.0),
    PushAction("arc_left", 0.28, 0.65),
    PushAction("arc_right", 0.28, -0.65),
    PushAction("reverse", -0.14, 0.0),
)


@dataclass(frozen=True)
class PushStateDiscretizer:
    angle_bins: int = 8

    def state(
        self,
        pose: Pose2D,
        crate_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        bumper_active: bool,
    ) -> tuple[int, int, int, int, int, int]:
        crate_dx = crate_xy[0] - pose.x
        crate_dy = crate_xy[1] - pose.y
        crate_distance = math.hypot(crate_dx, crate_dy)
        crate_bearing = wrap_angle(math.atan2(crate_dy, crate_dx) - pose.yaw)

        push_dx = goal_xy[0] - crate_xy[0]
        push_dy = goal_xy[1] - crate_xy[1]
        push_yaw = math.atan2(push_dy, push_dx)
        push_heading_error = wrap_angle(push_yaw - pose.yaw)
        push_norm = max(math.hypot(push_dx, push_dy), 1e-6)
        ux = push_dx / push_norm
        uy = push_dy / push_norm
        robot_from_crate_x = pose.x - crate_xy[0]
        robot_from_crate_y = pose.y - crate_xy[1]
        along_push_axis = robot_from_crate_x * ux + robot_from_crate_y * uy
        lateral = robot_from_crate_x * -uy + robot_from_crate_y * ux
        goal_distance = math.hypot(push_dx, push_dy)

        return (
            _bin(crate_distance, (0.55, 0.8, 1.15, 1.7)),
            _angle_bin(crate_bearing, self.angle_bins),
            _angle_bin(push_heading_error, self.angle_bins),
            _bin(lateral, (-0.28, 0.28)),
            _bin(along_push_axis, (-0.95, -0.25, 0.2)),
            int(bumper_active) + 2 * _bin(goal_distance, (0.25, 0.5, 0.9)),
        )


class QTablePushPolicy:
    """Tabular Q-learning policy for local crate pushing."""

    def __init__(
        self,
        actions: tuple[PushAction, ...] = DEFAULT_PUSH_ACTIONS,
        discretizer: PushStateDiscretizer | None = None,
        q_table: dict[tuple[int, ...], list[float]] | None = None,
    ) -> None:
        self.actions = actions
        self.discretizer = discretizer or PushStateDiscretizer()
        self.q_table = q_table or {}

    def command(
        self,
        pose: Pose2D,
        crate_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        bumper_active: bool,
    ) -> DifferentialCommand:
        state = self.discretizer.state(pose, crate_xy, goal_xy, bumper_active)
        return self.actions[self.greedy_action_index(state)].command()

    def greedy_action_index(self, state: tuple[int, ...]) -> int:
        values = self.values(state)
        return int(np.argmax(values))

    def epsilon_greedy_action_index(
        self,
        state: tuple[int, ...],
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, len(self.actions)))
        return self.greedy_action_index(state)

    def values(self, state: tuple[int, ...]) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]

    def update(
        self,
        state: tuple[int, ...],
        action_index: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
        alpha: float,
        gamma: float,
    ) -> None:
        values = self.values(state)
        bootstrap = 0.0 if done else max(self.values(next_state))
        values[action_index] += alpha * (reward + gamma * bootstrap - values[action_index])

    def save(self, path: Path) -> None:
        payload = {
            "actions": [action.__dict__ for action in self.actions],
            "angle_bins": self.discretizer.angle_bins,
            "q_table": {_state_key(state): values for state, values in sorted(self.q_table.items())},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "QTablePushPolicy":
        payload = json.loads(path.read_text())
        actions = tuple(PushAction(**action) for action in payload["actions"])
        discretizer = PushStateDiscretizer(angle_bins=int(payload.get("angle_bins", 8)))
        q_table = {
            _parse_state_key(state_key): [float(value) for value in values]
            for state_key, values in payload["q_table"].items()
        }
        return cls(actions=actions, discretizer=discretizer, q_table=q_table)


class CratePushRLEnv:
    """Fast local-control RL environment backed by the warehouse dynamics."""

    def __init__(self, seed: int = 0, dt: float = 0.1, max_steps: int = 80) -> None:
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.max_steps = max_steps
        self.discretizer = PushStateDiscretizer()
        self.sim = Warehouse2DSim(seed=seed)
        self.steps = 0

    def reset(self) -> tuple[int, ...]:
        self.sim = Warehouse2DSim(seed=int(self.rng.integers(0, 2**31 - 1)))
        self.steps = 0
        self._randomize_local_start()
        return self.observe()

    def observe(self) -> tuple[int, ...]:
        return self.discretizer.state(
            self.sim.pose,
            (self.sim.crate.cx, self.sim.crate.cy),
            (self.sim.goal_zone.cx, self.sim.goal_zone.cy),
            self.sim.front_bumper_contacting_crate(),
        )

    def step(self, action: PushAction) -> tuple[tuple[int, ...], float, bool, dict[str, float | bool]]:
        previous_pose = self.sim.pose
        previous_goal_distance = self.sim.crate_goal_distance()
        previous_approach_distance = self._approach_target_distance(previous_pose)

        bumper = self.sim.step_dynamics(action.command(), self.dt)
        self.steps += 1

        new_goal_distance = self.sim.crate_goal_distance()
        new_approach_distance = self._approach_target_distance(self.sim.pose)
        moved = previous_pose.distance_to((self.sim.pose.x, self.sim.pose.y))
        progress_to_goal = previous_goal_distance - new_goal_distance
        approach_progress = previous_approach_distance - new_approach_distance
        success = self.sim.crate_in_goal()
        done = success or self.steps >= self.max_steps

        reward = -0.025
        reward += progress_to_goal * 26.0
        if not bumper:
            reward += approach_progress * 0.9
        reward += 0.04 if bumper else 0.0
        if action.linear_mps > 0.05 and moved < 0.004 and not bumper:
            reward -= 0.08
        if success:
            reward += 8.0

        return (
            self.observe(),
            float(reward),
            done,
            {
                "success": success,
                "crate_goal_distance": new_goal_distance,
                "progress_to_goal": progress_to_goal,
            },
        )

    def _approach_target_distance(self, pose: Pose2D) -> float:
        crate_xy = (self.sim.crate.cx, self.sim.crate.cy)
        goal_xy = (self.sim.goal_zone.cx, self.sim.goal_zone.cy)
        push_dx = goal_xy[0] - crate_xy[0]
        push_dy = goal_xy[1] - crate_xy[1]
        norm = max(math.hypot(push_dx, push_dy), 1e-6)
        target = (crate_xy[0] - 0.65 * push_dx / norm, crate_xy[1] - 0.65 * push_dy / norm)
        return pose.distance_to(target)

    def _randomize_local_start(self) -> None:
        crate_xy = (self.sim.crate.cx, self.sim.crate.cy)
        goal_xy = (self.sim.goal_zone.cx, self.sim.goal_zone.cy)
        push_dx = goal_xy[0] - crate_xy[0]
        push_dy = goal_xy[1] - crate_xy[1]
        push_yaw = math.atan2(push_dy, push_dx)
        ux = math.cos(push_yaw)
        uy = math.sin(push_yaw)
        lateral_x = -uy
        lateral_y = ux

        for _ in range(30):
            behind = float(self.rng.uniform(0.56, 1.05))
            lateral = float(self.rng.uniform(-0.32, 0.32))
            x = crate_xy[0] - ux * behind + lateral_x * lateral
            y = crate_xy[1] - uy * behind + lateral_y * lateral
            yaw = wrap_angle(push_yaw + float(self.rng.normal(0.0, 0.58)))
            if not self.sim._collides(x, y):
                self.sim.pose = Pose2D(x, y, yaw)
                return
        self.sim.pose = Pose2D(crate_xy[0] - ux * 0.75, crate_xy[1] - uy * 0.75, push_yaw)


def train_push_policy(
    episodes: int = 300,
    seed: int = 7,
    alpha: float = 0.28,
    gamma: float = 0.94,
    epsilon_start: float = 0.45,
    epsilon_end: float = 0.04,
    max_steps: int = 80,
) -> tuple[QTablePushPolicy, list[dict[str, float | int | bool]]]:
    rng = np.random.default_rng(seed)
    env = CratePushRLEnv(seed=seed, max_steps=max_steps)
    policy = QTablePushPolicy()
    history: list[dict[str, float | int | bool]] = []

    for episode in range(episodes):
        progress = episode / max(episodes - 1, 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        state = env.reset()
        total_reward = 0.0
        done = False
        info: dict[str, float | bool] = {"success": False, "crate_goal_distance": env.sim.crate_goal_distance()}

        while not done:
            action_index = policy.epsilon_greedy_action_index(state, epsilon, rng)
            next_state, reward, done, info = env.step(policy.actions[action_index])
            policy.update(state, action_index, reward, next_state, done, alpha=alpha, gamma=gamma)
            state = next_state
            total_reward += reward

        history.append(
            {
                "episode": episode,
                "epsilon": round(epsilon, 4),
                "total_reward": round(total_reward, 4),
                "steps": env.steps,
                "success": bool(info["success"]),
                "crate_goal_distance": round(float(info["crate_goal_distance"]), 4),
                "visited_states": len(policy.q_table),
            }
        )
    return policy, history


def evaluate_policy(
    policy: QTablePushPolicy,
    episodes: int = 25,
    seed: int = 97,
    max_steps: int = 80,
) -> dict[str, float | int]:
    env = CratePushRLEnv(seed=seed, max_steps=max_steps)
    successes = 0
    steps_total = 0
    reward_total = 0.0
    final_distances: list[float] = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        info: dict[str, float | bool] = {"success": False, "crate_goal_distance": env.sim.crate_goal_distance()}
        while not done:
            action = policy.actions[policy.greedy_action_index(state)]
            state, reward, done, info = env.step(action)
            episode_reward += reward
        successes += int(bool(info["success"]))
        steps_total += env.steps
        reward_total += episode_reward
        final_distances.append(float(info["crate_goal_distance"]))
    return {
        "episodes": episodes,
        "success_rate": round(successes / max(episodes, 1), 4),
        "mean_steps": round(steps_total / max(episodes, 1), 2),
        "mean_reward": round(reward_total / max(episodes, 1), 4),
        "mean_final_crate_goal_error_m": round(float(np.mean(final_distances)), 4),
    }


def _bin(value: float, thresholds: tuple[float, ...]) -> int:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return index
    return len(thresholds)


def _angle_bin(angle: float, bins: int) -> int:
    wrapped = wrap_angle(angle)
    normalized = (wrapped + math.pi) / (2.0 * math.pi)
    return min(bins - 1, max(0, int(normalized * bins)))


def _state_key(state: tuple[int, ...]) -> str:
    return ",".join(str(part) for part in state)


def _parse_state_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(","))
