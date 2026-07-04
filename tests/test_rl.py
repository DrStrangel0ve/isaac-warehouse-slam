from warehouse_slam.rl import CratePushRLEnv, QTablePushPolicy, train_push_policy


def test_q_policy_update_save_and_load(tmp_path):
    policy = QTablePushPolicy()
    state = (0, 1, 2, 1, 0, 1)
    next_state = (0, 1, 2, 1, 0, 3)
    policy.update(state, 2, reward=1.0, next_state=next_state, done=False, alpha=0.5, gamma=0.9)

    path = tmp_path / "q.json"
    policy.save(path)
    loaded = QTablePushPolicy.load(path)

    assert loaded.greedy_action_index(state) == 2
    assert loaded.values(state)[2] > 0.0


def test_crate_push_rl_env_steps():
    env = CratePushRLEnv(seed=3, max_steps=10)
    policy = QTablePushPolicy()
    state = env.reset()
    next_state, reward, done, info = env.step(policy.actions[2])

    assert len(state) == 6
    assert len(next_state) == 6
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "crate_goal_distance" in info


def test_training_loop_populates_q_table():
    policy, history = train_push_policy(episodes=8, seed=11, max_steps=20)

    assert len(history) == 8
    assert len(policy.q_table) > 0
    assert all("total_reward" in row for row in history)
