import numpy as np

from chess_rl.rollout.replay_buffer import ReplayBuffer


def _fake_game(n, offset=0):
    states = [np.full((2, 2), offset + i, dtype=np.float32) for i in range(n)]
    value_targets = [float(offset + i) for i in range(n)]
    action_indices = [offset + i for i in range(n)]
    masks = [np.ones(3, dtype=np.float32) for _ in range(n)]
    return states, value_targets, action_indices, masks


def test_add_game_and_len():
    buffer = ReplayBuffer(capacity=10)
    buffer.add_game(*_fake_game(4))
    assert len(buffer) == 4


def test_capacity_overwrites_oldest():
    buffer = ReplayBuffer(capacity=5)
    buffer.add_game(*_fake_game(5, offset=0))
    buffer.add_game(*_fake_game(3, offset=100))

    assert len(buffer) == 5
    states, _, _, _ = buffer.sample(5)
    values_present = {s.flat[0] for s in states}
    # 첫 판의 앞 3개(0,1,2)는 두 번째 판(100,101,102)에 덮어써져서 더 이상 없어야 함.
    assert values_present == {3.0, 4.0, 100.0, 101.0, 102.0}


def test_sample_shapes_and_batch_smaller_than_capacity():
    buffer = ReplayBuffer(capacity=100)
    buffer.add_game(*_fake_game(6))

    states, value_targets, action_indices, masks = buffer.sample(4)
    assert states.shape == (4, 2, 2)
    assert value_targets.shape == (4,)
    assert action_indices.shape == (4,)
    assert masks.shape == (4, 3)

    states_all, *_ = buffer.sample(1000)
    assert states_all.shape[0] == 6
