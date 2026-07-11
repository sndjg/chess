"""여러 판의 포지션을 섞어서 학습할 수 있도록 모아두는 고정 크기 버퍼.

포지션 단위(state, value_target, action_index, legal_mask)로 저장하며, capacity를
넘으면 오래된 포지션부터 덮어쓴다(원형 버퍼). 판 하나가 끝날 때마다 학습을 그 판
데이터로만 하지 않고, 여기 쌓인 여러 판 중에서 무작위로 배치를 뽑아 학습하기 위한 것.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self._states = []
        self._value_targets = []
        self._action_indices = []
        self._masks = []
        self._next_idx = 0

    def add_game(self, states, value_targets, action_indices, masks) -> None:
        for state, value_target, action_index, mask in zip(
            states, value_targets, action_indices, masks
        ):
            if len(self._states) < self.capacity:
                self._states.append(state)
                self._value_targets.append(value_target)
                self._action_indices.append(action_index)
                self._masks.append(mask)
            else:
                self._states[self._next_idx] = state
                self._value_targets[self._next_idx] = value_target
                self._action_indices[self._next_idx] = action_index
                self._masks[self._next_idx] = mask
            self._next_idx = (self._next_idx + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._states)

    def sample(self, batch_size: int):
        n = min(batch_size, len(self))
        indices = np.random.choice(len(self), size=n, replace=False)
        states = np.stack([self._states[i] for i in indices])
        value_targets = np.array(
            [self._value_targets[i] for i in indices], dtype=np.float32
        )
        action_indices = np.array(
            [self._action_indices[i] for i in indices], dtype=np.int64
        )
        masks = np.stack([self._masks[i] for i in indices])
        return states, value_targets, action_indices, masks
