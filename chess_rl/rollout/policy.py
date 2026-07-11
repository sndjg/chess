"""정책 인터페이스: 사람이든 network든 random이든 같은 방식으로 다룬다."""

import random
from typing import Protocol

import chess
import numpy as np
import torch

from chess_rl.engine.action_space import ALL_MOVES, MOVE_TO_INDEX
from chess_rl.engine.board import encode_board, legal_move_mask


class Policy(Protocol):
    def select_move(self, board: chess.Board) -> chess.Move: ...


class RandomPolicy:
    def select_move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))


class NetworkPolicy:
    """policy head 출력을 합법수로 masking한 뒤 softmax 샘플링."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        planes = encode_board(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        policy_logits, _ = self.model(x)
        mask = legal_move_mask(board, MOVE_TO_INDEX)

        logits = policy_logits.squeeze(0).cpu().numpy()
        logits = np.where(mask == 0, -np.inf, logits)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        index = np.random.choice(len(probs), p=probs)
        return ALL_MOVES[index]
