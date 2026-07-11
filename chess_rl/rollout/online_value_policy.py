"""판마다 실제 대국 결과로 policy/value head를 함께 학습하는 온라인 정책.

재현성 인프라(ExperimentConfig, seed 고정, run 디렉토리)는 의도적으로 쓰지 않는다 —
사람과의 실시간 대국에 맞춰 서버 프로세스 동안 계속 다르게 학습되는 게 이 기능의 목적이라
재현성을 포기함(서버 재시작 시 초기화).

value head: Monte-Carlo 회귀(그 국면 차례 관점의 최종 결과로 MSE).
policy head: 게임에 나온 모든 수(사람 몫 포함)를 그 판의 결과로 가중해서 강화 —
log-prob(실제 둔 수) * (return - value baseline). 사람이 둔 수도 포함하므로 순수
on-policy REINFORCE는 아니고, "이긴 판의 수는 흉내내고 진 판의 수는 피하는" 식의
결과-가중 모방 학습에 가깝다.
"""

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chess_rl.engine.action_space import ALL_MOVES, MOVE_TO_INDEX
from chess_rl.engine.board import encode_board, legal_move_mask


def _result_to_white_score(result: str) -> float:
    return {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(result, 0.0)


class OnlineValuePolicy:
    """실제로 두는 수는 policy head 샘플링, 판이 끝날 때마다 그 판 결과로 policy/value head를 함께 학습."""

    def __init__(self, model, lr: float = 1e-3, train_epochs: int = 5, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_epochs = train_epochs
        self.games_trained = 0

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        self.model.eval()
        policy_logits, _ = self._forward(board)
        mask = legal_move_mask(board, MOVE_TO_INDEX)

        logits = policy_logits.squeeze(0).cpu().numpy()
        logits = np.where(mask == 0, -np.inf, logits)

        index = np.argmax(logits)
        return ALL_MOVES[index]

    @torch.no_grad()
    def value_estimate_white_perspective(self, board: chess.Board) -> float:
        self.model.eval()
        _, value = self._forward(board)
        value = value.item()
        return value if board.turn == chess.WHITE else -value

    @torch.no_grad()
    def move_values(self, board: chess.Board) -> list:
        """현재 board에서 둘 수 있는 모든 수를, 그 수를 두는 쪽 관점의 가치로 평가해 내림차순 정렬."""
        self.model.eval()
        results = []
        for move in board.legal_moves:
            next_board = board.copy(stack=False)
            next_board.push(move)
            _, value = self._forward(next_board)
            value_for_mover = -value.item()  # next_board는 상대 관점이라 부호 반전
            results.append({"move": move.uci(), "value": value_for_mover})
        results.sort(key=lambda r: r["value"], reverse=True)
        return results

    def learn_from_game(self, moves: list, result: str) -> dict:
        """한 판(moves, result)을 처음부터 재생하며, 양쪽 수 전부에 대해 value/policy head를 함께 학습."""
        board = chess.Board()
        states = []
        value_targets = []
        action_indices = []
        masks = []

        white_score = _result_to_white_score(result)
        for move_uci in moves:
            target = white_score if board.turn == chess.WHITE else -white_score
            states.append(encode_board(board))
            value_targets.append(target)
            action_indices.append(MOVE_TO_INDEX[move_uci])
            masks.append(legal_move_mask(board, MOVE_TO_INDEX))
            board.push(chess.Move.from_uci(move_uci))

        x = torch.from_numpy(np.stack(states)).to(self.device)
        y = torch.tensor(value_targets, dtype=torch.float32, device=self.device)
        action_idx = torch.tensor(action_indices, dtype=torch.long, device=self.device)
        mask = torch.tensor(np.stack(masks), dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            _, pred_before = self.model(x)
            loss_before = F.mse_loss(pred_before, y).item()

        self.model.train()
        for _ in range(self.train_epochs):
            self.optimizer.zero_grad()
            policy_logits, pred = self.model(x)
            value_loss = F.mse_loss(pred, y)

            masked_logits = policy_logits.masked_fill(mask == 0, float("-inf"))
            log_probs = F.log_softmax(masked_logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(len(moves)), action_idx]
            with torch.no_grad():
                baseline = pred
            advantage = y - baseline
            policy_loss = -(selected_log_probs * advantage).mean()

            (value_loss + policy_loss).backward()
            self.optimizer.step()
        self.model.eval()

        with torch.no_grad():
            _, pred_after = self.model(x)
            loss_after = F.mse_loss(pred_after, y).item()

        self.games_trained += 1
        return {
            "num_positions": len(moves),
            "loss_before": loss_before,
            "loss_after": loss_after,
            "games_trained": self.games_trained,
        }

    def _forward(self, board: chess.Board):
        planes = encode_board(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        return self.model(x)
