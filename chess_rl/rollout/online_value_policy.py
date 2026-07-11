"""판마다 실제 대국 결과로 value head만 학습하는 온라인 정책.

재현성 인프라(ExperimentConfig, seed 고정, run 디렉토리)는 의도적으로 쓰지 않는다 —
사람과의 실시간 대국에 맞춰 서버 프로세스 동안 계속 다르게 학습되는 게 이 기능의 목적이라
재현성을 포기함(서버 재시작 시 초기화).

주의: value loss로만 backward하지만 policy/value head가 conv trunk를 공유하는 구조라,
gradient가 trunk를 거쳐 policy head의 출력에도 간접적으로 영향을 준다 — "value만 건드린다"는
말은 loss 항에 policy 관련 term이 없다는 뜻이지, policy 출력이 전혀 안 변한다는 뜻은 아니다.
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
    """실제로 두는 수는 policy head 샘플링, value head는 판이 끝날 때마다 그 판 결과로 회귀."""

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
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        index = np.random.choice(len(probs), p=probs)
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
        """한 판(moves, result)을 처음부터 재생하며, 각 국면의 value 타깃(그 국면 차례 관점의 결과)으로 회귀."""
        board = chess.Board()
        states = []
        targets = []
        white_score = _result_to_white_score(result)
        for move_uci in moves:
            target = white_score if board.turn == chess.WHITE else -white_score
            states.append(encode_board(board))
            targets.append(target)
            board.push(chess.Move.from_uci(move_uci))

        x = torch.from_numpy(np.stack(states)).to(self.device)
        y = torch.tensor(targets, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            _, pred_before = self.model(x)
            loss_before = F.mse_loss(pred_before, y).item()

        self.model.train()
        for _ in range(self.train_epochs):
            self.optimizer.zero_grad()
            _, pred = self.model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
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
