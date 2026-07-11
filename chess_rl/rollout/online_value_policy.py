"""판마다 실제 대국 결과로 policy/value head를 함께 학습하는 온라인 정책.

재현성 인프라(ExperimentConfig, seed 고정, run 디렉토리)는 의도적으로 쓰지 않는다 —
사람과의 실시간 대국에 맞춰 서버 프로세스 동안 계속 다르게 학습되는 게 이 기능의 목적이라
재현성을 포기함(서버 재시작 시 초기화).

value head: Monte-Carlo 회귀(그 국면 차례 관점의 최종 결과로 MSE).
policy head: 게임에 나온 모든 수(사람 몫 포함)를 그 판의 결과로 가중해서 강화 —
log-prob(실제 둔 수) * (return - value baseline). 사람이 둔 수도 포함하므로 순수
on-policy REINFORCE는 아니고, "이긴 판의 수는 흉내내고 진 판의 수는 피하는" 식의
결과-가중 모방 학습에 가깝다.

실제 수 선택(select_move)은 policy head를 직접 쓰지 않고 chess_rl.mcts.search.run()으로
매 수마다 MCTS 탐색을 돌린다(추론에도 MCTS를 포함시킨 버전 — 지연이 느껴지면
mcts_simulations를 낮추는 방향으로 조정 예정). 탐색 후 수 선택은 두 가지 모드:
deterministic=True(기본값, 사람과의 실제 대국)는 방문 횟수 argmax, deterministic=False
(체크포인트 간 평가 대국 등)는 방문 횟수 분포에서 샘플링 — 후자는 같은 두 정책끼리 여러
판 반복 대국시켜도 매번 다른 게임이 나오게 하기 위한 것. MCTS 탐색 target 통합(policy
head를 방문분포로 학습시키는 것 등)은 아직 안 함 — policy/value head 학습은 여전히
기존 REINFORCE/MSE 방식 그대로.

판이 끝나면 그 판의 포지션들을 ReplayBuffer(chess_rl.rollout.replay_buffer)에 쌓아두고,
그 판 데이터만으로 학습하는 대신 buffer에서 배치를 무작위로 샘플링해 학습한다 — 판 하나
분량으로 학습이 캡되는 문제를 완화하기 위함.

checkpoint_dir이 주어지면(family와 함께, 필수) checkpoint_every판마다
checkpoint_dir/family/ 아래에 모델 스냅샷을 저장한다(chess_rl.utils.checkpoint) —
chess_rl.eval.arena가 이 스냅샷들끼리(다른 family와도) 대국시켜 학습이 실제로
나아지고 있는지 상대적으로 평가하는 데 쓴다.
"""

from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chess_rl.engine.action_space import MOVE_TO_INDEX
from chess_rl.engine.board import encode_board, legal_move_mask
from chess_rl.mcts.search import run as mcts_run
from chess_rl.rollout.replay_buffer import ReplayBuffer
from chess_rl.utils.checkpoint import (
    list_checkpoints,
    save_checkpoint,
    touch_family_meta,
    write_family_meta,
)


def _result_to_white_score(result: str) -> float:
    return {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(result, 0.0)


class OnlineValuePolicy:
    """실제로 두는 수는 MCTS 방문분포 기반, 판이 끝날 때마다 그 판 결과로 policy/value head를 함께 학습."""

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        train_epochs: int = 5,
        device: str = "cpu",
        mcts_simulations: int = 200,
        replay_capacity: int = 5000,
        batch_size: int = 256,
        checkpoint_dir: str | None = None,
        family: str | None = None,
        training_method: str | None = None,
        checkpoint_every: int = 1,
    ):
        if checkpoint_dir is not None:
            if not family:
                raise ValueError(
                    "checkpoint_dir가 주어지면 family(학습 계보 식별자)도 함께 줘야 함"
                )
            if not training_method:
                raise ValueError(
                    "checkpoint_dir가 주어지면 training_method(학습 방식 서술)도 함께 줘야 함"
                )

        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_epochs = train_epochs
        self.games_trained = 0
        self.mcts_simulations = mcts_simulations
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.batch_size = batch_size
        self.family = family
        self.checkpoint_every = checkpoint_every

        if checkpoint_dir is not None:
            family_dir = str(Path(checkpoint_dir) / family)
            if list_checkpoints(family_dir):
                raise ValueError(
                    f"family '{family}' 디렉터리({family_dir})에 이미 checkpoint가 있음 — "
                    "같은 family를 재사용하면 games_trained가 1부터 다시 시작되어 기존 파일을 "
                    "덮어쓰게 됨. 새 family 이름을 쓸 것."
                )
            write_family_meta(family_dir, family, training_method)
            self.checkpoint_dir = family_dir
        else:
            self.checkpoint_dir = None

    def select_move(self, board: chess.Board, deterministic: bool = True) -> chess.Move:
        """deterministic=True(기본값, 사람과의 실제 대국용): 방문 횟수가 가장 많은 수.
        deterministic=False(체크포인트 간 평가 대국 등): 방문 횟수 분포에서 샘플링 —
        같은 두 정책끼리 반복 대국시켜도 매번 다른 게임이 나오게 하기 위함."""
        self.model.eval()
        result = mcts_run(
            board, self.model, num_simulations=self.mcts_simulations, device=self.device
        )
        visit_counts = result["visit_counts"]

        if deterministic:
            best_uci = max(visit_counts, key=visit_counts.get)
        else:
            ucis = list(visit_counts.keys())
            counts = np.array([visit_counts[uci] for uci in ucis], dtype=np.float64)
            probs = counts / counts.sum()
            best_uci = np.random.choice(ucis, p=probs)
        return chess.Move.from_uci(best_uci)

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
        """한 판(moves, result)을 replay buffer에 적립하고, buffer에서 샘플링한 배치로 학습."""
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

        self.replay_buffer.add_game(states, value_targets, action_indices, masks)

        batch_states, batch_value_targets, batch_action_indices, batch_masks = (
            self.replay_buffer.sample(self.batch_size)
        )
        x = torch.from_numpy(batch_states).to(self.device)
        y = torch.from_numpy(batch_value_targets).to(self.device)
        action_idx = torch.from_numpy(batch_action_indices).to(self.device)
        mask = torch.from_numpy(batch_masks).to(self.device)
        num_positions = len(batch_states)

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
            selected_log_probs = log_probs[torch.arange(num_positions), action_idx]
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

        checkpoint_path = None
        if (
            self.checkpoint_dir is not None
            and self.games_trained % self.checkpoint_every == 0
        ):
            checkpoint_path = save_checkpoint(
                self.model, self.checkpoint_dir, self.games_trained
            )
            touch_family_meta(self.checkpoint_dir)

        return {
            "num_positions": num_positions,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "games_trained": self.games_trained,
            "buffer_size": len(self.replay_buffer),
            "checkpoint_path": str(checkpoint_path)
            if checkpoint_path is not None
            else None,
        }

    def _forward(self, board: chess.Board):
        planes = encode_board(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        return self.model(x)
