"""사람 vs 정책 인터랙티브 대국 세션 상태."""

import dataclasses

import chess

from chess_rl.rollout.policy import Policy


@dataclasses.dataclass
class PlaySession:
    board: chess.Board
    ai_policy: Policy
    human_color: bool  # chess.WHITE / chess.BLACK
    moves: list = dataclasses.field(default_factory=list)  # UCI
    moves_san: list = dataclasses.field(default_factory=list)

    def push_move(self, move: chess.Move) -> None:
        """SAN은 반드시 push 전에 계산해야 한다(disambiguation/체크·메이트 표기가
        현재 board 상태를 기준으로 계산됨)."""
        self.moves_san.append(self.board.san(move))
        self.moves.append(move.uci())
        self.board.push(move)

    def to_state(self) -> dict:
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "human_color": "white" if self.human_color == chess.WHITE else "black",
            "legal_moves": [m.uci() for m in self.board.legal_moves],
            "game_over": self.board.is_game_over(),
            "result": self.board.result() if self.board.is_game_over() else None,
            "moves_san": self.moves_san,
        }
