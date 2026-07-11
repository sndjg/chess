"""사람 vs 정책 인터랙티브 대국 세션 상태."""

import dataclasses

import chess

from chess_rl.rollout.policy import Policy


@dataclasses.dataclass
class PlaySession:
    board: chess.Board
    ai_policy: Policy
    human_color: bool  # chess.WHITE / chess.BLACK
    moves: list = dataclasses.field(default_factory=list)

    def to_state(self) -> dict:
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "human_color": "white" if self.human_color == chess.WHITE else "black",
            "legal_moves": [m.uci() for m in self.board.legal_moves],
            "game_over": self.board.is_game_over(),
            "result": self.board.result() if self.board.is_game_over() else None,
        }
