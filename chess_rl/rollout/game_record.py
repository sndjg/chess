"""리플레이용 게임 기록: UCI 수 목록을 저장하고, 각 ply의 FEN을 계산해서 내려준다."""

import dataclasses
import json
from pathlib import Path

import chess


@dataclasses.dataclass
class GameRecord:
    moves: list[str]  # UCI 표기 (예: "e2e4")
    result: str  # "1-0", "0-1", "1/2-1/2", "*"

    def fens(self) -> list[str]:
        """시작 국면 포함, 길이는 len(moves) + 1."""
        board = chess.Board()
        fens = [board.fen()]
        for move_uci in self.moves:
            board.push(chess.Move.from_uci(move_uci))
            fens.append(board.fen())
        return fens

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def from_json(cls, path) -> "GameRecord":
        data = json.loads(Path(path).read_text())
        return cls(**data)
