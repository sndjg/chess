"""viz UI 검증용 무작위 대국 하나를 생성해 games/sample.json으로 저장한다."""

import random

from pathlib import Path

import chess

from chess_rl.utils.repro import set_seed
from chess_rl.rollout.game_record import GameRecord

if __name__ == "__main__":
    set_seed(0)
    board = chess.Board()
    moves = []
    while not board.is_game_over() and len(moves) < 200:
        move = random.choice(list(board.legal_moves))
        moves.append(move.uci())
        board.push(move)

    record = GameRecord(moves=moves, result=board.result())
    Path("games").mkdir(exist_ok=True)
    record.to_json("games/sample.json")
    print(f"saved {len(moves)} moves, result={record.result}")
