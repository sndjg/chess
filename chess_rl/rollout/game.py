"""정책 두 개(사람 포함 무엇이든)로 자동으로 한 판을 끝까지 진행."""

import chess

from chess_rl.rollout.game_record import GameRecord
from chess_rl.rollout.policy import Policy


def play_game(
    policy_white: Policy, policy_black: Policy, max_moves: int = 300
) -> GameRecord:
    board = chess.Board()
    moves = []
    while not board.is_game_over() and len(moves) < max_moves:
        policy = policy_white if board.turn == chess.WHITE else policy_black
        move = policy.select_move(board)
        moves.append(move.uci())
        board.push(move)
    return GameRecord(moves=moves, result=board.result())
