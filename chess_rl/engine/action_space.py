"""고정 크기 action space 정의: 64x64 (from, to) + underpromotion 조합.

AlphaZero 원 논문의 8x8x73 encoding 대신, 구현이 단순한
'from_square * 64 + to_square' 기반 index에 underpromotion 몇 개를 덧붙이는
방식을 사용한다. 학습 성능보다 이해/디버깅 용이성을 우선한 선택.
"""

import chess

_PROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)  # queen은 기본 promotion으로 별도 처리


def _build_move_list():
    moves = []
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            moves.append(chess.Move(from_sq, to_sq))
            moves.append(chess.Move(from_sq, to_sq, promotion=chess.QUEEN))
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            for promo in _PROMOTION_PIECES:
                moves.append(chess.Move(from_sq, to_sq, promotion=promo))
    return moves


ALL_MOVES = _build_move_list()
MOVE_TO_INDEX = {move.uci(): i for i, move in enumerate(ALL_MOVES)}
ACTION_SPACE_SIZE = len(ALL_MOVES)
