"""python-chess board를 감싸서 RL 학습에 필요한 encode/decode를 제공하는 얇은 wrapper."""

import chess
import numpy as np

NUM_PIECE_PLANES = 12  # 6 piece types x 2 colors


def encode_board(board: chess.Board) -> np.ndarray:
    """board.piece_map()을 (12, 8, 8) one-hot plane으로 변환."""
    planes = np.zeros((NUM_PIECE_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        plane_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        planes[plane_idx, rank, file] = 1.0
    return planes


def legal_move_mask(board: chess.Board, move_to_index: dict) -> np.ndarray:
    """현재 국면의 합법수에 대응하는 index만 1인 mask를 반환."""
    mask = np.zeros(len(move_to_index), dtype=np.float32)
    for move in board.legal_moves:
        mask[move_to_index[move.uci()]] = 1.0
    return mask


def terminal_value_for_side_to_move(board: chess.Board) -> float:
    """종료된 국면(board.is_game_over() 전제)의 실제 결과를, board.turn(다음에 둘
    차례인 쪽 — 게임이 끝났으니 실제로 두진 못하지만) 관점의 값으로 반환.

    network은 체크메이트 같은 종료 국면을 특별히 학습한 적이 없어서 raw forward
    pass로는 신뢰할 수 없다 — 종료 국면의 값은 항상 이 함수로 직접 계산해야 한다
    (network에 다시 물어보지 말 것). MCTS(chess_rl/mcts/search.py)와
    OnlineValuePolicy의 화면 표시용 평가(value_estimate_white_perspective,
    move_values) 양쪽에서 공유한다.
    """
    result = board.result()
    if result == "1/2-1/2":
        return 0.0
    white_won = result == "1-0"
    return 1.0 if white_won == (board.turn == chess.WHITE) else -1.0
