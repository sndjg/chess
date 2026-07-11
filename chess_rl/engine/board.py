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
