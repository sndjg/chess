import chess
import torch

from chess_rl.engine.action_space import ACTION_SPACE_SIZE, MOVE_TO_INDEX
from chess_rl.engine.board import encode_board, legal_move_mask
from chess_rl.model.network import PolicyValueNet


def test_encode_board_start_position():
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (12, 8, 8)
    assert planes.sum() == 32  # 시작 국면 기물 32개


def test_legal_move_mask_start_position():
    board = chess.Board()
    mask = legal_move_mask(board, MOVE_TO_INDEX)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.sum() == len(list(board.legal_moves))


def test_network_forward_shape():
    board = chess.Board()
    planes = encode_board(board)
    x = torch.from_numpy(planes).unsqueeze(0)  # (1, 12, 8, 8)

    net = PolicyValueNet(in_planes=12, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2)
    policy_logits, value = net(x)

    assert policy_logits.shape == (1, ACTION_SPACE_SIZE)
    assert value.shape == (1,)
