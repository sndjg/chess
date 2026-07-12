import math

import chess
import torch

from chess_rl.engine.action_space import ACTION_SPACE_SIZE, MOVE_TO_INDEX
from chess_rl.engine.board import encode_board, legal_move_mask
from chess_rl.model.network import MaterialBlendedPolicyValueNet, PolicyValueNet


def test_encode_board_start_position():
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (13, 8, 8)
    assert planes[:12].sum() == 32  # 시작 국면 기물 32개
    assert (planes[12] == 1.0).all()  # 백 차례


def test_encode_board_turn_plane_flips_with_side_to_move():
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))  # 흑 차례
    planes = encode_board(board)
    assert (planes[12] == 0.0).all()

    board.push(chess.Move.from_uci("e7e5"))  # 다시 백 차례
    planes = encode_board(board)
    assert (planes[12] == 1.0).all()


def test_legal_move_mask_start_position():
    board = chess.Board()
    mask = legal_move_mask(board, MOVE_TO_INDEX)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.sum() == len(list(board.legal_moves))


def test_network_forward_shape():
    board = chess.Board()
    planes = encode_board(board)
    x = torch.from_numpy(planes).unsqueeze(0)  # (1, 13, 8, 8)

    net = PolicyValueNet(
        in_planes=13, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2
    )
    policy_logits, value = net(x)

    assert policy_logits.shape == (1, ACTION_SPACE_SIZE)
    assert value.shape == (1,)


def _blended_net(material_weight=0.5):
    return MaterialBlendedPolicyValueNet(
        PolicyValueNet(
            in_planes=13, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2
        ),
        material_weight=material_weight,
    )


def _boards_to_batch(boards):
    import numpy as np

    return torch.from_numpy(np.stack([encode_board(b) for b in boards]))


def test_material_value_zero_at_start_position():
    net = _blended_net()
    x = _boards_to_batch([chess.Board()])
    material = net.material_value_for_side_to_move(x)
    assert material.item() == 0.0  # 양쪽 재료 동일


def test_material_value_sign_follows_side_to_move():
    # 흑 퀸을 제거한 국면 — 백이 재료 우위.
    board = chess.Board()
    board.remove_piece_at(chess.D8)  # 흑 퀸 제거

    net = _blended_net()

    board.turn = chess.WHITE
    material_white_to_move = net.material_value_for_side_to_move(
        _boards_to_batch([board])
    ).item()
    board.turn = chess.BLACK
    material_black_to_move = net.material_value_for_side_to_move(
        _boards_to_batch([board])
    ).item()

    expected = math.tanh(9.0 / 10.0)
    assert abs(material_white_to_move - expected) < 1e-6
    assert abs(material_black_to_move - (-expected)) < 1e-6


def test_blended_forward_mixes_nn_and_material():
    board = chess.Board()
    board.remove_piece_at(chess.D8)  # 백이 퀸만큼 재료 우위, 백 차례
    x = _boards_to_batch([board])

    inner = PolicyValueNet(
        in_planes=13, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2
    )
    blended = MaterialBlendedPolicyValueNet(inner, material_weight=0.5)
    inner.eval()
    blended.eval()

    with torch.no_grad():
        policy_inner, value_inner = inner(x)
        policy_blended, value_blended = blended(x)

    assert torch.equal(policy_inner, policy_blended)  # policy는 그대로 통과
    expected = 0.5 * value_inner.item() + 0.5 * math.tanh(9.0 / 10.0)
    assert abs(value_blended.item() - expected) < 1e-6


def test_material_weight_one_ignores_nn_value():
    board = chess.Board()
    board.remove_piece_at(chess.D8)
    x = _boards_to_batch([board])

    blended = _blended_net(material_weight=1.0)
    blended.eval()
    with torch.no_grad():
        _, value = blended(x)
    assert abs(value.item() - math.tanh(9.0 / 10.0)) < 1e-6
