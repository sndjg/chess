import chess

from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.model.network import PolicyValueNet
from chess_rl.rollout.game import play_game
from chess_rl.rollout.policy import NetworkPolicy, RandomPolicy
from chess_rl.utils.repro import set_seed


def test_random_policy_returns_legal_move():
    board = chess.Board()
    policy = RandomPolicy()
    move = policy.select_move(board)
    assert move in board.legal_moves


def test_network_policy_returns_legal_move():
    set_seed(0)
    board = chess.Board()
    model = PolicyValueNet(
        in_planes=12, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2
    )
    policy = NetworkPolicy(model)
    move = policy.select_move(board)
    assert move in board.legal_moves


def test_play_game_random_vs_random_terminates_with_valid_record():
    set_seed(0)
    record = play_game(RandomPolicy(), RandomPolicy(), max_moves=300)

    board = chess.Board()
    for move_uci in record.moves:
        assert chess.Move.from_uci(move_uci) in board.legal_moves
        board.push(chess.Move.from_uci(move_uci))

    assert record.result in ("1-0", "0-1", "1/2-1/2", "*")
    assert len(record.fens()) == len(record.moves) + 1
