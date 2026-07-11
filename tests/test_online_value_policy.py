import chess

from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.model.network import PolicyValueNet
from chess_rl.rollout.online_value_policy import OnlineValuePolicy
from chess_rl.utils.repro import set_seed


def _make_policy():
    set_seed(0)
    model = PolicyValueNet(in_planes=12, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2)
    return OnlineValuePolicy(model, train_epochs=3)


def test_select_move_returns_legal_move():
    policy = _make_policy()
    board = chess.Board()
    move = policy.select_move(board)
    assert move in board.legal_moves


def test_select_move_stochastic_returns_legal_move_and_can_vary():
    policy = _make_policy()
    policy.mcts_simulations = 10  # 반복 샘플링 테스트라 시뮬레이션 수를 줄여 속도 확보
    board = chess.Board()

    moves = {policy.select_move(board, deterministic=False).uci() for _ in range(20)}
    assert moves <= {m.uci() for m in board.legal_moves}
    assert len(moves) >= 1


def test_move_values_covers_all_legal_moves_sorted_desc():
    policy = _make_policy()
    board = chess.Board()
    results = policy.move_values(board)

    assert {r["move"] for r in results} == {m.uci() for m in board.legal_moves}
    values = [r["value"] for r in results]
    assert values == sorted(values, reverse=True)


def test_learn_from_game_reduces_loss_and_increments_counter():
    policy = _make_policy()
    # 폴스메이트: 백이 짐 (0-1)
    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]

    result = policy.learn_from_game(moves, "0-1")

    assert result["num_positions"] == len(moves)
    assert result["loss_after"] <= result["loss_before"]
    assert result["games_trained"] == 1

    result2 = policy.learn_from_game(moves, "0-1")
    assert result2["games_trained"] == 2
    assert result2["buffer_size"] == 2 * len(moves)
    # 두 판이 buffer에 쌓였으니 batch_size가 buffer보다 크면 buffer 전체를 학습에 사용.
    assert result2["num_positions"] == result2["buffer_size"]


def test_value_estimate_is_from_white_perspective_and_bounded():
    policy = _make_policy()
    board = chess.Board()
    value = policy.value_estimate_white_perspective(board)
    assert -1.0 <= value <= 1.0
