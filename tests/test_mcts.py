import chess

from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.mcts.search import run, run_batched, select_move_from_visit_counts
from chess_rl.model.network import PolicyValueNet
from chess_rl.utils.repro import set_seed


def _small_model():
    return PolicyValueNet(
        in_planes=13, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2
    )


def test_run_visit_counts_match_legal_moves_and_sum_to_num_simulations():
    set_seed(0)
    board = chess.Board()
    result = run(board, _small_model(), num_simulations=20)

    legal_ucis = {move.uci() for move in board.legal_moves}
    assert set(result["visit_counts"].keys()) == legal_ucis
    assert sum(result["visit_counts"].values()) == 20
    assert -1.0 <= result["root_value"] <= 1.0


def test_run_handles_near_terminal_position_without_crashing():
    set_seed(0)
    # 백이 Qh5#로 바로 메이트를 낼 수 있는 국면(Fool's mate 직전).
    board = chess.Board()
    for move_uci in ["f2f3", "e7e5", "g2g4"]:
        board.push(chess.Move.from_uci(move_uci))

    result = run(board, _small_model(), num_simulations=15)
    legal_ucis = {move.uci() for move in board.legal_moves}
    assert set(result["visit_counts"].keys()) == legal_ucis
    assert sum(result["visit_counts"].values()) == 15


def test_run_batched_matches_run_for_single_board():
    """run(board, ...)은 run_batched([board], ...)의 wrapper이므로 결과가 완전히 같아야 함."""
    set_seed(0)
    model = _small_model()
    board = chess.Board()

    set_seed(0)
    single = run(board, model, num_simulations=15)
    set_seed(0)
    (batched,) = run_batched([board], model, num_simulations=15)

    assert single == batched


def test_run_batched_handles_multiple_independent_boards():
    set_seed(0)
    model = _small_model()
    board_a = chess.Board()
    board_b = chess.Board()
    board_b.push(chess.Move.from_uci("e2e4"))  # 서로 다른 국면

    results = run_batched([board_a, board_b], model, num_simulations=10)

    assert len(results) == 2
    for board, result in zip([board_a, board_b], results):
        legal_ucis = {move.uci() for move in board.legal_moves}
        assert set(result["visit_counts"].keys()) == legal_ucis
        assert sum(result["visit_counts"].values()) == 10


def test_select_move_from_visit_counts_deterministic_picks_argmax():
    visit_counts = {"e2e4": 5, "d2d4": 12, "g1f3": 3}
    assert select_move_from_visit_counts(visit_counts, deterministic=True) == "d2d4"


def test_select_move_from_visit_counts_stochastic_returns_valid_key():
    visit_counts = {"e2e4": 5, "d2d4": 12, "g1f3": 3}
    for _ in range(20):
        assert (
            select_move_from_visit_counts(visit_counts, deterministic=False)
            in visit_counts
        )
