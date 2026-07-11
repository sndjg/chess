import chess

from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.mcts.search import run
from chess_rl.model.network import PolicyValueNet
from chess_rl.utils.repro import set_seed


def _small_model():
    return PolicyValueNet(in_planes=12, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2)


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
