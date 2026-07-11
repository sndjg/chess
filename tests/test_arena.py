from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.eval.arena import a_beats_b, find_new_frontier, play_match
from chess_rl.model.network import PolicyValueNet
from chess_rl.utils.checkpoint import save_checkpoint
from chess_rl.utils.repro import set_seed


def _small_model(seed):
    set_seed(seed)
    return PolicyValueNet(
        in_planes=12, action_space_size=ACTION_SPACE_SIZE, channels=8, num_blocks=1
    )


def test_play_match_counts_sum_to_num_games():
    model_a = _small_model(0)
    model_b = _small_model(1)

    match = play_match(model_a, model_b, num_games=6, mcts_simulations=5, max_moves=20)

    assert match["a_wins"] + match["b_wins"] + match["draws"] == 6


def test_play_match_does_not_double_move_a_board_within_a_round():
    """회귀 테스트: 한 라운드 안에서 model_a 그룹 처리 후 turn이 바뀐 board가 model_b
    그룹에도 잘못 끼어들어 같은 board가 한 라운드에 두 번 움직이던 버그(이미 게임이 끝난
    board가 run_batched에 다시 들어가 크래시)가 재발하지 않는지 확인. 이 정도 게임
    수/수순 길이(num_games=20, max_moves=40)에서 실제로 재현됐었다."""
    model_a = _small_model(0)
    model_b = _small_model(1)

    match = play_match(
        model_a, model_b, num_games=20, mcts_simulations=15, max_moves=40
    )
    assert match["a_wins"] + match["b_wins"] + match["draws"] == 20


def test_a_beats_b_scoring():
    assert a_beats_b({"a_wins": 3, "b_wins": 1, "draws": 0})
    assert not a_beats_b({"a_wins": 1, "b_wins": 3, "draws": 0})
    assert not a_beats_b(
        {"a_wins": 2, "b_wins": 2, "draws": 0}
    )  # 동점은 못 이긴 것으로 처리
    assert a_beats_b({"a_wins": 2, "b_wins": 1, "draws": 2})  # 2 + 1.0 > 1 + 1.0


def test_find_new_frontier_no_checkpoints_returns_minus_one():
    result = find_new_frontier(_small_model(0), old_checkpoints=[], start_idx=-1)
    assert result == {"frontier_idx": -1, "matches": []}


def test_find_new_frontier_walks_from_start_idx(tmp_path):
    model = _small_model(0)
    for n in (10, 20, 30):
        save_checkpoint(model, str(tmp_path), games_trained=n)

    from chess_rl.utils.checkpoint import list_checkpoints

    checkpoints = list_checkpoints(str(tmp_path))
    assert len(checkpoints) == 3

    result = find_new_frontier(
        _small_model(0),
        old_checkpoints=checkpoints,
        start_idx=-1,
        num_games=4,
        mcts_simulations=5,
        max_moves=20,
    )
    # 같은 모델끼리 붙이는 것이므로(동일 checkpoint 저장) 승패는 우연에 가깝지만,
    # frontier_idx가 유효 범위 안에 있고 matches가 비어있지 않은지만 확인.
    assert -1 <= result["frontier_idx"] <= 2
    assert len(result["matches"]) >= 1
