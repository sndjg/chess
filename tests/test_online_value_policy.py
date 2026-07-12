import time

import pytest
import chess
import torch

from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.model.network import PolicyValueNet
from chess_rl.rollout.online_value_policy import OnlineValuePolicy
from chess_rl.utils.checkpoint import list_checkpoints, read_family_meta
from chess_rl.utils.repro import set_seed


def _make_policy(**kwargs):
    set_seed(0)
    model = PolicyValueNet(
        in_planes=13, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2
    )
    return OnlineValuePolicy(model, train_epochs=3, **kwargs)


def test_checkpoint_dir_requires_family_and_training_method():
    set_seed(0)
    model = PolicyValueNet(
        in_planes=13, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2
    )
    with pytest.raises(ValueError):
        OnlineValuePolicy(model, checkpoint_dir="somewhere")

    set_seed(0)
    model = PolicyValueNet(
        in_planes=13, action_space_size=ACTION_SPACE_SIZE, channels=16, num_blocks=2
    )
    with pytest.raises(ValueError):
        OnlineValuePolicy(model, checkpoint_dir="somewhere", family="human_online")


def test_reusing_family_dir_with_existing_checkpoints_raises(tmp_path):
    _make_policy(
        checkpoint_dir=str(tmp_path),
        family="human_online",
        training_method="테스트용",
        checkpoint_every=1,
    ).learn_from_game(["f2f3", "e7e5", "g2g4", "d8h4"], "0-1")

    with pytest.raises(ValueError):
        _make_policy(
            checkpoint_dir=str(tmp_path),
            family="human_online",
            training_method="테스트용",
        )


def test_learn_from_game_saves_checkpoint_and_family_meta(tmp_path):
    policy = _make_policy(
        checkpoint_dir=str(tmp_path),
        family="human_online",
        training_method="테스트용",
        checkpoint_every=1,
    )
    result = policy.learn_from_game(["f2f3", "e7e5", "g2g4", "d8h4"], "0-1")
    assert result["checkpoint_path"] == str(
        tmp_path / "human_online" / "game_000001.pt"
    )

    checkpoints = list_checkpoints(str(tmp_path / "human_online"))
    assert [c.games_trained for c in checkpoints] == [1]
    assert checkpoints[0].family == "human_online"

    meta = read_family_meta(str(tmp_path / "human_online"))
    assert meta.family == "human_online"
    assert meta.method == "테스트용"
    assert meta.git_commit
    assert meta.started_at <= meta.last_updated_at


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


def test_value_estimate_on_checkmate_uses_actual_result_not_network_guess():
    """회귀 테스트: 체크메이트 국면은 network raw forward pass가 아니라 실제 결과를 써야
    한다 — 안 그러면 대국 내내 백에 유리하다고 나오다가 백이 메이트시키는 순간 network가
    (한 번도 학습해본 적 없는 종료 국면이라) 근거 없이 -1에 가까운 값을 내는 버그가 생김."""
    policy = _make_policy()
    board = chess.Board()
    # 폴스메이트: 백이 짐 (0-1)
    for move_uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push(chess.Move.from_uci(move_uci))
    assert board.is_game_over()

    value = policy.value_estimate_white_perspective(board)
    assert value == -1.0


def test_move_values_scores_mating_move_as_best_for_mover():
    policy = _make_policy()
    board = chess.Board()
    for move_uci in ["f2f3", "e7e5", "g2g4"]:
        board.push(chess.Move.from_uci(move_uci))

    results = policy.move_values(board)
    mating_move = next(r for r in results if r["move"] == "d8h4")
    assert mating_move["value"] == 1.0
    assert results[0]["move"] == "d8h4"  # 내림차순 정렬이니 1위여야 함


def test_inference_handle_is_independent_but_refreshes_on_use_after_training():
    """핸들의 model은 canonical과 독립된 복사본이지만(학습 도중 안전), canonical이
    갱신되면 다음 사용 시점에 최신 복사본으로 갈아탄다(lazy refresh) — 대국 도중에도
    직전 판의 학습 성과가 다음 수부터 반영되게."""
    policy = _make_policy()
    handle = policy.new_inference_handle()

    handle_params_before = [p.clone() for p in handle.model.parameters()]

    policy.learn_from_game(["f2f3", "e7e5", "g2g4", "d8h4"], "0-1")

    # canonical은 학습으로 바뀌었어야 함.
    canonical_changed = any(
        not torch.equal(before, after)
        for before, after in zip(handle_params_before, policy.model.parameters())
    )
    assert canonical_changed

    # 아직 사용 전이면 핸들의 model은 스냅샷 그대로(학습이 핸들을 직접 건드리지 않음).
    handle_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(handle_params_before, handle.model.parameters())
    )
    assert handle_unchanged

    # 핸들을 사용하는 순간 최신 canonical로 갱신돼야 함.
    handle.select_move(chess.Board())
    refreshed = all(
        torch.equal(a, b)
        for a, b in zip(handle.model.parameters(), policy.model.parameters())
    )
    assert refreshed
    # 갱신된 model은 여전히 canonical과 다른 객체여야 함(독립 복사본 유지).
    assert handle.model is not policy.model


def test_concurrent_learn_interrupts_running_training_and_preserves_both():
    """학습이 진행 중일 때 또 다른 판이 끝나면: 진행 중이던 학습은 조기 중단 후 진행분을
    병합하고, 새 학습이 이어서 돈다 — 두 판 모두 games_trained에 반영."""
    import threading as th

    policy = _make_policy()
    policy.train_epochs = 3000  # 첫 학습이 충분히 오래 돌아서 겹칠 수 있게

    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    results = {}

    def _first():
        results["first"] = policy.learn_from_game(moves, "0-1")

    t = th.Thread(target=_first)
    t.start()
    time.sleep(0.3)  # 첫 학습이 학습 루프에 들어갈 시간
    results["second"] = policy.learn_from_game(moves, "0-1")
    t.join(timeout=60)

    assert results["first"]["interrupted"] is True
    assert results["first"]["epochs_run"] < 3000
    assert results["second"]["interrupted"] is False
    assert policy.games_trained == 2


def test_search_move_with_candidates_consistent_with_search():
    policy = _make_policy()
    policy.mcts_simulations = 20
    board = chess.Board()

    move, candidates = policy.search_move_with_candidates(board)

    assert move in board.legal_moves
    assert {c["move"] for c in candidates} == {m.uci() for m in board.legal_moves}
    assert sum(c["visits"] for c in candidates) == 20
    # 방문 횟수 내림차순 정렬이고, deterministic 선택은 최다 방문 수와 일치해야 함.
    visits = [c["visits"] for c in candidates]
    assert visits == sorted(visits, reverse=True)
    assert candidates[0]["visits"] == max(visits)
    assert move.uci() in {
        c["move"] for c in candidates if c["visits"] == candidates[0]["visits"]
    }
    assert all(-1.0 <= c["value"] <= 1.0 for c in candidates)


def test_inference_handle_search_move_with_candidates():
    policy = _make_policy()
    policy.mcts_simulations = 10
    handle = policy.new_inference_handle()
    board = chess.Board()

    move, candidates = handle.search_move_with_candidates(board)
    assert move in board.legal_moves
    assert len(candidates) == len(list(board.legal_moves))


def test_inference_handle_delegates_learn_from_game_and_games_trained_to_trainer():
    policy = _make_policy()
    handle = policy.new_inference_handle()

    assert handle.games_trained == 0
    result = handle.learn_from_game(["f2f3", "e7e5", "g2g4", "d8h4"], "0-1")

    assert result["games_trained"] == 1
    assert policy.games_trained == 1
    assert handle.games_trained == 1  # 트레이너로 위임되니 바로 반영됨
