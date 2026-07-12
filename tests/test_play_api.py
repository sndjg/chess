import time

import chess
from fastapi.testclient import TestClient

from chess_rl.viz.server import create_app


class ScriptedPolicy:
    """미리 정해둔 수를 순서대로 반환하는 테스트용 정책."""

    def __init__(self, moves: list):
        self._moves = list(moves)

    def select_move(self, board: chess.Board) -> chess.Move:
        return chess.Move.from_uci(self._moves.pop(0))


def test_play_page_served(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )
    res = client.get("/play")
    assert res.status_code == 200
    assert b"board.js" in res.content


def test_comparison_endpoint_starts_idle(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )
    data = client.get("/api/comparison").json()

    assert data["status"] == "idle"
    assert data["own_family"].startswith("human_online_")
    assert data["best_beaten_games_trained"] is None
    assert data["history"] == []


def test_logs_endpoint_records_training_line(tmp_path):
    scripted_learning = ScriptedLearningPolicy(["e7e5", "d8h4"])
    client = TestClient(
        create_app(
            games_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            extra_policies={"learning": lambda: scripted_learning},
        )
    )
    new_res = client.post(
        "/api/play/new", json={"human_color": "white", "policy": "learning"}
    )
    session_id = new_res.json()["session_id"]

    client.post(f"/api/play/{session_id}/move", json={"move": "f2f3"})
    client.post(f"/api/play/{session_id}/move", json={"move": "g2g4"})

    # 학습은 백그라운드 스레드에서 진행되므로 [train] 로그가 찍힐 때까지 잠깐 대기.
    deadline = time.time() + 5
    lines = []
    while time.time() < deadline:
        lines = client.get("/api/logs").json()["lines"]
        if any("[train]" in entry["message"] for entry in lines):
            break
        time.sleep(0.05)

    assert any(
        "[train]" in entry["message"] and "학습 완료" in entry["message"]
        for entry in lines
    )
    assert all({"timestamp", "level", "message"} <= entry.keys() for entry in lines)


def test_new_game_human_white_ai_does_not_move_first(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )
    res = client.post(
        "/api/play/new", json={"human_color": "white", "policy": "random"}
    )
    data = res.json()

    assert res.status_code == 200
    assert data["turn"] == "white"
    assert data["human_color"] == "white"
    assert data["ai_move"] is None
    assert data["fen"] == chess.Board().fen()


def test_new_game_human_black_ai_moves_first(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )
    res = client.post(
        "/api/play/new", json={"human_color": "black", "policy": "random"}
    )
    data = res.json()

    assert res.status_code == 200
    assert data["ai_move"] is not None
    assert data["turn"] == "black"


def test_legal_move_applies_and_ai_responds(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )
    new_res = client.post(
        "/api/play/new", json={"human_color": "white", "policy": "random"}
    )
    session_id = new_res.json()["session_id"]

    res = client.post(f"/api/play/{session_id}/move", json={"move": "e2e4"})
    data = res.json()

    assert res.status_code == 200
    assert data["human_move"] == "e2e4"
    assert data["human_move_san"] == "e4"
    assert data["ai_move"] is not None
    assert data["ai_move_san"] is not None
    assert data["turn"] == "white"  # 백(사람) -> 흑(AI) 이후 다시 백 차례
    assert data["moves_san"] == [data["human_move_san"], data["ai_move_san"]]


def test_illegal_move_returns_400(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )
    new_res = client.post(
        "/api/play/new", json={"human_color": "white", "policy": "random"}
    )
    session_id = new_res.json()["session_id"]

    res = client.post(f"/api/play/{session_id}/move", json={"move": "e2e5"})
    assert res.status_code == 400


def test_unknown_session_returns_404(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )
    res = client.post("/api/play/nonexistent/move", json={"move": "e2e4"})
    assert res.status_code == 404


def test_finished_game_is_saved_to_games_dir(tmp_path):
    """무작위 대국은 체크메이트까지 매우 오래 걸릴 수 있어(엔진 없이 랜덤으로는 잘 안 끝남),
    폴스메이트(Fool's Mate) 수순을 스크립트로 고정해 결정론적으로 빠르게 종료시킨다."""
    client = TestClient(
        create_app(
            games_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            extra_policies={"scripted": lambda: ScriptedPolicy(["e7e5", "d8h4"])},
        )
    )
    new_res = client.post(
        "/api/play/new", json={"human_color": "white", "policy": "scripted"}
    )
    session_id = new_res.json()["session_id"]

    client.post(f"/api/play/{session_id}/move", json={"move": "f2f3"})
    res = client.post(f"/api/play/{session_id}/move", json={"move": "g2g4"})
    data = res.json()

    assert data["ai_move"] == "d8h4"
    assert data["ai_move_san"] == "Qh4#"
    assert data["game_over"] is True
    assert data["result"] == "0-1"
    assert data["moves_san"] == ["f3", "e5", "g4", "Qh4#"]
    assert (tmp_path / f"play_{session_id}.json").exists()


def test_learning_policy_exposes_candidate_moves_value_and_training(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            extra_policies={"scripted": lambda: ScriptedPolicy(["e7e5", "d8h4"])},
        )
    )
    new_res = client.post(
        "/api/play/new", json={"human_color": "white", "policy": "learning"}
    )
    data = new_res.json()
    assert new_res.status_code == 200
    assert data["ai_move"] is None  # 사람이 백이라 AI는 아직 안 둠
    assert data["ai_candidate_moves"] is None
    session_id = data["session_id"]

    res = client.post(f"/api/play/{session_id}/move", json={"move": "e2e4"})
    data = res.json()

    assert res.status_code == 200
    assert data["ai_move"] is not None
    assert data["fen_before_ai_move"] is not None
    candidates = data["ai_candidate_moves"]
    assert candidates is not None
    assert len(candidates) > 0
    # MCTS 탐색 통계 기반 후보: 방문 횟수 내림차순 정렬, value는 Q([-1,1]).
    visits = [c["visits"] for c in candidates]
    assert visits == sorted(visits, reverse=True)
    assert all(-1.0 <= c["value"] <= 1.0 for c in candidates)
    assert -1.0 <= data["value_after_human_move"] <= 1.0
    assert -1.0 <= data["value_after_ai_move"] <= 1.0
    assert data["training"] is None  # 게임이 아직 안 끝남


class ScriptedLearningPolicy(ScriptedPolicy):
    """learn_from_game 호출 여부/인자를 기록하는, 정해진 수만 두는 테스트용 학습 정책."""

    def __init__(self, moves: list):
        super().__init__(moves)
        self.learn_calls = []
        self.games_trained = 0

    def value_estimate_white_perspective(self, board: chess.Board) -> float:
        return 0.0

    def move_values(self, board: chess.Board) -> list:
        return [{"move": m.uci(), "value": 0.0} for m in board.legal_moves]

    def learn_from_game(self, moves: list, result: str) -> dict:
        self.learn_calls.append((list(moves), result))
        self.games_trained += 1
        return {
            "num_positions": len(moves),
            "loss_before": 1.0,
            "loss_after": 0.5,
            "games_trained": self.games_trained,
        }


def test_learning_policy_trains_value_head_on_game_end(tmp_path):
    """폴스메이트로 판을 끝내서 learn_from_game이 (백그라운드에서) 실제로 호출되는지 확인.

    학습이 백그라운드 스레드로 옮겨져서 응답은 학습 완료를 기다리지 않고 바로 반환된다
    (training은 항상 None) — 마지막 수가 학습 끝날 때까지 보드에 반영 안 되던 문제의 수정."""
    scripted_learning = ScriptedLearningPolicy(["e7e5", "d8h4"])
    client = TestClient(
        create_app(
            games_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            extra_policies={"learning": lambda: scripted_learning},
        )
    )
    new_res = client.post(
        "/api/play/new", json={"human_color": "white", "policy": "learning"}
    )
    session_id = new_res.json()["session_id"]

    client.post(f"/api/play/{session_id}/move", json={"move": "f2f3"})
    res = client.post(f"/api/play/{session_id}/move", json={"move": "g2g4"})
    data = res.json()

    assert data["game_over"] is True
    assert data["training"] is None  # 학습은 백그라운드 — 응답에 결과를 싣지 않음

    deadline = time.time() + 5
    while time.time() < deadline and not scripted_learning.learn_calls:
        time.sleep(0.05)
    assert scripted_learning.learn_calls == [(["f2f3", "e7e5", "g2g4", "d8h4"], "0-1")]


def test_games_trained_present_across_new_sessions_with_same_learning_instance(
    tmp_path,
):
    """새로고침(=새 세션 생성)해도 같은 learning 인스턴스의 누적 games_trained가 그대로 보여야 한다."""
    scripted_learning = ScriptedLearningPolicy(["e7e5", "d8h4"])
    scripted_learning.games_trained = 3  # 이전에 이미 3판 학습했다고 가정
    client = TestClient(
        create_app(
            games_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            extra_policies={"learning": lambda: scripted_learning},
        )
    )

    res = client.post(
        "/api/play/new", json={"human_color": "white", "policy": "learning"}
    )
    data = res.json()

    assert data["games_trained"] == 3
