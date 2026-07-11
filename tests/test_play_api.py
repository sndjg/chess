import chess
from fastapi.testclient import TestClient

from chess_rl.viz import server as viz_server
from chess_rl.viz.server import create_app


class ScriptedPolicy:
    """미리 정해둔 수를 순서대로 반환하는 테스트용 정책."""

    def __init__(self, moves: list):
        self._moves = list(moves)

    def select_move(self, board: chess.Board) -> chess.Move:
        return chess.Move.from_uci(self._moves.pop(0))


def test_play_page_served(tmp_path):
    client = TestClient(create_app(games_dir=str(tmp_path)))
    res = client.get("/play")
    assert res.status_code == 200
    assert b"board.js" in res.content


def test_new_game_human_white_ai_does_not_move_first(tmp_path):
    client = TestClient(create_app(games_dir=str(tmp_path)))
    res = client.post("/api/play/new", json={"human_color": "white", "policy": "random"})
    data = res.json()

    assert res.status_code == 200
    assert data["turn"] == "white"
    assert data["human_color"] == "white"
    assert data["ai_move"] is None
    assert data["fen"] == chess.Board().fen()


def test_new_game_human_black_ai_moves_first(tmp_path):
    client = TestClient(create_app(games_dir=str(tmp_path)))
    res = client.post("/api/play/new", json={"human_color": "black", "policy": "random"})
    data = res.json()

    assert res.status_code == 200
    assert data["ai_move"] is not None
    assert data["turn"] == "black"


def test_legal_move_applies_and_ai_responds(tmp_path):
    client = TestClient(create_app(games_dir=str(tmp_path)))
    new_res = client.post("/api/play/new", json={"human_color": "white", "policy": "random"})
    session_id = new_res.json()["session_id"]

    res = client.post(f"/api/play/{session_id}/move", json={"move": "e2e4"})
    data = res.json()

    assert res.status_code == 200
    assert data["human_move"] == "e2e4"
    assert data["ai_move"] is not None
    assert data["turn"] == "white"  # 백(사람) -> 흑(AI) 이후 다시 백 차례


def test_illegal_move_returns_400(tmp_path):
    client = TestClient(create_app(games_dir=str(tmp_path)))
    new_res = client.post("/api/play/new", json={"human_color": "white", "policy": "random"})
    session_id = new_res.json()["session_id"]

    res = client.post(f"/api/play/{session_id}/move", json={"move": "e2e5"})
    assert res.status_code == 400


def test_unknown_session_returns_404(tmp_path):
    client = TestClient(create_app(games_dir=str(tmp_path)))
    res = client.post("/api/play/nonexistent/move", json={"move": "e2e4"})
    assert res.status_code == 404


def test_finished_game_is_saved_to_games_dir(tmp_path, monkeypatch):
    """무작위 대국은 체크메이트까지 매우 오래 걸릴 수 있어(엔진 없이 랜덤으로는 잘 안 끝남),
    폴스메이트(Fool's Mate) 수순을 스크립트로 고정해 결정론적으로 빠르게 종료시킨다."""
    monkeypatch.setitem(viz_server.POLICIES, "scripted", lambda: ScriptedPolicy(["e7e5", "d8h4"]))

    client = TestClient(create_app(games_dir=str(tmp_path)))
    new_res = client.post("/api/play/new", json={"human_color": "white", "policy": "scripted"})
    session_id = new_res.json()["session_id"]

    client.post(f"/api/play/{session_id}/move", json={"move": "f2f3"})
    res = client.post(f"/api/play/{session_id}/move", json={"move": "g2g4"})
    data = res.json()

    assert data["ai_move"] == "d8h4"
    assert data["game_over"] is True
    assert data["result"] == "0-1"
    assert (tmp_path / f"play_{session_id}.json").exists()
