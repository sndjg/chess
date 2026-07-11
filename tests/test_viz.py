import chess
from fastapi.testclient import TestClient

from chess_rl.rollout.game_record import GameRecord
from chess_rl.viz.server import create_app


def _make_sample_record() -> GameRecord:
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3"]
    for move_uci in moves:
        board.push(chess.Move.from_uci(move_uci))
    return GameRecord(moves=moves, result="*")


def test_game_record_fens_length_and_start_position():
    record = _make_sample_record()
    fens = record.fens()
    assert len(fens) == len(record.moves) + 1
    assert fens[0] == chess.Board().fen()


def test_game_record_json_roundtrip(tmp_path):
    record = _make_sample_record()
    path = tmp_path / "game.json"
    record.to_json(path)
    loaded = GameRecord.from_json(path)
    assert loaded == record


def test_api_list_and_get_game(tmp_path):
    record = _make_sample_record()
    record.to_json(tmp_path / "sample.json")

    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )

    games = client.get("/api/games").json()
    assert games == ["sample"]

    data = client.get("/api/games/sample").json()
    assert data["moves"] == record.moves
    assert data["result"] == record.result
    assert len(data["fens"]) == len(record.moves) + 1


def test_api_get_missing_game_returns_404(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )
    res = client.get("/api/games/nonexistent")
    assert res.status_code == 404


def test_index_page_served(tmp_path):
    client = TestClient(
        create_app(
            games_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "checkpoints")
        )
    )
    res = client.get("/")
    assert res.status_code == 200
    assert b"Self-play" in res.content
