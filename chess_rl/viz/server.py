"""self-play 대국 리플레이용 로컬 FastAPI 서버."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chess_rl.viz.game_record import GameRecord

STATIC_DIR = Path(__file__).parent / "static"


def create_app(games_dir: str = "games") -> FastAPI:
    games_path = Path(games_dir)
    app = FastAPI()

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/games")
    def list_games():
        if not games_path.exists():
            return []
        return sorted(p.stem for p in games_path.glob("*.json"))

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str):
        path = games_path / f"{game_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"game '{game_id}' not found")
        record = GameRecord.from_json(path)
        return {"moves": record.moves, "result": record.result, "fens": record.fens()}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
