"""self-play 대국 리플레이 + 사람 vs 정책 인터랙티브 대국용 로컬 FastAPI 서버."""

import uuid
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chess_rl.rollout.game_record import GameRecord
from chess_rl.rollout.policy import RandomPolicy
from chess_rl.viz.play_session import PlaySession

STATIC_DIR = Path(__file__).parent / "static"

POLICIES = {"random": RandomPolicy}


class NewGameRequest(BaseModel):
    human_color: str = "white"  # "white" | "black"
    policy: str = "random"


class MoveRequest(BaseModel):
    move: str  # UCI 표기


def create_app(games_dir: str = "games") -> FastAPI:
    games_path = Path(games_dir)
    app = FastAPI()
    sessions: dict[str, PlaySession] = {}

    def apply_ai_move(session: PlaySession) -> str | None:
        if session.board.is_game_over():
            return None
        move = session.ai_policy.select_move(session.board)
        session.moves.append(move.uci())
        session.board.push(move)
        return move.uci()

    def save_if_over(session_id: str, session: PlaySession) -> None:
        if not session.board.is_game_over():
            return
        games_path.mkdir(parents=True, exist_ok=True)
        record = GameRecord(moves=session.moves, result=session.board.result())
        record.to_json(games_path / f"play_{session_id}.json")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/play")
    def play_page():
        return FileResponse(STATIC_DIR / "play.html")

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

    @app.post("/api/play/new")
    def new_game(body: NewGameRequest):
        if body.policy not in POLICIES:
            raise HTTPException(status_code=400, detail=f"unknown policy '{body.policy}'")
        if body.human_color not in ("white", "black"):
            raise HTTPException(status_code=400, detail="human_color must be 'white' or 'black'")

        human_color = chess.WHITE if body.human_color == "white" else chess.BLACK
        session_id = uuid.uuid4().hex
        session = PlaySession(board=chess.Board(), ai_policy=POLICIES[body.policy](), human_color=human_color)
        sessions[session_id] = session

        ai_move = None
        if session.board.turn != human_color:
            ai_move = apply_ai_move(session)

        return {"session_id": session_id, "ai_move": ai_move, **session.to_state()}

    @app.get("/api/play/{session_id}")
    def get_play_state(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
        return session.to_state()

    @app.post("/api/play/{session_id}/move")
    def make_move(session_id: str, body: MoveRequest):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session '{session_id}' not found")
        if session.board.is_game_over():
            raise HTTPException(status_code=400, detail="game is already over")
        if session.board.turn != session.human_color:
            raise HTTPException(status_code=400, detail="not human's turn")

        try:
            move = chess.Move.from_uci(body.move)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid uci '{body.move}'")

        if move not in session.board.legal_moves:
            queen_promo = chess.Move.from_uci(body.move + "q")
            if queen_promo in session.board.legal_moves:
                move = queen_promo  # 프로모션은 v1에서 퀸 고정
            else:
                raise HTTPException(status_code=400, detail=f"illegal move '{body.move}'")

        session.moves.append(move.uci())
        session.board.push(move)

        ai_move = apply_ai_move(session)
        save_if_over(session_id, session)

        return {"human_move": move.uci(), "ai_move": ai_move, **session.to_state()}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
