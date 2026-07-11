"""self-play 대국 리플레이 + 사람 vs 정책 인터랙티브 대국용 로컬 FastAPI 서버."""

import uuid
from pathlib import Path

import chess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.model.network import PolicyValueNet
from chess_rl.rollout.game_record import GameRecord
from chess_rl.rollout.online_value_policy import OnlineValuePolicy
from chess_rl.rollout.policy import RandomPolicy
from chess_rl.viz.play_session import PlaySession

STATIC_DIR = Path(__file__).parent / "static"


class NewGameRequest(BaseModel):
    human_color: str = "white"  # "white" | "black"
    policy: str = "random"


class MoveRequest(BaseModel):
    move: str  # UCI 표기


def create_app(games_dir: str = "games", extra_policies: dict | None = None) -> FastAPI:
    games_path = Path(games_dir)
    app = FastAPI()
    sessions: dict[str, PlaySession] = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # "learning" 정책은 서버 프로세스가 살아있는 동안 같은 인스턴스를 계속 재사용해야
    # 판을 거듭할수록 학습이 누적된다 (재시작하면 초기화 — docs/IDEAS.md 참고).
    learning_policy = OnlineValuePolicy(
        PolicyValueNet(in_planes=12, action_space_size=ACTION_SPACE_SIZE, channels=64, num_blocks=4),
        device=device,
        checkpoint_dir="checkpoints/online_value",
        family="human_online",
        training_method=(
            "사람과의 실시간 대국(viz /play). 매 수 MCTS(root Dirichlet noise 없음) 탐색 후 "
            "방문분포 argmax로 둠. 판 종료 시 그 판의 포지션(사람 수 포함)을 replay buffer에 "
            "적립하고, buffer에서 샘플링한 배치로 policy는 REINFORCE(결과-가중, value baseline), "
            "value는 MSE로 함께 학습."
        ),
        checkpoint_every=1,
    )
    POLICY_PROVIDERS = {
        "random": lambda: RandomPolicy(),
        "learning": lambda: learning_policy,
    }
    if extra_policies:
        POLICY_PROVIDERS.update(extra_policies)  # 테스트 등에서 정책을 주입할 때 사용

    def apply_ai_move(session: PlaySession) -> dict:
        if session.board.is_game_over():
            return {"move": None, "candidate_moves": None, "value": None}

        candidate_moves = None
        if hasattr(session.ai_policy, "move_values"):
            candidate_moves = session.ai_policy.move_values(session.board)

        move = session.ai_policy.select_move(session.board)
        session.moves.append(move.uci())
        session.board.push(move)

        value = None
        if hasattr(session.ai_policy, "value_estimate_white_perspective"):
            value = session.ai_policy.value_estimate_white_perspective(session.board)

        return {"move": move.uci(), "candidate_moves": candidate_moves, "value": value}

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
        if body.policy not in POLICY_PROVIDERS:
            raise HTTPException(status_code=400, detail=f"unknown policy '{body.policy}'")
        if body.human_color not in ("white", "black"):
            raise HTTPException(status_code=400, detail="human_color must be 'white' or 'black'")

        human_color = chess.WHITE if body.human_color == "white" else chess.BLACK
        session_id = uuid.uuid4().hex
        session = PlaySession(board=chess.Board(), ai_policy=POLICY_PROVIDERS[body.policy](), human_color=human_color)
        sessions[session_id] = session

        fen_before_ai_move = None
        ai_result = {"move": None, "candidate_moves": None, "value": None}
        if session.board.turn != human_color:
            fen_before_ai_move = session.board.fen()
            ai_result = apply_ai_move(session)

        return {
            "session_id": session_id,
            "ai_move": ai_result["move"],
            "ai_candidate_moves": ai_result["candidate_moves"],
            "fen_before_ai_move": fen_before_ai_move,
            "value_after_ai_move": ai_result["value"],
            "training": None,
            "games_trained": getattr(session.ai_policy, "games_trained", None),
            **session.to_state(),
        }

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

        value_after_human_move = None
        if hasattr(session.ai_policy, "value_estimate_white_perspective"):
            value_after_human_move = session.ai_policy.value_estimate_white_perspective(session.board)

        fen_before_ai_move = session.board.fen()
        ai_result = apply_ai_move(session)

        training = None
        if session.board.is_game_over() and hasattr(session.ai_policy, "learn_from_game"):
            training = session.ai_policy.learn_from_game(session.moves, session.board.result())

        save_if_over(session_id, session)

        return {
            "human_move": move.uci(),
            "ai_move": ai_result["move"],
            "ai_candidate_moves": ai_result["candidate_moves"],
            "fen_before_ai_move": fen_before_ai_move,
            "value_after_human_move": value_after_human_move,
            "value_after_ai_move": ai_result["value"],
            "training": training,
            "games_trained": getattr(session.ai_policy, "games_trained", None),
            **session.to_state(),
        }

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
