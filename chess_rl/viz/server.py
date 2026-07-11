"""self-play 대국 리플레이 + 사람 vs 정책 인터랙티브 대국용 로컬 FastAPI 서버."""

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.eval.arena import play_match
from chess_rl.model.network import PolicyValueNet
from chess_rl.rollout.game_record import GameRecord
from chess_rl.rollout.online_value_policy import OnlineValuePolicy
from chess_rl.rollout.policy import RandomPolicy
from chess_rl.utils.checkpoint import (
    list_checkpoints,
    load_checkpoint,
    read_family_meta,
)
from chess_rl.viz.play_session import PlaySession

STATIC_DIR = Path(__file__).parent / "static"


class NewGameRequest(BaseModel):
    human_color: str = "white"  # "white" | "black"
    policy: str = "random"


class MoveRequest(BaseModel):
    move: str  # UCI 표기


def create_app(
    games_dir: str = "games",
    extra_policies: dict | None = None,
    checkpoint_dir: str = "checkpoints/online_value",
) -> FastAPI:
    games_path = Path(games_dir)
    app = FastAPI()
    sessions: dict[str, PlaySession] = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # "learning" 정책은 서버 프로세스가 살아있는 동안 같은 인스턴스를 계속 재사용해야
    # 판을 거듭할수록 학습이 누적된다 (재시작하면 초기화 — docs/IDEAS.md 참고). 재시작마다
    # 완전히 무관한 새 계보가 시작되는 것이므로, family 이름에 프로세스 시작 시각을 붙여
    # 매번 고유하게 만든다 — 안 그러면 같은 family 디렉터리에 이미 checkpoint가 있어서
    # OnlineValuePolicy가 시작 시 에러를 던진다.
    family = "human_online_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    learning_policy = OnlineValuePolicy(
        PolicyValueNet(
            in_planes=12, action_space_size=ACTION_SPACE_SIZE, channels=64, num_blocks=4
        ),
        device=device,
        checkpoint_dir=checkpoint_dir,
        family=family,
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

    # 지금 학습 중인 family(위 family)가 새 checkpoint를 찍을 때마다, 가장 최근에 시작된
    # *다른* family의 최신 checkpoint와 100판 붙여서 "다른 계보 대비 얼마나 나아졌는지"를
    # 백그라운드에서 갱신한다. 매 수마다 도는 실시간 대국과 같은 GPU를 쓰므로 무거울 수
    # 있음 — 일단 켜두고 실제로 얼마나 부담되는지 관찰하기로 함(docs/IDEAS.md 성능 TODO).
    comparison_lock = threading.Lock()
    comparison_state = {
        "status": "idle",  # idle | running | done | no_opponent | error
        "own_family": family,
        "own_games_trained": None,
        "opponent_family": None,
        "opponent_games_trained": None,
        "result": None,
        "updated_at": None,
        "error": None,
    }

    def _find_latest_other_family(base_dir: str, exclude_family: str):
        base = Path(base_dir)
        if not base.exists():
            return None

        candidates = []
        for sub in base.iterdir():
            if not sub.is_dir() or sub.name == exclude_family:
                continue
            checkpoints = list_checkpoints(str(sub))
            if not checkpoints:
                continue
            meta = read_family_meta(str(sub))
            candidates.append((meta.started_at, sub.name, checkpoints[-1]))

        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        _, opponent_family, latest_checkpoint = candidates[-1]
        return opponent_family, latest_checkpoint

    def _run_comparison(own_checkpoint_path: str, own_games_trained: int) -> None:
        with comparison_lock:
            if comparison_state["status"] == "running":
                return  # 이미 갱신 중이면 이번 checkpoint는 건너뜀(쌓이지 않게)
            comparison_state["status"] = "running"

        try:
            found = _find_latest_other_family(checkpoint_dir, exclude_family=family)
            if found is None:
                with comparison_lock:
                    comparison_state.update(
                        status="no_opponent",
                        own_games_trained=own_games_trained,
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    )
                return

            opponent_family, opponent_checkpoint = found
            own_model = load_checkpoint(own_checkpoint_path, device)
            opponent_model = load_checkpoint(opponent_checkpoint.path, device)
            match = play_match(
                own_model,
                opponent_model,
                num_games=100,
                mcts_simulations=200,
                device=device,
            )

            with comparison_lock:
                comparison_state.update(
                    status="done",
                    own_games_trained=own_games_trained,
                    opponent_family=opponent_family,
                    opponent_games_trained=opponent_checkpoint.games_trained,
                    result=match,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    error=None,
                )
        except Exception as e:
            with comparison_lock:
                comparison_state.update(
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )

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
            raise HTTPException(
                status_code=400, detail=f"unknown policy '{body.policy}'"
            )
        if body.human_color not in ("white", "black"):
            raise HTTPException(
                status_code=400, detail="human_color must be 'white' or 'black'"
            )

        human_color = chess.WHITE if body.human_color == "white" else chess.BLACK
        session_id = uuid.uuid4().hex
        session = PlaySession(
            board=chess.Board(),
            ai_policy=POLICY_PROVIDERS[body.policy](),
            human_color=human_color,
        )
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
            raise HTTPException(
                status_code=404, detail=f"session '{session_id}' not found"
            )
        return session.to_state()

    @app.post("/api/play/{session_id}/move")
    def make_move(session_id: str, body: MoveRequest):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail=f"session '{session_id}' not found"
            )
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
                raise HTTPException(
                    status_code=400, detail=f"illegal move '{body.move}'"
                )

        session.moves.append(move.uci())
        session.board.push(move)

        value_after_human_move = None
        if hasattr(session.ai_policy, "value_estimate_white_perspective"):
            value_after_human_move = session.ai_policy.value_estimate_white_perspective(
                session.board
            )

        fen_before_ai_move = session.board.fen()
        ai_result = apply_ai_move(session)

        training = None
        if session.board.is_game_over() and hasattr(
            session.ai_policy, "learn_from_game"
        ):
            training = session.ai_policy.learn_from_game(
                session.moves, session.board.result()
            )
            checkpoint_path = training.pop("checkpoint_path", None)
            if checkpoint_path:
                threading.Thread(
                    target=_run_comparison,
                    args=(checkpoint_path, training["games_trained"]),
                    daemon=True,
                ).start()

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

    @app.get("/api/comparison")
    def get_comparison():
        with comparison_lock:
            return dict(comparison_state)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
