"""self-play 대국 리플레이 + 사람 vs 정책 인터랙티브 대국용 로컬 FastAPI 서버."""

import threading
import time
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
from chess_rl.eval.arena import find_new_frontier
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
    # OnlineValuePolicy가 시작 시 에러를 던진다. 사람이 눈으로 구분하는 이름이라 로컬
    # 시각을 쓴다(family_meta.json의 started_at/last_updated_at 등 기계가 읽는 타임스탬프는
    # UTC 유지 — 아래 comparison_state 쪽 참고).
    family = "human_online_" + datetime.now().strftime("%Y%m%dT%H%M%S")
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
    # *다른* family의 checkpoint들을 상대로 find_new_frontier()를 돌려서 "다른 계보의
    # 어디까지 이기는지"를 백그라운드에서 갱신한다. 매 수마다 도는 실시간 대국과 같은
    # GPU를 쓰므로 무거울 수 있음 — 일단 켜두고 실제로 얼마나 부담되는지 관찰하기로 함
    # (docs/IDEAS.md 성능 TODO).
    comparison_lock = threading.Lock()
    comparison_state = {
        "status": "idle",  # idle | running | done | no_opponent | error
        "own_family": family,
        "own_games_trained": None,
        "opponent_family": None,
        "best_beaten_games_trained": None,  # 지금까지 이긴 것 중 가장 강한(=games_trained 큰) 상대
        "history": [],  # 매치가 끝날 때마다 하나씩 추가: {opponent_games_trained, won, best_beaten_games_trained}
        "updated_at": None,
        "error": None,
    }
    # find_new_frontier가 "직전에 어디까지 갔는지"부터 이어서 걷도록, opponent family별
    # 마지막 frontier index를 기억해둔다. 상대 family가 바뀌면(더 최근 family가 새로
    # 생기면) 그 family에 대해서는 처음부터(-1) 다시 걷는다.
    frontier_state = {"opponent_family": None, "idx": -1}

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
            candidates.append((meta.started_at, sub.name, checkpoints))

        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        _, opponent_family, checkpoints = candidates[-1]
        return opponent_family, checkpoints

    def _on_match(match_entry: dict, won: bool) -> None:
        """매치 하나가 끝날 때마다(walk 전체가 끝나기 전에도) 바로 화면에 반영."""
        print(
            f"[comparison] vs {match_entry['opponent_family']}"
            f"({match_entry['opponent_games_trained']}판): "
            f"{match_entry['a_wins']}승 {match_entry['b_wins']}패 {match_entry['draws']}무 "
            f"({'승' if won else '패'}), {match_entry['elapsed_seconds']:.1f}초",
            flush=True,
        )
        with comparison_lock:
            if won and (
                comparison_state["best_beaten_games_trained"] is None
                or match_entry["opponent_games_trained"]
                > comparison_state["best_beaten_games_trained"]
            ):
                comparison_state["best_beaten_games_trained"] = match_entry[
                    "opponent_games_trained"
                ]
            comparison_state["history"].append(
                {
                    "opponent_games_trained": match_entry["opponent_games_trained"],
                    "won": won,
                    "elapsed_seconds": match_entry["elapsed_seconds"],
                    "best_beaten_games_trained": comparison_state[
                        "best_beaten_games_trained"
                    ],
                }
            )
            comparison_state["updated_at"] = datetime.now(timezone.utc).isoformat()

    def _run_comparison(own_checkpoint_path: str, own_games_trained: int) -> None:
        with comparison_lock:
            if comparison_state["status"] == "running":
                return  # 이미 갱신 중이면 이번 checkpoint는 건너뜀(쌓이지 않게)
            comparison_state["status"] = "running"
            comparison_state["own_games_trained"] = own_games_trained

        print(
            f"[comparison] {family} {own_games_trained}판 checkpoint — 비교 시작",
            flush=True,
        )
        started_at = time.time()
        try:
            found = _find_latest_other_family(checkpoint_dir, exclude_family=family)
            if found is None:
                print("[comparison] 비교할 다른 family 없음", flush=True)
                with comparison_lock:
                    comparison_state.update(
                        status="no_opponent",
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    )
                return

            opponent_family, opponent_checkpoints = found
            with comparison_lock:
                if frontier_state["opponent_family"] != opponent_family:
                    # 상대 family가 바뀌었으니 이전 진행 상황(history/best)은 더는
                    # 의미가 없음 — 새 family 기준으로 처음부터 다시 걷는다.
                    frontier_state["opponent_family"] = opponent_family
                    frontier_state["idx"] = -1
                    comparison_state["opponent_family"] = opponent_family
                    comparison_state["best_beaten_games_trained"] = None
                    comparison_state["history"] = []
                start_idx = frontier_state["idx"]

            own_model = load_checkpoint(own_checkpoint_path, device)
            result = find_new_frontier(
                own_model,
                opponent_checkpoints,
                start_idx,
                num_games=100,
                mcts_simulations=200,
                device=device,
                on_match=_on_match,
            )

            with comparison_lock:
                frontier_state["idx"] = result["frontier_idx"]
                comparison_state.update(
                    status="done",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    error=None,
                )
            print(
                f"[comparison] 완료 — frontier_idx={result['frontier_idx']}, "
                f"매치 {len(result['matches'])}회, 총 {time.time() - started_at:.1f}초",
                flush=True,
            )
        except Exception as e:
            print(f"[comparison] 실패: {type(e).__name__}: {e}", flush=True)
            with comparison_lock:
                comparison_state.update(
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )

    def apply_ai_move(session: PlaySession) -> dict:
        if session.board.is_game_over():
            return {
                "move": None,
                "move_san": None,
                "candidate_moves": None,
                "value": None,
            }

        candidate_moves = None
        if hasattr(session.ai_policy, "move_values"):
            candidate_moves = session.ai_policy.move_values(session.board)

        move = session.ai_policy.select_move(session.board)
        move_san = session.board.san(move)
        session.push_move(move)

        value = None
        if hasattr(session.ai_policy, "value_estimate_white_perspective"):
            value = session.ai_policy.value_estimate_white_perspective(session.board)

        return {
            "move": move.uci(),
            "move_san": move_san,
            "candidate_moves": candidate_moves,
            "value": value,
        }

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
        ai_result = {
            "move": None,
            "move_san": None,
            "candidate_moves": None,
            "value": None,
        }
        if session.board.turn != human_color:
            fen_before_ai_move = session.board.fen()
            ai_result = apply_ai_move(session)

        return {
            "session_id": session_id,
            "ai_move": ai_result["move"],
            "ai_move_san": ai_result["move_san"],
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

        human_move_san = session.board.san(move)
        session.push_move(move)

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
            "human_move_san": human_move_san,
            "ai_move": ai_result["move"],
            "ai_move_san": ai_result["move_san"],
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
