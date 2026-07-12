"""self-play 대국 리플레이 + 사람 vs 정책 인터랙티브 대국용 로컬 FastAPI 서버."""

import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.engine.board import NUM_INPUT_PLANES
from chess_rl.eval.arena import find_new_frontier
from chess_rl.model.network import MaterialBlendedPolicyValueNet, PolicyValueNet
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

    # 서버 프로세스 stdout(uvicorn 로그, [comparison] 진행 상황 등)을 화면 하단 로그
    # 패널에서도 볼 수 있게 최근 것만 메모리에 따로 들고 있는다 — 터미널/파일 로그를
    # 못 보는 상황(다른 기기에서 접속 등)에서도 진행 상황을 확인하기 위함. level로
    # 화면에서 드롭다운 필터링 가능(GET /api/logs).
    log_lines: deque[dict] = deque(maxlen=500)

    def _log(message: str, level: str = "info") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level.upper()} {message}", flush=True)
        log_lines.append({"timestamp": timestamp, "level": level, "message": message})

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
        MaterialBlendedPolicyValueNet(
            PolicyValueNet(
                in_planes=NUM_INPUT_PLANES,
                action_space_size=ACTION_SPACE_SIZE,
                channels=64,
                num_blocks=4,
            ),
            material_weight=0.5,
        ),
        device=device,
        train_epochs=25000,
        checkpoint_dir=checkpoint_dir,
        family=family,
        training_method=(
            "사람과의 실시간 대국(viz /play). 매 수 MCTS(root Dirichlet noise 없음) 탐색 후 "
            "방문분포 argmax로 둠. 판 종료 시 그 판의 포지션(사람 수 포함)을 replay buffer에 "
            "적립하고, buffer에서 샘플링한 배치로 policy는 REINFORCE(결과-가중, value baseline), "
            "value는 MSE로 함께 학습. train_epochs=25000(2000이 실측 ~5초뿐이라 재상향, 판당 "
            "약 1분 학습 실험 — 고정 배치 하나로 도는 구조라 과적합 위험 관찰 필요). "
            "인코딩은 13-plane(기물 12 + 차례 1 — 이전 family들의 12-plane 인코딩은 차례 정보가 "
            "없어 value 학습에 결함), value는 MaterialBlendedPolicyValueNet으로 신경망 출력과 "
            "재료 점수 휴리스틱(tanh(재료차/10))을 0.5:0.5 가중합."
        ),
        checkpoint_every=1,
    )
    POLICY_PROVIDERS = {
        "random": lambda: RandomPolicy(),
        # 매 대국마다 canonical 모델의 독립 복사본을 새로 받는다 — learn_from_game()이
        # 오래 걸리는 동안에도(train_epochs가 클수록 더) 다른 대국이 안전하게 계속 추론할 수 있게
        # (online_value_policy.py 모듈 docstring '동시성' 절 참고).
        "learning": lambda: learning_policy.new_inference_handle(),
    }
    _log(
        f"서버 시작 — family={family}, device={device}, "
        f"train_epochs={learning_policy.train_epochs}"
    )
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

    # 비교가 이미 진행 중일 때 새로 생긴 checkpoint는 여기 쌓인다. 도착 순서 = games_trained
    # 오름차순이므로 리스트 끝에서 꺼내면(pop, LIFO) "가장 최신 것부터" 처리되고, 그 뒤에도
    # 계속 남아있으면(유휴 시간이 나면) 순서대로 나머지도 다 처리된다 — 아무것도 안 버림.
    pending_checkpoints: list[tuple[str, int]] = []

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
        _log(
            f"[comparison] vs {match_entry['opponent_family']}"
            f"({match_entry['opponent_games_trained']}판): "
            f"{match_entry['a_wins']}승 {match_entry['b_wins']}패 {match_entry['draws']}무 "
            f"({'승' if won else '패'}), {match_entry['elapsed_seconds']:.1f}초"
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
                    "own_games_trained": comparison_state["own_games_trained"],
                    "opponent_games_trained": match_entry["opponent_games_trained"],
                    "won": won,
                    "elapsed_seconds": match_entry["elapsed_seconds"],
                    "best_beaten_games_trained": comparison_state[
                        "best_beaten_games_trained"
                    ],
                }
            )
            comparison_state["updated_at"] = datetime.now(timezone.utc).isoformat()

    def _schedule_comparison(checkpoint_path: str, games_trained: int) -> None:
        """새 checkpoint가 생길 때마다 호출됨. 이미 비교 중이면 대기열에 쌓아두기만 하고
        (실행 중인 스레드가 끝나고 나서 이어서 처리), 유휴 상태면 새 스레드를 띄운다."""
        with comparison_lock:
            if comparison_state["status"] == "running":
                pending_checkpoints.append((checkpoint_path, games_trained))
                _log(
                    f"[comparison] {games_trained}판 checkpoint 대기열에 추가 "
                    f"(현재 비교 진행 중, 대기 {len(pending_checkpoints)}개)"
                )
                return
            comparison_state["status"] = "running"
            comparison_state["own_games_trained"] = games_trained

        threading.Thread(
            target=_run_comparison_chain,
            args=(checkpoint_path, games_trained),
            daemon=True,
        ).start()

    def _run_comparison_chain(checkpoint_path: str, games_trained: int) -> None:
        """checkpoint 하나를 비교하고, 대기열이 남아있으면(최신 것부터) 계속 이어서 처리하다가
        대기열이 비면 멈춘다."""
        while True:
            _run_comparison_once(checkpoint_path, games_trained)

            with comparison_lock:
                if not pending_checkpoints:
                    if comparison_state["status"] == "running":
                        comparison_state["status"] = "done"
                    break
                # 도착 순서 = games_trained 오름차순이므로 리스트 끝(pop)이 가장 최신.
                checkpoint_path, games_trained = pending_checkpoints.pop()
                comparison_state["own_games_trained"] = games_trained
                comparison_state["status"] = "running"
            _log(
                f"[comparison] 대기열에서 {games_trained}판 checkpoint 이어서 비교 "
                f"(남은 대기 {len(pending_checkpoints)}개)"
            )

    def _run_comparison_once(own_checkpoint_path: str, own_games_trained: int) -> None:
        _log(f"[comparison] {family} {own_games_trained}판 checkpoint — 비교 시작")
        started_at = time.time()
        try:
            found = _find_latest_other_family(checkpoint_dir, exclude_family=family)
            if found is None:
                _log("[comparison] 비교할 다른 family 없음")
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
                # num_games는 배치로 한 번에 도니 늘려도 비용이 거의 안 붙지만,
                # mcts_simulations는 시뮬레이션 루프가 순차라 직접 비례해서 느려짐
                # (실측: sims=200일 때 100판 매치 1회에 785초) — 여기부터 줄인다.
                mcts_simulations=50,
                device=device,
                on_match=_on_match,
            )

            with comparison_lock:
                frontier_state["idx"] = result["frontier_idx"]
                # status="done"은 여기서 안 정함 — 대기열에 더 있으면 _run_comparison_chain이
                # 곧바로 다음 checkpoint로 이어갈 거라 "running"을 유지해야 함.
                comparison_state.update(
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    error=None,
                )
            _log(
                f"[comparison] 완료 — frontier_idx={result['frontier_idx']}, "
                f"매치 {len(result['matches'])}회, 총 {time.time() - started_at:.1f}초"
            )
        except Exception as e:
            _log(f"[comparison] 실패: {type(e).__name__}: {e}", level="error")
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

        # search_move_with_candidates가 있으면(OnlineValuePolicy 계열) MCTS를 한 번만
        # 돌려서 수 선택과 화살표 후보를 같은 탐색에서 얻는다 — 화살표(후보 평가)와 실제
        # 둔 수의 근거가 일치하고, 탐색+후보평가를 따로 하던 중복 계산도 없어짐.
        candidate_moves = None
        if hasattr(session.ai_policy, "search_move_with_candidates"):
            move, candidate_moves = session.ai_policy.search_move_with_candidates(
                session.board
            )
        else:
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
        _log(f"새 게임 시작 — policy={body.policy}, human_color={body.human_color}")

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

    def _handle_game_end(session_id: str, session: PlaySession) -> None:
        """게임이 끝났으면 로그 + 백그라운드 학습 트리거 + 기록 저장. 안 끝났으면 no-op."""
        if not session.board.is_game_over():
            return

        _log(f"게임 종료 — 결과 {session.board.result()}, {len(session.moves)}수")

        if hasattr(session.ai_policy, "learn_from_game"):
            # 학습(train_epochs가 크면 분 단위)을 요청 처리 안에서 동기로 돌리면 마지막
            # 수가 학습이 끝날 때까지 보드에 반영 안 되는 문제가 있어 백그라운드로 뺀다 —
            # 결과(loss 등)는 로그 패널([train] 라인)로 확인. 학습 자체의 동시성은
            # OnlineValuePolicy 쪽 복사/병합 구조가 보장(모듈 docstring '동시성' 절).
            moves_snapshot = list(session.moves)
            game_result = session.board.result()
            ai_policy = session.ai_policy

            def _train_in_background() -> None:
                _log(
                    f"[train] {family} 학습 시작 — {len(moves_snapshot)}수, 결과 {game_result}"
                )
                train_started_at = time.time()
                training = ai_policy.learn_from_game(moves_snapshot, game_result)
                train_elapsed = time.time() - train_started_at
                _log(
                    f"[train] {family} {training['games_trained']}판째 학습 완료 — "
                    f"loss {training['loss_before']:.4f} → {training['loss_after']:.4f} "
                    f"(buffer={training.get('buffer_size', 'n/a')}), {train_elapsed:.1f}초"
                )
                checkpoint_path = training.get("checkpoint_path")
                if checkpoint_path:
                    _schedule_comparison(checkpoint_path, training["games_trained"])

            threading.Thread(target=_train_in_background, daemon=True).start()

        save_if_over(session_id, session)

    @app.post("/api/play/{session_id}/move")
    def make_move(session_id: str, body: MoveRequest):
        """사람 수만 적용하고 바로 반환 — AI 응수는 별도 엔드포인트(/ai-move)로.

        클라이언트가 사람 수를 먼저 렌더링한 뒤 AI 수는 계산되는 대로 이어서 렌더링할
        수 있게 하기 위한 분리(AI 탐색이 수 초 걸리는 동안 사람 수가 보드에 안 보이는
        문제 해결).
        """
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

        _handle_game_end(session_id, session)

        return {
            "human_move": move.uci(),
            "human_move_san": human_move_san,
            "value_after_human_move": value_after_human_move,
            "training": None,  # 학습은 백그라운드 진행 — 결과는 /api/logs의 [train] 라인으로
            "games_trained": getattr(session.ai_policy, "games_trained", None),
            **session.to_state(),
        }

    @app.post("/api/play/{session_id}/ai-move")
    def make_ai_move(session_id: str):
        """AI 응수 한 수를 계산·적용하고 반환 — /move로 사람 수를 적용한 뒤 호출."""
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail=f"session '{session_id}' not found"
            )
        if session.board.is_game_over():
            raise HTTPException(status_code=400, detail="game is already over")
        if session.board.turn == session.human_color:
            raise HTTPException(status_code=400, detail="it's the human's turn")

        fen_before_ai_move = session.board.fen()
        ai_result = apply_ai_move(session)

        _handle_game_end(session_id, session)

        return {
            "ai_move": ai_result["move"],
            "ai_move_san": ai_result["move_san"],
            "ai_candidate_moves": ai_result["candidate_moves"],
            "fen_before_ai_move": fen_before_ai_move,
            "value_after_ai_move": ai_result["value"],
            "training": None,
            "games_trained": getattr(session.ai_policy, "games_trained", None),
            **session.to_state(),
        }

    @app.get("/api/comparison")
    def get_comparison():
        with comparison_lock:
            return dict(comparison_state)

    @app.get("/api/logs")
    def get_logs():
        return {"lines": list(log_lines)}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
