# chess_rl 프로젝트 지침

AlphaZero 스타일 self-play 체스 강화학습 호비 프로젝트.

## 실행 환경
- conda 환경: `chess` (python 3.11, torch 2.5.1+cu121, GPU: RTX 4060)
- 설치: `conda run -n chess pip install -e ".[dev]"`
- 테스트: `conda run -n chess pytest -q`

## 메모리
- 이 프로젝트에서는 global memory 시스템(`~/.claude/projects/.../memory/`)을 사용하지 않는다. 기억해야 할 사항(피드백, 진행 상황, 설계 결정 등)은 이 `CLAUDE.md`나 repo 내부 문서에 직접 기록할 것.

## 작업 방식
- 새 모듈(mcts, rollout, train 등)을 작성하기 전에 설계 방향(자료구조, 알고리즘 선택 등)을 먼저 짧게 제안하고 합의한 뒤 코드를 작성할 것. 확인 없이 여러 모듈을 한 번에 구현하지 말 것.
- 초기 스캐폴드(engine/board.py, engine/action_space.py, model/network.py, tests/test_engine.py)는 이미 합의된 상태로 유지.

## 구조
```
chess_rl/
  engine/     # python-chess wrapper: board encoding, action space
  model/      # policy+value network (ResNet)
  mcts/       # MCTS 탐색 (미구현 — 다음 설계 대상)
  rollout/    # 대국 데이터 생성. self-play(양쪽 동일 정책)는 이 안의 특수 케이스 —
              # 서로 다른 두 정책끼리 붙는 대국(스타일 모방/대련용 등)도 지원할 수 있게 설계
    policy.py     # Policy 프로토콜 + RandomPolicy, NetworkPolicy
    online_value_policy.py  # OnlineValuePolicy: 판마다 실제 결과로 value head만 온라인 학습 (재현성 의도적 포기)
    game.py       # play_game(policy_white, policy_black) -> GameRecord
    game_record.py  # GameRecord: UCI 수 목록 저장 + ply별 FEN 계산
  train/      # 학습 루프 (미구현)
  configs/    # ExperimentConfig용 yaml 설정
  config.py   # ExperimentConfig dataclass (yaml 로드/저장)
  utils/
    repro.py  # seed 고정, git commit hash/dirty tree 체크, pip freeze 스냅샷
    run.py    # run 디렉토리 생성 + 재현성 메타데이터 저장
  viz/        # self-play 대국 리플레이용 로컬 웹 UI (FastAPI + vanilla JS)
tests/
scripts/
  smoke_run.py           # 재현성 인프라 자체의 end-to-end 스모크 테스트
  generate_sample_game.py  # viz 검증용 무작위 대국 생성 -> games/sample.json
  serve_viz.py           # viz 로컬 서버 실행 (기본 포트 8000)
runs/           # (gitignore) run별 checkpoints/tensorboard/meta
games/          # (gitignore) 리플레이용 게임 기록(JSON)
```

## viz (리플레이 + 인터랙티브 플레이 UI)
- `GameRecord`(`chess_rl/rollout/game_record.py`)가 UCI 수 목록을 저장하고, 각 ply의 FEN을 계산해서 내려준다.
- `chess_rl.viz.server.create_app(games_dir)`가 FastAPI 앱을 만든다:
  - 리플레이: `GET /`(화면), `GET /api/games`(목록), `GET /api/games/{id}`(moves+result+fens)
  - 인터랙티브 플레이: `GET /play`(화면), `POST /api/play/new`(정책+사람 색 지정해 세션 생성, 사람이 흑이면 AI가 먼저 둠), `POST /api/play/{id}/move`(사람 수 적용 → 불법이면 400, 프로모션 미지정 시 자동 퀸 처리 → 합법이면 AI 응답까지 적용), `GET /api/play/{id}`(현재 상태 재조회).
  - 세션은 프로세스 메모리에만 보관(`chess_rl/viz/play_session.py`). 게임이 끝나면 자동으로 `games/play_<session_id>.json`에 `GameRecord` 포맷으로 저장되어 리플레이 화면에도 바로 뜬다.
  - 상대 정책 두 가지 등록(`server.py`의 `POLICY_PROVIDERS` dict): `random`(RandomPolicy, 매 게임 새 인스턴스), `learning`(OnlineValuePolicy, **서버 프로세스 동안 같은 인스턴스를 계속 재사용** — 판을 거듭할수록 학습 누적, 재시작하면 초기화). `NetworkPolicy`는 구현은 돼 있지만(`rollout/policy.py`) 아직 UI에 안 붙여둠.
  - `learning` 정책 사용 시 응답에 추가로 담기는 필드: `ai_candidate_moves`(AI가 자기 차례에 실제로 두기 *직전*, 둘 수 있는 모든 수를 그 수를 두는 쪽 관점 가치로 평가해 내림차순 정렬한 목록), `fen_before_ai_move`(그 평가 시점의 FEN), `value_after_human_move`/`value_after_ai_move`(백 관점 value 추정치), `games_trained`(이 서버 프로세스에서 지금까지 학습한 판 수 — **매 응답에 항상 포함**돼야 새로고침해도 카운터가 0으로 보이는 표시 버그가 안 생김), `training`(게임이 이번 호출로 끝났을 때만 `{num_positions, loss_before, loss_after, games_trained}`, 안 끝났으면 `None`).
  - `extra_policies` 파라미터(`create_app(games_dir, extra_policies=...)`)로 테스트용 정책을 주입할 수 있음 — `POLICY_PROVIDERS`가 `create_app()` 내부 지역 변수라 모듈 레벨 monkeypatch로는 주입이 안 됨.
- 프론트는 외부 CDN 없이 vanilla HTML/CSS/JS(`chess_rl/viz/static/`)로, 보드 렌더링 로직은 `board.js`에 공용화(리플레이의 `app.js`, 플레이의 `play.js`가 공유). 플레이 화면은 두 보드 — 왼쪽은 실제 대국(칸 두 번 클릭으로 수 둠), 오른쪽은 `learning` 정책일 때 AI가 직전에 검토한 후보 수를 화살표(가치 높을수록 진하게)로 보여주는 "생각 스냅샷". 프로모션은 폰이 마지막 랭크 도달 시 기물 선택 버튼(Q/R/B/N)이 뜸. 색 선택에 "랜덤"도 있음(클라이언트에서 결정).
- 설치: `pip install -e ".[dev,viz]"`. 실행: `conda run -n chess python scripts/serve_viz.py --port <port>` (8000이 이미 사용 중일 수 있어 포트 확인 필요).
- 다른 기기(예: Tailscale로 연결된 휴대폰)에서 접근하려면 `--host 0.0.0.0` 지정. 이 경우 Tailscale뿐 아니라 노트북이 연결된 다른 네트워크에도 노출되니 주의. 접속 주소는 `tailscale ip -4`로 확인한 IP:포트.
- E2E 테스트(`tests/test_viz_e2e.py`)는 headless Chromium(Playwright)으로 실제 렌더링/상호작용을 검증한다. 최초 1회 `conda run -n chess python -m playwright install chromium` 필요.

## 재현성 정책
- 모든 실험 실행은 `chess_rl.utils.run.create_run_dir()`을 거쳐 `runs/<timestamp>_<name>/`을 생성한다.
- 각 run의 `meta/`에 config.yaml 스냅샷, git commit hash, `pip freeze` 결과를 저장한다.
- 기본적으로 git working tree가 dirty하면 실행을 거부한다 (`DirtyWorkingTreeError`). 의도적으로 허용하려면 `allow_dirty=True` / `--allow-dirty`.
- seed는 `chess_rl.utils.repro.set_seed()`로 python/numpy/torch(+cuda)를 한 번에 고정한다.
- metric 추적은 TensorBoard(`runs/<run>/tensorboard/`)를 사용한다.

## 아이디어 백로그
- 당장 구현하지 않지만 설계에 영향을 줄 아이디어는 `docs/IDEAS.md`에 기록한다. (예: 스타일 모방/대련용 정책 — 이 함의로 모듈 이름을 selfplay 대신 rollout으로 정함)

## 진행 상황
- [x] repo 구조, pyproject.toml, conda 환경(`chess`) 세팅
- [x] engine: board encoding (12,8,8 plane), 고정 action space (64x64 + underpromotion)
- [x] model: policy+value ResNet
- [x] 재현성 인프라: ExperimentConfig, seed 고정, run 디렉토리 + 메타데이터 스냅샷, TensorBoard 연결 (scripts/smoke_run.py로 검증)
- [x] viz: 대국 리플레이 + 사람 vs 정책 인터랙티브 플레이 로컬 웹 UI (FastAPI + vanilla JS)
- [x] rollout: Policy 인터페이스(RandomPolicy, NetworkPolicy), play_game(), viz의 사람 vs AI 플레이로 실사용 검증
- [x] OnlineValuePolicy: 판마다 실제 결과로 value head만 학습하는 재미용 AI, viz에 화살표/게이지/차트/loss 비교 시각화 (재현성 의도적 포기)
- [ ] mcts
- [ ] train
