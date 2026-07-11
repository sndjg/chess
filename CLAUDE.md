# chess_rl 프로젝트 지침

AlphaZero 스타일 self-play 체스 강화학습 호비 프로젝트.

## 실행 환경
- conda 환경: `chess` (python 3.11, torch 2.5.1+cu121, GPU: RTX 4060)
- 설치: `conda run -n chess pip install -e ".[dev]"`
- 테스트: `conda run -n chess pytest -q`

## 메모리
- 이 프로젝트에서는 global memory 시스템(`~/.claude/projects/.../memory/`)을 사용하지 않는다. 기억해야 할 사항(피드백, 진행 상황, 설계 결정 등)은 이 `CLAUDE.md`나 repo 내부 문서에 직접 기록할 것.

## 작업 방식
- 새 모듈(mcts, selfplay, train 등)을 작성하기 전에 설계 방향(자료구조, 알고리즘 선택 등)을 먼저 짧게 제안하고 합의한 뒤 코드를 작성할 것. 확인 없이 여러 모듈을 한 번에 구현하지 말 것.
- 초기 스캐폴드(engine/board.py, engine/action_space.py, model/network.py, tests/test_engine.py)는 이미 합의된 상태로 유지.

## 구조
```
chess_rl/
  engine/     # python-chess wrapper: board encoding, action space
  model/      # policy+value network (ResNet)
  mcts/       # MCTS 탐색 (미구현 — 다음 설계 대상)
  selfplay/   # self-play 데이터 생성 (미구현)
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

## viz (리플레이 UI)
- `GameRecord`(`chess_rl/viz/game_record.py`)가 UCI 수 목록을 저장하고, 각 ply의 FEN을 계산해서 내려준다.
- `chess_rl.viz.server.create_app(games_dir)`가 FastAPI 앱을 만든다: `GET /api/games`(목록), `GET /api/games/{id}`(moves+result+fens).
- 프론트는 외부 CDN 없이 vanilla HTML/CSS/JS(`chess_rl/viz/static/`)로 8x8 보드를 렌더링하고 슬라이더로 리플레이.
- self-play가 게임을 이 JSON 포맷(`{"moves": [...], "result": ...}`)으로 저장하면 그대로 붙는다.
- 설치: `pip install -e ".[dev,viz]"`. 실행: `conda run -n chess python scripts/serve_viz.py --port <port>` (8000이 이미 사용 중일 수 있어 포트 확인 필요).
- 다른 기기(예: Tailscale로 연결된 휴대폰)에서 접근하려면 `--host 0.0.0.0` 지정. 이 경우 Tailscale뿐 아니라 노트북이 연결된 다른 네트워크에도 노출되니 주의. 접속 주소는 `tailscale ip -4`로 확인한 IP:포트.
- E2E 테스트(`tests/test_viz_e2e.py`)는 headless Chromium(Playwright)으로 실제 렌더링/상호작용을 검증한다. 최초 1회 `conda run -n chess python -m playwright install chromium` 필요.

## 재현성 정책
- 모든 실험 실행은 `chess_rl.utils.run.create_run_dir()`을 거쳐 `runs/<timestamp>_<name>/`을 생성한다.
- 각 run의 `meta/`에 config.yaml 스냅샷, git commit hash, `pip freeze` 결과를 저장한다.
- 기본적으로 git working tree가 dirty하면 실행을 거부한다 (`DirtyWorkingTreeError`). 의도적으로 허용하려면 `allow_dirty=True` / `--allow-dirty`.
- seed는 `chess_rl.utils.repro.set_seed()`로 python/numpy/torch(+cuda)를 한 번에 고정한다.
- metric 추적은 TensorBoard(`runs/<run>/tensorboard/`)를 사용한다.

## 진행 상황
- [x] repo 구조, pyproject.toml, conda 환경(`chess`) 세팅
- [x] engine: board encoding (12,8,8 plane), 고정 action space (64x64 + underpromotion)
- [x] model: policy+value ResNet
- [x] 재현성 인프라: ExperimentConfig, seed 고정, run 디렉토리 + 메타데이터 스냅샷, TensorBoard 연결 (scripts/smoke_run.py로 검증)
- [x] viz: self-play 대국 리플레이 로컬 웹 UI (FastAPI + vanilla JS), 샘플 대국으로 API/화면 동작 검증
- [ ] mcts
- [ ] selfplay
- [ ] train
