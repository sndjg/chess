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
  mcts/       # AlphaZero 스타일 MCTS 최소 버전 (PUCT + network eval + backup)
    node.py       # Node: edge(prior/visit_count/value_sum) 통계, transposition table 없음
    search.py     # run(board, model, num_simulations) -> {visit_counts, root_value}
  rollout/    # 대국 데이터 생성. self-play(양쪽 동일 정책)는 이 안의 특수 케이스 —
              # 서로 다른 두 정책끼리 붙는 대국(스타일 모방/대련용 등)도 지원할 수 있게 설계
    policy.py     # Policy 프로토콜 + RandomPolicy, NetworkPolicy
    online_value_policy.py  # OnlineValuePolicy: 판마다 실제 결과로 policy+value head를 함께 온라인 학습 (재현성 의도적 포기)
    replay_buffer.py  # ReplayBuffer: 여러 판의 포지션을 모아뒀다가 배치로 샘플링(고정 크기 원형 버퍼)
    game.py       # play_game(policy_white, policy_black) -> GameRecord
    game_record.py  # GameRecord: UCI 수 목록 저장 + ply별 FEN 계산
  eval/       # 학습 진행을 상대적으로 평가(체크포인트끼리 대국) — docs/IDEAS.md '실력 측정 문제' 참고
    arena.py      # play_match(): N판 대국(색 50/50 교대), find_new_frontier(): 새 체크포인트가
                  # 과거 체크포인트(다른 family 포함) 중 어디까지 이기는지 frontier 추적
  train/      # 학습 루프 (미구현)
  configs/    # ExperimentConfig용 yaml 설정
  config.py   # ExperimentConfig dataclass (yaml 로드/저장)
  utils/
    repro.py  # seed 고정, git commit hash/dirty tree 체크, pip freeze 스냅샷
    run.py    # run 디렉토리 생성 + 재현성 메타데이터 저장
    checkpoint.py  # OnlineValuePolicy 체크포인트 저장/조회/로드 + family_meta.json(학습 방식/git
                   # commit/시작·마지막 갱신 시각) — 아래 '체크포인트 & 상대 평가' 참고
  viz/        # self-play 대국 리플레이용 로컬 웹 UI (FastAPI + vanilla JS)
tests/
scripts/
  smoke_run.py           # 재현성 인프라 자체의 end-to-end 스모크 테스트
  generate_sample_game.py  # viz 검증용 무작위 대국 생성 -> games/sample.json
  serve_viz.py           # viz 로컬 서버 실행 (기본 포트 8000)
runs/           # (gitignore) run별 checkpoints/tensorboard/meta
games/          # (gitignore) 리플레이용 게임 기록(JSON)
checkpoints/    # (gitignore) OnlineValuePolicy 체크포인트, family(학습 계보)별 하위 디렉터리
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

## 체크포인트 & 상대 평가 (OnlineValuePolicy)
- `OnlineValuePolicy(checkpoint_dir=..., family=..., training_method=...)`를 주면 `checkpoint_every`(기본 1)판마다 `checkpoint_dir/family/game_{n:06d}.pt`에 모델 전체를 저장한다(`chess_rl/utils/checkpoint.py`, state_dict 대신 모델 통째로 pickle — 이 정책은 이미 재현성을 포기했으므로 로드 단순함을 우선).
- **family = 학습 계보 식별자**. 목적이 "이 계보가 저 계보(예: 사람과 같이 학습 vs 순수 self-play) 중 어디까지 이기는지" 상대 비교이지, 같은 계보 안에서의 진행 비교가 아니라서 games_trained만으로는 계보 구분이 안 됨 — `list_checkpoints()`가 디렉터리 이름을 family로 채워 넣어 결과에 항상 딸려 나오게 함.
- `family` 디렉터리에는 `family_meta.json`(`FamilyMeta`: family, method, git_commit, started_at, last_updated_at)도 같이 저장됨 — `write_family_meta`가 최초 1회 생성(git commit hash + 시작 시각 기록), `learn_from_game`이 checkpoint를 저장할 때마다 `touch_family_meta`로 `last_updated_at` 갱신(온라인 정책은 서버가 살아있는 한 계속 학습하므로 명시적 "종료 시점"은 없음 — 오래 안 갱신됐으면 사실상 중단된 것으로 간주하는 용도).
- **같은 family 이름을 재사용하면 에러**: `OnlineValuePolicy.__init__`이 해당 family 디렉터리에 이미 checkpoint가 있으면 `ValueError`를 던짐 — 안 그러면 games_trained가 다시 1부터 시작돼 기존 파일을 덮어씀(서버 재시작 시 학습 계보가 완전히 새로 시작되는 것과 맞물린 위험). 재시작 후 다시 checkpoint를 쌓으려면 매번 다른 family 이름을 줘야 함.
- viz 서버의 `learning` 정책은 `family="human_online_<프로세스 시작 시각>"`(예: `human_online_20260712T071200`), `checkpoint_every=1`로 등록됨(`chess_rl/viz/server.py`) — 재시작마다 완전히 새 계보가 시작되는 것이므로 매번 고유한 family가 되도록 시각을 붙임(고정 이름을 재사용하면 위 가드에 걸려 서버가 시작 시 죽는다).
- `chess_rl/eval/arena.py`: `play_match(model_a, model_b, num_games=100, ...)`가 색 50/50 교대로 대국시켜 승/패/무 집계(무승부·max_moves 내 미종료는 0.5점 처리), `find_new_frontier(new_model, old_checkpoints, start_idx, ...)`가 새 체크포인트가 (보통 다른 family의) 과거 체크포인트 중 어디까지 이기는지, 직전 frontier에서 시작해 이기면 위로/지면 아래로 걸어가며 추적(전수/이분 탐색 아님).
- **실시간 비교 시각화(`/play` 화면)**: `server.py`가 `learning` 정책이 새 checkpoint를 찍을 때마다(매판, checkpoint_every=1) 백그라운드 스레드로 `_run_comparison`을 띄워, 가장 최근에 시작된 *다른* family의 최신 checkpoint와 100판(rigorous, mcts_simulations=200) 붙인다. 결과는 `comparison_state`(스레드 락으로 보호)에 저장되고 `GET /api/comparison`으로 조회, `/play` 화면이 5초 간격으로 폴링해 보드 아래 패널에 표시(`static/play.js`의 `pollComparison`). 이미 갱신 중이면 새 checkpoint가 나와도 건너뛴다(안 쌓이게). 실시간 대국과 같은 GPU를 공유해서 부하가 얼마나 되는지는 아직 실측 전 — `docs/IDEAS.md` 성능 TODO 참고.

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
- [x] OnlineValuePolicy: 판마다 실제 결과로 policy+value head를 함께 학습하는 재미용 AI, viz에 화살표/게이지/차트/loss 비교 시각화 (재현성 의도적 포기). select_move는 MCTS 방문분포 기반(deterministic=사람 대국용 argmax / stochastic=평가 대국용 샘플링), 학습은 replay buffer에서 샘플링한 배치로.
- [~] mcts: `chess_rl/mcts/`에 최소 버전 구현됨 — PUCT selection + network 기반 expansion/evaluation + backup, `run(board, model, num_simulations)`이 방문분포+root value 반환. transposition table 없음, temperature/Dirichlet noise 없음(self-play용, 아직 미구현), 성능 최적화(배치 leaf 평가 등)는 프로파일링 후 결정 예정(`docs/IDEAS.md`). MCTS 탐색 target으로 policy head를 학습시키는 통합은 아직 안 함(policy/value 학습은 여전히 REINFORCE/MSE).
- [x] eval: `chess_rl/eval/arena.py` + `chess_rl/utils/checkpoint.py` — OnlineValuePolicy 체크포인트를 family(학습 계보)별로 저장하고, 새 체크포인트가 (다른 family 포함) 과거 체크포인트 중 어디까지 이기는지 frontier로 추적하는 상대 평가. 위 '체크포인트 & 상대 평가' 참고.
- [ ] train
