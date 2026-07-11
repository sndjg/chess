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
  configs/    # 하이퍼파라미터 설정
tests/
scripts/
```

## 진행 상황
- [x] repo 구조, pyproject.toml, conda 환경(`chess`) 세팅
- [x] engine: board encoding (12,8,8 plane), 고정 action space (64x64 + underpromotion)
- [x] model: policy+value ResNet
- [ ] mcts
- [ ] selfplay
- [ ] train
