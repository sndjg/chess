# chess_rl

AlphaZero 스타일 self-play 체스 강화학습 호비 프로젝트.

## 환경

```bash
conda activate chess
pip install -e ".[dev]"
```

GPU(CUDA) 환경에서는 PyPI의 torch wheel이 CUDA 런타임을 함께 받아오므로 별도 index-url 없이 `pip install -e .`만으로 GPU가 인식된다.

## 구조

```
chess_rl/
  engine/     # python-chess wrapper: board encoding, action space
  model/      # policy+value network (ResNet)
  mcts/       # MCTS 탐색 (구현 예정)
  selfplay/   # self-play 데이터 생성 (구현 예정)
  train/      # 학습 루프 (구현 예정)
  configs/    # 하이퍼파라미터 설정
tests/
scripts/
```

## 현재 상태

- `engine`: board -> (12, 8, 8) plane encoding, 고정 action space(64x64 기반 + underpromotion) 완료.
- `model`: policy+value ResNet 완료.
- `mcts`, `selfplay`, `train`: 아직 미구현 — 다음 단계에서 설계.
